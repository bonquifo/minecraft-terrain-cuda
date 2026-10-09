package mccuda.client;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.lwjgl.system.MemoryUtil;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.platform.NativeImage;

import net.fabricmc.api.ClientModInitializer;
import net.minecraft.client.CloudStatus;
import net.minecraft.client.InactivityFpsLimit;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Options;
import net.minecraft.client.Screenshot;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.tutorial.TutorialSteps;
import net.minecraft.world.Difficulty;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPresets;

/**
 * Records the real Minecraft client flying along the GPU flythrough's camera path
 * (-Dmccuda.path) through a fresh world whose terrain inside the dumped region comes
 * from the GPU (see GpuDump and the worldgen mixins).
 *
 * Title screen -> create the world (spectator, same seed) -> pre-pass along the path so
 * the server generates every chunk in view -> back to the start, wait until everything
 * in view is meshed -> record: one video frame per rendered frame at t = k / 30 s (so a
 * slow frame never skips path time), piped to ffmpeg -> quit.
 */
public final class Recorder implements ClientModInitializer {
    private enum State { OFF, BOOT, LOADING, PREPASS, RETURN, RECORD, FINISH, DONE }

    public static final int FPS = 30, RENDER_DISTANCE = 24;
    private static final String WORLD = "mccuda-gpu-terrain";
    private static final String[] COMMANDS = {"time set noon", "weather clear", "gamerule advance_time false", "gamerule advance_weather false",
        "gamerule spawn_mobs false"};

    /** Structures change the terrain around them (the Beardifier), which the GPU pipeline leaves out; off unless -Dmccuda.structures=true. */
    private static final boolean STRUCTURES = Boolean.getBoolean("mccuda.structures");
    private static State state = State.OFF;
    private static CameraPath path;
    private static Path out;
    private static int frames, frame, posedFrame = -1, settle, stall, ticks;
    private static double prepassT;
    private static double[] pose;
    private static FrameSink sink;
    private static long phaseStart, prepassMs, settleMs, recordStart;
    private static final List<String> frameLog = new ArrayList<>();

    @Override
    public void onInitializeClient() {
        String p = System.getProperty("mccuda.path");
        if (p == null) return;
        try {
            path = new CameraPath(Path.of(p));
        } catch (IOException e) {
            throw new RuntimeException("cannot read camera path " + p, e);
        }
        out = Path.of(System.getProperty("mccuda.out"));
        frames = (int) Math.ceil(path.duration() * FPS) + 1;
        state = State.BOOT;
        log("recorder armed: %d frames along %s -> %s", frames, p, out);
    }

    static void log(String fmt, Object... args) {
        System.out.println("[mccuda] " + String.format(Locale.ROOT, fmt, args));
    }

    private static void configure(Options o) {
        o.renderDistance().set(RENDER_DISTANCE);
        o.fov().set(70);
        o.bobView().set(false);
        o.cloudStatus().set(CloudStatus.OFF);   // clouds drift with game time, which differs between runs
        o.enableVsync().set(false);
        o.framerateLimit().set(FPS);
        o.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
        o.pauseOnLostFocus = false;
        o.tutorialStep = TutorialSteps.NONE;
    }

    private static void deleteWorld(Path dir) {
        if (!Files.exists(dir)) return;
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path f : s.sorted(Comparator.reverseOrder()).toList()) Files.delete(f);
        } catch (IOException e) {
            throw new RuntimeException("cannot delete old world " + dir, e);
        }
    }

    /** Fraction of the chunks within (render distance - 2) of (x, z) that the client has received. */
    private static double loadedFraction(Minecraft mc, double x, double z) {
        int r = RENDER_DISTANCE - 2, cx = (int) Math.floor(x) >> 4, cz = (int) Math.floor(z) >> 4, have = 0, all = 0;
        for (int dz = -r; dz <= r; dz++) {
            for (int dx = -r; dx <= r; dx++) {
                if (dx * dx + dz * dz > r * r) continue;
                all++;
                if (mc.level.getChunkSource().getChunk(cx + dx, cz + dz, ChunkStatus.FULL, false) != null) have++;
            }
        }
        return (double) have / all;
    }

    /** Chunks the client holds (anything it can draw) that lie outside the dumped region. */
    private static int outsideRegion(Minecraft mc, GpuDump dump) {
        int r = RENDER_DISTANCE + 3, cx = (int) Math.floor(pose[0]) >> 4, cz = (int) Math.floor(pose[2]) >> 4, n = 0;
        for (int dz = -r; dz <= r; dz++)
            for (int dx = -r; dx <= r; dx++)
                if (!dump.covers(cx + dx, cz + dz) && mc.level.getChunkSource().getChunk(cx + dx, cz + dz, ChunkStatus.FULL, false) != null) n++;
        return n;
    }

    private static void place(Minecraft mc, double x, double y, double z, float yRot, float xRot) {
        mc.player.snapTo(x, y, z, yRot, xRot);
        mc.player.setDeltaMovement(0, 0, 0);
    }

    public static void tick(Minecraft mc) {
        if (state == State.OFF || state == State.DONE) return;
        GpuDump dump = GpuDump.get();
        ticks++;
        switch (state) {
            case BOOT -> {
                if (mc.gui.overlay() != null || !(mc.gui.screen() instanceof TitleScreen)) return;
                configure(mc.options);
                deleteWorld(mc.gameDirectory.toPath().resolve("saves").resolve(WORLD));
                LevelSettings settings = new LevelSettings(WORLD, GameType.SPECTATOR, new LevelSettings.DifficultySettings(Difficulty.PEACEFUL, false, false), true,
                    WorldDataConfiguration.DEFAULT);
                log("creating world '%s', seed %d, structures %s", WORLD, dump.seed, STRUCTURES ? "on" : "off");
                state = State.LOADING;
                mc.createWorldOpenFlows().createFreshLevel(WORLD, settings, new WorldOptions(dump.seed, STRUCTURES, false), WorldPresets::createNormalWorldDimensions,
                    mc.gui.screen());
            }
            case LOADING -> {
                if (mc.level == null || mc.player == null || mc.gui.screen() != null) return;
                for (String c : COMMANDS) mc.getConnection().sendCommand(c);
                if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
                state = State.PREPASS;
                prepassT = 0;
                phaseStart = System.nanoTime();
                log("in the world; pre-pass along the path");
            }
            case PREPASS -> {
                CameraPath.Sample s = path.sample(prepassT);
                place(mc, s.x(), 200, s.z(), 0, 90);
                double f = loadedFraction(mc, s.x(), s.z());
                // never faster than real time; wait for the server to catch up, but not forever
                if (f >= 0.97 || ++stall > 400) {
                    prepassT += 1.0 / 20;
                    stall = 0;
                }
                if (ticks % 40 == 0)
                    log("pre-pass t=%.2f/%.1f loaded=%.3f gpu chunks=%d gpu biomes=%d vanilla chunks=%d verified: %d biome chunks (%d bad), %d block chunks (%d bad; %d with structure adaptation, %d of them differ; %d without differ)",
                        prepassT, path.duration(), f, dump.servedBlocks.get(), dump.servedBiomes.get(), dump.vanillaChunks.get(), dump.checkedBiomeChunks.get(),
                        dump.biomeMismatches.get(), dump.checkedBlockChunks.get(), dump.blockMismatches.get(), dump.adaptedChunks.get(), dump.adaptedBadChunks.get(), dump.plainBadChunks.get());
                if (prepassT > path.duration()) {
                    prepassMs = (System.nanoTime() - phaseStart) / 1_000_000;
                    phaseStart = System.nanoTime();
                    state = State.RETURN;
                    settle = 0;
                    stall = 0;
                    log("pre-pass done in %.1f s; back to the start", prepassMs / 1000.0);
                }
            }
            case RETURN -> {
                CameraPath.Sample s = path.sample(0);
                place(mc, s.x(), dump.ground(s.x(), s.z()), s.z(), 0, 0);
                double f = loadedFraction(mc, s.x(), s.z());
                settle = f >= 0.999 && mc.levelRenderer.hasRenderedAllSections() ? settle + 1 : 0;
                if (ticks % 40 == 0) log("settling: loaded=%.4f all sections rendered=%b", f, mc.levelRenderer.hasRenderedAllSections());
                if (settle >= 60 || ++stall > 2400) {
                    settleMs = (System.nanoTime() - phaseStart) / 1_000_000;
                    path.reset();
                    frame = 0;
                    posedFrame = -1;
                    sink = new FrameSink(out);
                    // stop entities and fluids, so that two runs show the same world state
                    mc.getConnection().sendCommand("tick freeze");
                    recordStart = System.nanoTime();
                    state = State.RECORD;
                    log("recording %d frames (settled in %.1f s%s)", frames, settleMs / 1000.0, settle >= 60 ? "" : ", timed out");
                }
            }
            case RECORD -> {
                if (pose != null) place(mc, pose[0], pose[1] - 1.62, pose[2], (float) pose[3], (float) pose[4]);
            }
            case FINISH -> {
                if (!sink.drained()) return;
                sink.close();
                writeStats(dump);
                state = State.DONE;
                log("done: %d frames -> %s", frames, out);
                mc.stop();
            }
            default -> {}
        }
    }

    /** The camera pose for the frame being rendered, or null when not recording. */
    public static double[] cameraPose() {
        if (state != State.RECORD) return null;
        if (posedFrame != frame) {
            GpuDump dump = GpuDump.get();
            pose = path.pose((double) frame / FPS, dump::ground, 1.0 / FPS);
            posedFrame = frame;
        }
        return pose;
    }

    public static void onWorldRendered(GameRenderer renderer) {
        if (state != State.RECORD || posedFrame != frame) return;
        Minecraft mc = Minecraft.getInstance();
        GpuDump dump = GpuDump.get();
        frameLog.add(String.format(Locale.ROOT, "%d,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d,%.1f", frame, (double) frame / FPS, pose[0], pose[1], pose[2], pose[3],
            pose[4], mc.getFps(), mc.level.getChunkSource().getLoadedChunksCount(), outsideRegion(mc, dump), dump.servedBlocks.get(), dump.servedBiomes.get(),
            dump.vanillaChunks.get(),
            (System.nanoTime() - recordStart) / 1e6));
        sink.capture(renderer.mainRenderTarget());
        frame++;
        if (frame >= frames) state = State.FINISH;
    }

    private static void writeStats(GpuDump dump) {
        try {
            Files.writeString(out.resolveSibling(name(out) + ".csv"),
                "frame,t,x,y,z,yrot,xrot,client_fps,client_chunks,client_chunks_outside_region,gpu_chunks,gpu_biomes,vanilla_chunks,wall_ms\n" + String.join("\n", frameLog) + "\n",
                StandardCharsets.UTF_8);
            String json = String.format(Locale.ROOT,
                "{\"served_from_gpu\": %b, \"frames\": %d, \"fps\": %d, \"width\": %d, \"height\": %d, \"render_distance\": %d, \"seed\": %d, \"region\": {\"x0\": %d, \"z0\": %d, \"size\": %d},"
                    + " \"gpu_chunks\": %d, \"gpu_biomes\": %d, \"vanilla_chunks\": %d, \"prepass_s\": %.1f, \"settle_s\": %.1f, \"record_s\": %.1f,"
                    + " \"verify\": {\"biome_chunks\": %d, \"biomes\": %d, \"biome_mismatches\": %d, \"block_chunks\": %d, \"blocks\": %d, \"block_mismatches\": %d,"
                    + " \"structure_adapted_chunks\": %d, \"structure_adapted_chunks_differing\": %d, \"structure_adapted_mismatches\": %d, \"other_chunks_differing\": %d},"
                    + " \"structures\": %b}\n",
                dump.serve, frames, FPS, sink.width, sink.height, RENDER_DISTANCE, dump.seed, dump.x0, dump.z0, dump.region, dump.servedBlocks.get(), dump.servedBiomes.get(),
                dump.vanillaChunks.get(), prepassMs / 1000.0, settleMs / 1000.0, (System.nanoTime() - recordStart) / 1e9, dump.checkedBiomeChunks.get(),
                dump.checkedBiomes.get(), dump.biomeMismatches.get(), dump.checkedBlockChunks.get(), dump.checkedBlocks.get(), dump.blockMismatches.get(), dump.adaptedChunks.get(), dump.adaptedBadChunks.get(),
                dump.adaptedMismatches.get(), dump.plainBadChunks.get(), STRUCTURES);
            Files.writeString(out.resolveSibling(name(out) + ".json"), json, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String name(Path p) {
        String n = p.getFileName().toString();
        int dot = n.lastIndexOf('.');
        return dot < 0 ? n : n.substring(0, dot);
    }

    /** Frames read back from the GPU in submission order and piped to ffmpeg (lossless-ish H.264) on a writer thread. */
    static final class FrameSink {
        private final Path file;
        private final BlockingQueue<byte[]> queue = new ArrayBlockingQueue<>(24);
        private final AtomicInteger written = new AtomicInteger();
        private int submitted;
        private Process ffmpeg;
        private Thread writer;
        int width, height;

        FrameSink(Path file) {
            this.file = file;
        }

        void capture(RenderTarget target) {
            submitted++;
            Screenshot.takeScreenshot(target, this::accept);
        }

        private void accept(NativeImage img) {
            try (img) {
                if (ffmpeg == null) start(img.getWidth(), img.getHeight());
                if (img.getWidth() != width || img.getHeight() != height) throw new IllegalStateException("frame size changed during recording");
                byte[] rgba = new byte[width * height * 4];
                MemoryUtil.memByteBuffer(img.getPointer(), rgba.length).get(rgba);
                queue.put(rgba);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        private void start(int w, int h) {
            width = w;
            height = h;
            try {
                Files.createDirectories(file.getParent());
                ffmpeg = new ProcessBuilder("ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", w + "x" + h, "-r", String.valueOf(FPS),
                    "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", file.toString()).redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.INHERIT).start();
            } catch (IOException e) {
                throw new RuntimeException("cannot start ffmpeg", e);
            }
            OutputStream os = ffmpeg.getOutputStream();
            writer = new Thread(() -> {
                try {
                    while (true) {
                        byte[] b = queue.take();
                        if (b.length == 0) break;
                        os.write(b);
                        written.incrementAndGet();
                    }
                    os.close();
                } catch (IOException | InterruptedException e) {
                    log("frame writer failed: %s", e);
                }
            }, "mccuda-frame-writer");
            writer.start();
            log("ffmpeg: %dx%d -> %s", w, h, file);
        }

        boolean drained() {
            return written.get() == submitted;
        }

        void close() {
            try {
                queue.put(new byte[0]);
                writer.join();
                int rc = ffmpeg.waitFor();
                if (rc != 0) throw new IllegalStateException("ffmpeg exited with " + rc);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
