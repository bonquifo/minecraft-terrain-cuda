package mccuda.oracle;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import net.minecraft.world.level.levelgen.densityfunction.DensityVolume;
import net.minecraft.world.level.levelgen.densityfunction.ScopedDensityBuffer;

/**
 * Vanilla throughput for the same work our GPU pipeline does: per chunk, the
 * final density over the chunk volume plus doFill's substance decision (core
 * configuration, aquifers disabled). Each worker thread uses its own NoiseChunk,
 * exactly like vanilla's buildTerrain tasks; the RandomState is shared, as in a
 * server. Output: one CSV row per timed repetition.
 *
 * args: threads=<n> region=<chunks per side> reps=<n> warmup=<n> seed=<long> x0=<chunk> z0=<chunk> tag=<name>
 */
final class VanillaBench {
    static void run(Main.Vanilla v, List<String> args, Path outDir) throws Exception {
        int threads = intArg(args, "threads", 1);
        int region = intArg(args, "region", 32);
        int reps = intArg(args, "reps", 10);
        int warmup = intArg(args, "warmup", 3);
        long seed = Long.parseLong(strArg(args, "seed", "0"));
        int x0 = intArg(args, "x0", 0), z0 = intArg(args, "z0", 0);
        String tag = strArg(args, "tag", "vanilla");

        NoiseGeneratorSettings core = ChunkDump.coreSettings(v.overworld);
        Aquifer.FluidPicker picker = ChunkDump.fluidPicker(core);
        RandomState rs = RandomState.create(v.noises, seed, core);
        int minY = core.noiseSettings().minY(), height = core.noiseSettings().height();
        int total = region * region;

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<double[]> rows = new ArrayList<>();
        long checksum = 0;
        for (int rep = -warmup; rep < reps; rep++) {
            AtomicInteger next = new AtomicInteger();
            long start = System.nanoTime();
            List<Future<Long>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(pool.submit(() -> {
                    long solid = 0;
                    int i;
                    while ((i = next.getAndIncrement()) < total) {
                        int cx = x0 + i % region, cz = z0 + i / region;
                        DensityVolume vol = new DensityVolume(16, height, 16, cx * 16, minY, cz * 16);
                        try (NoiseChunk nc = new NoiseChunk(rs, Beardifier.EMPTY, core, picker, Blender.empty(), vol);
                             ScopedDensityBuffer buf = nc.cachingSamplers().get(core.noiseRouter().finalDensity()).sampleVolume(vol)) {
                            Aquifer aquifer = nc.aquifer();
                            for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) for (int y = height - 1; y >= 0; y--) {
                                BlockState s = aquifer.computeSubstance(vol.blockX(x), vol.blockY(y), vol.blockZ(z), buf.get(vol.indexUnchecked(x, y, z)));
                                if (s == null) solid++;
                            }
                        }
                    }
                    return solid;
                }));
            }
            long solid = 0;
            for (Future<Long> f : futures) solid += f.get();
            double seconds = (System.nanoTime() - start) / 1e9;
            if (rep >= 0) {
                rows.add(new double[] {rep, seconds, total / seconds});
                System.out.printf("bench-vanilla %s threads=%d rep %d: %.3f s, %.1f chunks/s%n", tag, threads, rep, seconds, total / seconds);
            }
            checksum = solid;
        }
        pool.shutdown();

        Files.createDirectories(outDir);
        Path csv = outDir.resolve(tag + "_t" + threads + ".csv");
        try (Writer w = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            w.write("impl,threads,region,seed,x0,z0,rep,seconds,chunks_per_s,solid_blocks\n");
            for (double[] r : rows) {
                w.write(String.format("%s,%d,%d,%d,%d,%d,%d,%.6f,%.3f,%d%n", tag, threads, region, seed, x0, z0, (int) r[0], r[1], r[2], checksum));
            }
        }
        System.out.println("bench-vanilla: wrote " + csv);
    }

    static int intArg(List<String> args, String key, int def) {
        return Integer.parseInt(strArg(args, key, Integer.toString(def)));
    }

    static String strArg(List<String> args, String key, String def) {
        for (String a : args) if (a.startsWith(key + "=")) return a.substring(key.length() + 1);
        return def;
    }

    private VanillaBench() {}

    static void unused() throws IOException {}
}
