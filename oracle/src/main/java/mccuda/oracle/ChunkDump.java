package mccuda.oracle;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Writer;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.google.gson.stream.JsonWriter;

import net.minecraft.core.Holder;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunctions;
import net.minecraft.world.level.levelgen.densityfunction.DensitySampler;
import net.minecraft.world.level.levelgen.densityfunction.DensityVolume;
import net.minecraft.world.level.levelgen.densityfunction.ScopedDensityBuffer;

/**
 * Level-1/2/3 chunk reference data.
 *
 * L2 final density: exactly what NoiseBasedChunkGenerator.doFill reads, i.e. the
 *    final density sampled over the 16 x height x 16 chunk volume through a
 *    NoiseChunk's caching samplers (Beardifier.EMPTY, empty blender).
 * L3 block fill: doFill's substance decision for the core configuration, i.e.
 *    vanilla settings with aquifers disabled. doFill's loop body is
 *    aquifer.computeSubstance(...) falling back to the default block; there is no
 *    arithmetic in it, so it is reproduced here around vanilla's own aquifer and
 *    fluid picker (obtained by reflection from NoiseBasedChunkGenerator).
 * L1 named functions: each registry-named density function referenced by the
 *    final density, sampled over the chunk volume in the same kind of context.
 *
 * Layout of every per-chunk array: index = ((y - minY) * 16 + z) * 16 + x.
 */
final class ChunkDump {
    static final String[] PALETTE = {"minecraft:air", "minecraft:stone", "minecraft:water", "minecraft:lava"};

    static void dump(Main.Vanilla v, Path root, boolean quickOnly) throws Exception {
        NoiseGeneratorSettings stock = v.overworld;
        NoiseGeneratorSettings core = coreSettings(stock);
        Aquifer.FluidPicker picker = fluidPicker(core);
        int minY = core.noiseSettings().minY(), height = core.noiseSettings().height();
        int perChunk = 16 * 16 * height;

        Map<String, DensityFunction> named = new LinkedHashMap<>();
        collectNamed(stock.noiseRouter().finalDensity(), named, new IdentityHashMap<>());

        Files.createDirectories(root.resolve("density"));
        Files.createDirectories(root.resolve("fill"));
        Files.createDirectories(root.resolve("nodes"));
        List<Map<String, Object>> manifestSeeds = new ArrayList<>();
        Map<String, String> hashes = new LinkedHashMap<>();
        long totalChunks = 0, selfCheckMismatches = 0;
        long t0 = System.nanoTime();

        for (Corpus.Seed seed : Corpus.seeds()) {
            if (quickOnly && !seed.quick()) continue;
            List<Corpus.Chunk> chunks = Corpus.chunks(seed, quickOnly);
            RandomState rsCore = RandomState.create(v.noises, seed.seed(), core);
            RandomState rsStock = RandomState.create(v.noises, seed.seed(), stock);
            ByteBuffer density = ByteBuffer.allocate(chunks.size() * perChunk * 4).order(ByteOrder.LITTLE_ENDIAN);
            byte[] fill = new byte[chunks.size() * perChunk];
            List<String> nodeDumps = new ArrayList<>();

            for (int ci = 0; ci < chunks.size(); ci++) {
                Corpus.Chunk c = chunks.get(ci);
                DensityVolume vol = new DensityVolume(16, height, 16, c.cx() * 16, minY, c.cz() * 16);
                float[] dens = new float[perChunk];
                try (NoiseChunk nc = new NoiseChunk(rsCore, Beardifier.EMPTY, core, picker, Blender.empty(), vol)) {
                    DensitySampler.Bound fd = nc.cachingSamplers().get(core.noiseRouter().finalDensity());
                    Aquifer aquifer = nc.aquifer();
                    BlockState defaultState = core.defaultBlock();
                    try (ScopedDensityBuffer buf = fd.sampleVolume(vol)) {
                        for (int z = 0; z < 16; z++) {
                            for (int x = 0; x < 16; x++) {
                                for (int y = height - 1; y >= 0; y--) {
                                    float d = buf.get(vol.indexUnchecked(x, y, z));
                                    int idx = (y * 16 + z) * 16 + x;
                                    dens[idx] = d;
                                    BlockState state = aquifer.computeSubstance(vol.blockX(x), vol.blockY(y), vol.blockZ(z), d);
                                    if (state == null) state = defaultState;
                                    fill[ci * perChunk + idx] = paletteIndex(state);
                                }
                            }
                        }
                    }
                }
                for (int i = 0; i < perChunk; i++) density.putFloat((ci * perChunk + i) * 4, dens[i]);

                // Self-check: the stock settings (aquifers enabled) must give the same final density.
                try (NoiseChunk nc = new NoiseChunk(rsStock, Beardifier.EMPTY, stock, picker, Blender.empty(), vol)) {
                    try (ScopedDensityBuffer buf = nc.cachingSamplers().get(stock.noiseRouter().finalDensity()).sampleVolume(vol)) {
                        for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) for (int y = 0; y < height; y++) {
                            float d = buf.get(vol.indexUnchecked(x, y, z));
                            if (Float.floatToRawIntBits(d) != Float.floatToRawIntBits(dens[(y * 16 + z) * 16 + x])) selfCheckMismatches++;
                        }
                    }
                }

                // L1: named functions for the first, a border and a wrap chunk of the quick seeds.
                if (seed.quick() && (ci == 4 || c.tag().equals("border") || c.tag().equals("wrap"))) {
                    Path nd = root.resolve("nodes").resolve("s" + seed.index() + "_c" + ci);
                    Files.createDirectories(nd);
                    try (NoiseChunk nc = new NoiseChunk(rsCore, Beardifier.EMPTY, core, picker, Blender.empty(), vol)) {
                        for (Map.Entry<String, DensityFunction> e : named.entrySet()) {
                            ByteBuffer nb = ByteBuffer.allocate(perChunk * 4).order(ByteOrder.LITTLE_ENDIAN);
                            try (ScopedDensityBuffer buf = nc.cachingSamplers().get(e.getValue()).sampleVolume(vol)) {
                                for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) for (int y = 0; y < height; y++) {
                                    nb.putFloat(((y * 16 + z) * 16 + x) * 4, buf.get(vol.indexUnchecked(x, y, z)));
                                }
                            }
                            Path f = nd.resolve(fileName(e.getKey()) + ".f32");
                            Files.write(f, nb.array());
                            hashes.put(root.relativize(f).toString().replace('\\', '/'), sha256(nb.array()));
                        }
                    }
                    nodeDumps.add("s" + seed.index() + "_c" + ci);
                }
                totalChunks++;
            }

            Path df = root.resolve("density").resolve("s" + seed.index() + ".f32");
            Path ff = root.resolve("fill").resolve("s" + seed.index() + ".u8");
            Files.write(df, density.array());
            Files.write(ff, fill);
            hashes.put(root.relativize(df).toString().replace('\\', '/'), sha256(density.array()));
            hashes.put(root.relativize(ff).toString().replace('\\', '/'), sha256(fill));

            Map<String, Object> ms = new LinkedHashMap<>();
            ms.put("index", seed.index());
            ms.put("seed", Long.toString(seed.seed()));
            ms.put("quick", seed.quick());
            ms.put("chunks", chunks);
            ms.put("node_dumps", nodeDumps);
            manifestSeeds.add(ms);
            System.out.printf("dump-chunks: seed %d (%d) %d chunks%n", seed.index(), seed.seed(), chunks.size());
        }

        if (selfCheckMismatches != 0) {
            throw new IllegalStateException("self-check failed: aquifer-disabled settings changed final density in " + selfCheckMismatches + " blocks");
        }
        writeManifest(root.resolve("manifest.json"), manifestSeeds, named, hashes, minY, height, quickOnly);
        System.out.printf("dump-chunks: %d chunks in %.1f s, self-check (stock vs aquifer-disabled final density) 0 mismatches%n",
            totalChunks, (System.nanoTime() - t0) / 1e9);
    }

    /** Vanilla overworld settings with aquifers disabled; everything else is shared with the stock record. */
    static NoiseGeneratorSettings coreSettings(NoiseGeneratorSettings s) {
        return new NoiseGeneratorSettings(s.noiseSettings(), s.defaultBlock(), s.defaultFluid(), s.noiseRouter(), s.materialRule(), s.spawnTarget(),
            s.seaLevel(), s.disableMobGeneration(), Optional.empty(), s.useLegacyRandomSource(), s.debugFunctions());
    }

    /** Vanilla's global fluid picker (lava below y=-54, default fluid below sea level). */
    static Aquifer.FluidPicker fluidPicker(NoiseGeneratorSettings settings) throws ReflectiveOperationException {
        Method m = NoiseBasedChunkGenerator.class.getDeclaredMethod("createFluidPicker", NoiseGeneratorSettings.class);
        m.setAccessible(true);
        return (Aquifer.FluidPicker) m.invoke(null, settings);
    }

    static byte paletteIndex(BlockState state) {
        if (state.is(Blocks.AIR)) return 0;
        if (state.is(Blocks.STONE)) return 1;
        if (state.is(Blocks.WATER)) return 2;
        if (state.is(Blocks.LAVA)) return 3;
        throw new IllegalStateException("unexpected block state in core fill: " + state);
    }

    static void collectNamed(DensityFunction f, Map<String, DensityFunction> out, Map<DensityFunction, Boolean> seen) {
        if (seen.put(f, Boolean.TRUE) != null) return;
        if (f instanceof DensityFunctions.HolderHolder(Holder<DensityFunction> holder)) {
            holder.unwrapKey().ifPresent(key -> out.putIfAbsent(key.identifier().toString(), holder.value()));
            collectNamed(holder.value(), out, seen);
            return;
        }
        f.rewriteChildren(child -> { collectNamed(child, out, seen); return child; });
    }

    static String fileName(String name) {
        return name.replace(':', '_').replace('/', '.');
    }

    static String sha256(byte[] data) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
    }

    private static void writeManifest(Path out, List<Map<String, Object>> seeds, Map<String, DensityFunction> named, Map<String, String> hashes,
                                      int minY, int height, boolean quick) throws IOException {
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
            json.setIndent(" ");
            json.beginObject();
            json.name("format").value("mccuda-oracle/1");
            json.name("minecraft").value(Main.MC_VERSION);
            json.name("java").value(System.getProperty("java.version"));
            json.name("quick").value(quick);
            json.name("min_y").value(minY);
            json.name("height").value(height);
            json.name("layout").value("index = ((y - min_y) * 16 + z) * 16 + x; density float32 LE, fill uint8 palette index");
            json.name("palette").beginArray();
            for (String p : PALETTE) json.value(p);
            json.endArray();
            json.name("named_functions").beginArray();
            for (String n : named.keySet()) json.beginObject().name("name").value(n).name("file").value(fileName(n) + ".f32").endObject();
            json.endArray();
            json.name("seeds").beginArray();
            for (Map<String, Object> s : seeds) {
                json.beginObject();
                json.name("index").value((int) s.get("index"));
                json.name("seed").value((String) s.get("seed"));
                json.name("quick").value((boolean) s.get("quick"));
                json.name("chunks").beginArray();
                @SuppressWarnings("unchecked") List<Corpus.Chunk> chunks = (List<Corpus.Chunk>) s.get("chunks");
                for (Corpus.Chunk c : chunks) json.beginArray().value(c.cx()).value(c.cz()).value(c.tag()).endArray();
                json.endArray();
                json.name("node_dumps").beginArray();
                @SuppressWarnings("unchecked") List<String> nd = (List<String>) s.get("node_dumps");
                for (String n : nd) json.value(n);
                json.endArray();
                json.endObject();
            }
            json.endArray();
            json.name("sha256").beginObject();
            for (Map.Entry<String, String> e : hashes.entrySet()) json.name(e.getKey()).value(e.getValue());
            json.endObject();
            json.endObject();
        }
    }

    static void writeAll(OutputStream os, byte[] b) throws IOException {
        os.write(b);
    }
}
