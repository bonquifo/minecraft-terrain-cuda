package mccuda.oracle;

import java.io.Writer;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.IdMapper;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.chunk.PalettedContainerFactory;
import net.minecraft.world.level.chunk.ProtoChunk;
import net.minecraft.world.level.chunk.Strategy;
import net.minecraft.world.level.chunk.UpgradeData;
import net.minecraft.world.level.chunk.status.ChunkStatus;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.Beardifier;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;
import net.minecraft.world.level.levelgen.densityfunction.DensityVolume;
import net.minecraft.world.level.levelgen.material.rule.MaterialRule;

/**
 * Vanilla throughput for the full TERRAIN step without carvers, i.e. the same
 * work as our GPU pipeline, over a square region of chunks:
 *
 *   phase 1  BIOMES: doCreateBiomes for every chunk of the region plus a 1-chunk
 *            border (the border is needed by BiomeManager's zoom at the edges)
 *   phase 2  per region chunk, what NoiseBasedChunkGenerator.buildTerrain does
 *            before carvers: a NoiseChunk, doFill (aquifers on) and buildSurface
 *
 * Both phases run on a fixed thread pool sharing one RandomState, like a server's
 * worldgen workers. The R-tree's per-thread last-result cache is left alone
 * (live-server behaviour) unless reset=1. With dump=1 the region's final blocks
 * of the last repetition are written for cross-checking (needs threads=1 and
 * reset=1 to be comparable with our generators).
 *
 * args: threads=<n> region=<chunks per side> reps=<n> warmup=<n> seed=<long> x0=<chunk> z0=<chunk> tag=<name> [reset=1] [dump=1]
 */
final class TerrainBench {
    static void run(Main.Vanilla v, List<String> args, Path outDir) throws Exception {
        int threads = VanillaBench.intArg(args, "threads", 1);
        int region = VanillaBench.intArg(args, "region", 16);
        int reps = VanillaBench.intArg(args, "reps", 5);
        int warmup = VanillaBench.intArg(args, "warmup", 2);
        long seed = Long.parseLong(VanillaBench.strArg(args, "seed", "0"));
        int x0 = VanillaBench.intArg(args, "x0", 0), z0 = VanillaBench.intArg(args, "z0", 0);
        String tag = VanillaBench.strArg(args, "tag", "vanilla-terrain");
        boolean reset = VanillaBench.intArg(args, "reset", 0) != 0;
        boolean dump = VanillaBench.intArg(args, "dump", 0) != 0;

        TerrainModel model = new TerrainModel(v);
        Path scratch = Files.createTempFile("mccuda_terrain", ".json");
        model.export(scratch);   // completes the palette (same order as overworld_terrain.json)
        Files.deleteIfExists(scratch);
        NoiseGeneratorSettings stock = v.overworld;
        NoiseBasedChunkGenerator generator = model.generator;
        Aquifer.FluidPicker picker = ChunkDump.fluidPicker(stock);
        MaterialRule materialRule = stock.materialRule().value();
        int minY = stock.noiseSettings().minY(), height = stock.noiseSettings().height();

        IdMapper<Holder<Biome>> biomeIds = new IdMapper<>();
        var biomeLookup = v.lookup.lookupOrThrow(Registries.BIOME);
        for (var e : biomeLookup.listElements().toList()) biomeIds.add(e);
        Holder<Biome> plains = biomeLookup.getOrThrow(Biomes.PLAINS);
        PalettedContainerFactory factory = new PalettedContainerFactory(Strategy.createForBlockStates(Block.BLOCK_STATE_REGISTRY),
            Blocks.AIR.defaultBlockState(), null, Strategy.createForBiomes(biomeIds), plains, null);
        LevelHeightAccessor heightAccessor = LevelHeightAccessor.create(minY, height);

        Method doCreateBiomes = ChunkGenerator.class.getDeclaredMethod("doCreateBiomes", Blender.class, RandomState.class, ChunkAccess.class);
        doCreateBiomes.setAccessible(true);
        Method doFill = NoiseBasedChunkGenerator.class.getDeclaredMethod("doFill", NoiseChunk.class, ChunkAccess.class);
        doFill.setAccessible(true);
        Method buildSurface = NoiseBasedChunkGenerator.class.getDeclaredMethod("buildSurface", ChunkAccess.class, NoiseChunk.class, RandomState.class,
            BiomeManager.class, Set.class, MaterialRule.class);
        buildSurface.setAccessible(true);
        ThreadLocal<?> rtreeLastResult = TerrainDump.rtreeLastResult(model.biomeSource);

        RandomState rs = RandomState.create(v.noises, seed, stock);
        long zoomSeed = BiomeManager.obfuscateSeed(seed);
        int border = region + 2;
        int interior = region * region;

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<double[]> rows = new ArrayList<>();
        ConcurrentHashMap<Long, ProtoChunk> chunks = new ConcurrentHashMap<>();
        for (int rep = -warmup; rep < reps; rep++) {
            chunks.clear();
            long start = System.nanoTime();
            // phase 1: biomes for the region plus border
            AtomicInteger next = new AtomicInteger();
            List<Future<?>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(pool.submit(() -> {
                    int i;
                    while ((i = next.getAndIncrement()) < border * border) {
                        ChunkPos pos = new ChunkPos(x0 - 1 + i % border, z0 - 1 + i / border);
                        ProtoChunk proto = new ProtoChunk(pos, UpgradeData.EMPTY, heightAccessor, factory, null);
                        if (reset) rtreeLastResult.remove();
                        doCreateBiomes.invoke(generator, Blender.empty(), rs, proto);
                        proto.setPersistedStatus(ChunkStatus.BIOMES);
                        chunks.put(ChunkPos.pack(pos.x(), pos.z()), proto);
                    }
                    return null;
                }));
            }
            for (Future<?> f : futures) f.get();
            long mid = System.nanoTime();
            // phase 2: noise fill + surface for the region
            AtomicInteger next2 = new AtomicInteger();
            futures.clear();
            for (int t = 0; t < threads; t++) {
                futures.add(pool.submit(() -> {
                    int i;
                    while ((i = next2.getAndIncrement()) < interior) {
                        int cx = x0 + i % region, cz = z0 + i / region;
                        ProtoChunk center = chunks.get(ChunkPos.pack(cx, cz));
                        BiomeManager biomeManager = new BiomeManager(
                            (qx, qy, qz) -> chunks.get(ChunkPos.pack(qx >> 2, qz >> 2)).getNoiseBiome(qx, qy, qz), zoomSeed);
                        Set<Holder<Biome>> possible = new ObjectOpenHashSet<>();
                        for (int dz = -1; dz <= 1; dz++)
                            for (int dx = -1; dx <= 1; dx++) chunks.get(ChunkPos.pack(cx + dx, cz + dz)).collectBiomesInPalette(possible);
                        DensityVolume vol = new DensityVolume(16, height, 16, cx * 16, minY, cz * 16);
                        try (NoiseChunk nc = new NoiseChunk(rs, Beardifier.EMPTY, stock, picker, Blender.empty(), vol)) {
                            doFill.invoke(generator, nc, center);
                            buildSurface.invoke(generator, center, nc, rs, biomeManager, possible, materialRule);
                        }
                    }
                    return null;
                }));
            }
            for (Future<?> f : futures) f.get();
            long end = System.nanoTime();
            double seconds = (end - start) / 1e9, biomeSeconds = (mid - start) / 1e9, terrainSeconds = (end - mid) / 1e9;
            if (rep >= 0) {
                rows.add(new double[] {rep, seconds, interior / seconds, biomeSeconds, terrainSeconds});
                System.out.printf("bench-terrain %s threads=%d rep %d: %.3f s (biomes %.3f s, fill+surface %.3f s), %.1f chunks/s%n", tag, threads, rep,
                    seconds, biomeSeconds, terrainSeconds, interior / seconds);
            }
        }
        pool.shutdown();

        Files.createDirectories(outDir);
        if (dump) {
            int perChunk = 16 * 16 * height;
            ByteBuffer terrain = ByteBuffer.allocate(interior * perChunk * 2).order(ByteOrder.LITTLE_ENDIAN);
            BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
            for (int i = 0; i < interior; i++) {
                int cx = x0 + i % region, cz = z0 + i / region;
                ProtoChunk c = chunks.get(ChunkPos.pack(cx, cz));
                for (int y = 0; y < height; y++) for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) {
                    BlockState s = c.getBlockState(pos.set(cx * 16 + x, minY + y, cz * 16 + z));
                    int idx = model.palette.indexOf(s);
                    if (idx < 0) throw new IllegalStateException("block state outside the terrain palette: " + s);
                    terrain.putShort((i * perChunk + (y * 16 + z) * 16 + x) * 2, (short) idx);
                }
            }
            Path f = outDir.resolve(tag + "_terrain.u16");
            Files.write(f, terrain.array());
            System.out.println("bench-terrain: wrote " + f);
        }
        Path csv = outDir.resolve(tag + "_t" + threads + ".csv");
        try (Writer w = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            w.write("impl,threads,region,seed,x0,z0,rep,seconds,chunks_per_s,biome_seconds,terrain_seconds\n");
            for (double[] r : rows) {
                w.write(String.format("%s,%d,%d,%d,%d,%d,%d,%.6f,%.3f,%.6f,%.6f%n", tag, threads, region, seed, x0, z0, (int) r[0], r[1], r[2], r[3], r[4]));
            }
        }
        System.out.println("bench-terrain: wrote " + csv);
    }

    private TerrainBench() {}
}
