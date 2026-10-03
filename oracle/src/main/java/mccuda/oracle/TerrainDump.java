package mccuda.oracle;

import java.io.Writer;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.gson.stream.JsonWriter;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.IdMapper;
import net.minecraft.core.registries.Registries;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.biome.Climate;
import net.minecraft.world.level.biome.MultiNoiseBiomeSource;
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
 * Full TERRAIN-step reference (without carvers) on real ProtoChunks, using
 * vanilla's own private steps via reflection:
 *
 *   biomes   ChunkGenerator.doCreateBiomes for the chunk and its 8 neighbours
 *            (the R-tree's per-thread last-result cache is reset before each chunk,
 *            so a chunk's biomes do not depend on which chunk was generated before
 *            it; in a live server they can, on exact parameter ties)
 *   aquifer  NoiseBasedChunkGenerator.doFill with the stock settings (aquifers on)
 *   terrain  NoiseBasedChunkGenerator.buildSurface (material rules incl. ore veins,
 *            eroded-badlands and frozen-ocean extensions), with a BiomeManager over
 *            the 3x3 neighbourhood and possibleBiomes collected like ChunkStatusTasks
 *
 * Outputs per seed: biomes (u8 per quart, index = (qy * 4 + qz) * 4 + qx),
 * aquifer fill and final terrain (u16 palette index per block, canonical layout).
 */
final class TerrainDump {
    static void dump(Main.Vanilla v, Path gameData, Path root, boolean quick) throws Exception {
        TerrainModel model = new TerrainModel(v);
        model.export(gameData.resolve("overworld_terrain.json"));   // also completes the palette
        NoiseGeneratorSettings stock = v.overworld;
        NoiseBasedChunkGenerator generator = model.generator;
        Aquifer.FluidPicker picker = ChunkDump.fluidPicker(stock);
        MaterialRule materialRule = stock.materialRule().value();
        int minY = stock.noiseSettings().minY(), height = stock.noiseSettings().height();
        int perChunk = 16 * 16 * height, quartsPerChunk = 4 * (height / 4) * 4;

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
        ThreadLocal<?> rtreeLastResult = rtreeLastResult(model.biomeSource);

        Path out = root.resolve("terrain");
        Files.createDirectories(out);
        Map<String, String> hashes = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        int total = 0;
        StringBuilder seedsJson = new StringBuilder();

        for (Corpus.Seed seed : Corpus.seeds()) {
            if (quick && !seed.quick()) continue;
            List<Corpus.Chunk> chunks = Corpus.chunks(seed, quick);
            RandomState rs = RandomState.create(v.noises, seed.seed(), stock);
            long zoomSeed = BiomeManager.obfuscateSeed(seed.seed());
            ByteBuffer biomes = ByteBuffer.allocate(chunks.size() * quartsPerChunk);
            ByteBuffer aquifer = ByteBuffer.allocate(chunks.size() * perChunk * 2).order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer terrain = ByteBuffer.allocate(chunks.size() * perChunk * 2).order(ByteOrder.LITTLE_ENDIAN);

            for (int ci = 0; ci < chunks.size(); ci++) {
                Corpus.Chunk c = chunks.get(ci);
                Map<Long, ProtoChunk> region = new HashMap<>();
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        ChunkPos pos = new ChunkPos(c.cx() + dx, c.cz() + dz);
                        ProtoChunk proto = new ProtoChunk(pos, UpgradeData.EMPTY, heightAccessor, factory, null);
                        rtreeLastResult.remove();
                        doCreateBiomes.invoke(generator, Blender.empty(), rs, proto);
                        proto.setPersistedStatus(ChunkStatus.BIOMES);
                        region.put(ChunkPos.pack(pos.x(), pos.z()), proto);
                    }
                }
                ProtoChunk center = region.get(ChunkPos.pack(c.cx(), c.cz()));
                BiomeManager biomeManager = new BiomeManager(
                    (qx, qy, qz) -> region.get(ChunkPos.pack(qx >> 2, qz >> 2)).getNoiseBiome(qx, qy, qz), zoomSeed);
                Set<Holder<Biome>> possible = new ObjectOpenHashSet<>();
                for (ProtoChunk p : region.values()) p.collectBiomesInPalette(possible);

                for (int qy = 0; qy < height / 4; qy++) for (int qz = 0; qz < 4; qz++) for (int qx = 0; qx < 4; qx++) {
                    Holder<Biome> b = center.getNoiseBiome((c.cx() << 2) + qx, (minY >> 2) + qy, (c.cz() << 2) + qz);
                    biomes.put(ci * quartsPerChunk + (qy * 4 + qz) * 4 + qx, (byte) model.biomeIndex(b));
                }

                DensityVolume vol = new DensityVolume(16, height, 16, c.cx() * 16, minY, c.cz() * 16);
                try (NoiseChunk nc = new NoiseChunk(rs, Beardifier.EMPTY, stock, picker, Blender.empty(), vol)) {
                    doFill.invoke(generator, nc, center);
                    readBlocks(center, model, c, minY, height, aquifer, ci * perChunk);
                    buildSurface.invoke(generator, center, nc, rs, biomeManager, possible, materialRule);
                    readBlocks(center, model, c, minY, height, terrain, ci * perChunk);
                }
                total++;
            }
            String[] names = {"biomes", "aquifer", "terrain"};
            ByteBuffer[] bufs = {biomes, aquifer, terrain};
            String[] ext = {".u8", ".u16", ".u16"};
            for (int i = 0; i < 3; i++) {
                Path f = out.resolve(names[i] + "_s" + seed.index() + ext[i]);
                Files.write(f, bufs[i].array());
                hashes.put(root.relativize(f).toString().replace('\\', '/'), ChunkDump.sha256(bufs[i].array()));
            }
            System.out.printf("dump-terrain: seed %d (%d) %d chunks%n", seed.index(), seed.seed(), chunks.size());
        }

        try (Writer w = Files.newBufferedWriter(root.resolve("manifest_terrain.json"), StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
            json.setIndent(" ");
            json.beginObject();
            json.name("format").value("mccuda-terrain-oracle/1");
            json.name("minecraft").value(Main.MC_VERSION);
            json.name("note").value("doCreateBiomes (R-tree cache reset per chunk) + doFill (aquifers on) + buildSurface; no carvers");
            json.name("palette").beginArray();
            for (BlockState s : model.palette) json.value(BlockStateParser.serialize(s));
            json.endArray();
            json.name("biome_names").beginArray();
            for (String n : model.biomeNames) json.value(n);
            json.endArray();
            json.name("sha256").beginObject();
            for (var e : hashes.entrySet()) json.name(e.getKey()).value(e.getValue());
            json.endObject();
            json.endObject();
        }
        System.out.printf("dump-terrain: %d chunks in %.1f s, palette %d states%n", total, (System.nanoTime() - t0) / 1e9, model.palette.size());
    }

    private static void readBlocks(ProtoChunk chunk, TerrainModel model, Corpus.Chunk c, int minY, int height, ByteBuffer out, int base) {
        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        for (int y = 0; y < height; y++) for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) {
            BlockState s = chunk.getBlockState(pos.set(c.cx() * 16 + x, minY + y, c.cz() * 16 + z));
            int idx = model.palette.indexOf(s);
            if (idx < 0) throw new IllegalStateException("block state outside the terrain palette: " + s + " at " + pos);
            out.putShort((base + (y * 16 + z) * 16 + x) * 2, (short) idx);
        }
    }

    @SuppressWarnings("unchecked")
    static ThreadLocal<?> rtreeLastResult(MultiNoiseBiomeSource source) throws ReflectiveOperationException {
        Method pm = MultiNoiseBiomeSource.class.getDeclaredMethod("parameters");
        pm.setAccessible(true);
        Climate.ParameterList<Holder<Biome>> list = (Climate.ParameterList<Holder<Biome>>) pm.invoke(source);
        Field index = Climate.ParameterList.class.getDeclaredField("index");
        index.setAccessible(true);
        Object rtree = index.get(list);
        Field last = rtree.getClass().getDeclaredField("lastResult");
        last.setAccessible(true);
        return (ThreadLocal<?>) last.get(rtree);
    }
}
