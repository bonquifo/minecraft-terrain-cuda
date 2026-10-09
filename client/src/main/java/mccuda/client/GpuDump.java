package mccuda.client;

import java.io.IOException;
import java.io.Reader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.Property;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.structure.TerrainAdjustment;

/**
 * The GPU generator's output for one region (mccuda_bench --dump): blocks as palette
 * indices in the canonical layout and quart biomes, row-major by chunk. Chunks inside
 * the region get their blocks and biomes from here instead of vanilla's noise fill,
 * surface rules and biome placement; everything after (carvers, features, light) is the
 * game's own.
 */
public final class GpuDump {
    private static GpuDump instance;
    private static boolean loaded;

    public final long seed;
    /** false: a vanilla control run; the dump only supplies the camera's ground height, every chunk is vanilla's. */
    public final boolean serve = !"false".equals(System.getProperty("mccuda.serve"));
    final int x0, z0, region, height, minY, chunkBytes, biomeBytes;
    final MappedByteBuffer terrain, biomes;
    final BlockState[] palette;
    final String[] biomeNames;
    final int airIndex;
    public final AtomicInteger servedBlocks = new AtomicInteger(), servedBiomes = new AtomicInteger(), vanillaChunks = new AtomicInteger();
    /** Control run: the game's own biomes and post-surface blocks for region chunks, compared with the dump. */
    public final AtomicInteger checkedBiomeChunks = new AtomicInteger(), checkedBlockChunks = new AtomicInteger();
    public final AtomicLong checkedBiomes = new AtomicLong(), biomeMismatches = new AtomicLong(), checkedBlocks = new AtomicLong(),
        blockMismatches = new AtomicLong();
    /** The block check split by whether a structure reshapes the chunk's density (the Beardifier, outside this project's scope). */
    public final AtomicInteger adaptedChunks = new AtomicInteger(), adaptedBadChunks = new AtomicInteger(), plainBadChunks = new AtomicInteger();
    public final AtomicLong adaptedMismatches = new AtomicLong();
    private volatile Holder<Biome>[] biomeHolders;

    public static synchronized GpuDump get() {
        if (!loaded) {
            loaded = true;
            String dir = System.getProperty("mccuda.dump");
            if (dir != null) {
                try {
                    instance = new GpuDump(Path.of(dir), Path.of(System.getProperty("mccuda.terrain")));
                    System.out.printf("[mccuda] GPU dump: region %d x %d at chunk (%d, %d), %s%n", instance.region, instance.region, instance.x0, instance.z0,
                        instance.serve ? "serving its chunks" : "vanilla control run (not serving)");
                } catch (IOException e) {
                    throw new RuntimeException("cannot load GPU dump " + dir, e);
                }
            }
        }
        return instance;
    }

    private GpuDump(Path dir, Path terrainJson) throws IOException {
        JsonObject meta;
        try (Reader r = Files.newBufferedReader(dir.resolve("meta.json"))) {
            meta = JsonParser.parseReader(r).getAsJsonObject();
        }
        seed = meta.get("seed").getAsLong();
        x0 = meta.get("x0").getAsInt();
        z0 = meta.get("z0").getAsInt();
        region = meta.get("region").getAsInt();
        height = meta.get("height").getAsInt();
        minY = meta.get("min_y").getAsInt();
        chunkBytes = meta.get("chunk_bytes").getAsInt();
        biomeBytes = meta.get("biome_bytes").getAsInt();
        try (FileChannel t = FileChannel.open(dir.resolve("terrain.u8"), StandardOpenOption.READ);
             FileChannel b = FileChannel.open(dir.resolve("biomes.u8"), StandardOpenOption.READ)) {
            terrain = t.map(FileChannel.MapMode.READ_ONLY, 0, t.size());
            biomes = b.map(FileChannel.MapMode.READ_ONLY, 0, b.size());
        }
        JsonObject tm;
        try (Reader r = Files.newBufferedReader(terrainJson)) {
            tm = JsonParser.parseReader(r).getAsJsonObject();
        }
        JsonArray pal = tm.getAsJsonArray("palette");
        palette = new BlockState[pal.size()];
        int air = -1;
        for (int i = 0; i < pal.size(); i++) {
            palette[i] = parseState(pal.get(i).getAsString());
            if (palette[i].isAir()) air = i;
        }
        airIndex = air;
        JsonArray names = tm.getAsJsonArray("biome_names");
        biomeNames = new String[names.size()];
        for (int i = 0; i < names.size(); i++) biomeNames[i] = names.get(i).getAsString();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static BlockState parseState(String s) {
        int br = s.indexOf('[');
        Block block = BuiltInRegistries.BLOCK.getValue(Identifier.parse(br < 0 ? s : s.substring(0, br)));
        BlockState state = block.defaultBlockState();
        if (br >= 0) {
            for (String kv : s.substring(br + 1, s.length() - 1).split(",")) {
                String[] p = kv.split("=");
                Property prop = block.getStateDefinition().getProperty(p[0]);
                state = state.setValue(prop, (Comparable) prop.getValue(p[1]).orElseThrow());
            }
        }
        return state;
    }

    /** True when this chunk is filled from the dump. */
    public boolean serves(ChunkPos pos) {
        return serve && covers(pos.x(), pos.z());
    }

    public boolean covers(ChunkPos pos) {
        return covers(pos.x(), pos.z());
    }

    public boolean covers(int cx, int cz) {
        int rx = cx - x0, rz = cz - z0;
        return rx >= 0 && rz >= 0 && rx < region && rz < region;
    }

    /** Top non-air block + 1 of the GPU terrain at (x, z), water included: the ground the GPU flythrough (viz/fly.cu) follows. */
    public double ground(double x, double z) {
        int bx = (int) Math.floor(x), bz = (int) Math.floor(z);
        if (!covers(bx >> 4, bz >> 4)) return 64.0;
        long base = (long) ((bz >> 4) - z0) * region * chunkBytes + (long) ((bx >> 4) - x0) * chunkBytes + (bz & 15) * 16 + (bx & 15);
        int y = height - 1;
        while (y > 0 && (terrain.get((int) (base + (long) y * 256)) & 0xFF) == airIndex) y--;
        return minY + y + 1;
    }

    private int index(ChunkPos pos) {
        return (pos.z() - z0) * region + (pos.x() - x0);
    }

    /** Sets every non-air block of the chunk from the dump and primes the worldgen heightmaps. */
    public void fillBlocks(ChunkAccess chunk) {
        long base = (long) index(chunk.getPos()) * chunkBytes;
        LevelChunkSection[] sections = chunk.getSections();
        for (int y = 0; y < height; y++) {
            LevelChunkSection sec = sections[y >> 4];
            int ly = y & 15;
            for (int z = 0; z < 16; z++) {
                for (int x = 0; x < 16; x++) {
                    int v = terrain.get((int) (base + ((long) y * 16 + z) * 16 + x)) & 0xFF;
                    if (v != airIndex) sec.setBlockState(x, ly, z, palette[v], false);
                }
            }
        }
        Heightmap.primeHeightmaps(chunk, EnumSet.of(Heightmap.Types.OCEAN_FLOOR_WG, Heightmap.Types.WORLD_SURFACE_WG));
        servedBlocks.incrementAndGet();
    }

    /** Fills the chunk's quart biomes from the dump. */
    public void fillBiomes(ChunkAccess chunk, BiomeSource source) {
        Holder<Biome>[] holders = biomeHolders;
        if (holders == null) holders = resolveBiomes(source);
        final Holder<Biome>[] h = holders;
        final ChunkPos pos = chunk.getPos();
        final long base = (long) index(pos) * biomeBytes;
        final int qMinY = minY >> 2, qh = height / 4;
        chunk.fillBiomesFromNoise((qx, qy, qz) -> {
            int cqy = Math.min(Math.max(qy - qMinY, 0), qh - 1);
            int v = biomes.get((int) (base + (cqy * 4 + (qz & 3)) * 4 + (qx & 3))) & 0xFF;
            return h[v];
        });
        servedBiomes.incrementAndGet();
    }

    /** Compares the chunk's quart biomes, as the game generated them, with the dump. */
    public void verifyBiomes(ChunkAccess chunk, BiomeSource source) {
        Holder<Biome>[] h = biomeHolders;
        if (h == null) h = resolveBiomes(source);
        // read the sections directly: the chunk's own getter refuses until its status reaches BIOMES
        LevelChunkSection[] sections = chunk.getSections();
        long base = (long) index(chunk.getPos()) * biomeBytes, bad = 0;
        int qh = height / 4;
        for (int qy = 0; qy < qh; qy++)
            for (int qz = 0; qz < 4; qz++)
                for (int qx = 0; qx < 4; qx++)
                    if (sections[qy >> 2].getNoiseBiome(qx, qy & 3, qz) != h[biomes.get((int) (base + (qy * 4 + qz) * 4 + qx)) & 0xFF]) bad++;
        checkedBiomes.addAndGet(16L * qh);
        biomeMismatches.addAndGet(bad);
        checkedBiomeChunks.incrementAndGet();
    }

    /** Compares every block of the chunk after the game's surface pass (the end of the TERRAIN step) with the dump. */
    public void verifyBlocks(ChunkAccess chunk) {
        long base = (long) index(chunk.getPos()) * chunkBytes, bad = 0;
        LevelChunkSection[] sections = chunk.getSections();
        for (int y = 0; y < height; y++) {
            LevelChunkSection sec = sections[y >> 4];
            int ly = y & 15;
            for (int z = 0; z < 16; z++)
                for (int x = 0; x < 16; x++)
                    if (sec.getBlockState(x, ly, z) != palette[terrain.get((int) (base + ((long) y * 16 + z) * 16 + x)) & 0xFF]) bad++;
        }
        boolean adapted = chunk.getAllReferences().entrySet().stream().anyMatch(e -> e.getKey().terrainAdaptation() != TerrainAdjustment.NONE && !e.getValue().isEmpty());
        checkedBlocks.addAndGet(256L * height);
        blockMismatches.addAndGet(bad);
        checkedBlockChunks.incrementAndGet();
        if (adapted) {
            adaptedChunks.incrementAndGet();
            adaptedMismatches.addAndGet(bad);
            if (bad > 0) adaptedBadChunks.incrementAndGet();
        } else if (bad > 0) {
            plainBadChunks.incrementAndGet();
            System.out.printf("[mccuda] mismatch without structure adaptation: chunk %s, %d blocks%n", chunk.getPos(), bad);
        }
    }

    @SuppressWarnings("unchecked")
    private synchronized Holder<Biome>[] resolveBiomes(BiomeSource source) {
        if (biomeHolders != null) return biomeHolders;
        Map<String, Holder<Biome>> byName = new HashMap<>();
        for (Holder<Biome> b : source.possibleBiomes()) b.unwrapKey().ifPresent(k -> byName.put(k.identifier().toString(), b));
        List<String> missing = new ArrayList<>();
        Holder<Biome>[] out = new Holder[biomeNames.length];
        for (int i = 0; i < biomeNames.length; i++) {
            out[i] = byName.get(biomeNames[i]);
            if (out[i] == null) missing.add(biomeNames[i]);
        }
        if (!missing.isEmpty()) System.out.println("[mccuda] biomes not offered by the biome source (unused by this region if never referenced): " + missing);
        biomeHolders = out;
        return out;
    }
}
