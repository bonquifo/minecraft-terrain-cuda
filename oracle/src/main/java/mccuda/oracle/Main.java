package mccuda.oracle;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.minecraft.SharedConstants;
import net.minecraft.core.HolderGetter;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

/**
 * Reference oracle entry point. Every value this program writes is computed by
 * vanilla 26.3 classes; the oracle only drives them and serialises results.
 *
 * Commands:
 *   export-ir            optimised overworld density graph (our own JSON IR)
 *   dump-rng             Xoroshiro128++ / legacy LCG / seed-hash reference streams
 *   dump-noise  [quick]  per-seed octave tables and noise sample values
 *   dump-chunks [quick]  final density (L2), block fill (L3) and named-function volumes (L1)
 *   dump-terrain [quick] biomes, aquifer fill and full terrain (doFill + buildSurface)
 *   bench-vanilla ...    vanilla throughput (see VanillaBench)
 *   all                  export-ir, dump-rng, dump-noise, dump-chunks
 */
public final class Main {
    public static final String MC_VERSION = "26.3";

    public static void main(String[] args) throws Exception {
        List<String> argv = new ArrayList<>(Arrays.asList(args));
        Path dataRoot = Path.of("data");
        int di = argv.indexOf("--data");
        if (di >= 0) {
            dataRoot = Path.of(argv.get(di + 1));
            argv.remove(di + 1);
            argv.remove(di);
        }
        if (argv.isEmpty() || argv.getFirst().equals("help")) {
            System.out.println("usage: <export-ir|dump-rng|dump-noise|dump-chunks|bench-vanilla|all> [args] --data <dir>");
            return;
        }

        long t0 = System.nanoTime();
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
        String version = SharedConstants.getCurrentVersion().name();
        if (!version.equals(MC_VERSION)) {
            throw new IllegalStateException("Oracle pinned to " + MC_VERSION + " but game reports " + version);
        }
        Vanilla vanilla = new Vanilla();
        System.out.printf("oracle: minecraft %s, java %s, bootstrap %.1f s%n", version, System.getProperty("java.version"), (System.nanoTime() - t0) / 1e9);

        Path gameData = dataRoot.resolve("gamedata").resolve(MC_VERSION);
        Path oracleData = dataRoot.resolve("oracle").resolve(MC_VERSION);
        Files.createDirectories(gameData);
        Files.createDirectories(oracleData);

        String cmd = argv.getFirst();
        List<String> rest = argv.subList(1, argv.size());
        switch (cmd) {
            case "export-ir" -> exportIr(vanilla, gameData);
            case "dump-terrain" -> {
                boolean quick = rest.contains("quick");
                TerrainDump.dump(vanilla, gameData, oracleData.resolve(quick ? "quick" : "full"), quick);
            }
            case "dump-rng" -> RngDump.dump(oracleData.resolve("rng.json"));
            case "dump-noise" -> NoiseDump.dump(vanilla, oracleData.resolve("noise"), rest.contains("quick"));
            case "dump-chunks" -> {
                boolean quick = rest.contains("quick");
                ChunkDump.dump(vanilla, oracleData.resolve(quick ? "quick" : "full"), quick);
            }
            case "bench-vanilla" -> VanillaBench.run(vanilla, rest, dataRoot.resolve("bench"));
            case "all" -> {
                exportIr(vanilla, gameData);
                RngDump.dump(oracleData.resolve("rng.json"));
                NoiseDump.dump(vanilla, oracleData.resolve("noise"), false);
                ChunkDump.dump(vanilla, oracleData.resolve("quick"), true);
                ChunkDump.dump(vanilla, oracleData.resolve("full"), false);
                TerrainDump.dump(vanilla, gameData, oracleData.resolve("quick"), true);
                TerrainDump.dump(vanilla, gameData, oracleData.resolve("full"), false);
            }
            default -> throw new IllegalArgumentException("unknown command " + cmd);
        }
        System.out.printf("oracle: %s done in %.1f s%n", cmd, (System.nanoTime() - t0) / 1e9);
    }

    /** Terrain model first: its material-rule walk registers the extra IR roots and noises. */
    static void exportIr(Vanilla vanilla, Path gameData) throws Exception {
        TerrainModel model = new TerrainModel(vanilla);
        model.export(gameData.resolve("overworld_terrain.json"));
        new IrExporter(vanilla, model).export(gameData.resolve("overworld_ir.json"));
    }

    /** Vanilla registries and the overworld settings, shared by all commands. */
    public static final class Vanilla {
        public final HolderLookup.Provider lookup = VanillaRegistries.createWorldLookup();
        public final HolderGetter<NormalNoise> noises = lookup.lookupOrThrow(Registries.NOISE);
        public final NoiseGeneratorSettings overworld = lookup.lookupOrThrow(Registries.NOISE_SETTINGS)
            .getOrThrow(NoiseGeneratorSettings.OVERWORLD).value();
    }
}
