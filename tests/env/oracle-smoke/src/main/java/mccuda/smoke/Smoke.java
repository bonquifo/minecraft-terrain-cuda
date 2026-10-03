package mccuda.smoke;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.TreeMap;

import net.minecraft.SharedConstants;
import net.minecraft.core.HolderGetter;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.registries.VanillaRegistries;
import net.minecraft.server.Bootstrap;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.densityfunction.DensityBuffer;
import net.minecraft.world.level.levelgen.densityfunction.DensityBufferPool;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.DensitySampler;
import net.minecraft.world.level.levelgen.densityfunction.DensityVolume;
import net.minecraft.world.level.levelgen.densityfunction.DfRewriteRule;
import net.minecraft.world.level.levelgen.densityfunction.SamplerContext;
import net.minecraft.world.level.levelgen.densityfunction.op.BinaryFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.InterpolatedFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.UnaryFunction;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

/**
 * Phase 0 smoke test: bootstrap vanilla 26.3 headlessly, evaluate the overworld
 * noise router through vanilla's own samplers, and summarise its structure.
 * Prints only derived facts (counts, sample values), never game data.
 */
public final class Smoke {
    private static final int[][] POINTS = {
        {0, 64, 0}, {0, -64, 0}, {0, 319, 0}, {-1, 63, -1}, {15, 40, 16},
        {-123456, 12, 654321}, {29_999_999, 70, -29_999_999}, {-29_999_984, -60, 29_999_984},
    };

    public static void main(String[] args) {
        long t0 = System.nanoTime();
        SharedConstants.tryDetectVersion();
        Bootstrap.bootStrap();
        HolderLookup.Provider lookup = VanillaRegistries.createWorldLookup();
        HolderGetter<NormalNoise> noises = lookup.lookupOrThrow(Registries.NOISE);
        NoiseGeneratorSettings settings = lookup.lookupOrThrow(Registries.NOISE_SETTINGS)
            .getOrThrow(NoiseGeneratorSettings.OVERWORLD).value();
        System.out.printf("minecraft %s (id %s), java %s, bootstrap %.1f s%n",
            SharedConstants.getCurrentVersion().name(), SharedConstants.getCurrentVersion().id(),
            System.getProperty("java.version"), (System.nanoTime() - t0) / 1e9);

        System.out.printf("overworld: minY=%d height=%d seaLevel=%d defaultBlock=%s defaultFluid=%s aquifers=%s legacyRandom=%s%n",
            settings.noiseSettings().minY(), settings.noiseSettings().height(), settings.seaLevel(),
            settings.defaultBlock().getBlock(), settings.defaultFluid().getBlock(),
            settings.aquifers().isPresent(), settings.useLegacyRandomSource());

        NoiseRouter router = settings.noiseRouter();
        DensityFunction finalDensity = router.finalDensity();
        Map<String, Integer> hist = new TreeMap<>();
        Map<DensityFunction, Boolean> unique = new IdentityHashMap<>();
        int treeSize = walk(finalDensity, hist, unique);
        System.out.printf("final_density graph: %d nodes as a tree, %d distinct objects%n", treeSize, unique.size());
        hist.forEach((k, v) -> System.out.printf("  %-42s %d%n", k, v));

        for (String seedArg : args) {
            long seed = Long.parseLong(seedArg.trim());
            RandomState rs = RandomState.create(noises, seed, settings);
            System.out.printf("seed %d%n", seed);
            for (int[] p : POINTS) {
                float v = rs.sampleBlockValueUncached(finalDensity, p[0], p[1], p[2]);
                System.out.printf("  final_density(%d, %d, %d) = %.9g  bits 0x%08x%n",
                    p[0], p[1], p[2], v, Float.floatToRawIntBits(v));
            }
            compareVolumeToPoints(rs, finalDensity, 0, 0, true);
            compareVolumeToPoints(rs, finalDensity, 0, 0, false);
            compareVolumeToPoints(rs, finalDensity, -1_874_999, 1_874_998, true);
        }

        // Rough single-thread cost of vanilla's volume path; not a benchmark.
        RandomState rs = RandomState.create(noises, 0L, settings);
        DensitySampler sampler = rs.getSampler(finalDensity);
        DensityBufferPool pool = rs.acquireDensityBufferPool();
        SamplerContext ctx = SamplerContext.builder().useBufferArena(pool).enableCaches().build();
        int minY = settings.noiseSettings().minY(), h = settings.noiseSettings().height();
        DensityBuffer buf = DensityBuffer.createUnpooled(16 * h * 16);
        int n = 0;
        long start = 0;
        for (int i = 0; i < 400; i++) {
            if (i == 100) { start = System.nanoTime(); n = 0; }
            DensityVolume vol = new DensityVolume(16, h, 16, (i % 20) * 16, minY, (i / 20) * 16, 1, 1, 1);
            sampler.sampleVolume(ctx, buf, vol);
            n++;
        }
        double ms = (System.nanoTime() - start) / 1e6 / n;
        System.out.printf("vanilla final_density volume sample: %.3f ms per 16x%dx16 chunk (1 thread, after warm-up)%n", ms, h);
        rs.releaseDensityBufferPool(pool);
        System.out.println("SMOKE: PASS");
    }

    /** Samples one chunk column volume and checks it against per-point sampling. */
    private static void compareVolumeToPoints(RandomState rs, DensityFunction f, int chunkX, int chunkZ, boolean caches) {
        int minY = -64, h = 384;
        int bx = chunkX * 16, bz = chunkZ * 16;
        DensityVolume vol = new DensityVolume(16, h, 16, bx, minY, bz, 1, 1, 1);
        DensityBuffer buf = DensityBuffer.createUnpooled(vol.size());
        DensityBufferPool pool = rs.acquireDensityBufferPool();
        try {
            SamplerContext.Builder b = SamplerContext.builder().useBufferArena(pool);
            if (caches) b.enableCaches();
            SamplerContext ctx = b.build();
            rs.getSampler(f).sampleVolume(ctx, buf, vol);
            int mismatches = 0, solid = 0, signFlips = 0, maxUlp = 0, cornerMismatches = 0;
            float maxAbs = 0f;
            for (int x = 0; x < 16; x++) for (int y = 0; y < h; y++) for (int z = 0; z < 16; z++) {
                int idx = vol.indexOfBlock(bx + x, minY + y, bz + z);
                float va = buf.get(idx);
                float vp = rs.sampleBlockValueUncached(f, bx + x, minY + y, bz + z);
                if (Float.floatToRawIntBits(va) != Float.floatToRawIntBits(vp)) {
                    mismatches++;
                    if (x % 4 == 0 && y % 8 == 0 && z % 4 == 0) cornerMismatches++;
                    if ((va > 0f) != (vp > 0f)) signFlips++;
                    maxAbs = Math.max(maxAbs, Math.abs(va - vp));
                    maxUlp = Math.max(maxUlp, Math.abs(Float.floatToRawIntBits(va) - Float.floatToRawIntBits(vp)));
                }
                if (va > 0f) solid++;
            }
            System.out.printf("  chunk (%d, %d) caches=%s: volume vs point: %d/%d bitwise mismatches (%d at cell corners), max |diff| %.3g, max %d ulp, %d solid/air flips; %d blocks with density > 0%n",
                chunkX, chunkZ, caches, mismatches, vol.size(), cornerMismatches, maxAbs, maxUlp, signFlips, solid);
        } catch (RuntimeException e) {
            System.out.printf("  chunk (%d, %d): volume sampling failed: %s%n", chunkX, chunkZ, e);
        } finally {
            rs.releaseDensityBufferPool(pool);
        }
    }

    private static int walk(DensityFunction f, Map<String, Integer> hist, Map<DensityFunction, Boolean> unique) {
        DensityFunction g = DfRewriteRule.INLINE_REFERENCE.rewrite(f);
        if (g != f) {
            hist.merge("(reference)", 1, Integer::sum);
            f = g;
        }
        unique.put(f, Boolean.TRUE);
        hist.merge(label(f), 1, Integer::sum);
        int[] count = {1};
        f.rewriteChildren(child -> { count[0] += walk(child, hist, unique); return child; });
        return count[0];
    }

    private static String label(DensityFunction f) {
        String name = f.getClass().getName();
        name = name.substring(name.lastIndexOf('.') + 1);
        if (f instanceof UnaryFunction u) return name + ":" + u.type();
        if (f instanceof BinaryFunction b) return name + ":" + b.type();
        if (f instanceof InterpolatedFunction i) return name + ":cell " + i.cellSizeXz() + "x" + i.cellSizeY();
        return name;
    }
}
