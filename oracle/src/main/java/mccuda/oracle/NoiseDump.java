package mccuda.oracle;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;

import com.google.gson.stream.JsonWriter;

import net.minecraft.core.Holder;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.levelgen.PositionalRandomFactory;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.densityfunction.DensityBuffer;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunctions;
import net.minecraft.world.level.levelgen.densityfunction.DensityVolume;
import net.minecraft.world.level.levelgen.densityfunction.generator.NoiseFunction;
import net.minecraft.world.level.levelgen.densityfunction.generator.ShiftNoiseFunction;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.GradientNoise;
import net.minecraft.world.level.levelgen.synth.Noise;
import net.minecraft.world.level.levelgen.synth.NoiseStack;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import net.minecraft.world.level.levelgen.synth.SmearedPerlinNoise;

/**
 * Level-0 noise reference: for each seed, the octave tables vanilla builds for
 * every noise the overworld final density uses (permutation, offsets, layer
 * frequency/amplitude, smear scale), plus point samples (Noise.get) and volume
 * samples (Noise.addToVolume) taken through vanilla's own objects.
 */
final class NoiseDump {
    static final int POINTS = 48;

    static void dump(Main.Vanilla v, Path dir, boolean quickOnly) throws Exception {
        Files.createDirectories(dir);
        Map<String, Holder<NormalNoise>> noiseKeys = new TreeMap<>();
        Map<String, BlendedNoise> blended = new TreeMap<>();
        collect(v.overworld.noiseRouter().finalDensity(), noiseKeys, blended, new IdentityHashMap<>());

        for (Corpus.Seed seed : Corpus.seeds()) {
            if (quickOnly && !seed.quick()) continue;
            RandomState rs = RandomState.create(v.noises, seed.seed(), v.overworld);
            Path out = dir.resolve("s" + seed.index() + ".json");
            try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
                json.beginObject();
                json.name("seed").value(Long.toString(seed.seed()));
                json.name("noises").beginObject();
                for (Map.Entry<String, Holder<NormalNoise>> e : noiseKeys.entrySet()) {
                    Noise noise = rs.getOrCreateNoise(e.getValue().unwrapKey().orElseThrow());
                    json.name(e.getKey());
                    writeNoise(json, noise, seed.seed() ^ e.getKey().hashCode());
                }
                json.endObject();
                json.name("blended").beginObject();
                PositionalRandomFactory random = randomStateRandom(rs);
                for (Map.Entry<String, BlendedNoise> e : blended.entrySet()) {
                    BlendedNoise.FbmSet fbm = e.getValue().createFbmSet(random.fromHashOf(BlendedNoise.NOISE_SEED));
                    json.name(e.getKey()).beginObject();
                    json.name("min_limit");
                    writeNoise(json, fbm.minLimitNoise(), seed.seed() + 1);
                    json.name("max_limit");
                    writeNoise(json, fbm.maxLimitNoise(), seed.seed() + 2);
                    json.name("main");
                    writeNoise(json, fbm.mainNoise(), seed.seed() + 3);
                    json.endObject();
                }
                json.endObject();
                json.endObject();
            }
        }
        System.out.println("dump-noise: " + noiseKeys.size() + " noises, " + blended.size() + " blended -> " + dir);
    }

    private static void collect(DensityFunction f, Map<String, Holder<NormalNoise>> noises, Map<String, BlendedNoise> blended, Map<DensityFunction, Boolean> seen) {
        if (seen.put(f, Boolean.TRUE) != null) return;
        switch (f) {
            case DensityFunctions.HolderHolder(Holder<DensityFunction> holder) -> collect(holder.value(), noises, blended, seen);
            case NoiseFunction n -> noises.put(key(n.noise()), n.noise());
            case ShiftNoiseFunction s -> noises.put(key(s.offsetNoise()), s.offsetNoise());
            case BlendedNoise b -> blended.put(blendedKey(b), b);
            default -> { }
        }
        f.rewriteChildren(child -> { collect(child, noises, blended, seen); return child; });
    }

    static String key(Holder<NormalNoise> h) {
        return h.unwrapKey().orElseThrow().identifier().toString();
    }

    static String blendedKey(BlendedNoise b) {
        return String.format("%s|%s|%s|%s|%s", IrExporter.dbits(b.xzScale()), IrExporter.dbits(b.yScale()), IrExporter.dbits(b.xzFactor()),
            IrExporter.dbits(b.yFactor()), IrExporter.dbits(b.smearScaleMultiplier()));
    }

    private static void writeNoise(JsonWriter json, Noise noise, long sampleSeed) throws Exception {
        json.beginObject();
        json.name("class").value(noise.getClass().getName());
        json.name("layers").beginArray();
        for (Object layer : layers((NoiseStack) noise)) {
            Noise ln = (Noise) recordGet(layer, "noise");
            double frequency = (double) recordGet(layer, "frequency");
            float amplitude = (float) recordGet(layer, "amplitude");
            json.beginObject();
            json.name("class").value(ln.getClass().getSimpleName());
            json.name("frequency").value(IrExporter.dbits(frequency));
            json.name("amplitude").value(IrExporter.fbits(amplitude));
            GradientNoise g = (GradientNoise) ln;
            json.name("offset").beginArray()
                .value(IrExporter.dbits(getDouble(g, GradientNoise.class, "offsetX")))
                .value(IrExporter.dbits(getDouble(g, GradientNoise.class, "offsetY")))
                .value(IrExporter.dbits(getDouble(g, GradientNoise.class, "offsetZ")))
                .endArray();
            byte[] perms = (byte[]) field(GradientNoise.class, "perms").get(g);
            StringBuilder hex = new StringBuilder();
            for (byte b : perms) hex.append(String.format("%02x", b & 0xFF));
            json.name("perms").value(hex.toString());
            if (ln instanceof SmearedPerlinNoise) {
                json.name("smear").value(IrExporter.dbits(getDouble(ln, SmearedPerlinNoise.class, "fudgeYScale")));
            }
            json.endObject();
        }
        json.endArray();

        // Point samples: coordinates spread from small to beyond the wrap fold.
        SplittableRandom r = new SplittableRandom(sampleSeed);
        json.name("points").beginArray();
        for (int i = 0; i < POINTS; i++) {
            double scale = i < 16 ? 100.0 : (i < 32 ? 1.0e6 : 4.0e7);
            double x = (r.nextDouble() * 2 - 1) * scale;
            double y = (r.nextDouble() * 2 - 1) * (i < 32 ? 400.0 : 4.0e7);
            double z = (r.nextDouble() * 2 - 1) * scale;
            json.beginArray().value(IrExporter.dbits(x)).value(IrExporter.dbits(y)).value(IrExporter.dbits(z))
                .value(IrExporter.fbits(noise.get(x, y, z))).endArray();
        }
        json.endArray();

        // Volume samples through addToVolume (the path chunk generation uses).
        int[][] vols = {{-20, -64, 37, 1, 1, 1}, {1000, 0, -1000, 4, 8, 4}, {-29_999_984, -64, 29_999_968, 4, 8, 4}, {98_048, 64, -98_064, 1, 1, 1}};
        double[][] scales = {{1.0, 1.0}, {0.25, 0.0}, {171.103, 85.5515}, {0.5, 0.125}};
        json.name("volumes").beginArray();
        for (int[] vd : vols) {
            for (double[] sc : scales) {
                DensityVolume vol = new DensityVolume(3, 5, 2, vd[0], vd[1], vd[2], vd[3], vd[4], vd[5]);
                DensityBuffer buf = DensityBuffer.createUnpooled(vol.size());
                buf.fill(0.0F);
                float amp = 1.0F;
                noise.addToVolume(buf, vol, sc[0], sc[1], amp);
                json.beginObject();
                json.name("volume").beginArray();
                for (int k : new int[] {3, 5, 2, vd[0], vd[1], vd[2], vd[3], vd[4], vd[5]}) json.value(k);
                json.endArray();
                json.name("xz_scale").value(IrExporter.dbits(sc[0]));
                json.name("y_scale").value(IrExporter.dbits(sc[1]));
                json.name("amplitude").value(IrExporter.fbits(amp));
                json.name("values").beginArray();
                for (int i = 0; i < vol.size(); i++) json.value(IrExporter.fbits(buf.get(i)));
                json.endArray();
                json.endObject();
            }
        }
        json.endArray();
        json.endObject();
    }

    private static Object[] layers(NoiseStack stack) throws ReflectiveOperationException {
        return (Object[]) field(NoiseStack.class, "layers").get(stack);
    }

    static Object recordGet(Object record, String name) throws ReflectiveOperationException {
        Method m = record.getClass().getDeclaredMethod(name);
        m.setAccessible(true);
        return m.invoke(record);
    }

    static double getDouble(Object o, Class<?> owner, String name) throws ReflectiveOperationException {
        return field(owner, name).getDouble(o);
    }

    static Field field(Class<?> owner, String name) throws ReflectiveOperationException {
        Field f = owner.getDeclaredField(name);
        f.setAccessible(true);
        return f;
    }

    static PositionalRandomFactory randomStateRandom(RandomState rs) throws ReflectiveOperationException {
        return (PositionalRandomFactory) field(RandomState.class, "random").get(rs);
    }

    static Identifier terrainSeed() {
        return BlendedNoise.NOISE_SEED;
    }
}
