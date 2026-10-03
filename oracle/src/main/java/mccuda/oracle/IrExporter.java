package mccuda.oracle;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.google.gson.stream.JsonWriter;
import com.mojang.serialization.MapCodec;

import net.minecraft.core.Holder;
import net.minecraft.util.CubicSpline;
import net.minecraft.util.Interval;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunctions;
import net.minecraft.world.level.levelgen.densityfunction.DensitySampler;
import net.minecraft.world.level.levelgen.densityfunction.DfRewriteRule;
import net.minecraft.world.level.levelgen.densityfunction.generator.ConstantFunction;
import net.minecraft.world.level.levelgen.densityfunction.generator.GradientFunction;
import net.minecraft.world.level.levelgen.densityfunction.generator.NoiseFunction;
import net.minecraft.world.level.levelgen.densityfunction.generator.ShiftNoiseFunction;
import net.minecraft.world.level.levelgen.densityfunction.generator.SimpleDensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.BinaryFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.BlendDensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.CacheFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.ClampFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.IntervalSelectFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.InterpolatedFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.LerpFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.RangeChoiceFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.SliceFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.SplineFunction;
import net.minecraft.world.level.levelgen.densityfunction.op.UnaryFunction;
import net.minecraft.world.level.levelgen.synth.BlendedNoise;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

/**
 * Exports the overworld density graph after the same compile-time rewrites that
 * vanilla's DensityFunctionCompiler applies (reference inlining, cache
 * deduplication by structural equality, SLICE_UNIFORM_AXES). The rewrite rules
 * themselves are vanilla's public rules; only the cache bookkeeping is mirrored.
 *
 * The output is our own JSON format. Floats are written as raw IEEE bits (int),
 * doubles as raw bits (hex string), so nothing is lost in transit. Each node also
 * carries vanilla's computed range() and domainAxes(), which drive vanilla's
 * compile-time MIN/MAX pruning.
 */
final class IrExporter {
    /** Our stand-in for the compiler's private PreparedCache. */
    record ExportCache(int id, DensityFunction input) implements DensityFunction {
        @Override public DensitySampler compileSampler(CompileContext context) { throw new UnsupportedOperationException(); }
        @Override public DensityFunction rewriteChildren(DfRewriteRule rule) { return this; }
        @Override public Interval range() { return input.range(); }
        @Override public int domainAxes() { return input.domainAxes(); }
        @Override public MapCodec<? extends DensityFunction> codec() { throw new UnsupportedOperationException(); }
    }

    private final Main.Vanilla vanilla;
    private final Map<DensityFunction, ExportCache> prepared = new HashMap<>();
    private int nextCacheId;
    private final DfRewriteRule optimizer = DfRewriteRule.sequence(new DfRewriteRule() {
        @Override
        public DensityFunction rewrite(DensityFunction function) {
            function = DfRewriteRule.INLINE_REFERENCE.rewrite(function);
            return function instanceof CacheFunction cache ? reuseOrPrepare(cache) : function.rewriteChildren(this);
        }
    }, DfRewriteRule.SLICE_UNIFORM_AXES);

    // Structural (record equals) numbering: evaluation is pure, so equal subtrees share a node.
    private final Map<DensityFunction, Integer> ids = new HashMap<>();
    private final List<DensityFunction> order = new ArrayList<>();
    private final Map<String, Holder<NormalNoise>> noiseDefs = new TreeMap<>();
    private final Map<String, Integer> histogram = new TreeMap<>();

    IrExporter(Main.Vanilla vanilla) {
        this.vanilla = vanilla;
    }

    private ExportCache reuseOrPrepare(CacheFunction cache) {
        ExportCache p = prepared.get(cache.input());
        if (p == null) {
            int id = nextCacheId++;
            p = new ExportCache(id, optimizer.rewrite(cache.input()));
            prepared.put(cache.input(), p);
        }
        return p;
    }

    void export(Path out) throws IOException {
        DensityFunction rawFinal = vanilla.overworld.noiseRouter().finalDensity();
        Map<String, DensityFunction> namedRaw = new LinkedHashMap<>();
        collectNamed(rawFinal, namedRaw, new IdentityHashMap<>());

        Map<String, DensityFunction> roots = new LinkedHashMap<>();
        roots.put("final_density", optimizer.rewrite(rawFinal));
        for (Map.Entry<String, DensityFunction> e : namedRaw.entrySet()) {
            roots.put(e.getKey(), optimizer.rewrite(e.getValue()));
        }
        for (DensityFunction root : roots.values()) number(root);

        Files.createDirectories(out.getParent());
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
            json.setIndent(" ");
            json.beginObject();
            json.name("format").value("mccuda-ir/1");
            json.name("minecraft").value(Main.MC_VERSION);
            json.name("settings").value("minecraft:overworld");
            json.name("min_y").value(vanilla.overworld.noiseSettings().minY());
            json.name("height").value(vanilla.overworld.noiseSettings().height());
            json.name("sea_level").value(vanilla.overworld.seaLevel());
            json.name("roots").beginObject();
            for (Map.Entry<String, DensityFunction> e : roots.entrySet()) json.name(e.getKey()).value(ids.get(e.getValue()));
            json.endObject();
            json.name("nodes").beginArray();
            for (DensityFunction f : order) writeNode(json, f);
            json.endArray();
            json.name("noises").beginObject();
            for (Map.Entry<String, Holder<NormalNoise>> e : noiseDefs.entrySet()) {
                json.name(e.getKey());
                writeNoiseDef(json, e.getValue().value());
            }
            json.endObject();
            json.endObject();
        }
        System.out.printf("export-ir: %d nodes, %d caches, %d roots, %d noises -> %s%n", order.size(), nextCacheId, roots.size(), noiseDefs.size(), out);
        histogram.forEach((k, v) -> System.out.printf("  %-28s %d%n", k, v));
    }

    /** Collects every registry-named density function reachable from the raw graph. */
    private static void collectNamed(DensityFunction f, Map<String, DensityFunction> out, Map<DensityFunction, Boolean> seen) {
        if (seen.put(f, Boolean.TRUE) != null) return;
        if (f instanceof DensityFunctions.HolderHolder(Holder<DensityFunction> holder)) {
            holder.unwrapKey().ifPresent(key -> out.putIfAbsent(key.identifier().toString(), holder.value()));
            collectNamed(holder.value(), out, seen);
            return;
        }
        f.rewriteChildren(child -> { collectNamed(child, out, seen); return child; });
    }

    /** Assigns ids in post-order so every child id is smaller than its parent's. */
    private void number(DensityFunction f) {
        if (ids.containsKey(f)) return;
        for (DensityFunction child : children(f)) number(child);
        ids.put(f, order.size());
        order.add(f);
        histogram.merge(typeName(f), 1, Integer::sum);
    }

    private static List<DensityFunction> children(DensityFunction f) {
        List<DensityFunction> out = new ArrayList<>();
        switch (f) {
            case ExportCache c -> out.add(c.input());
            case SplineFunction s -> s.spline().forEachCoordinate(coord -> out.add(coord.function()));
            default -> f.rewriteChildren(child -> { out.add(child); return child; });
        }
        return out;
    }

    private static String typeName(DensityFunction f) {
        return switch (f) {
            case ExportCache c -> "cache";
            case ConstantFunction c -> "constant";
            case SimpleDensityFunction s -> "simple";
            case GradientFunction g -> "gradient";
            case NoiseFunction n -> "noise";
            case ShiftNoiseFunction.ShiftA s -> "shift_a";
            case ShiftNoiseFunction.ShiftB s -> "shift_b";
            case ShiftNoiseFunction.Shift s -> "shift";
            case BlendedNoise b -> "blended_noise";
            case BinaryFunction b -> "binary";
            case UnaryFunction u -> "unary";
            case ClampFunction c -> "clamp";
            case LerpFunction l -> "lerp";
            case RangeChoiceFunction r -> "range_choice";
            case IntervalSelectFunction i -> "interval_select";
            case SplineFunction s -> "spline";
            case InterpolatedFunction i -> "interpolated";
            case SliceFunction s -> "slice";
            case BlendDensityFunction b -> "blend_density";
            default -> throw new IllegalStateException("IR exporter does not support " + f.getClass().getName() + ": " + f);
        };
    }

    private void writeNode(JsonWriter json, DensityFunction f) throws IOException {
        json.beginObject();
        json.name("id").value(ids.get(f));
        json.name("t").value(typeName(f));
        Interval range = f.range();
        json.name("range").beginArray().value(fbits(range.min())).value(fbits(range.max())).endArray();
        json.name("axes").value(f.domainAxes());
        switch (f) {
            case ExportCache c -> {
                json.name("cache_id").value(c.id());
                json.name("in").value(ids.get(c.input()));
            }
            case ConstantFunction c -> json.name("v").value(fbits(c.value()));
            case SimpleDensityFunction s -> json.name("kind").value(s.id());
            case GradientFunction g -> {
                json.name("axis").value(g.axis().getSerializedName());
                json.name("tiling").value(g.tiling().getSerializedName());
                json.name("from").value(g.fromCoordinate());
                json.name("to").value(g.toCoordinate());
                json.name("from_value").value(fbits(g.fromValue()));
                json.name("to_value").value(fbits(g.toValue()));
            }
            case NoiseFunction n -> {
                json.name("noise").value(noiseKey(n.noise()));
                json.name("xz_scale").value(dbits(n.xzScale()));
                json.name("y_scale").value(dbits(n.yScale()));
                json.name("shift_x").value(ids.get(n.shiftX()));
                json.name("shift_y").value(ids.get(n.shiftY()));
                json.name("shift_z").value(ids.get(n.shiftZ()));
                json.name("shift_x_zero").value(n.shiftX().equals(DensityFunctions.zero()));
                json.name("shift_y_zero").value(n.shiftY().equals(DensityFunctions.zero()));
                json.name("shift_z_zero").value(n.shiftZ().equals(DensityFunctions.zero()));
            }
            case ShiftNoiseFunction.ShiftA s -> json.name("noise").value(noiseKey(s.offsetNoise()));
            case ShiftNoiseFunction.ShiftB s -> json.name("noise").value(noiseKey(s.offsetNoise()));
            case ShiftNoiseFunction.Shift s -> json.name("noise").value(noiseKey(s.offsetNoise()));
            case BlendedNoise b -> {
                json.name("xz_scale").value(dbits(b.xzScale()));
                json.name("y_scale").value(dbits(b.yScale()));
                json.name("xz_factor").value(dbits(b.xzFactor()));
                json.name("y_factor").value(dbits(b.yFactor()));
                json.name("smear_scale_multiplier").value(dbits(b.smearScaleMultiplier()));
            }
            case BinaryFunction b -> {
                json.name("op").value(b.type().id);
                json.name("l").value(ids.get(b.left()));
                json.name("r").value(ids.get(b.right()));
            }
            case UnaryFunction u -> {
                json.name("op").value(u.type().id);
                json.name("in").value(ids.get(u.input()));
            }
            case ClampFunction c -> {
                json.name("in").value(ids.get(c.input()));
                json.name("min").value(fbits(c.min()));
                json.name("max").value(fbits(c.max()));
            }
            case LerpFunction l -> {
                json.name("alpha").value(ids.get(l.alpha()));
                json.name("first").value(ids.get(l.first()));
                json.name("second").value(ids.get(l.second()));
            }
            case RangeChoiceFunction r -> {
                json.name("in").value(ids.get(r.input()));
                json.name("min_inclusive").value(fbits(r.minInclusive()));
                json.name("max_exclusive").value(fbits(r.maxExclusive()));
                json.name("when_in").value(ids.get(r.whenInRange()));
                json.name("when_out").value(ids.get(r.whenOutOfRange()));
            }
            case IntervalSelectFunction i -> {
                json.name("in").value(ids.get(i.input()));
                json.name("thresholds").beginArray();
                for (int k = 0; k < i.thresholds().size(); k++) json.value(fbits(i.thresholds().getFloat(k)));
                json.endArray();
                json.name("functions").beginArray();
                for (DensityFunction g : i.functions()) json.value(ids.get(g));
                json.endArray();
            }
            case SplineFunction s -> {
                json.name("spline");
                writeSpline(json, s.spline());
            }
            case InterpolatedFunction i -> {
                json.name("in").value(ids.get(i.input()));
                json.name("cell_xz").value(i.cellSizeXz());
                json.name("cell_y").value(i.cellSizeY());
            }
            case SliceFunction s -> {
                json.name("axis").value(s.axis().getSerializedName());
                json.name("coord").value(s.coordinate());
                json.name("in").value(ids.get(s.input()));
            }
            case BlendDensityFunction b -> json.name("in").value(ids.get(b.input()));
            default -> throw new IllegalStateException("unreachable");
        }
        json.endObject();
    }

    private void writeSpline(JsonWriter json, CubicSpline<SplineFunction.Coordinate> spline) throws IOException {
        json.beginObject();
        switch (spline) {
            case CubicSpline.Constant<SplineFunction.Coordinate> c -> json.name("k").value(fbits(c.value()));
            case CubicSpline.Multipoint<SplineFunction.Coordinate> m -> {
                json.name("coord").value(ids.get(m.coordinate().function()));
                json.name("loc").beginArray();
                for (float v : m.locations()) json.value(fbits(v));
                json.endArray();
                json.name("der").beginArray();
                for (float v : m.derivatives()) json.value(fbits(v));
                json.endArray();
                json.name("val").beginArray();
                for (CubicSpline<SplineFunction.Coordinate> v : m.values()) writeSpline(json, v);
                json.endArray();
            }
            default -> throw new IllegalStateException("unknown spline " + spline);
        }
        json.endObject();
    }

    private String noiseKey(Holder<NormalNoise> holder) {
        String key = holder.unwrapKey().orElseThrow(() -> new IllegalStateException("unnamed noise")).identifier().toString();
        noiseDefs.putIfAbsent(key, holder);
        return key;
    }

    private static void writeNoiseDef(JsonWriter json, NormalNoise noise) throws IOException {
        try {
            var pf = NormalNoise.class.getDeclaredField("parameters");
            pf.setAccessible(true);
            Object p = pf.get(noise);
            Class<?> pc = p.getClass();
            double baseAmplitude = (double) accessor(pc, "baseAmplitude").invoke(p);
            int baseOctave = (int) accessor(pc, "baseOctave").invoke(p);
            int octaveCount = (int) accessor(pc, "octaveCount").invoke(p);
            Enum<?> normalize = (Enum<?>) accessor(pc, "normalize").invoke(p);
            it.unimi.dsi.fastutil.doubles.DoubleList mods = (it.unimi.dsi.fastutil.doubles.DoubleList) accessor(pc, "amplitudeModifiers").invoke(p);
            json.beginObject();
            json.name("base_amplitude").value(dbits(baseAmplitude));
            json.name("base_octave").value(baseOctave);
            json.name("octave_count").value(octaveCount);
            json.name("normalize").value(normalize.name().toLowerCase());
            json.name("amplitude_modifiers").beginArray();
            for (int i = 0; i < mods.size(); i++) json.value(dbits(mods.getDouble(i)));
            json.endArray();
            Interval r = noise.range();
            json.name("range").beginArray().value(fbits(r.min())).value(fbits(r.max())).endArray();
            json.endObject();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    private static Method accessor(Class<?> c, String name) throws NoSuchMethodException {
        Method m = c.getDeclaredMethod(name);
        m.setAccessible(true);
        return m;
    }

    static long fbits(float f) {
        return Integer.toUnsignedLong(Float.floatToRawIntBits(f));
    }

    static String dbits(double d) {
        return String.format("0x%016x", Double.doubleToRawLongBits(d));
    }
}
