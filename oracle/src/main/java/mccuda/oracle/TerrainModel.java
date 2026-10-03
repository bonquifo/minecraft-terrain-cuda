package mccuda.oracle;

import java.io.IOException;
import java.io.Writer;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.google.gson.stream.JsonWriter;
import com.mojang.datafixers.util.Pair;

import net.minecraft.commands.arguments.blocks.BlockStateParser;
import net.minecraft.core.Holder;
import net.minecraft.core.HolderSet;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.level.LevelHeightAccessor;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.Climate;
import net.minecraft.world.level.biome.MultiNoiseBiomeSource;
import net.minecraft.world.level.biome.MultiNoiseBiomeSourceParameterList;
import net.minecraft.world.level.biome.MultiNoiseBiomeSourceParameterLists;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Aquifer;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.Noises;
import net.minecraft.world.level.levelgen.VerticalAnchor;
import net.minecraft.world.level.levelgen.WorldGenerationContext;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.material.condition.AbovePreliminarySurfaceCondition;
import net.minecraft.world.level.levelgen.material.condition.BiomeCondition;
import net.minecraft.world.level.levelgen.material.condition.HoleCondition;
import net.minecraft.world.level.levelgen.material.condition.MaterialCondition;
import net.minecraft.world.level.levelgen.material.condition.NoiseThresholdCondition;
import net.minecraft.world.level.levelgen.material.condition.NotCondition;
import net.minecraft.world.level.levelgen.material.condition.SteepCondition;
import net.minecraft.world.level.levelgen.material.condition.StoneDepthCondition;
import net.minecraft.world.level.levelgen.material.condition.TemperatureCondition;
import net.minecraft.world.level.levelgen.material.condition.VerticalGradientCondition;
import net.minecraft.world.level.levelgen.material.condition.WaterCondition;
import net.minecraft.world.level.levelgen.material.condition.YCondition;
import net.minecraft.world.level.levelgen.material.rule.BandlandsRule;
import net.minecraft.world.level.levelgen.material.rule.BlockRule;
import net.minecraft.world.level.levelgen.material.rule.ConditionRule;
import net.minecraft.world.level.levelgen.material.rule.MaterialRule;
import net.minecraft.world.level.levelgen.material.rule.OreVeinRule;
import net.minecraft.world.level.levelgen.material.rule.SequenceRule;
import net.minecraft.world.level.levelgen.synth.NormalNoise;

/**
 * Everything beyond the final density that the full TERRAIN step needs, exported
 * from vanilla objects into our own JSON (data/gamedata/26.3/overworld_terrain.json):
 * the material-rule tree (anchors resolved), the block-state palette, the multi-noise
 * biome parameter list (in vanilla order, which fixes the R-tree), biome climate
 * settings, and the names of the extra density-function roots and noises, which
 * IrExporter adds to the IR.
 */
final class TerrainModel {
    final Main.Vanilla v;
    final MultiNoiseBiomeSource biomeSource;
    final NoiseBasedChunkGenerator generator;
    final WorldGenerationContext genContext;
    final Map<String, DensityFunction> extraRoots = new LinkedHashMap<>();
    final Map<String, Holder<NormalNoise>> extraNoises = new TreeMap<>();
    final List<BlockState> palette = new ArrayList<>();
    final List<String> biomeNames = new ArrayList<>();
    private int oreCount;

    TerrainModel(Main.Vanilla v) {
        this.v = v;
        Holder<MultiNoiseBiomeSourceParameterList> preset = v.lookup.lookupOrThrow(Registries.MULTI_NOISE_BIOME_SOURCE_PARAMETER_LIST)
            .getOrThrow(MultiNoiseBiomeSourceParameterLists.OVERWORLD);
        this.biomeSource = MultiNoiseBiomeSource.createFromPreset(preset);
        this.generator = new NoiseBasedChunkGenerator(biomeSource, Holder.direct(v.overworld));
        this.genContext = new WorldGenerationContext(generator, LevelHeightAccessor.create(v.overworld.noiseSettings().minY(), v.overworld.noiseSettings().height()));

        for (BlockState s : new BlockState[] {Blocks.AIR.defaultBlockState(), v.overworld.defaultBlock(), v.overworld.defaultFluid(), Blocks.LAVA.defaultBlockState(),
            Blocks.TERRACOTTA.defaultBlockState(), Blocks.DYED_TERRACOTTA.white().defaultBlockState(), Blocks.DYED_TERRACOTTA.orange().defaultBlockState(),
            Blocks.DYED_TERRACOTTA.yellow().defaultBlockState(), Blocks.DYED_TERRACOTTA.brown().defaultBlockState(), Blocks.DYED_TERRACOTTA.red().defaultBlockState(),
            Blocks.DYED_TERRACOTTA.lightGray().defaultBlockState(), Blocks.PACKED_ICE.defaultBlockState(), Blocks.SNOW_BLOCK.defaultBlockState()}) {
            paletteIndex(s);
        }

        NoiseRouter router = v.overworld.noiseRouter();
        extraRoots.put("climate/temperature", router.temperature());
        extraRoots.put("climate/vegetation", router.vegetation());
        extraRoots.put("climate/continents", router.continents());
        extraRoots.put("climate/erosion", router.erosion());
        extraRoots.put("climate/depth", router.depth());
        extraRoots.put("climate/ridges", router.ridges());
        extraRoots.put("surface/chunk_surface_level", router.chunkSurfaceLevel());
        Aquifer.Config aq = v.overworld.aquifers().orElseThrow();
        extraRoots.put("aquifer/barrier", aq.barrierNoise());
        extraRoots.put("aquifer/fluid_level_floodedness", aq.fluidLevelFloodednessNoise());
        extraRoots.put("aquifer/fluid_level_spread", aq.fluidLevelSpreadNoise());
        extraRoots.put("aquifer/lava", aq.lavaNoise());
        extraRoots.put("aquifer/exclusion", aq.exclusion());
        extraRoots.put("aquifer/surface_level", aq.surfaceLevel());

        for (ResourceKey<NormalNoise> k : List.of(Noises.CLAY_BANDS_OFFSET, Noises.SURFACE, Noises.SURFACE_SECONDARY, Noises.BADLANDS_PILLAR,
            Noises.BADLANDS_PILLAR_ROOF, Noises.BADLANDS_SURFACE, Noises.ICEBERG_PILLAR, Noises.ICEBERG_PILLAR_ROOF, Noises.ICEBERG_SURFACE)) {
            extraNoises.put(k.identifier().toString(), v.noises.getOrThrow(k));
        }
        for (var e : v.lookup.lookupOrThrow(Registries.BIOME).listElements().toList()) biomeNames.add(e.key().identifier().toString());
        biomeNames.sort(String::compareTo);
    }

    MaterialRule materialRule() {
        return v.overworld.materialRule().value();
    }

    int paletteIndex(BlockState s) {
        int i = palette.indexOf(s);
        if (i < 0) {
            palette.add(s);
            i = palette.size() - 1;
        }
        return i;
    }

    int biomeIndex(Holder<Biome> b) {
        int i = biomeNames.indexOf(b.unwrapKey().orElseThrow().identifier().toString());
        if (i < 0) throw new IllegalStateException("unknown biome " + b);
        return i;
    }

    /** Writes the terrain model; the rule walk also registers ore-vein density roots and threshold noises. */
    void export(Path out) throws IOException {
        Files.createDirectories(out.getParent());
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
            json.setIndent(" ");
            json.beginObject();
            json.name("format").value("mccuda-terrain/1");
            json.name("minecraft").value(Main.MC_VERSION);
            json.name("gen_min_y").value(genContext.getMinGenY());
            json.name("gen_depth").value(genContext.getGenDepth());
            json.name("sea_level").value(v.overworld.seaLevel());
            json.name("material_rule");
            writeRule(json, materialRule());
            json.name("palette").beginArray();
            for (BlockState s : palette) json.value(BlockStateParser.serialize(s));
            json.endArray();
            json.name("default_block").value(paletteIndex(v.overworld.defaultBlock()));
            json.name("default_fluid").value(paletteIndex(v.overworld.defaultFluid()));
            json.name("biome_names").beginArray();
            for (String n : biomeNames) json.value(n);
            json.endArray();
            writeBiomes(json);
            writeParameterList(json);
            json.name("roots").beginArray();
            for (String k : extraRoots.keySet()) json.value(k);
            json.endArray();
            json.endObject();
        }
        System.out.printf("terrain model: palette %d states, %d biomes, %d ore veins -> %s%n", palette.size(), biomeNames.size(), oreCount, out);
    }

    private void writeRule(JsonWriter json, MaterialRule rule) throws IOException {
        json.beginObject();
        switch (rule) {
            case MaterialRule.HolderHolder(Holder<MaterialRule> h) -> {
                json.name("t").value("ref");
                json.name("rule");
                writeRule(json, h.value());
            }
            case BlockRule b -> {
                json.name("t").value("block");
                json.name("state").value(paletteIndex(b.resultState()));
            }
            case SequenceRule s -> {
                json.name("t").value("sequence");
                json.name("rules").beginArray();
                for (MaterialRule r : s.sequence()) writeRule(json, r);
                json.endArray();
            }
            case ConditionRule c -> {
                json.name("t").value("condition");
                json.name("if");
                writeCondition(json, c.ifTrue());
                json.name("then");
                writeRule(json, c.thenRun());
            }
            case BandlandsRule b -> json.name("t").value("bandlands");
            case OreVeinRule o -> {
                int i = oreCount++;
                json.name("t").value("ore_vein");
                json.name("ore").value(paletteIndex(o.oreBlock()));
                json.name("raw_ore").value(paletteIndex(o.rawOreBlock()));
                json.name("filler").value(paletteIndex(o.fillerBlock()));
                json.name("raw_ore_chance").value(IrExporter.fbits(o.rawOreChance()));
                String base = "ore_vein/" + i + "/";
                extraRoots.put(base + "density", o.density());
                extraRoots.put(base + "richness", o.richness());
                extraRoots.put(base + "filler_gap", o.fillerGap());
                json.name("density").value(base + "density");
                json.name("richness").value(base + "richness");
                json.name("filler_gap").value(base + "filler_gap");
            }
            default -> throw new IllegalStateException("unsupported material rule " + rule);
        }
        json.endObject();
    }

    private void writeCondition(JsonWriter json, MaterialCondition c) throws IOException {
        json.beginObject();
        switch (c) {
            case MaterialCondition.HolderHolder(Holder<MaterialCondition> h) -> {
                json.name("t").value("ref");
                json.name("cond");
                writeCondition(json, h.value());
            }
            case BiomeCondition b -> {
                json.name("t").value("biome");
                json.name("biomes").beginArray();
                HolderSet<Biome> set = b.biomes();
                for (Holder<Biome> h : set) json.value(biomeIndex(h));
                json.endArray();
            }
            case NoiseThresholdCondition n -> {
                json.name("t").value("noise_threshold");
                json.name("noise").value(n.noise().identifier().toString());
                extraNoises.put(n.noise().identifier().toString(), v.noises.getOrThrow(n.noise()));
                json.name("min").value(IrExporter.dbits(n.minThreshold()));
                json.name("max").value(IrExporter.dbits(n.maxThreshold()));
                json.name("is_3d").value(n.is3d());
            }
            case VerticalGradientCondition g -> {
                json.name("t").value("vertical_gradient");
                json.name("random_name").value(g.randomName().toString());
                json.name("true_at_and_below").value(g.trueAtAndBelow().resolveY(genContext));
                json.name("false_at_and_above").value(g.falseAtAndAbove().resolveY(genContext));
            }
            case YCondition y -> {
                json.name("t").value("y_above");
                json.name("anchor").value(y.anchor().resolveY(genContext));
                json.name("surface_depth_multiplier").value(y.surfaceDepthMultiplier());
                json.name("add_stone_depth").value(y.addStoneDepth());
            }
            case WaterCondition wc -> {
                json.name("t").value("water");
                json.name("offset").value(wc.offset());
                json.name("surface_depth_multiplier").value(wc.surfaceDepthMultiplier());
                json.name("add_stone_depth").value(wc.addStoneDepth());
            }
            case NotCondition n -> {
                json.name("t").value("not");
                json.name("cond");
                writeCondition(json, n.target());
            }
            case StoneDepthCondition s -> {
                json.name("t").value("stone_depth");
                json.name("offset").value(s.offset());
                json.name("add_surface_depth").value(s.addSurfaceDepth());
                json.name("secondary_depth_range").value(s.secondaryDepthRange());
                json.name("ceiling").value(s.surfaceType().name().equals("CEILING"));
            }
            case TemperatureCondition t -> json.name("t").value("temperature");
            case SteepCondition s -> json.name("t").value("steep");
            case HoleCondition h -> json.name("t").value("hole");
            case AbovePreliminarySurfaceCondition a -> json.name("t").value("above_preliminary_surface");
            default -> throw new IllegalStateException("unsupported material condition " + c);
        }
        json.endObject();
    }

    private void writeBiomes(JsonWriter json) throws IOException {
        json.name("biomes").beginArray();
        var lookup = v.lookup.lookupOrThrow(Registries.BIOME);
        for (String name : biomeNames) {
            Holder<Biome> h = lookup.listElements().filter(e -> e.key().identifier().toString().equals(name)).findFirst().orElseThrow();
            Biome b = h.value();
            try {
                Field f = Biome.class.getDeclaredField("climateSettings");
                f.setAccessible(true);
                Object cs = f.get(b);
                Method mod = cs.getClass().getDeclaredMethod("temperatureModifier");
                mod.setAccessible(true);
                Enum<?> m = (Enum<?>) mod.invoke(cs);
                json.beginObject();
                json.name("name").value(name);
                json.name("temperature").value(IrExporter.fbits(b.getBaseTemperature()));
                json.name("temperature_modifier").value(m.name().toLowerCase());
                json.endObject();
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException(e);
            }
        }
        json.endArray();
    }

    @SuppressWarnings("unchecked")
    private void writeParameterList(JsonWriter json) throws IOException {
        try {
            Method pm = MultiNoiseBiomeSource.class.getDeclaredMethod("parameters");
            pm.setAccessible(true);
            Climate.ParameterList<Holder<Biome>> list = (Climate.ParameterList<Holder<Biome>>) pm.invoke(biomeSource);
            json.name("biome_parameters").beginArray();
            for (Pair<Climate.ParameterPoint, Holder<Biome>> p : list.values()) {
                Climate.ParameterPoint pt = p.getFirst();
                json.beginObject();
                json.name("biome").value(biomeIndex(p.getSecond()));
                json.name("p").beginArray();
                for (Climate.Parameter q : new Climate.Parameter[] {pt.temperature(), pt.humidity(), pt.continentalness(), pt.erosion(), pt.depth(), pt.weirdness()}) {
                    json.value(q.min()).value(q.max());
                }
                json.endArray();
                json.name("offset").value(pt.offset());
                json.endObject();
            }
            json.endArray();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    static VerticalAnchor unused() { return VerticalAnchor.bottom(); }
}
