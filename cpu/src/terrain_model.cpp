#include "mccuda/terrain_model.h"

#include <fstream>
#include <stdexcept>

#include "mccuda/javamath.h"
#include "nlohmann/json.hpp"

namespace mcc {

namespace {

using json = nlohmann::json;

double jd(const json& j) { return bits2d(std::stoull(j.get<std::string>(), nullptr, 16)); }

struct Parser {
    TerrainModel& m;

    int cond(const json& j) {
        std::string t = j["t"];
        if (t == "ref") return cond(j["cond"]);
        Condition c{};
        if (t == "biome") {
            c.type = CondType::Biome;
            for (const auto& b : j["biomes"]) {
                int i = b.get<int>();
                if (i >= 128) throw std::runtime_error("biome index >= 128");
                c.biome_mask[i >> 6] |= 1ull << (i & 63);
            }
        } else if (t == "noise_threshold") {
            c.type = CondType::NoiseThreshold;
            c.noise = j["noise"];
            c.min = jd(j["min"]);
            c.max = jd(j["max"]);
            c.is_3d = j["is_3d"];
        } else if (t == "vertical_gradient") {
            c.type = CondType::VerticalGradient;
            c.random_name = j["random_name"];
            c.true_at_and_below = j["true_at_and_below"];
            c.false_at_and_above = j["false_at_and_above"];
        } else if (t == "y_above") {
            c.type = CondType::YAbove;
            c.anchor = j["anchor"];
            c.surface_depth_multiplier = j["surface_depth_multiplier"];
            c.add_stone_depth = j["add_stone_depth"];
        } else if (t == "water") {
            c.type = CondType::Water;
            c.offset = j["offset"];
            c.surface_depth_multiplier = j["surface_depth_multiplier"];
            c.add_stone_depth = j["add_stone_depth"];
        } else if (t == "not") {
            c.type = CondType::Not;
            c.child = cond(j["cond"]);
        } else if (t == "stone_depth") {
            c.type = CondType::StoneDepth;
            c.offset = j["offset"];
            c.add_surface_depth = j["add_surface_depth"];
            c.secondary_depth_range = j["secondary_depth_range"];
            c.ceiling = j["ceiling"];
        } else if (t == "temperature") c.type = CondType::Temperature;
        else if (t == "steep") c.type = CondType::Steep;
        else if (t == "hole") c.type = CondType::Hole;
        else if (t == "above_preliminary_surface") c.type = CondType::AbovePreliminarySurface;
        else throw std::runtime_error("unknown condition " + t);
        m.conds.push_back(std::move(c));
        return static_cast<int>(m.conds.size()) - 1;
    }

    int rule(const json& j) {
        std::string t = j["t"];
        if (t == "ref") return rule(j["rule"]);
        Rule r{};
        if (t == "block") {
            r.type = RuleType::Block;
            r.state = j["state"];
        } else if (t == "sequence") {
            r.type = RuleType::Sequence;
            for (const auto& c : j["rules"]) r.rules.push_back(rule(c));
        } else if (t == "condition") {
            r.type = RuleType::Condition;
            r.cond = cond(j["if"]);
            r.then = rule(j["then"]);
        } else if (t == "bandlands") {
            r.type = RuleType::Bandlands;
        } else if (t == "ore_vein") {
            r.type = RuleType::OreVein;
            r.ore = j["ore"];
            r.raw_ore = j["raw_ore"];
            r.filler = j["filler"];
            r.raw_ore_chance = bits2f(static_cast<uint32_t>(j["raw_ore_chance"].get<uint64_t>()));
            r.density = j["density"];
            r.richness = j["richness"];
            r.filler_gap = j["filler_gap"];
        } else {
            throw std::runtime_error("unknown rule " + t);
        }
        m.rules.push_back(std::move(r));
        return static_cast<int>(m.rules.size()) - 1;
    }
};

}  // namespace

int TerrainModel::block(const std::string& name) const {
    for (size_t i = 0; i < palette.size(); ++i)
        if (palette[i].name == name) return static_cast<int>(i);
    return -1;
}

TerrainModel load_terrain_model(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open terrain model " + path);
    json doc = json::parse(in);
    if (doc["format"] != "mccuda-terrain/1") throw std::runtime_error("unexpected terrain model format");
    TerrainModel m;
    m.gen_min_y = doc["gen_min_y"];
    m.gen_depth = doc["gen_depth"];
    m.sea_level = doc["sea_level"];
    for (const auto& p : doc["palette"]) {
        BlockInfo b;
        b.name = p.get<std::string>();
        b.air = b.name == "minecraft:air";
        b.water = b.name.rfind("minecraft:water", 0) == 0;
        b.lava = b.name.rfind("minecraft:lava", 0) == 0;
        b.fluid = b.water || b.lava;
        m.palette.push_back(b);
    }
    m.default_block = doc["default_block"];
    m.default_fluid = doc["default_fluid"];
    m.air = m.block("minecraft:air");
    m.water = m.block("minecraft:water[level=0]");
    m.lava = m.block("minecraft:lava[level=0]");
    m.packed_ice = m.block("minecraft:packed_ice");
    m.snow_block = m.block("minecraft:snow_block");
    const char* terracotta[7] = {"minecraft:terracotta", "minecraft:white_terracotta", "minecraft:orange_terracotta", "minecraft:yellow_terracotta",
                                 "minecraft:brown_terracotta", "minecraft:red_terracotta", "minecraft:light_gray_terracotta"};
    for (int i = 0; i < 7; ++i) m.terracotta[i] = m.block(terracotta[i]);
    for (int v : {m.air, m.water, m.lava, m.packed_ice, m.snow_block, m.terracotta[0], m.terracotta[6]})
        if (v < 0) throw std::runtime_error("terrain palette lacks a required block state");

    for (const auto& n : doc["biome_names"]) m.biome_names.push_back(n);
    for (const auto& b : doc["biomes"]) {
        BiomeInfo bi;
        bi.name = b["name"];
        bi.temperature = bits2f(static_cast<uint32_t>(b["temperature"].get<uint64_t>()));
        bi.frozen_modifier = b["temperature_modifier"] == "frozen";
        m.biomes.push_back(bi);
    }
    for (size_t i = 0; i < m.biome_names.size(); ++i) {
        if (m.biomes[i].name != m.biome_names[i]) throw std::runtime_error("biome order mismatch");
        if (m.biome_names[i] == "minecraft:eroded_badlands") m.biome_eroded_badlands = static_cast<int>(i);
        if (m.biome_names[i] == "minecraft:frozen_ocean") m.biome_frozen_ocean = static_cast<int>(i);
        if (m.biome_names[i] == "minecraft:deep_frozen_ocean") m.biome_deep_frozen_ocean = static_cast<int>(i);
    }
    for (const auto& p : doc["biome_parameters"]) {
        BiomeParam bp{};
        for (int d = 0; d < 6; ++d) {
            bp.min[d] = p["p"][static_cast<size_t>(2 * d)].get<int64_t>();
            bp.max[d] = p["p"][static_cast<size_t>(2 * d + 1)].get<int64_t>();
        }
        bp.min[6] = bp.max[6] = p["offset"].get<int64_t>();
        bp.biome = p["biome"];
        m.params.push_back(bp);
    }
    Parser parser{m};
    m.root_rule = parser.rule(doc["material_rule"]);
    return m;
}

}  // namespace mcc
