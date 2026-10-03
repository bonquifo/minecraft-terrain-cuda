// The full-TERRAIN model exported by the oracle (overworld_terrain.json):
// material-rule tree, block palette, biome parameter list and biome climate.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mcc {

enum class RuleType : uint8_t { Block, Sequence, Condition, Bandlands, OreVein };
enum class CondType : uint8_t {
    Biome, NoiseThreshold, VerticalGradient, YAbove, Water, Temperature, Steep, Not, Hole, AbovePreliminarySurface, StoneDepth,
};

struct Condition {
    CondType type;
    // biome
    uint64_t biome_mask[2] = {0, 0};          // bit per biome index (< 128)
    // noise threshold
    std::string noise;
    double min = 0, max = 0;
    bool is_3d = false;
    // vertical gradient
    std::string random_name;
    int true_at_and_below = 0, false_at_and_above = 0;
    // y_above / water / stone_depth
    int anchor = 0, surface_depth_multiplier = 0, offset = 0, secondary_depth_range = 0;
    bool add_stone_depth = false, add_surface_depth = false, ceiling = false;
    int child = -1;                            // not
};

struct Rule {
    RuleType type;
    int state = -1;                            // block
    std::vector<int> rules;                    // sequence
    int cond = -1, then = -1;                  // condition
    // ore vein
    int ore = -1, raw_ore = -1, filler = -1;
    float raw_ore_chance = 0.0f;
    std::string density, richness, filler_gap; // IR root names
};

struct BiomeInfo {
    std::string name;
    float temperature;
    bool frozen_modifier;
};

struct BiomeParam {
    int64_t min[7], max[7];                    // 6 climate params + offset (min == max)
    int biome;
};

struct BlockInfo {
    std::string name;
    bool air, water, lava, fluid;
};

struct TerrainModel {
    int gen_min_y = 0, gen_depth = 0, sea_level = 0;
    std::vector<BlockInfo> palette;
    int default_block = -1, default_fluid = -1, air = -1, water = -1, lava = -1, packed_ice = -1, snow_block = -1;
    int terracotta[7] = {-1, -1, -1, -1, -1, -1, -1};   // terracotta, white, orange, yellow, brown, red, light gray
    std::vector<std::string> biome_names;
    std::vector<BiomeInfo> biomes;
    std::vector<BiomeParam> params;            // vanilla order
    std::vector<Rule> rules;
    std::vector<Condition> conds;
    int root_rule = -1;
    int biome_eroded_badlands = -1, biome_frozen_ocean = -1, biome_deep_frozen_ocean = -1;

    int block(const std::string& name) const;
};

TerrainModel load_terrain_model(const std::string& path);

}  // namespace mcc
