// Full TERRAIN step (without carvers) on the CPU, mirroring vanilla 26.3:
//   biomes (doCreateBiomes) -> final density + doFill with NoiseBasedAquifer
//   -> MaterialSystem.buildSurface (material rules, ore veins, extensions).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mccuda/biome.h"
#include "mccuda/ir.h"
#include "mccuda/rng.h"
#include "mccuda/seed_tables.h"
#include "mccuda/terrain_model.h"
#include "mccuda/volume_eval.h"

namespace mcc {

// Legacy 2D simplex noise (SimplexNoise with discardNoiseOffset) used by biome temperature.
struct Simplex2D {
    uint8_t perm[256];
    double ox = 0, oy = 0;
    float get(double x, double y) const;
};

struct BiomeTemperature {
    Simplex2D temperature;           // LegacyRandomSource(1234)
    Simplex2D frozen[3];             // LegacyRandomSource(3456), NoiseStack layers 1, 0.5, 0.25
    float frozen_amp[3];
    Simplex2D biome_info;            // LegacyRandomSource(2345)
    BiomeTemperature();
    float frozen_get(double x, double y) const;
    float height_adjusted(const BiomeInfo& b, int x, int y, int z, int sea_level) const;   // Biome.getHeightAdjustedTemperature
};

// Per-seed state shared by all chunks (RandomState-equivalent beyond the noise tables).
struct TerrainSeed {
    const IR& ir;
    const TerrainModel& tm;
    SeedTables tables;
    BiomeTree tree;
    int64_t seed;
    int64_t zoom_seed;
    XoroshiroPositional random;                // RandomState.random
    XoroshiroPositional aquifer_random;        // getOrCreateRandomFactory("minecraft:aquifer")
    XoroshiroPositional ore_random;            // getOrCreateRandomFactory("minecraft:ore")
    std::map<std::string, XoroshiroPositional> gradient_random;   // vertical-gradient random names
    int clay_bands[192];                       // palette indices
    BiomeTemperature temp;
    TerrainSeed(const IR& ir, const TerrainModel& tm, int64_t seed);
};

struct ChunkTerrain {
    std::vector<uint8_t> biomes;     // (qy * 4 + qz) * 4 + qx
    std::vector<uint16_t> aquifer;   // after doFill
    std::vector<uint16_t> terrain;   // after buildSurface
    std::vector<float> density;
};

// Generates one chunk. Neighbour biomes (for BiomeManager zoom at chunk edges)
// are computed internally from scratch, each with a fresh R-tree candidate.
void cpu_generate_terrain(TerrainSeed& ts, int cx, int cz, ChunkTerrain& out);

// Biomes of one chunk exactly as vanilla's doCreateBiomes stores them.
void cpu_chunk_biomes(TerrainSeed& ts, VolumeEvaluator& eval, int cx, int cz, uint8_t* out);

}  // namespace mcc
