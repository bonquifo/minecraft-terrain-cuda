// GPU implementation of the full TERRAIN step (biomes, density, aquifers,
// material rules; no carvers), bit-exact with vanilla 26.3.
#pragma once

#include <cstdint>
#include <memory>

#include "mccuda/ir.h"
#include "mccuda/terrain_model.h"
#include "mccuda_cuda/generator.h"

namespace mcc::cuda {

struct TerrainTiming {
    double h2d_ms = 0, biome_ms = 0, density_ms = 0, aquifer_ms = 0, surface_ms = 0, d2h_ms = 0, wall_ms = 0;
};

class FullTerrainGenerator {
public:
    FullTerrainGenerator(const IR& ir, const TerrainModel& tm, int max_batch_chunks = 256);
    ~FullTerrainGenerator();
    FullTerrainGenerator(const FullTerrainGenerator&) = delete;
    FullTerrainGenerator& operator=(const FullTerrainGenerator&) = delete;

    void set_seed(int64_t seed);
    // terrain_out: u8 palette index per block (canonical layout), chunks back to back.
    // biomes_out (u8 per quart) and aquifer_out (u8 per block, after doFill) are optional.
    void generate(const ChunkXZ* chunks, int count, uint8_t* terrain_out, uint8_t* biomes_out, uint8_t* aquifer_out, TerrainTiming* timing = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mcc::cuda
