// GPU implementation of the full TERRAIN step (biomes, density, aquifers,
// material rules; no carvers), bit-exact with vanilla 26.3.
#pragma once

#include <cstdint>
#include <memory>

#include "mccuda/ir.h"
#include "mccuda/terrain_model.h"
#include "mccuda_cuda/generator.h"

namespace mcc::cuda {

// Accumulated per stage (CUDA events on the default stream) plus host wall time.
//   biome    climate + R-tree search + tie chain        noise    density/ore corner values
//   aquifer  surface-level grid + aquifer cells          fill     final density + aquifer substance per block
//   surface  heights + material rules + extensions       h2d/d2h  transfers (d2h: terrain, plus biomes/aquifer if requested)
struct TerrainTiming {
    double h2d_ms = 0, biome_ms = 0, noise_ms = 0, aquifer_ms = 0, fill_ms = 0, surface_ms = 0, d2h_ms = 0, wall_ms = 0;
    double kernel_ms() const { return biome_ms + noise_ms + aquifer_ms + fill_ms + surface_ms; }
};

class FullTerrainGenerator {
public:
    // pipeline_slots: batches in flight (2: transfers and host work of one batch overlap
    // the next batch's kernels; 1: strictly sequential, for per-stage timing).
    FullTerrainGenerator(const IR& ir, const TerrainModel& tm, int max_batch_chunks = 256, int pipeline_slots = 2);
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

// Device self-test of the FP32-filtered BiomeManager zoom against the plain FP64 one.
struct ZoomSelfTest {
    int64_t checked, mismatches, fallbacks;   // fallbacks: near ties resolved with FP64
};
ZoomSelfTest zoom_selftest(int64_t world_seed, int64_t count);

// Page-locked host memory (faster device-to-host copies for terrain_out).
uint8_t* alloc_pinned(size_t bytes);
void free_pinned(void* p);

}  // namespace mcc::cuda
