// GPU terrain generator (core pipeline: final density + aquifer-disabled fill).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/seed_tables.h"

namespace mcc::cuda {

struct ChunkXZ {
    int32_t x, z;
};

struct Timing {
    double h2d_ms = 0, kernel_ms = 0, d2h_ms = 0, wall_ms = 0;
};

struct VersionInfo {
    int id;
    std::string name;
    std::string description;
};

std::vector<VersionInfo> versions();

class TerrainGenerator {
public:
    TerrainGenerator(const IR& ir, int version, int max_batch_chunks = 1024);
    ~TerrainGenerator();
    TerrainGenerator(const TerrainGenerator&) = delete;
    TerrainGenerator& operator=(const TerrainGenerator&) = delete;

    void set_seed(const SeedTables& tables);
    // density_out may be null. Outputs use the canonical layout, chunks back to back.
    void generate(const ChunkXZ* chunks, int count, float* density_out, uint8_t* fill_out, Timing* timing = nullptr);
    int version() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs the generated evaluators on the host (no GPU): isolates code generation
// from device compilation when debugging.
void generated_host_chunk(const IR& ir, const SeedTables& tables, int cx, int cz, float* density, uint8_t* fill);

// Compares every memoised node of the generated evaluators with the CPU
// reference interpreter for one chunk's columns and corners; returns #mismatches.
int debug_compare_generated(const IR& ir, const SeedTables& tables, int cx, int cz);

void print_device();

}  // namespace mcc::cuda
