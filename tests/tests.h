#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/seed_tables.h"
#include "mccuda/terrain_model.h"
#include "oracle_data.h"

namespace mcc::test {

// A chunk generator under test: fills density (may be skipped if the
// implementation only produces blocks) and fill for each chunk, in the
// canonical layout ((y - minY) * 16 + z) * 16 + x, chunks back to back.
struct Impl {
    std::string name;
    bool produces_density = true;
    std::function<void(const IR&, const SeedTables&, const std::vector<std::pair<int, int>>&, std::vector<float>&, std::vector<uint8_t>&)> generate;
};

std::vector<Impl> implementations(const IR& ir);

// A full-terrain generator under test: biomes (u8 per quart), aquifer fill and
// final terrain (u16 palette index per block), chunks back to back.
struct TerrainImpl {
    std::string name;
    std::function<void(const IR&, const TerrainModel&, int64_t, const std::vector<std::pair<int, int>>&, std::vector<uint8_t>&, std::vector<uint16_t>&,
                       std::vector<uint16_t>&)>
        generate;
};
std::vector<TerrainImpl> terrain_implementations();
std::vector<TerrainImpl> cuda_terrain_implementations();

bool run_unit();
bool run_rng(const std::string& path);
bool run_noise(const IR& ir, const std::string& dir, bool quick);
bool run_chunks(const IR& ir, const OracleSet& set, const Impl& impl);
bool run_named_functions(const IR& ir, const OracleSet& set);
bool run_terrain(const IR& ir, const TerrainModel& tm, const OracleSet& set, const TerrainImpl& impl);

}  // namespace mcc::test
