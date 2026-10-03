#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/seed_tables.h"
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

bool run_unit();
bool run_rng(const std::string& path);
bool run_noise(const IR& ir, const std::string& dir, bool quick);
bool run_chunks(const IR& ir, const OracleSet& set, const Impl& impl);
bool run_named_functions(const IR& ir, const OracleSet& set);

}  // namespace mcc::test
