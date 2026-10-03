// Reader for the oracle dump format (see oracle/ and docs/dump-format.md).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mcc::test {

struct OracleChunk {
    int cx, cz;
    std::string tag;
};

struct OracleSeed {
    int index;
    int64_t seed;
    bool quick;
    std::vector<OracleChunk> chunks;
    std::vector<std::string> node_dumps;   // "s<seed>_c<chunk>"
};

struct OracleSet {
    std::string dir;
    int min_y = 0, height = 0;
    std::vector<OracleSeed> seeds;
    std::vector<std::pair<std::string, std::string>> named;   // (function name, file name)

    size_t chunk_floats() const { return static_cast<size_t>(16 * 16 * height); }
    std::vector<float> density(const OracleSeed& s) const;
    std::vector<uint8_t> fill(const OracleSeed& s) const;
    std::vector<float> node(const std::string& dump, const std::string& file) const;
};

OracleSet load_oracle_set(const std::string& dir);
std::vector<uint8_t> read_file(const std::string& path);

}  // namespace mcc::test
