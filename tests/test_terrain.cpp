// Full-TERRAIN tests: biomes, aquifer fill (doFill) and final terrain
// (buildSurface) per chunk, bit-for-bit against the oracle's terrain dumps.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "check.h"
#include "mccuda/terrain.h"
#include "nlohmann/json.hpp"
#include "tests.h"

namespace mcc::test {

bool run_terrain(const IR& ir, const TerrainModel& tm, const OracleSet& set, const TerrainImpl& impl) {
    std::printf("[terrain] %s  impl=%s\n", set.dir.c_str(), impl.name.c_str());
    std::ifstream in(set.dir + "/manifest_terrain.json");
    if (!in) throw std::runtime_error("no terrain manifest in " + set.dir + " (run oracle/regen.ps1)");
    nlohmann::json m = nlohmann::json::parse(in);
    if (m["palette"].size() != tm.palette.size()) throw std::runtime_error("terrain palette size differs from the model");
    for (size_t i = 0; i < tm.palette.size(); ++i)
        if (m["palette"][i].get<std::string>() != tm.palette[i].name) throw std::runtime_error("terrain palette differs from the model");

    Check cb("biomes (" + impl.name + ")"), ca("aquifer fill (" + impl.name + ")"), ct("terrain (" + impl.name + ")");
    const size_t per = set.chunk_floats();
    const size_t qper = static_cast<size_t>(16 * set.height / 4);
    double seconds = 0;
    size_t chunks_total = 0;
    for (const OracleSeed& s : set.seeds) {
        auto biomes = read_file(set.dir + "/terrain/biomes_s" + std::to_string(s.index) + ".u8");
        auto aq = read_file(set.dir + "/terrain/aquifer_s" + std::to_string(s.index) + ".u16");
        auto te = read_file(set.dir + "/terrain/terrain_s" + std::to_string(s.index) + ".u16");
        std::vector<std::pair<int, int>> pos;
        for (const auto& c : s.chunks) pos.emplace_back(c.cx, c.cz);
        std::vector<uint8_t> gb;
        std::vector<uint16_t> ga, gt;
        auto t0 = std::chrono::steady_clock::now();
        impl.generate(ir, tm, s.seed, pos, gb, ga, gt);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (size_t ci = 0; ci < s.chunks.size(); ++ci) {
            const OracleChunk& c = s.chunks[ci];
            for (size_t q = 0; q < qper; ++q) {
                ++cb.total;
                uint8_t want = biomes[ci * qper + q], got = gb[ci * qper + q];
                if (want != got)
                    cb.fail("seed#%d chunk(%d,%d)[%s] quart(%zu,%zu,%zu): want %s got %s", s.index, c.cx, c.cz, c.tag.c_str(), q % 4, q / 16, (q / 4) % 4,
                            tm.biome_names[want].c_str(), tm.biome_names[got].c_str());
            }
            for (size_t i = 0; i < per; ++i) {
                size_t o = ci * per + i;
                uint16_t wa = static_cast<uint16_t>(aq[2 * o] | (aq[2 * o + 1] << 8));
                uint16_t wt = static_cast<uint16_t>(te[2 * o] | (te[2 * o + 1] << 8));
                int x = static_cast<int>(i % 16), z = static_cast<int>((i / 16) % 16), y = static_cast<int>(i / 256) + set.min_y;
                ++ca.total;
                if (wa != ga[o])
                    ca.fail("seed#%d chunk(%d,%d)[%s] block(%d,%d,%d): want %s got %s", s.index, c.cx, c.cz, c.tag.c_str(), c.cx * 16 + x, y, c.cz * 16 + z,
                            tm.palette[wa].name.c_str(), tm.palette[ga[o]].name.c_str());
                ++ct.total;
                if (wt != gt[o])
                    ct.fail("seed#%d chunk(%d,%d)[%s] block(%d,%d,%d): want %s got %s", s.index, c.cx, c.cz, c.tag.c_str(), c.cx * 16 + x, y, c.cz * 16 + z,
                            tm.palette[wt].name.c_str(), tm.palette[gt[o]].name.c_str());
            }
        }
        chunks_total += s.chunks.size();
    }
    std::printf("  %zu chunks, generator time %.2f s\n", chunks_total, seconds);
    cb.print();
    ca.print();
    ct.print();
    return cb.ok() && ca.ok() && ct.ok();
}

std::vector<TerrainImpl> terrain_implementations() {
    std::vector<TerrainImpl> v;
    v.push_back({"cpu", [](const IR& ir, const TerrainModel& tm, int64_t seed, const std::vector<std::pair<int, int>>& chunks, std::vector<uint8_t>& b,
                           std::vector<uint16_t>& a, std::vector<uint16_t>& t) {
                     TerrainSeed ts(ir, tm, seed);
                     for (auto [cx, cz] : chunks) {
                         ChunkTerrain out;
                         cpu_generate_terrain(ts, cx, cz, out);
                         b.insert(b.end(), out.biomes.begin(), out.biomes.end());
                         a.insert(a.end(), out.aquifer.begin(), out.aquifer.end());
                         t.insert(t.end(), out.terrain.begin(), out.terrain.end());
                     }
                 }});
    for (auto& c : cuda_terrain_implementations()) v.push_back(c);
    return v;
}

}  // namespace mcc::test
