// Levels 1-3: named density functions (L1), final density (L2) and block fill
// (L3) per chunk, compared bit-for-bit with the oracle.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "check.h"
#include "mccuda/volume_eval.h"
#include "tests.h"

namespace mcc::test {

namespace {

const char* kBlockNames[] = {"air", "stone", "water", "lava"};

int ulp_distance(float a, float b) {
    int32_t ia = static_cast<int32_t>(f2bits(a)), ib = static_cast<int32_t>(f2bits(b));
    if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
    if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
    int64_t d = static_cast<int64_t>(ia) - ib;
    return static_cast<int>(d < 0 ? -d : d);
}

}  // namespace

bool run_chunks(const IR& ir, const OracleSet& set, const Impl& impl) {
    std::printf("[chunks] %s  impl=%s\n", set.dir.c_str(), impl.name.c_str());
    Check cd("L2 final density (" + impl.name + ")"), cf("L3 block fill (" + impl.name + ")");
    const size_t per = set.chunk_floats();
    size_t chunks_total = 0;
    double seconds = 0.0;
    for (const OracleSeed& s : set.seeds) {
        SeedTables tables = build_seed_tables(ir, s.seed);
        std::vector<std::pair<int, int>> pos;
        for (const auto& c : s.chunks) pos.emplace_back(c.cx, c.cz);
        std::vector<float> density;
        std::vector<uint8_t> fill;
        auto t0 = std::chrono::steady_clock::now();
        impl.generate(ir, tables, pos, density, fill);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> want_d = set.density(s);
        std::vector<uint8_t> want_f = set.fill(s);
        if (fill.size() != want_f.size()) throw std::runtime_error("fill output size mismatch");
        for (size_t ci = 0; ci < s.chunks.size(); ++ci) {
            const OracleChunk& c = s.chunks[ci];
            for (size_t i = 0; i < per; ++i) {
                size_t o = ci * per + i;
                int x = static_cast<int>(i % 16), z = static_cast<int>((i / 16) % 16), y = static_cast<int>(i / 256) + set.min_y;
                if (impl.produces_density) {
                    ++cd.total;
                    if (f2bits(density[o]) != f2bits(want_d[o]))
                        cd.fail("seed#%d chunk(%d,%d)[%s] block(%d,%d,%d): want %.9g (0x%08x) got %.9g (0x%08x), %d ulp",
                                s.index, c.cx, c.cz, c.tag.c_str(), c.cx * 16 + x, y, c.cz * 16 + z, want_d[o], f2bits(want_d[o]),
                                density[o], f2bits(density[o]), ulp_distance(want_d[o], density[o]));
                }
                ++cf.total;
                if (fill[o] != want_f[o])
                    cf.fail("seed#%d chunk(%d,%d)[%s] block(%d,%d,%d): want %s got %s", s.index, c.cx, c.cz, c.tag.c_str(),
                            c.cx * 16 + x, y, c.cz * 16 + z, kBlockNames[want_f[o] & 3], kBlockNames[fill[o] & 3]);
            }
        }
        chunks_total += s.chunks.size();
    }
    std::printf("  %zu chunks over %zu seeds, generator time %.2f s\n", chunks_total, set.seeds.size(), seconds);
    if (impl.produces_density) cd.print();
    cf.print();
    return (cd.ok() || !impl.produces_density) && cf.ok();
}

bool run_named_functions(const IR& ir, const OracleSet& set) {
    std::printf("[named] L1 named density functions, CPU reference port\n");
    Check c("L1 named functions");
    int dumps = 0;
    for (const OracleSeed& s : set.seeds) {
        if (s.node_dumps.empty()) continue;
        SeedTables tables = build_seed_tables(ir, s.seed);
        VolumeEvaluator eval(ir, tables);
        for (const std::string& dump : s.node_dumps) {
            size_t ci = static_cast<size_t>(std::stoi(dump.substr(dump.find("_c") + 2)));
            const OracleChunk& ch = s.chunks.at(ci);
            Volume v{16, set.height, 16, ch.cx * 16, set.min_y, ch.cz * 16, 1, 1, 1};
            std::vector<float> buf(static_cast<size_t>(v.size()));
            for (const auto& [name, file] : set.named) {
                eval.clear_caches();
                eval.sample(ir.root(name), v, buf.data());
                std::vector<float> want = set.node(dump, file);
                uint64_t bad = 0;
                for (int z = 0; z < 16; ++z)
                    for (int x = 0; x < 16; ++x)
                        for (int y = 0; y < set.height; ++y) {
                            size_t o = static_cast<size_t>((y * 16 + z) * 16 + x);
                            float got = buf[static_cast<size_t>(v.index(x, y, z))];
                            ++c.total;
                            if (f2bits(got) != f2bits(want[o])) {
                                if (bad++ == 0)
                                    c.fail("%s %s block(%d,%d,%d): want %.9g got %.9g", dump.c_str(), name.c_str(), ch.cx * 16 + x,
                                           set.min_y + y, ch.cz * 16 + z, want[o], got);
                                else
                                    ++c.failed;
                            }
                        }
            }
            ++dumps;
        }
    }
    std::printf("  %d chunk dumps x %zu functions\n", dumps, set.named.size());
    c.print();
    return c.ok() && dumps > 0;
}

}  // namespace mcc::test
