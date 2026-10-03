// mccuda_tests: one runner for every correctness level.
//
//   mccuda_tests [--data DIR] [--set quick|full] [--impl NAME|all] [--only unit,rng,noise,named,chunks]
//
// Exit code 0 only if every selected check is bit-exact.
#include <cstdio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mccuda/ir.h"
#include "mccuda/volume_eval.h"
#include "mccuda_cuda/generator.h"
#include "mccuda_cuda/terrain_gpu.h"
#include "tests.h"

#ifndef MCCUDA_DATA_DIR
#define MCCUDA_DATA_DIR "data"
#endif

using namespace mcc;
using namespace mcc::test;

namespace mcc::test {
std::vector<Impl> cuda_implementations(const IR& ir);   // cuda_impls.cpp
}

std::vector<Impl> mcc::test::implementations(const IR& ir) {
    std::vector<Impl> v;
    v.push_back({"cpu", true, [](const IR& ir, const SeedTables& t, const std::vector<std::pair<int, int>>& chunks, std::vector<float>& d, std::vector<uint8_t>& f) {
                     const size_t per = static_cast<size_t>(16 * 16 * ir.height);
                     d.assign(per * chunks.size(), 0.0f);
                     f.assign(per * chunks.size(), 0);
                     VolumeEvaluator eval(ir, t);
                     for (size_t i = 0; i < chunks.size(); ++i)
                         cpu_generate_chunk(ir, eval, chunks[i].first, chunks[i].second, d.data() + i * per, f.data() + i * per);
                 }});
    for (auto& c : cuda_implementations(ir)) v.push_back(c);
    return v;
}

int main(int argc, char** argv) {
    std::string data = MCCUDA_DATA_DIR, set_name = "quick", impl_name = "all";
    std::set<std::string> only = {"unit", "rng", "noise", "named", "chunks", "terrain", "zoom"};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--data") data = next();
        else if (a == "--set") set_name = next();
        else if (a == "--impl") impl_name = next();
        else if (a == "--only") {
            only.clear();
            std::stringstream ss(next());
            std::string tok;
            while (std::getline(ss, tok, ',')) only.insert(tok);
        } else throw std::runtime_error("unknown argument " + a);
    }
    try {
        const std::string ver = "26.3";
        const std::string ir_path = data + "/gamedata/" + ver + "/overworld_ir.json";
        const std::string oracle = data + "/oracle/" + ver;
        bool ok = true;
        if (only.count("unit")) ok &= run_unit();
        if (only.count("rng")) ok &= run_rng(oracle + "/rng.json");
        IR ir = load_ir(ir_path);
        if (only.count("noise")) ok &= run_noise(ir, oracle + "/noise", set_name == "quick");
        OracleSet set = load_oracle_set(oracle + "/" + set_name);
        if (only.count("debug-gen")) {
            const OracleSeed& s0 = set.seeds.front();
            SeedTables t = build_seed_tables(ir, s0.seed);
            for (const auto& c : s0.chunks) {
                std::printf("[debug-gen] seed %lld chunk (%d,%d)\n", static_cast<long long>(s0.seed), c.cx, c.cz);
                int bad = cuda::debug_compare_generated(ir, t, c.cx, c.cz);
                if (bad) { ok = false; break; }
            }
        }
        if (only.count("zoom")) {
            // GPU zoom (FP32 filter + FP64 near-tie fallback) vs the plain FP64 BiomeManager zoom.
            std::printf("[zoom] FP32-filtered BiomeManager zoom vs FP64 reference, random blocks\n");
            int64_t checked = 0, bad = 0, fb = 0;
            for (const OracleSeed& s : set.seeds) {
                cuda::ZoomSelfTest r = cuda::zoom_selftest(s.seed, set_name == "quick" ? (int64_t{1} << 24) : (int64_t{1} << 26));
                checked += r.checked;
                bad += r.mismatches;
                fb += r.fallbacks;
            }
            std::printf("  zoom fast path                     %s  (%lld checked, %lld mismatches; %lld near ties took the FP64 path)\n",
                        bad == 0 ? "PASS" : "FAIL", static_cast<long long>(checked), static_cast<long long>(bad), static_cast<long long>(fb));
            ok &= bad == 0;
        }
        if (only.count("named")) ok &= run_named_functions(ir, set);
        if (only.count("chunks")) {
            int ran = 0;
            for (const Impl& impl : implementations(ir)) {
                if (impl_name != "all" && impl_name != impl.name) continue;
                ok &= run_chunks(ir, set, impl);
                ++ran;
            }
            if (ran == 0) throw std::runtime_error("no implementation named " + impl_name);
        }
        if (only.count("terrain")) {
            TerrainModel tm = load_terrain_model(data + "/gamedata/" + ver + "/overworld_terrain.json");
            int ran = 0;
            for (const TerrainImpl& impl : terrain_implementations()) {
                if (impl_name != "all" && impl_name != impl.name) continue;
                ok &= run_terrain(ir, tm, set, impl);
                ++ran;
            }
            if (ran == 0) std::printf("[terrain] no implementation named %s\n", impl_name.c_str());
        }
        std::printf("\nRESULT: %s\n", ok ? "PASS (bit-exact)" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
