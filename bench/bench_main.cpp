// mccuda_bench: throughput of our generators over a square region of chunks.
//
//   mccuda_bench --mode terrain-gpu|terrain-cpu|core-gpu|core-cpu [options]
//     --region R      chunks per side (default 16)        --x0/--z0  region origin (chunk coords)
//     --seed S        world seed (default 123456789)      --reps N / --warmup W
//     --batch B       GPU chunks per launch (default 256) --threads T  CPU worker threads (default 1)
//     --version V     core-gpu kernel version (default 0) --verify F  compare the region with a vanilla dump (u16)
//     --pageable      GPU output to pageable instead of pinned host memory
//     --streams S     GPU batches in flight (default 2: transfers overlap the next batch's kernels)
//     --out FILE      CSV to append to (default bench/results/bench.csv)   --tag NAME
//
// Region semantics match the vanilla bench (oracle bench-terrain): every chunk of
// the region gets the full TERRAIN step (no carvers); biomes are computed for the
// region plus a 1-chunk border, once per chunk. The GPU walks the region in square
// tiles of floor(sqrt(batch)) chunks per side, one tile per launch when batch is a
// perfect square, and recomputes the biomes of the tile borders it shares.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/terrain.h"
#include "mccuda/volume_eval.h"
#include "mccuda_cuda/generator.h"
#include "mccuda_cuda/terrain_gpu.h"

#ifndef MCCUDA_DATA_DIR
#define MCCUDA_DATA_DIR "data"
#endif

using namespace mcc;

namespace {

struct Options {
    std::string mode = "terrain-gpu", data = MCCUDA_DATA_DIR, out = "bench/results/bench.csv", tag, verify;
    int region = 16, x0 = 0, z0 = 0, reps = 5, warmup = 2, batch = 256, threads = 1, version = 0, streams = 2;
    int64_t seed = 123456789;
    bool pinned = true;
};

struct Row {
    double seconds = 0;
    cuda::TerrainTiming t;
};

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Region chunks in tile order: tile rows, then tiles, each tile row-major.
std::vector<cuda::ChunkXZ> tile_order(const Options& o, int tile) {
    std::vector<cuda::ChunkXZ> v;
    for (int tz = 0; tz < o.region; tz += tile)
        for (int tx = 0; tx < o.region; tx += tile)
            for (int z = tz; z < std::min(tz + tile, o.region); ++z)
                for (int x = tx; x < std::min(tx + tile, o.region); ++x) v.push_back({o.x0 + x, o.z0 + z});
    return v;
}

template <typename F>
void parallel_for(int threads, int count, F&& f) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            int i;
            while ((i = next.fetch_add(1)) < count) f(t, i);
        });
    for (auto& th : pool) th.join();
}

// Compares a region (row-major chunk order, u16 or u8 palette indices) with a vanilla dump.
template <typename T>
size_t verify_region(const Options& o, const std::vector<T>& got, size_t per) {
    std::ifstream in(o.verify, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + o.verify);
    std::vector<uint16_t> want(per * o.region * o.region);
    in.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 2));
    if (static_cast<size_t>(in.gcount()) != want.size() * 2) throw std::runtime_error("vanilla dump has the wrong size: " + o.verify);
    size_t bad = 0;
    for (size_t i = 0; i < want.size(); ++i)
        if (want[i] != static_cast<uint16_t>(got[i])) ++bad;
    std::printf("verify vs %s: %zu blocks, %zu mismatches -> %s\n", o.verify.c_str(), want.size(), bad, bad == 0 ? "bit-exact" : "MISMATCH");
    return bad;
}

void write_rows(const Options& o, const std::string& impl, const std::vector<Row>& rows) {
    std::filesystem::path p(o.out);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    bool fresh = !std::filesystem::exists(p);
    std::ofstream f(p, std::ios::app);
    if (fresh)
        f << "impl,tag,mode,threads,batch,region,seed,x0,z0,rep,seconds,chunks_per_s,h2d_ms,biome_ms,noise_ms,aquifer_ms,fill_ms,surface_ms,d2h_ms,kernel_ms,"
             "streams\n";
    const int chunks = o.region * o.region;
    for (size_t r = 0; r < rows.size(); ++r) {
        const Row& w = rows[r];
        char line[512];
        std::snprintf(line, sizeof line, "%s,%s,%s,%d,%d,%d,%lld,%d,%d,%zu,%.6f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n", impl.c_str(),
                      o.tag.c_str(), o.mode.c_str(), o.threads, o.batch, o.region, static_cast<long long>(o.seed), o.x0, o.z0, r, w.seconds,
                      chunks / w.seconds, w.t.h2d_ms, w.t.biome_ms, w.t.noise_ms, w.t.aquifer_ms, w.t.fill_ms, w.t.surface_ms, w.t.d2h_ms, w.t.kernel_ms(),
                      o.streams);
        f << line;
    }
}

void report(const Options& o, const std::string& impl, const std::vector<Row>& rows) {
    const int chunks = o.region * o.region;
    std::vector<double> rate;
    for (const Row& r : rows) rate.push_back(chunks / r.seconds);
    double mean = 0, sd = 0;
    for (double x : rate) mean += x;
    mean /= rate.size();
    for (double x : rate) sd += (x - mean) * (x - mean);
    sd = rate.size() > 1 ? std::sqrt(sd / (rate.size() - 1)) : 0.0;
    std::printf("%s: %d chunks x %zu reps: %.1f chunks/s (sd %.1f, %.2f%%), %.3f ms/chunk\n", impl.c_str(), chunks, rows.size(), mean, sd,
                100.0 * sd / mean, 1000.0 / mean);
    write_rows(o, impl, rows);
}

void bench_terrain_gpu(const Options& o, const IR& ir, const TerrainModel& tm) {
    const size_t per = static_cast<size_t>(16 * 16 * ir.height);
    const int tile = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(o.batch))));
    std::vector<cuda::ChunkXZ> order = tile_order(o, tile);
    const size_t bytes = per * order.size();
    uint8_t* out = o.pinned ? cuda::alloc_pinned(bytes) : new uint8_t[bytes];
    std::vector<Row> rows;
    {
        cuda::FullTerrainGenerator gen(ir, tm, o.batch, o.streams);
        double t0 = now_s();
        gen.set_seed(o.seed);
        std::printf("terrain-gpu: seed setup %.3f s (host tables + upload, once per seed, not in the timings)\n", now_s() - t0);
        for (int rep = -o.warmup; rep < o.reps; ++rep) {
            Row r;
            double s = now_s();
            gen.generate(order.data(), static_cast<int>(order.size()), out, nullptr, nullptr, &r.t);
            r.seconds = now_s() - s;
            if (rep >= 0) rows.push_back(r);
        }
        cuda::BiomeTieStats ts = gen.biome_tie_stats();
        const double q = ts.quarts ? static_cast<double>(ts.quarts) : 1.0;
        std::printf("biome ties: the R-tree's last-result rule changed the leaf of %lld of %lld quart lookups (%.2f ppm), the biome of %lld (%.2f ppm)\n",
                    static_cast<long long>(ts.overrides), static_cast<long long>(ts.quarts), 1e6 * static_cast<double>(ts.overrides) / q,
                    static_cast<long long>(ts.biome_changes), 1e6 * static_cast<double>(ts.biome_changes) / q);
        if (!o.tag.empty() && o.tag.rfind("ties", 0) == 0) {
            std::filesystem::path tp = std::filesystem::path(o.out).parent_path() / "gpu_biome_ties.csv";
            bool fresh = !std::filesystem::exists(tp);
            std::ofstream tf(tp, std::ios::app);
            if (fresh) tf << "seed,region,quarts,leaf_overrides,biome_changes\n";
            tf << o.seed << ',' << o.region << ',' << ts.quarts << ',' << ts.overrides << ',' << ts.biome_changes << '\n';
        }
    }
    report(o, "cuda-terrain", rows);
    if (!o.verify.empty()) {
        std::vector<uint8_t> rm(bytes);
        for (size_t i = 0; i < order.size(); ++i) {
            size_t dst = static_cast<size_t>(order[i].z - o.z0) * o.region + (order[i].x - o.x0);
            std::copy(out + i * per, out + (i + 1) * per, rm.begin() + dst * per);
        }
        if (verify_region(o, rm, per)) throw std::runtime_error("GPU output differs from vanilla");
    }
    if (o.pinned) cuda::free_pinned(out);
    else delete[] out;
}

void bench_terrain_cpu(const Options& o, const IR& ir, const TerrainModel& tm) {
    const size_t per = static_cast<size_t>(16 * 16 * ir.height);
    const int qper = 16 * (ir.height / 4);
    const int border = o.region + 2, nchunks = o.region * o.region;
    TerrainSeed ts(ir, tm, o.seed);
    std::vector<uint8_t> biomes(static_cast<size_t>(border) * border * qper);
    std::vector<uint16_t> terrain(per * nchunks);
    std::vector<Row> rows;
    for (int rep = -o.warmup; rep < o.reps; ++rep) {
        Row r;
        double s = now_s();
        std::vector<std::unique_ptr<VolumeEvaluator>> evals(o.threads);
        parallel_for(o.threads, border * border, [&](int t, int i) {
            if (!evals[t]) evals[t] = std::make_unique<VolumeEvaluator>(ir, ts.tables);
            cpu_chunk_biomes(ts, *evals[t], o.x0 - 1 + i % border, o.z0 - 1 + i / border, biomes.data() + static_cast<size_t>(i) * qper);
        });
        double mid = now_s();
        parallel_for(o.threads, nchunks, [&](int t, int i) {
            int x = i % o.region, z = i / o.region;
            const uint8_t* nb[9];
            for (int dz = 0; dz < 3; ++dz)
                for (int dx = 0; dx < 3; ++dx) nb[dz * 3 + dx] = biomes.data() + static_cast<size_t>((z + dz) * border + x + dx) * qper;
            ChunkTerrain c;
            if (!evals[t]) evals[t] = std::make_unique<VolumeEvaluator>(ir, ts.tables);   // a thread may have had no biome chunk
            cpu_terrain_from_biomes(ts, *evals[t], o.x0 + x, o.z0 + z, nb, c);
            std::copy(c.terrain.begin(), c.terrain.end(), terrain.begin() + static_cast<size_t>(i) * per);
        });
        r.seconds = now_s() - s;
        r.t.biome_ms = (mid - s) * 1000.0;
        r.t.surface_ms = (now_s() - mid) * 1000.0;   // fill + surface (not split on the CPU)
        if (rep >= 0) rows.push_back(r);
    }
    report(o, "cpu-port-terrain", rows);
    if (!o.verify.empty() && verify_region(o, terrain, per)) throw std::runtime_error("CPU output differs from vanilla");
}

void bench_core_gpu(const Options& o, const IR& ir) {
    const size_t per = static_cast<size_t>(16 * 16 * ir.height);
    const int tile = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(o.batch))));
    std::vector<cuda::ChunkXZ> order = tile_order(o, tile);
    uint8_t* out = o.pinned ? cuda::alloc_pinned(per * order.size()) : new uint8_t[per * order.size()];
    SeedTables tables = build_seed_tables(ir, o.seed);
    std::vector<Row> rows;
    {
        cuda::TerrainGenerator gen(ir, o.version, o.batch);
        gen.set_seed(tables);
        for (int rep = -o.warmup; rep < o.reps; ++rep) {
            Row r;
            cuda::Timing t;
            double s = now_s();
            gen.generate(order.data(), static_cast<int>(order.size()), nullptr, out, &t);
            r.seconds = now_s() - s;
            r.t.h2d_ms = t.h2d_ms;
            r.t.fill_ms = t.kernel_ms;
            r.t.d2h_ms = t.d2h_ms;
            if (rep >= 0) rows.push_back(r);
        }
    }
    report(o, "cuda-core-v" + std::to_string(o.version), rows);
    if (o.pinned) cuda::free_pinned(out);
    else delete[] out;
}

void bench_core_cpu(const Options& o, const IR& ir) {
    const size_t per = static_cast<size_t>(16 * 16 * ir.height);
    const int nchunks = o.region * o.region;
    SeedTables tables = build_seed_tables(ir, o.seed);
    std::vector<Row> rows;
    for (int rep = -o.warmup; rep < o.reps; ++rep) {
        Row r;
        double s = now_s();
        std::vector<std::unique_ptr<VolumeEvaluator>> evals(o.threads);
        std::vector<std::vector<float>> dens(o.threads, std::vector<float>(per));
        std::vector<std::vector<uint8_t>> fill(o.threads, std::vector<uint8_t>(per));
        parallel_for(o.threads, nchunks, [&](int t, int i) {
            if (!evals[t]) evals[t] = std::make_unique<VolumeEvaluator>(ir, tables);
            cpu_generate_chunk(ir, *evals[t], o.x0 + i % o.region, o.z0 + i / o.region, dens[t].data(), fill[t].data());
        });
        r.seconds = now_s() - s;
        if (rep >= 0) rows.push_back(r);
    }
    report(o, "cpu-port-core", rows);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
            if (a == "--mode") o.mode = next();
            else if (a == "--data") o.data = next();
            else if (a == "--out") o.out = next();
            else if (a == "--tag") o.tag = next();
            else if (a == "--verify") o.verify = next();
            else if (a == "--region") o.region = std::stoi(next());
            else if (a == "--x0") o.x0 = std::stoi(next());
            else if (a == "--z0") o.z0 = std::stoi(next());
            else if (a == "--reps") o.reps = std::stoi(next());
            else if (a == "--warmup") o.warmup = std::stoi(next());
            else if (a == "--batch") o.batch = std::stoi(next());
            else if (a == "--threads") o.threads = std::stoi(next());
            else if (a == "--version") o.version = std::stoi(next());
            else if (a == "--seed") o.seed = std::stoll(next());
            else if (a == "--pageable") o.pinned = false;
            else if (a == "--streams") o.streams = std::stoi(next());
            else throw std::runtime_error("unknown argument " + a);
        }
        const std::string ver = "26.3";
        IR ir = load_ir(o.data + "/gamedata/" + ver + "/overworld_ir.json");
        cuda::print_device();
        std::printf("mode %s, region %dx%d at (%d,%d), seed %lld, reps %d (+%d warmup), batch %d, threads %d, streams %d\n", o.mode.c_str(), o.region,
                    o.region, o.x0, o.z0, static_cast<long long>(o.seed), o.reps, o.warmup, o.batch, o.threads, o.streams);
        if (o.mode == "terrain-gpu" || o.mode == "terrain-cpu") {
            TerrainModel tm = load_terrain_model(o.data + "/gamedata/" + ver + "/overworld_terrain.json");
            if (o.mode == "terrain-gpu") bench_terrain_gpu(o, ir, tm);
            else bench_terrain_cpu(o, ir, tm);
        } else if (o.mode == "core-gpu") {
            bench_core_gpu(o, ir);
        } else if (o.mode == "core-cpu") {
            bench_core_cpu(o, ir);
        } else {
            throw std::runtime_error("unknown mode " + o.mode);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
