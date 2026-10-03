#include "mccuda_cuda/generator.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "mccuda/volume_eval.h"
#include "mccuda_cuda/router_tables.h"
#include "mccuda_router.cuh"   // generated

namespace mcc::cuda {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kCols = 25;   // 5 x 5 corner columns per chunk (4-block cells)

RouterTables make_tables(const SeedTables& t, const PerlinOctave* octaves, const StackLayer* layers) {
    RouterTables T{};
    T.octaves = octaves;
    T.layers = layers;
    for (int i = 0; i < gen::kNumNoises; ++i) {
        auto it = t.noise.find(gen::kNoiseKeys[i]);
        if (it == t.noise.end()) throw std::runtime_error(std::string("seed tables lack noise ") + gen::kNoiseKeys[i]);
        T.stacks[i] = {it->second.first_layer, it->second.count};
    }
    for (int i = 0; i < gen::kNumBlended; ++i) {
        const BlendedSet& b = t.blended.at(gen::kBlendedNodeIds[i]);
        T.blended[i] = {{b.min_limit.first_layer, b.min_limit.count}, {b.max_limit.first_layer, b.max_limit.count}, {b.main.first_layer, b.main.count}};
    }
    return T;
}

// ---- v0: naive kernels ---------------------------------------------------------

__global__ void k_columns_v0(RouterTables T, const ChunkXZ* chunks, int n, float* colvals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * kCols) return;
    int c = idx / kCols, k = idx % kCols;
    int x = chunks[c].x * 16 + (k % 5) * 4;
    int z = chunks[c].z * 16 + (k / 5) * 4;
    gen::eval_columns(T, x, z, colvals + static_cast<size_t>(idx) * gen::kNumColumns);
}

__global__ void k_corners_v0(RouterTables T, const ChunkXZ* chunks, int n, const float* colvals, float* corners, int min_y, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * kCols * rows) return;
    int row = idx % rows;
    int k = (idx / rows) % kCols;
    int c = idx / (rows * kCols);
    int x = chunks[c].x * 16 + (k % 5) * 4;
    int z = chunks[c].z * 16 + (k / 5) * 4;
    int y = min_y + row * 8;
    float out[gen::kNumChannels];
    gen::eval_corners(T, x, y, z, colvals + (static_cast<size_t>(c) * kCols + k) * gen::kNumColumns, out);
    for (int ch = 0; ch < gen::kNumChannels; ++ch)
        corners[((static_cast<size_t>(c) * gen::kNumChannels + ch) * kCols + k) * rows + row] = out[ch];
}

__global__ void k_blocks_v0(RouterTables T, const ChunkXZ* chunks, int n, const float* corners, int min_y, int height, int rows, int sea_level,
                            float* density, uint8_t* fill) {
    size_t per = static_cast<size_t>(height) * 256;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(n) * per) return;
    int c = static_cast<int>(idx / per);
    int i = static_cast<int>(idx % per);
    int x = i & 15, z = (i >> 4) & 15, yy = i >> 8;
    int cx = x >> 2, xi = x & 3, cz = z >> 2, zi = z & 3, cy = yy >> 3, yi = yy & 7;
    float ch[gen::kNumChannels];
    for (int k = 0; k < gen::kNumChannels; ++k) {
        const float* b = corners + (static_cast<size_t>(c) * gen::kNumChannels + k) * kCols * rows;
        const int c00 = (cz * 5 + cx) * rows, c10 = (cz * 5 + cx + 1) * rows, c01 = ((cz + 1) * 5 + cx) * rows, c11 = ((cz + 1) * 5 + cx + 1) * rows;
        ch[k] = interp_fill(b[c00 + cy], b[c10 + cy], b[c00 + cy + 1], b[c10 + cy + 1], b[c01 + cy], b[c11 + cy], b[c01 + cy + 1], b[c11 + cy + 1],
                            xi, yi, zi, 0, 0.25f, 0.125f);
    }
    int bx = chunks[c].x * 16 + x, by = min_y + yy, bz = chunks[c].z * 16 + z;
    float d = gen::eval_block(T, bx, by, bz, ch);
    if (density) density[idx] = d;
    fill[idx] = core_fill(d, by, sea_level);
}

}  // namespace

std::vector<VersionInfo> versions() {
    return {
        {0, "cuda-v0", "naive: one thread per column / corner / block, generated evaluators, tables in global memory"},
    };
}

struct TerrainGenerator::Impl {
    const IR& ir;
    int version;
    int max_batch;
    int rows;
    size_t per;
    RouterTables tables{};
    PerlinOctave* d_octaves = nullptr;
    StackLayer* d_layers = nullptr;
    ChunkXZ* d_chunks = nullptr;
    float* d_colvals = nullptr;
    float* d_corners = nullptr;
    float* d_density = nullptr;
    uint8_t* d_fill = nullptr;
    cudaEvent_t e0{}, e1{}, e2{}, e3{};

    Impl(const IR& ir_, int v, int mb) : ir(ir_), version(v), max_batch(mb) {
        if (ir.height % 8 != 0 || ir.min_y % 8 != 0) throw std::runtime_error("height/min_y must be multiples of 8");
        rows = ir.height / 8 + 1;
        per = static_cast<size_t>(ir.height) * 256;
        check(cudaMalloc(&d_chunks, sizeof(ChunkXZ) * mb), "malloc chunks");
        check(cudaMalloc(&d_colvals, sizeof(float) * mb * kCols * (gen::kNumColumns > 0 ? gen::kNumColumns : 1)), "malloc colvals");
        check(cudaMalloc(&d_corners, sizeof(float) * mb * kCols * rows * gen::kNumChannels), "malloc corners");
        check(cudaMalloc(&d_density, sizeof(float) * mb * per), "malloc density");
        check(cudaMalloc(&d_fill, mb * per), "malloc fill");
        cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
        // Safety net for any evaluator frame the compiler does not keep in registers.
        check(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024), "stack limit");
    }

    ~Impl() {
        cudaFree(d_octaves); cudaFree(d_layers); cudaFree(d_chunks); cudaFree(d_colvals);
        cudaFree(d_corners); cudaFree(d_density); cudaFree(d_fill);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    }

    void set_seed(const SeedTables& t) {
        cudaFree(d_octaves); cudaFree(d_layers);
        check(cudaMalloc(&d_octaves, sizeof(PerlinOctave) * t.octaves.size()), "malloc octaves");
        check(cudaMalloc(&d_layers, sizeof(StackLayer) * t.layers.size()), "malloc layers");
        check(cudaMemcpy(d_octaves, t.octaves.data(), sizeof(PerlinOctave) * t.octaves.size(), cudaMemcpyHostToDevice), "upload octaves");
        check(cudaMemcpy(d_layers, t.layers.data(), sizeof(StackLayer) * t.layers.size(), cudaMemcpyHostToDevice), "upload layers");
        tables = make_tables(t, d_octaves, d_layers);
    }

    void run_batch(const ChunkXZ* chunks, int n, float* density_out, uint8_t* fill_out, Timing* tm) {
        cudaEventRecord(e0);
        check(cudaMemcpy(d_chunks, chunks, sizeof(ChunkXZ) * n, cudaMemcpyHostToDevice), "upload chunks");
        cudaEventRecord(e1);
        const int tb = 128;
        k_columns_v0<<<(n * kCols + tb - 1) / tb, tb>>>(tables, d_chunks, n, d_colvals);
        k_corners_v0<<<(n * kCols * rows + tb - 1) / tb, tb>>>(tables, d_chunks, n, d_colvals, d_corners, ir.min_y, rows);
        size_t blocks = static_cast<size_t>(n) * per;
        k_blocks_v0<<<static_cast<unsigned>((blocks + 255) / 256), 256>>>(tables, d_chunks, n, d_corners, ir.min_y, ir.height, rows, ir.sea_level,
                                                                          density_out ? d_density : nullptr, d_fill);
        check(cudaGetLastError(), "kernel launch");
        cudaEventRecord(e2);
        if (density_out) check(cudaMemcpy(density_out, d_density, sizeof(float) * blocks, cudaMemcpyDeviceToHost), "download density");
        check(cudaMemcpy(fill_out, d_fill, blocks, cudaMemcpyDeviceToHost), "download fill");
        cudaEventRecord(e3);
        check(cudaEventSynchronize(e3), "sync");
        if (tm) {
            float a, b, c;
            cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&c, e2, e3);
            tm->h2d_ms += a; tm->kernel_ms += b; tm->d2h_ms += c;
        }
    }
};

TerrainGenerator::TerrainGenerator(const IR& ir, int version, int max_batch) : impl_(std::make_unique<Impl>(ir, version, max_batch)) {
    if (version != 0) throw std::runtime_error("unknown version " + std::to_string(version));
}

TerrainGenerator::~TerrainGenerator() = default;

int TerrainGenerator::version() const { return impl_->version; }

void TerrainGenerator::set_seed(const SeedTables& t) { impl_->set_seed(t); }

void TerrainGenerator::generate(const ChunkXZ* chunks, int count, float* density_out, uint8_t* fill_out, Timing* timing) {
    auto t0 = std::chrono::steady_clock::now();
    for (int off = 0; off < count; off += impl_->max_batch) {
        int n = std::min(impl_->max_batch, count - off);
        impl_->run_batch(chunks + off, n, density_out ? density_out + static_cast<size_t>(off) * impl_->per : nullptr,
                         fill_out + static_cast<size_t>(off) * impl_->per, timing);
    }
    if (timing) timing->wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void generated_host_chunk(const IR& ir, const SeedTables& t, int cx, int cz, float* density, uint8_t* fill) {
    RouterTables T = make_tables(t, t.octaves.data(), t.layers.data());
    const int rows = ir.height / 8 + 1;
    std::vector<float> col(static_cast<size_t>(kCols) * (gen::kNumColumns > 0 ? gen::kNumColumns : 1));
    std::vector<float> corners(static_cast<size_t>(gen::kNumChannels) * kCols * rows);
    for (int k = 0; k < kCols; ++k) gen::eval_columns(T, cx * 16 + (k % 5) * 4, cz * 16 + (k / 5) * 4, col.data() + static_cast<size_t>(k) * gen::kNumColumns);
    for (int k = 0; k < kCols; ++k)
        for (int r = 0; r < rows; ++r) {
            float out[gen::kNumChannels];
            gen::eval_corners(T, cx * 16 + (k % 5) * 4, ir.min_y + r * 8, cz * 16 + (k / 5) * 4, col.data() + static_cast<size_t>(k) * gen::kNumColumns, out);
            for (int ch = 0; ch < gen::kNumChannels; ++ch) corners[(static_cast<size_t>(ch) * kCols + k) * rows + r] = out[ch];
        }
    for (int yy = 0; yy < ir.height; ++yy)
        for (int z = 0; z < 16; ++z)
            for (int x = 0; x < 16; ++x) {
                int ccx = x >> 2, xi = x & 3, ccz = z >> 2, zi = z & 3, cy = yy >> 3, yi = yy & 7;
                float ch[gen::kNumChannels];
                for (int k = 0; k < gen::kNumChannels; ++k) {
                    const float* b = corners.data() + static_cast<size_t>(k) * kCols * rows;
                    const int c00 = (ccz * 5 + ccx) * rows, c10 = (ccz * 5 + ccx + 1) * rows, c01 = ((ccz + 1) * 5 + ccx) * rows, c11 = ((ccz + 1) * 5 + ccx + 1) * rows;
                    ch[k] = interp_fill(b[c00 + cy], b[c10 + cy], b[c00 + cy + 1], b[c10 + cy + 1], b[c01 + cy], b[c11 + cy], b[c01 + cy + 1], b[c11 + cy + 1],
                                        xi, yi, zi, 0, 0.25f, 0.125f);
                }
                int by = ir.min_y + yy;
                float d = gen::eval_block(T, cx * 16 + x, by, cz * 16 + z, ch);
                size_t o = static_cast<size_t>((yy * 16 + z) * 16 + x);
                if (density) density[o] = d;
                if (fill) fill[o] = core_fill(d, by, ir.sea_level);
            }
}

int debug_compare_generated(const IR& ir, const SeedTables& t, int cx, int cz) {
    RouterTables T = make_tables(t, t.octaves.data(), t.layers.data());
    VolumeEvaluator ev(ir, t);
    auto ref = [&](int node, int x, int y, int z) {
        float v;
        Volume vol{1, 1, 1, x, y, z, 1, 1, 1};
        ev.clear_caches();
        ev.sample(node, vol, &v);
        return v;
    };
    int bad = 0;
    std::vector<float> mv(4096);
    std::vector<uint8_t> mh_storage(4096);
    bool* mh = reinterpret_cast<bool*>(mh_storage.data());
    float col[25][gen::kNumColumns > 0 ? gen::kNumColumns : 1];
    for (int k = 0; k < kCols; ++k) {
        int x = cx * 16 + (k % 5) * 4, z = cz * 16 + (k / 5) * 4;
        gen::eval_columns_dbg(T, x, z, col[k], mv.data(), mh);
        for (int r = 0; r < gen::kReachCount_ColumnEval; ++r) {
            if (!mh[r]) continue;
            int id = gen::kReach_ColumnEval[r];
            float want = ref(id, x, 0, z);
            if (f2bits(want) != f2bits(mv[r]) && bad < 12) {
                std::printf("  column (%d,%d) node %d: generated %.9g (0x%08x) reference %.9g (0x%08x)\n", x, z, id, mv[r], f2bits(mv[r]), want, f2bits(want));
                ++bad;
            }
        }
    }
    const int rows = ir.height / 8 + 1;
    for (int k = 0; k < kCols && bad < 12; ++k)
        for (int row = 0; row < rows && bad < 12; ++row) {
            int x = cx * 16 + (k % 5) * 4, z = cz * 16 + (k / 5) * 4, y = ir.min_y + row * 8;
            float out[gen::kNumChannels];
            gen::eval_corners_dbg(T, x, y, z, col[k], out, mv.data(), mh);
            for (int r = 0; r < gen::kReachCount_CornerEval; ++r) {
                if (!mh[r]) continue;
                int id = gen::kReach_CornerEval[r];
                float want = ref(id, x, y, z);
                if (f2bits(want) != f2bits(mv[r]) && bad < 12) {
                    std::printf("  corner (%d,%d,%d) node %d (%d): generated %.9g (0x%08x) reference %.9g (0x%08x)\n", x, y, z, id,
                                static_cast<int>(ir.at(id).type), mv[r], f2bits(mv[r]), want, f2bits(want));
                    ++bad;
                }
            }
        }
    return bad;
}

void print_device() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) std::printf("device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
}

}  // namespace mcc::cuda
