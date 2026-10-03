// GPU TERRAIN step: biomes -> density + aquifer fill -> material rules.
// Every stage mirrors the CPU reference port (cpu/src/terrain.cpp), which is
// itself verified bit-for-bit against vanilla; comments name the vanilla method.
#include "mccuda_cuda/terrain_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mccuda/terrain.h"
#include "mccuda/volume_eval.h"
#include "mccuda_cuda/router_tables.h"
#include "mccuda_cuda/terrain_device.h"

namespace mcc::cuda {
struct SurfaceCtx;
}

namespace mcc::gen {
using SurfaceCtx = mcc::cuda::SurfaceCtx;
}

#if defined(__CUDACC__)
#define MCC_RULE_FN __device__ __noinline__
#else
#define MCC_RULE_FN inline
#endif

namespace mcc::cuda {

constexpr int kWayBelowMinY = -2032 << 4;
constexpr int kGridX = 27, kGridZ = 19;          // surface-level quart grid per chunk
constexpr int kGridCells = kGridX * kGridZ;
constexpr int kMaxGradientRandoms = 4;
constexpr int kMaxOreRules = 4;

enum BlockFlag : uint8_t { kFlagAir = 1, kFlagWater = 2, kFlagLava = 4, kFlagFluid = 8 };

struct TerrainTables {
    RouterTables rt;
    const BiomeTree::FlatNode* tree;
    const int32_t* tree_children;
    int32_t tree_root;
    const int32_t* leaf_biome;
    const int32_t* leaf_node;
    uint8_t block_flags[64];
    int air, water, lava, default_block, default_fluid, snow_block, packed_ice;
    int sea_level, min_y, height;
    int biome_eroded_badlands, biome_frozen_ocean, biome_deep_frozen_ocean;
    float biome_temp[128];
    uint8_t biome_frozen[128];
    dev::Simplex2D temp_noise, frozen[3], biome_info;
    float frozen_amp[3];
    int clay_bands[192];
    dev::Positional random, aquifer_random, ore_random, gradient_random[kMaxGradientRandoms];
    int64_t zoom_seed;
    int st_surface, st_surface_secondary, st_clay_offset, st_badlands_pillar, st_badlands_roof, st_badlands_surface, st_iceberg_pillar,
        st_iceberg_roof, st_iceberg_surface;
    int aq_min_grid_y, aq_grid_y;                  // aquifer grid (y extent is chunk-independent)
    int ore_state[kMaxOreRules][3];                // ore, raw ore, filler (copies of the generated tables)
    float ore_raw_chance[kMaxOreRules];
};

__constant__ TerrainTables c_tt;

// ---- material system ------------------------------------------------------------------
struct SurfaceCtx {
    // chunk / column
    int cx, cz, lx, lz;
    uint8_t* blocks;           // this chunk's blocks
    int* hm;                   // this column's live heightmap value (getFirstAvailable)
    const uint8_t* const* nb;  // 9 biome arrays (3x3 neighbourhood)
    const float* corners;      // channel corners of this chunk
    int corner_chunk, rows;
    float prelim;
    // per column / block context
    int x, y, z;
    int grad_x, grad_z, surface_depth;
    int stone_above, stone_below, water_height;
    bool secondary_set, biome_set, min_surface_set;
    double secondary;
    int biome_v, min_surface;
    bool ore_set;
    float ore_ch[16];

    __device__ int quart_biome(int qx, int qy, int qz) const {
        int dx = (qx >> 2) - cx, dz = (qz >> 2) - cz;
        int qmin = c_tt.min_y >> 2, qh = c_tt.height / 4;
        int cqy = min(max(qy, qmin), qmin + qh - 1) - qmin;
        return nb[(dz + 1) * 3 + dx + 1][(cqy * 4 + (qz & 3)) * 4 + (qx & 3)];
    }
    __device__ int biome_at(int bx, int by, int bz) const {
        int qx, qy, qz;
        dev::zoom_quart(c_tt.zoom_seed, bx, by, bz, qx, qy, qz);
        return quart_biome(qx, qy, qz);
    }
    __device__ double noise_stack(int st, double nx, double ny, double nz) const {
        const StackRef2& s = c_tt.rt.stacks[st];
        return static_cast<double>(stack_get(c_tt.rt.layers + s.first, s.count, c_tt.rt.octaves, nx, ny, nz));
    }
    __device__ int biome() {
        if (!biome_set) { biome_v = biome_at(x, y, z); biome_set = true; }
        return biome_v;
    }
    __device__ double noise2d(int st) const { return noise_stack(st, x, 0.0, z); }
    __device__ double noise3d(int st) const { return noise_stack(st, x, y, z); }
    __device__ bool vertical_gradient(int k, int true_at, int false_at) const {
        if (y <= true_at) return true;
        if (y >= false_at) return false;
        double t = (static_cast<double>(y) - true_at) / static_cast<double>(false_at - true_at);
        double probability = 1.0 + t * (0.0 - 1.0);
        dev::Xoro r = c_tt.gradient_random[k].at(x, y, z);
        return static_cast<double>(dev::xoro_next_float(r)) < probability;
    }
    __device__ double surface_secondary() {
        if (!secondary_set) { secondary = noise_stack(c_tt.st_surface_secondary, x, 0.0, z); secondary_set = true; }
        return secondary;
    }
    __device__ bool stone_depth(int offset, bool add, int range, bool ceiling) {
        int stone = ceiling ? stone_below : stone_above;
        int sd = add ? surface_depth : 0;
        int sec = 0;
        if (range != 0) {
            double t = (surface_secondary() - -1.0) / (1.0 - -1.0);
            sec = java_d2i(0.0 + t * (static_cast<double>(range) - 0.0));
        }
        return stone <= 1 + offset + sd + sec;
    }
    __device__ float temperature(int b, int bx, int by, int bz) const {
        float adjusted = c_tt.biome_temp[b];
        if (c_tt.biome_frozen[b]) {
            const double freq[3] = {1.0, 0.5, 0.25};
            float fv = 0.0f;
            for (int i = 0; i < 3; ++i) fv += c_tt.frozen_amp[i] * dev::simplex_get(c_tt.frozen[i], static_cast<double>(bx) * 0.05 * freq[i], static_cast<double>(bz) * 0.05 * freq[i]);
            double large = static_cast<double>(fv * 7.0f);
            double edge = static_cast<double>(dev::simplex_get(c_tt.biome_info, static_cast<double>(bx) * 0.2, static_cast<double>(bz) * 0.2));
            if (large + edge < 0.3) {
                double small = static_cast<double>(dev::simplex_get(c_tt.biome_info, static_cast<double>(bx) * 0.09, static_cast<double>(bz) * 0.09));
                if (small < 0.8) adjusted = 0.2f;
            }
        }
        int snow_level = c_tt.sea_level + 17;
        if (by > snow_level) {
            float v = dev::simplex_get(c_tt.temp_noise, static_cast<double>(static_cast<float>(bx) / 8.0f), static_cast<double>(static_cast<float>(bz) / 8.0f)) * 8.0f;
            float t = v + static_cast<float>(by);
            t = t - static_cast<float>(snow_level);
            t = t * 0.05f;
            t = t / 40.0f;
            return adjusted - t;
        }
        return adjusted;
    }
    __device__ bool cold_enough_to_snow() { return !(temperature(biome(), x, y, z) >= 0.15f); }
    __device__ int min_surface_level() {
        if (!min_surface_set) { min_surface = mth_floor(static_cast<double>(prelim)) + surface_depth - 8; min_surface_set = true; }
        return min_surface;
    }
    __device__ int band() const {
        int offset = java_round_f(static_cast<float>(noise_stack(c_tt.st_clay_offset, x, 0.0, z)) * 4.0f);
        return c_tt.clay_bands[(y + offset + 192) % 192];
    }
    __device__ int ore_vein(int k);   // defined after the generated header

    // column access
    __device__ size_t bidx(int by) const { return static_cast<size_t>(((by - c_tt.min_y) * 16 + lz) * 16 + lx); }
    __device__ int get(int by) const { return (by < c_tt.min_y || by >= c_tt.min_y + c_tt.height) ? c_tt.air : blocks[bidx(by)]; }
    __device__ bool air(int s) const { return c_tt.block_flags[s] & kFlagAir; }
    __device__ bool fluid(int s) const { return c_tt.block_flags[s] & kFlagFluid; }
    __device__ void set(int by, int state, bool write) {
        if (by < c_tt.min_y || by >= c_tt.min_y + c_tt.height) return;
        if (write) blocks[bidx(by)] = static_cast<uint8_t>(state);
        int first = *hm;
        if (by <= first - 2) return;
        if (!air(state)) {
            if (by >= first) *hm = by + 1;
        } else if (first - 1 == by) {
            for (int yy = by - 1; yy >= c_tt.min_y; --yy)
                if (!air(get(yy))) { *hm = yy + 1; return; }
            *hm = c_tt.min_y;
        }
    }
};

}  // namespace mcc::cuda

#define MCC_WITH_RULES
#include "mccuda_router.cuh"   // generated: evaluators + material rules

namespace mcc::cuda {

constexpr int kAllChannels = gen::kNumChannels + gen::kNumOreChannels;
static_assert(kAllChannels <= 16, "SurfaceCtx::ore_ch too small");
constexpr int kCols = 25;

__device__ __forceinline__ void interp_channels(const float* corners, int c, int rows, int x, int yy, int z, int first, int count, float* out) {
    int cx = x >> 2, xi = x & 3, cz = z >> 2, zi = z & 3, cy = yy >> 3, yi = yy & 7;
    for (int k = 0; k < count; ++k) {
        const float* b = corners + (static_cast<size_t>(c) * kAllChannels + first + k) * kCols * rows;
        const int c00 = (cz * 5 + cx) * rows, c10 = (cz * 5 + cx + 1) * rows, c01 = ((cz + 1) * 5 + cx) * rows, c11 = ((cz + 1) * 5 + cx + 1) * rows;
        out[k] = interp_fill(b[c00 + cy], b[c10 + cy], b[c00 + cy + 1], b[c10 + cy + 1], b[c01 + cy], b[c11 + cy], b[c01 + cy + 1], b[c11 + cy + 1],
                             xi, yi, zi, 0, 0.25f, 0.125f);
    }
}

__device__ int SurfaceCtx::ore_vein(int k) {
    if (!ore_set) {
        interp_channels(corners, corner_chunk, rows, lx, y - c_tt.min_y, lz, 0, kAllChannels, ore_ch);
        ore_set = true;
    }
    float density = gen::eval_ore(c_tt.rt, x, y, z, ore_ch, 2 * k);
    if (density <= 0.0f) return -1;
    dev::Xoro r = c_tt.ore_random.at(x, y, z);
    if (dev::xoro_next_float(r) > density) return -1;
    float richness = gen::eval_ore(c_tt.rt, x, y, z, ore_ch, 2 * k + 1);
    if (dev::xoro_next_float(r) < richness && gen::eval_point(c_tt.rt, x, y, z, 5 + k) < 0.0f)
        return dev::xoro_next_float(r) < c_tt.ore_raw_chance[k] ? c_tt.ore_state[k][1] : c_tt.ore_state[k][0];
    return c_tt.ore_state[k][2];
}



namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

// ---- biomes --------------------------------------------------------------------------
__device__ int64_t tree_distance(int node, const int64_t* t) {
    const BiomeTree::FlatNode& n = c_tt.tree[node];
    int64_t d = 0;
    for (int i = 0; i < 7; ++i) {
        int64_t above = t[i] - n.max[i], below = n.min[i] - t[i];
        int64_t p = above > 0 ? above : (below > 0 ? below : 0);
        d += p * p;
    }
    return d;
}

// Climate.RTree.SubTree.search with an empty candidate (iterative DFS with an explicit stack).
__device__ int tree_search_first_min(const int64_t* t, int64_t* out_dist) {
    struct Frame { int node; int child; };
    Frame stack[16];
    int sp = 0;
    int64_t min_d = INT64_MAX;
    int closest = -1;
    // Recursive form: search(node, candidate) visits children in order, descends into a child
    // only if min_d > dist(child bbox), and accepts a leaf only if strictly closer.
    stack[sp++] = {c_tt.tree_root, 0};
    while (sp > 0) {
        Frame& f = stack[sp - 1];
        const BiomeTree::FlatNode& n = c_tt.tree[f.node];
        if (n.leaf >= 0) {                                 // only reached for the root being a leaf
            int64_t d = tree_distance(f.node, t);
            if (min_d > d) { min_d = d; closest = f.node; }
            --sp;
            continue;
        }
        if (f.child >= n.child_count) { --sp; continue; }
        int c = c_tt.tree_children[n.first_child + f.child];
        ++f.child;
        int64_t cd = tree_distance(c, t);
        if (min_d > cd) {
            const BiomeTree::FlatNode& cn = c_tt.tree[c];
            if (cn.leaf >= 0) {
                min_d = cd;
                closest = c;
            } else {
                stack[sp++] = {c, 0};
            }
        }
    }
    *out_dist = min_d;
    return c_tt.tree[closest].leaf;
}

__global__ void k_biome_columns(const ChunkXZ* bchunks, int nb, float* bcol) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * 16) return;
    int b = idx / 16, k = idx % 16;
    int x = bchunks[b].x * 16 + (k % 4) * 4, z = bchunks[b].z * 16 + (k / 4) * 4;
    gen::eval_climate_columns(c_tt.rt, x, z, bcol + static_cast<size_t>(idx) * gen::kNumClimateColumns);
}

__global__ void k_biome_quarts(const ChunkXZ* bchunks, int nb, const float* bcol, int64_t* targets, int32_t* first_min, int64_t* min_dist) {
    const int qh = c_tt.height / 4;
    const int per = 16 * qh;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nb * per) return;
    int b = idx / per, q = idx % per;
    int qx = q & 3, qz = (q >> 2) & 3, qy = q >> 4;
    int x = bchunks[b].x * 16 + qx * 4, y = c_tt.min_y + qy * 4, z = bchunks[b].z * 16 + qz * 4;
    float clim[6];
    gen::eval_climate(c_tt.rt, x, y, z, bcol + (static_cast<size_t>(b) * 16 + qz * 4 + qx) * gen::kNumClimateColumns, clim);
    int64_t* t = targets + static_cast<size_t>(idx) * 7;
    for (int k = 0; k < 6; ++k) {
        float v = clim[k] * 10000.0f;                          // Climate.quantizeCoord
        int64_t qv;
        if (v != v) qv = 0;
        else if (v >= 9.2233720368547758e18f) qv = INT64_MAX;
        else if (v <= -9.2233720368547758e18f) qv = INT64_MIN;
        else qv = static_cast<int64_t>(v);
        t[k] = qv;
    }
    t[6] = 0;
    int64_t d;
    first_min[idx] = tree_search_first_min(t, &d);
    min_dist[idx] = d;
}

// The R-tree's last-result candidate decides exact ties: the result is the candidate
// if it is at minimal distance, otherwise the first minimal leaf in DFS order.
__global__ void k_biome_chain(int nb, const int64_t* targets, const int32_t* first_min, const int64_t* min_dist, uint8_t* biomes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const int qh = c_tt.height / 4;
    const int per = 16 * qh;
    int prev = -1;
    for (int sy = 0; sy < qh / 4; ++sy)
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y)
                for (int z = 0; z < 4; ++z) {
                    int qy = sy * 4 + y;
                    int q = (qy * 4 + z) * 4 + x;
                    size_t i = static_cast<size_t>(b) * per + q;
                    int leaf = first_min[i];
                    if (prev >= 0 && tree_distance(c_tt.leaf_node[prev], targets + i * 7) == min_dist[i]) leaf = prev;
                    prev = leaf;
                    biomes[i] = static_cast<uint8_t>(c_tt.leaf_biome[leaf]);
                }
}

// ---- density ------------------------------------------------------------------------

__global__ void k_columns(const ChunkXZ* chunks, int n, float* col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * kCols) return;
    int c = idx / kCols, k = idx % kCols;
    gen::eval_columns(c_tt.rt, chunks[c].x * 16 + (k % 5) * 4, chunks[c].z * 16 + (k / 5) * 4, col + static_cast<size_t>(idx) * gen::kNumColumns);
}

__global__ void k_corners(const ChunkXZ* chunks, int n, const float* col, float* corners, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * kCols * rows) return;
    int row = idx % rows, k = (idx / rows) % kCols, c = idx / (rows * kCols);
    int x = chunks[c].x * 16 + (k % 5) * 4, z = chunks[c].z * 16 + (k / 5) * 4, y = c_tt.min_y + row * 8;
    const float* cv = col + (static_cast<size_t>(c) * kCols + k) * gen::kNumColumns;
    float out[kAllChannels];
    gen::eval_corners(c_tt.rt, x, y, z, cv, out);
    if (gen::kNumOreChannels > 0) gen::eval_ore_corners(c_tt.rt, x, y, z, cv, out + gen::kNumChannels);
    for (int ch = 0; ch < kAllChannels; ++ch) corners[((static_cast<size_t>(c) * kAllChannels + ch) * kCols + k) * rows + row] = out[ch];
}

// ---- surface-level grid (aquifer) and preliminary surface corners -----------------------
__global__ void k_surface_grid(const ChunkXZ* chunks, int n, int32_t* grid, float* prelim_corners) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int per = kGridCells + 4;
    if (idx >= n * per) return;
    int c = idx / per, e = idx % per;
    int cx = chunks[c].x, cz = chunks[c].z;
    if (e < kGridCells) {
        int qx = 4 * cx - 16 + e % kGridX, qz = 4 * cz - 8 + e / kGridX;
        // Positions inside NoiseBasedAquifer's constructor prefill use the volume path.
        bool prefill = qx >= 4 * cx - 4 && qx <= 4 * cx + 6 && qz >= 4 * cz - 4 && qz <= 4 * cz + 6;
        float v = prefill ? gen::fts_volume(c_tt.rt, qx * 4, qz * 4) : gen::fts_point(c_tt.rt, qx * 4, qz * 4);
        grid[static_cast<size_t>(c) * kGridCells + e] = mth_floor(static_cast<double>(v));
    } else {
        int k = e - kGridCells;
        prelim_corners[static_cast<size_t>(c) * 4 + k] = gen::fts_volume(c_tt.rt, cx * 16 + (k & 1) * 16, cz * 16 + (k >> 1) * 16);
    }
}

// ---- aquifer -----------------------------------------------------------------------
struct FluidStatus {
    int level;
    int type;
};

__device__ __forceinline__ int fs_at(FluidStatus s, int y) { return y < s.level ? s.type : c_tt.air; }

__device__ __forceinline__ FluidStatus global_fluid(int y) {
    int lava_level = c_tt.sea_level < -54 ? c_tt.sea_level : -54;
    return y < lava_level ? FluidStatus{-54, c_tt.lava} : FluidStatus{c_tt.sea_level, c_tt.default_fluid};
}

__device__ __forceinline__ int grid_y(int b) { return floor_div(b, 12); }

struct AquiferChunk {
    int min_grid_x, min_grid_z;
    int skip_above_y;
};

__device__ __forceinline__ int surface_level_at(const int32_t* grid, int cx, int cz, int x, int z) {
    int qx = x >> 2, qz = z >> 2;
    return grid[(qx - (4 * cx - 16)) + (qz - (4 * cz - 8)) * kGridX];
}

__global__ void k_aquifer_skip(int n, const int32_t* grid, int* skip_above) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int m = INT_MIN;
    for (int gz = 4; gz <= 14; ++gz)        // quarts 4cz-4 .. 4cz+6
        for (int gx = 12; gx <= 22; ++gx)   // quarts 4cx-4 .. 4cx+6
            m = max(m, grid[static_cast<size_t>(c) * kGridCells + gz * kGridX + gx]);
    int adjusted = m + 8;
    int skip_grid_y = grid_y(adjusted + 12) + 1;
    skip_above[c] = skip_grid_y * 12 + 11 - 1;
}

// Location and fluid status of every aquifer grid cell of a chunk (3 x gridY x 3).
__global__ void k_aquifer_cells(const ChunkXZ* chunks, int n, const int32_t* grid, int4* cells) {
    const int per = 9 * c_tt.aq_grid_y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * per) return;
    int c = idx / per, i = idx % per;
    int cx = chunks[c].x, cz = chunks[c].z;
    int lx = i % 3, lz = (i / 3) % 3, ly = i / 9;          // getIndex: (y * gsz + z) * gsx + x
    int gx = cx - 1 + lx, gz = cz - 1 + lz, gy = c_tt.aq_min_grid_y + ly;
    dev::Xoro r = c_tt.aquifer_random.at(gx, gy, gz);
    int x = (gx << 4) + dev::xoro_next_int(r, 10);
    int y = gy * 12 + dev::xoro_next_int(r, 9);
    int z = (gz << 4) + dev::xoro_next_int(r, 10);
    const int* g = grid + static_cast<size_t>(c) * kGridCells;

    // computeFluid
    FluidStatus global = global_fluid(y);
    FluidStatus result;
    bool done = false;
    int lowest = INT_MAX;
    int top = y + 12, bottom = y - 12;
    bool center_under = false;
    const int offsets[13][2] = {{0, 0}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {-3, 0}, {-2, 0}, {-1, 0}, {1, 0}, {-2, 1}, {-1, 1}, {0, 1}, {1, 1}};
    for (int k = 0; k < 13 && !done; ++k) {
        int sx = x + (offsets[k][0] << 4), sz = z + (offsets[k][1] << 4);
        int level = surface_level_at(g, cx, cz, sx, sz);
        int adjusted = level + 8;
        bool start = offsets[k][0] == 0 && offsets[k][1] == 0;
        if (start && bottom > adjusted) { result = global; done = true; break; }
        bool pokes = top > adjusted;
        if (pokes || start) {
            FluidStatus at_surface = global_fluid(adjusted);
            if (fs_at(at_surface, adjusted) != c_tt.air) {
                if (start) center_under = true;
                if (pokes) { result = at_surface; done = true; break; }
            }
        }
        lowest = min(lowest, level);
    }
    if (!done) {
        double partially, fully;
        if (gen::eval_point(c_tt.rt, x, y, z, 4) > 0.0f) {
            partially = -1.0;
            fully = -1.0;
        } else {
            int below = lowest + 8 - y;
            double factor = 0.0;
            if (center_under) {
                double f = (static_cast<double>(below) - 0.0) / (64.0 - 0.0);
                factor = f < 0.0 ? 1.0 : (f > 1.0 ? 0.0 : 1.0 + f * (0.0 - 1.0));
            }
            double noise = mth_dclamp(static_cast<double>(gen::eval_point(c_tt.rt, x, y, z, 1)), -1.0, 1.0);
            double fully_t = -0.3 + ((factor - 1.0) / (0.0 - 1.0)) * (0.8 - -0.3);
            double partial_t = -0.8 + ((factor - 1.0) / (0.0 - 1.0)) * (0.4 - -0.8);
            partially = noise - partial_t;
            fully = noise - fully_t;
        }
        int level;
        if (fully > 0.0) level = global.level;
        else if (partially > 0.0) {
            int fcx = floor_div(x, 16), fcy = floor_div(y, 40), fcz = floor_div(z, 16);
            int middle = fcy * 40 + 20;
            double spread = static_cast<double>(gen::eval_point(c_tt.rt, fcx, fcy, fcz, 2) * 10.0f);
            int q = mth_floor(spread / 3.0) * 3;
            level = min(lowest, middle + q);
        } else level = kWayBelowMinY;
        int type = global.type;
        if (level <= -10 && level != kWayBelowMinY && global.type != c_tt.lava) {
            int tcx = floor_div(x, 64), tcy = floor_div(y, 40), tcz = floor_div(z, 64);
            double lava = static_cast<double>(gen::eval_point(c_tt.rt, tcx, tcy, tcz, 3));
            if (fabs(lava) > 0.3) type = c_tt.lava;
        }
        result = FluidStatus{level, type};
    }
    cells[idx] = make_int4(x, y, z, 0);
    cells[static_cast<size_t>(n) * per + idx] = make_int4(result.level, result.type, 0, 0);
}

__device__ double aquifer_pressure(int x, int y, int z, double& barrier, bool& barrier_set, FluidStatus a, FluidStatus b) {
    int t1 = fs_at(a, y), t2 = fs_at(b, y);
    if ((t1 == c_tt.lava && t2 == c_tt.water) || (t1 == c_tt.water && t2 == c_tt.lava)) return 2.0;
    int diff = abs(a.level - b.level);
    if (diff == 0) return 0.0;
    double average = 0.5 * static_cast<double>(a.level + b.level);
    double above = static_cast<double>(y) + 0.5 - average;
    double base = static_cast<double>(diff) / 2.0;
    double towards = base - fabs(above);
    double gradient;
    if (above > 0.0) {
        double center = 0.0 + towards;
        gradient = center > 0.0 ? center / 1.5 : center / 2.5;
    } else {
        double center = 3.0 + towards;
        gradient = center > 0.0 ? center / 3.0 : center / 10.0;
    }
    double noise;
    if (!(gradient < -2.0) && !(gradient > 2.0)) {
        if (!barrier_set) {
            barrier = static_cast<double>(gen::eval_point(c_tt.rt, x, y, z, 0));
            barrier_set = true;
        }
        noise = barrier;
    } else {
        noise = 0.0;
    }
    return 2.0 * (noise + gradient);
}

// NoiseBasedAquifer.computeSubstance; -1 = null (default block).
__device__ int aquifer_substance(int cx, int cz, const int4* cells, size_t cell_base, size_t status_base, int skip_above, int x, int y, int z, double density) {
    if (density > 0.0) return -1;
    FluidStatus global = global_fluid(y);
    if (y > skip_above) return fs_at(global, y);
    if (fs_at(global, y) == c_tt.lava) return c_tt.lava;
    int xa = (x - 5) >> 4, ya = grid_y(y + 1), za = (z - 5) >> 4;
    int d1 = INT_MAX, d2 = INT_MAX, d3 = INT_MAX, d4 = INT_MAX;
    int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
    for (int ox = 0; ox <= 1; ++ox)
        for (int oy = -1; oy <= 1; ++oy)
            for (int oz = 0; oz <= 1; ++oz) {
                int gx = xa + ox, gy = ya + oy, gz = za + oz;
                int index = ((gy - c_tt.aq_min_grid_y) * 3 + (gz - (cz - 1))) * 3 + (gx - (cx - 1));
                int4 loc = cells[cell_base + index];
                int dx = loc.x - x, dy = loc.y - y, dz = loc.z - z;
                int nd = dx * dx + dy * dy + dz * dz;
                if (d1 >= nd) { i4 = i3; i3 = i2; i2 = i1; i1 = index; d4 = d3; d3 = d2; d2 = d1; d1 = nd; }
                else if (d2 >= nd) { i4 = i3; i3 = i2; i2 = index; d4 = d3; d3 = d2; d2 = nd; }
                else if (d3 >= nd) { i4 = i3; i3 = index; d4 = d3; d3 = nd; }
                else if (d4 >= nd) { i4 = index; d4 = nd; }
            }
    (void)i4; (void)d4;
    auto status = [&](int index) { int4 s = cells[status_base + index]; return FluidStatus{s.x, s.y}; };
    FluidStatus s1 = status(i1);
    double sim12 = 1.0 - static_cast<double>(d2 - d1) / 25.0;
    int fluid = fs_at(s1, y);
    if (sim12 <= 0.0) return fluid;
    if (fluid == c_tt.water && fs_at(global_fluid(y - 1), y - 1) == c_tt.lava) return fluid;
    double barrier = 0.0;
    bool barrier_set = false;
    FluidStatus s2 = status(i2);
    double b12 = sim12 * aquifer_pressure(x, y, z, barrier, barrier_set, s1, s2);
    if (density + b12 > 0.0) return -1;
    FluidStatus s3 = status(i3);
    double sim13 = 1.0 - static_cast<double>(d3 - d1) / 25.0;
    if (sim13 > 0.0) {
        double b13 = sim12 * sim13 * aquifer_pressure(x, y, z, barrier, barrier_set, s1, s3);
        if (density + b13 > 0.0) return -1;
    }
    double sim23 = 1.0 - static_cast<double>(d3 - d2) / 25.0;
    if (sim23 > 0.0) {
        double b23 = sim12 * sim23 * aquifer_pressure(x, y, z, barrier, barrier_set, s2, s3);
        if (density + b23 > 0.0) return -1;
    }
    return fluid;
}

__global__ void k_blocks(const ChunkXZ* chunks, int n, const float* corners, int rows, const int4* cells, const int* skip_above, float* density,
                         uint8_t* blocks) {
    const int h = c_tt.height;
    const size_t per = static_cast<size_t>(h) * 256;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(n) * per) return;
    int c = static_cast<int>(idx / per);
    int i = static_cast<int>(idx % per);
    int x = i & 15, z = (i >> 4) & 15, yy = i >> 8;
    float ch[gen::kNumChannels];
    interp_channels(corners, c, rows, x, yy, z, 0, gen::kNumChannels, ch);
    int bx = chunks[c].x * 16 + x, by = c_tt.min_y + yy, bz = chunks[c].z * 16 + z;
    float d = gen::eval_block(c_tt.rt, bx, by, bz, ch);
    if (density) density[idx] = d;
    const int cell_per = 9 * c_tt.aq_grid_y;
    int s = aquifer_substance(chunks[c].x, chunks[c].z, cells, static_cast<size_t>(c) * cell_per, static_cast<size_t>(n) * cell_per + static_cast<size_t>(c) * cell_per,
                              skip_above[c], bx, by, bz, static_cast<double>(d));
    blocks[idx] = static_cast<uint8_t>(s < 0 ? c_tt.default_block : s);
}

__global__ void k_heights(int n, const uint8_t* blocks, int* h0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 256) return;
    int c = idx / 256, col = idx % 256;
    int x = col / 16, z = col % 16;                          // heightmap index x * 16 + z
    const uint8_t* b = blocks + static_cast<size_t>(c) * c_tt.height * 256;
    int hv = c_tt.min_y;
    for (int yy = c_tt.height - 1; yy >= 0; --yy)
        if (!(c_tt.block_flags[b[(yy * 16 + z) * 16 + x]] & kFlagAir)) { hv = c_tt.min_y + yy + 1; break; }
    h0[idx] = hv;
}

}  // namespace

namespace {

__device__ void eroded_badlands(SurfaceCtx& c, int bx, int bz, int height) {
    double a = fabs(c.noise_stack(c_tt.st_badlands_surface, bx, 0.0, bz) * 8.25);
    double b = static_cast<double>(static_cast<float>(c.noise_stack(c_tt.st_badlands_pillar, bx * 0.2, 0.0, bz * 0.2)) * 15.0f);
    double pillar = java_dmin(a, b);
    if (pillar <= 0.0) return;
    double floor_v = fabs(c.noise_stack(c_tt.st_badlands_roof, bx * 0.75, 0.0, bz * 0.75) * 1.5);
    double top = 64.0 + java_dmin(pillar * pillar * 2.5, ceil(floor_v * 50.0) + 24.0);
    int start_y = mth_floor(top);
    if (height > start_y) return;
    for (int yy = start_y; yy >= c_tt.min_y; --yy) {
        int old = c.get(yy);
        if (old == c_tt.default_block) break;
        if (c_tt.block_flags[old] & kFlagWater) return;
    }
    for (int yy = start_y; yy >= c_tt.min_y && c.air(c.get(yy)); --yy) c.set(yy, c_tt.default_block, true);
}

// Runs MaterialSystem.frozenOceanExtension; with write == false only the heightmap moves.
__device__ void frozen_ocean(SurfaceCtx& c, int min_surface, int biome, int bx, int bz, int height, bool write) {
    double a = fabs(c.noise_stack(c_tt.st_iceberg_surface, bx, 0.0, bz) * 8.25);
    double b = static_cast<double>(static_cast<float>(c.noise_stack(c_tt.st_iceberg_pillar, bx * 1.28, 0.0, bz * 1.28)) * 15.0f);
    double iceberg = java_dmin(a, b);
    if (iceberg <= 1.8) return;
    double roof = fabs(c.noise_stack(c_tt.st_iceberg_roof, bx * 1.17, 0.0, bz * 1.17) * 1.5);
    double top = java_dmin(iceberg * iceberg * 1.2, ceil(roof * 40.0) + 14.0);
    const int sea = c_tt.sea_level;
    if (c.temperature(biome, bx, sea, bz) > 0.1f) top -= 2.0;
    if (top <= 2.0) return;
    double bottom = static_cast<double>(sea) - top - 7.0;
    top += static_cast<double>(sea);
    dev::Xoro r = c_tt.random.at(bx, 0, bz);
    int max_snow = 2 + dev::xoro_next_int(r, 4);
    int min_snow_height = sea + 18 + dev::xoro_next_int(r, 10);
    int snow = 0;
    int itop = java_d2i(top), ibottom = java_d2i(bottom);
    // Dry runs must not let their own writes change later reads: track writes virtually.
    for (int yy = max(height, itop + 1); yy >= min_surface; --yy) {
        int cur = c.get(yy);
        bool place = false;
        if (c.air(cur) && yy < itop && dev::xoro_next_double(r) > 0.01) place = true;
        else if (cur == c_tt.water && yy > ibottom && yy < sea && dev::xoro_next_double(r) > 0.15) place = true;
        if (place) {
            int state = (snow <= max_snow && yy > min_snow_height) ? c_tt.snow_block : c_tt.packed_ice;
            if (state == c_tt.snow_block) ++snow;
            c.set(yy, state, write);
        }
    }
}

__device__ void init_ctx(SurfaceCtx& c, const ChunkXZ* chunks, int ci, int col, uint8_t* blocks, int* hm_live, const uint8_t* const* nb, const float* corners,
                         int rows, const float* prelim_corners) {
    c.cx = chunks[ci].x; c.cz = chunks[ci].z;
    c.lx = col / 16; c.lz = col % 16;
    c.blocks = blocks + static_cast<size_t>(ci) * c_tt.height * 256;
    c.hm = hm_live;
    c.nb = nb;
    c.corners = corners;
    c.corner_chunk = ci;
    c.rows = rows;
    const float* pc = prelim_corners + static_cast<size_t>(ci) * 4;
    // chunk_surface_level: InterpolatedFunction(16 x 1) over the 4 chunk corners.
    c.prelim = interp_fill(pc[0], pc[1], pc[0], pc[1], pc[2], pc[3], pc[2], pc[3], c.lx, 0, c.lz, 0, 1.0f / 16.0f, 1.0f / 1.0f);
    c.secondary_set = c.biome_set = c.min_surface_set = c.ore_set = false;
}

__device__ void update_xz(SurfaceCtx& c, int bx, int bz, int gx, int gz) {
    c.x = bx; c.z = bz; c.grad_x = gx; c.grad_z = gz;
    double nv = c.noise_stack(c_tt.st_surface, bx, 0.0, bz);
    dev::Xoro r = c_tt.random.at(bx, 0, bz);
    c.surface_depth = java_d2i(nv * 2.75 + 3.0 + dev::xoro_next_double(r) * 0.25);
    c.secondary_set = false;
    c.min_surface_set = false;
}

// Pass 1: eroded-badlands extension (writes) and the heights each column reaches
// after its badlands extension (h1) and after its frozen-ocean extension (h2).
__global__ void k_surface1(const ChunkXZ* chunks, int n, uint8_t* blocks, const int* h0, int* h1, int* h2, const int* nb_index, const uint8_t* bbiomes,
                           const float* corners, int rows, const float* prelim_corners) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 256) return;
    int ci = idx / 256, col = idx % 256;
    const int qper = 16 * (c_tt.height / 4);
    const uint8_t* nb[9];
    for (int k = 0; k < 9; ++k) nb[k] = bbiomes + static_cast<size_t>(nb_index[ci * 9 + k]) * qper;
    int hm = h0[idx];
    SurfaceCtx c;
    init_ctx(c, chunks, ci, col, blocks, &hm, nb, corners, rows, prelim_corners);
    int bx = c.cx * 16 + c.lx, bz = c.cz * 16 + c.lz;
    int starting = (hm - 1) + 1;
    int surface_biome = c.biome_at(bx, starting, bz);
    if (surface_biome == c_tt.biome_eroded_badlands) eroded_badlands(c, bx, bz, starting);
    h1[idx] = hm;
    if (surface_biome == c_tt.biome_frozen_ocean || surface_biome == c_tt.biome_deep_frozen_ocean) {
        update_xz(c, bx, bz, 0, 0);
        frozen_ocean(c, c.min_surface_level(), surface_biome, bx, bz, starting, false);
    }
    h2[idx] = hm;
}

// Pass 2: the material-rule walk per column, then the frozen-ocean extension.
__global__ void k_surface2(const ChunkXZ* chunks, int n, uint8_t* blocks, const int* h0, const int* h1, const int* h2, const int* nb_index,
                           const uint8_t* bbiomes, const float* corners, int rows, const float* prelim_corners) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 256) return;
    int ci = idx / 256, col = idx % 256;
    const int qper = 16 * (c_tt.height / 4);
    const uint8_t* nb[9];
    for (int k = 0; k < 9; ++k) nb[k] = bbiomes + static_cast<size_t>(nb_index[ci * 9 + k]) * qper;
    int x = col / 16, z = col % 16;
    // Height of another column as seen when vanilla processes (x, z): columns earlier in
    // the x-major, z-minor loop have finished both extensions; later ones none.
    auto H = [&](int ox, int oz) {
        int j = ci * 256 + ox * 16 + oz;
        int v;
        if (ox == x && oz == z) v = h1[j];
        else if (ox < x || (ox == x && oz < z)) v = h2[j];
        else v = h0[j];
        return v - 1;                                        // ChunkAccess.getHeight
    };
    int hm = h1[idx];
    SurfaceCtx c;
    init_ctx(c, chunks, ci, col, blocks, &hm, nb, corners, rows, prelim_corners);
    int bx = c.cx * 16 + x, bz = c.cz * 16 + z;
    int starting = (h0[idx] - 1) + 1;
    int surface_biome = c.biome_at(bx, starting, bz);
    int height = H(x, z) + 1;
    int gx = H(min(x + 1, 15), z) - H(max(x - 1, 0), z);
    int gz = H(x, min(z + 1, 15)) - H(x, max(z - 1, 0));
    update_xz(c, bx, bz, gx, gz);
    int stone_above = 0, water_height = INT_MIN, next_ceiling = INT_MAX;
    const int end_y = c_tt.min_y, max_y = c_tt.min_y + c_tt.height - 1;
    for (int y = height; y >= end_y; --y) {
        int old = c.get(y);
        if (c.air(old)) {
            stone_above = 0;
            water_height = INT_MIN;
        } else if (c.fluid(old)) {
            if (water_height == INT_MIN) water_height = y + 1;
        } else {
            if (next_ceiling >= y) {
                next_ceiling = kWayBelowMinY;
                for (int la = y - 1; la >= end_y - 1; --la) {
                    int s = c.get(la);
                    if (c.air(s) || c.fluid(s)) { next_ceiling = la + 1; break; }
                }
            }
            ++stone_above;
            c.stone_above = stone_above;
            c.stone_below = y - next_ceiling + 1;
            c.water_height = water_height;
            c.y = y;
            c.biome_set = false;
            c.ore_set = false;
            if (y >= end_y && y <= max_y) {
                int state = gen::apply_material_rules(c);
                if (state >= 0) c.set(y, state, true);
            }
        }
    }
    if (surface_biome == c_tt.biome_frozen_ocean || surface_biome == c_tt.biome_deep_frozen_ocean)
        frozen_ocean(c, c.min_surface_level(), surface_biome, bx, bz, starting, true);
}

}  // namespace

// ---- host side ---------------------------------------------------------------------------
struct FullTerrainGenerator::Impl {
    const IR& ir;
    const TerrainModel& tm;
    int max_batch;
    int rows;
    size_t per;
    int qper;
    std::unique_ptr<TerrainSeed> seed;
    PerlinOctave* d_octaves = nullptr;
    StackLayer* d_layers = nullptr;
    BiomeTree::FlatNode* d_tree = nullptr;
    int32_t* d_tree_children = nullptr;
    int32_t* d_leaf_biome = nullptr;
    int32_t* d_leaf_node = nullptr;
    // batch buffers
    ChunkXZ *d_chunks = nullptr, *d_bchunks = nullptr;
    int* d_nb = nullptr;
    float *d_bcol = nullptr, *d_col = nullptr, *d_corners = nullptr, *d_prelim = nullptr;
    int64_t *d_targets = nullptr, *d_mindist = nullptr;
    int32_t *d_firstmin = nullptr, *d_grid = nullptr;
    uint8_t *d_bbiomes = nullptr, *d_blocks = nullptr;
    int4* d_cells = nullptr;
    int *d_skip = nullptr, *d_h0 = nullptr, *d_h1 = nullptr, *d_h2 = nullptr;
    int bcap = 0;                  // biome-chunk capacity of the biome buffers
    cudaEvent_t ev[9];

    Impl(const IR& ir_, const TerrainModel& tm_, int mb) : ir(ir_), tm(tm_), max_batch(mb) {
        rows = ir.height / 8 + 1;
        per = static_cast<size_t>(ir.height) * 256;
        qper = 16 * (ir.height / 4);
        check(cudaMalloc(&d_chunks, sizeof(ChunkXZ) * mb), "malloc");
        check(cudaMalloc(&d_nb, sizeof(int) * mb * 9), "malloc");
        check(cudaMalloc(&d_col, sizeof(float) * mb * kCols * std::max(gen::kNumColumns, 1)), "malloc");
        check(cudaMalloc(&d_corners, sizeof(float) * mb * kCols * rows * kAllChannels), "malloc");
        check(cudaMalloc(&d_grid, sizeof(int32_t) * mb * kGridCells), "malloc");
        check(cudaMalloc(&d_prelim, sizeof(float) * mb * 4), "malloc");
        check(cudaMalloc(&d_skip, sizeof(int) * mb), "malloc");
        int aq_min = floor_div(ir.min_y + 1, 12) - 1, aq_max = floor_div(ir.min_y + ir.height - 1 + 1, 12) + 1;
        int cells = 9 * (aq_max - aq_min + 1);
        check(cudaMalloc(&d_cells, sizeof(int4) * mb * cells * 2), "malloc");
        check(cudaMalloc(&d_blocks, mb * per), "malloc");
        check(cudaMalloc(&d_h0, sizeof(int) * mb * 256), "malloc");
        check(cudaMalloc(&d_h1, sizeof(int) * mb * 256), "malloc");
        check(cudaMalloc(&d_h2, sizeof(int) * mb * 256), "malloc");
        for (auto& e : ev) cudaEventCreate(&e);
        check(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024), "stack limit");
        if (tm.palette.size() > 64 || tm.biomes.size() > 128) throw std::runtime_error("palette/biome tables too large");
    }

    ~Impl() {
        void* ptrs[] = {d_octaves, d_layers, d_tree, d_tree_children, d_leaf_biome, d_leaf_node, d_chunks, d_bchunks, d_nb, d_bcol, d_targets, d_mindist,
                        d_firstmin, d_bbiomes, d_col, d_corners, d_grid, d_prelim, d_skip, d_cells, d_blocks, d_h0, d_h1, d_h2};
        for (void* p : ptrs) cudaFree(p);
        for (auto& e : ev) cudaEventDestroy(e);
    }

    template <typename T>
    void upload(T*& d, const std::vector<T>& v) {
        cudaFree(d);
        check(cudaMalloc(&d, sizeof(T) * std::max<size_t>(v.size(), 1)), "malloc table");
        check(cudaMemcpy(d, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "upload table");
    }

    // Biome buffers grow to the largest neighbourhood union seen (a square region of
    // n chunks needs about (sqrt(n) + 2)^2 biome chunks, scattered chunks up to 9n).
    void ensure_biome_capacity(int nbk) {
        if (nbk <= bcap) return;
        void* old[] = {d_bchunks, d_bcol, d_targets, d_mindist, d_firstmin, d_bbiomes};
        for (void* p : old) cudaFree(p);
        const size_t q = static_cast<size_t>(nbk) * qper;
        check(cudaMalloc(&d_bchunks, sizeof(ChunkXZ) * nbk), "malloc");
        check(cudaMalloc(&d_bcol, sizeof(float) * nbk * 16 * std::max(gen::kNumClimateColumns, 1)), "malloc");
        check(cudaMalloc(&d_targets, sizeof(int64_t) * q * 7), "malloc");
        check(cudaMalloc(&d_mindist, sizeof(int64_t) * q), "malloc");
        check(cudaMalloc(&d_firstmin, sizeof(int32_t) * q), "malloc");
        check(cudaMalloc(&d_bbiomes, q), "malloc");
        bcap = nbk;
    }

    void set_seed(int64_t s) {
        seed = std::make_unique<TerrainSeed>(ir, tm, s);
        const SeedTables& t = seed->tables;
        upload(d_octaves, t.octaves);
        upload(d_layers, t.layers);
        upload(d_tree, seed->tree.flat());
        upload(d_tree_children, seed->tree.flat_children());
        std::vector<int32_t> leaf_biome, leaf_node;
        for (size_t i = 0; i < tm.params.size(); ++i) leaf_biome.push_back(tm.params[i].biome);
        leaf_node.assign(tm.params.size(), -1);
        for (size_t i = 0; i < seed->tree.flat().size(); ++i)
            if (seed->tree.flat()[i].leaf >= 0) leaf_node[static_cast<size_t>(seed->tree.flat()[i].leaf)] = static_cast<int32_t>(i);
        upload(d_leaf_biome, leaf_biome);
        upload(d_leaf_node, leaf_node);

        TerrainTables tt{};
        tt.rt.octaves = d_octaves;
        tt.rt.layers = d_layers;
        for (int i = 0; i < gen::kNumNoises; ++i) {
            const StackRef& r = t.noise.at(gen::kNoiseKeys[i]);
            tt.rt.stacks[i] = {r.first_layer, r.count};
        }
        for (int i = 0; i < gen::kNumBlended; ++i) {
            const BlendedSet& b = t.blended.at(gen::kBlendedNodeIds[i]);
            tt.rt.blended[i] = {{b.min_limit.first_layer, b.min_limit.count}, {b.max_limit.first_layer, b.max_limit.count}, {b.main.first_layer, b.main.count}};
        }
        tt.tree = d_tree;
        tt.tree_children = d_tree_children;
        tt.tree_root = seed->tree.root();
        tt.leaf_biome = d_leaf_biome;
        tt.leaf_node = d_leaf_node;
        for (size_t i = 0; i < tm.palette.size(); ++i) {
            const BlockInfo& b = tm.palette[i];
            tt.block_flags[i] = static_cast<uint8_t>((b.air ? kFlagAir : 0) | (b.water ? kFlagWater : 0) | (b.lava ? kFlagLava : 0) | (b.fluid ? kFlagFluid : 0));
        }
        tt.air = tm.air; tt.water = tm.water; tt.lava = tm.lava; tt.default_block = tm.default_block; tt.default_fluid = tm.default_fluid;
        tt.snow_block = tm.snow_block; tt.packed_ice = tm.packed_ice;
        tt.sea_level = tm.sea_level; tt.min_y = ir.min_y; tt.height = ir.height;
        tt.biome_eroded_badlands = tm.biome_eroded_badlands; tt.biome_frozen_ocean = tm.biome_frozen_ocean; tt.biome_deep_frozen_ocean = tm.biome_deep_frozen_ocean;
        for (size_t i = 0; i < tm.biomes.size(); ++i) { tt.biome_temp[i] = tm.biomes[i].temperature; tt.biome_frozen[i] = tm.biomes[i].frozen_modifier ? 1 : 0; }
        auto copy_simplex = [](dev::Simplex2D& d, const Simplex2D& s) { std::copy(s.perm, s.perm + 256, d.perm); };
        copy_simplex(tt.temp_noise, seed->temp.temperature);
        for (int i = 0; i < 3; ++i) { copy_simplex(tt.frozen[i], seed->temp.frozen[i]); tt.frozen_amp[i] = seed->temp.frozen_amp[i]; }
        copy_simplex(tt.biome_info, seed->temp.biome_info);
        std::copy(seed->clay_bands, seed->clay_bands + 192, tt.clay_bands);
        tt.random = {seed->random.lo(), seed->random.hi()};
        tt.aquifer_random = {seed->aquifer_random.lo(), seed->aquifer_random.hi()};
        tt.ore_random = {seed->ore_random.lo(), seed->ore_random.hi()};
        if (gen::kNumRandomNames > kMaxGradientRandoms) throw std::runtime_error("too many vertical-gradient random names");
        for (int i = 0; i < gen::kNumRandomNames; ++i) {
            const XoroshiroPositional& p = seed->gradient_random.at(gen::kRandomNames[i]);
            tt.gradient_random[i] = {p.lo(), p.hi()};
        }
        tt.zoom_seed = seed->zoom_seed;
        auto st = [&](const char* k) {
            for (int i = 0; i < gen::kNumNoises; ++i)
                if (std::string(gen::kNoiseKeys[i]) == k) return i;
            throw std::runtime_error(std::string("missing noise ") + k);
        };
        tt.st_surface = st("minecraft:surface");
        tt.st_surface_secondary = st("minecraft:surface_secondary");
        tt.st_clay_offset = st("minecraft:clay_bands_offset");
        tt.st_badlands_pillar = st("minecraft:badlands_pillar");
        tt.st_badlands_roof = st("minecraft:badlands_pillar_roof");
        tt.st_badlands_surface = st("minecraft:badlands_surface");
        tt.st_iceberg_pillar = st("minecraft:iceberg_pillar");
        tt.st_iceberg_roof = st("minecraft:iceberg_pillar_roof");
        tt.st_iceberg_surface = st("minecraft:iceberg_surface");
        tt.aq_min_grid_y = floor_div(ir.min_y + 1, 12) - 1;
        tt.aq_grid_y = floor_div(ir.min_y + ir.height - 1 + 1, 12) + 1 - tt.aq_min_grid_y + 1;
        if (gen::kNumOreRules > kMaxOreRules) throw std::runtime_error("too many ore-vein rules");
        for (int k = 0; k < gen::kNumOreRules; ++k) {
            for (int j = 0; j < 3; ++j) tt.ore_state[k][j] = gen::kOreRuleState[k][j];
            tt.ore_raw_chance[k] = bits2f(gen::kOreRuleRawChanceBits[k]);
        }
        check(cudaMemcpyToSymbol(c_tt, &tt, sizeof(tt)), "upload terrain tables");
    }

    void run_batch(const ChunkXZ* chunks, int n, uint8_t* terrain_out, uint8_t* biomes_out, uint8_t* aquifer_out, TerrainTiming* tm_) {
        // Biome chunks: the union of every output chunk's 3x3 neighbourhood.
        std::map<std::pair<int, int>, int> bindex;
        std::vector<ChunkXZ> bchunks;
        std::vector<int> nb(static_cast<size_t>(n) * 9);
        for (int i = 0; i < n; ++i)
            for (int dz = -1; dz <= 1; ++dz)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto key = std::make_pair(chunks[i].x + dx, chunks[i].z + dz);
                    auto it = bindex.find(key);
                    if (it == bindex.end()) {
                        it = bindex.emplace(key, static_cast<int>(bchunks.size())).first;
                        bchunks.push_back({key.first, key.second});
                    }
                    nb[static_cast<size_t>(i) * 9 + (dz + 1) * 3 + dx + 1] = it->second;
                }
        int nbk = static_cast<int>(bchunks.size());
        ensure_biome_capacity(nbk);
        cudaEventRecord(ev[0]);
        check(cudaMemcpy(d_chunks, chunks, sizeof(ChunkXZ) * n, cudaMemcpyHostToDevice), "upload chunks");
        check(cudaMemcpy(d_bchunks, bchunks.data(), sizeof(ChunkXZ) * nbk, cudaMemcpyHostToDevice), "upload biome chunks");
        check(cudaMemcpy(d_nb, nb.data(), sizeof(int) * nb.size(), cudaMemcpyHostToDevice), "upload neighbours");
        cudaEventRecord(ev[1]);
        const int tb = 128;
        auto grid = [](size_t threads, int block) { return static_cast<unsigned>((threads + block - 1) / block); };
        k_biome_columns<<<grid(static_cast<size_t>(nbk) * 16, tb), tb>>>(d_bchunks, nbk, d_bcol);
        k_biome_quarts<<<grid(static_cast<size_t>(nbk) * qper, tb), tb>>>(d_bchunks, nbk, d_bcol, d_targets, d_firstmin, d_mindist);
        k_biome_chain<<<grid(nbk, tb), tb>>>(nbk, d_targets, d_firstmin, d_mindist, d_bbiomes);
        check(cudaGetLastError(), "biome kernels");
        cudaEventRecord(ev[2]);
        k_columns<<<grid(static_cast<size_t>(n) * kCols, tb), tb>>>(d_chunks, n, d_col);
        k_corners<<<grid(static_cast<size_t>(n) * kCols * rows, tb), tb>>>(d_chunks, n, d_col, d_corners, rows);
        check(cudaGetLastError(), "noise kernels");
        cudaEventRecord(ev[3]);
        k_surface_grid<<<grid(static_cast<size_t>(n) * (kGridCells + 4), tb), tb>>>(d_chunks, n, d_grid, d_prelim);
        k_aquifer_skip<<<grid(n, tb), tb>>>(n, d_grid, d_skip);
        int cell_per = 9 * (floor_div(ir.min_y + ir.height, 12) + 1 - (floor_div(ir.min_y + 1, 12) - 1) + 1);
        k_aquifer_cells<<<grid(static_cast<size_t>(n) * cell_per, tb), tb>>>(d_chunks, n, d_grid, d_cells);
        check(cudaGetLastError(), "aquifer kernels");
        cudaEventRecord(ev[4]);
        k_blocks<<<grid(static_cast<size_t>(n) * per, 256), 256>>>(d_chunks, n, d_corners, rows, d_cells, d_skip, nullptr, d_blocks);
        check(cudaGetLastError(), "fill kernel");
        cudaEventRecord(ev[5]);
        if (aquifer_out) check(cudaMemcpy(aquifer_out, d_blocks, static_cast<size_t>(n) * per, cudaMemcpyDeviceToHost), "download aquifer");
        cudaEventRecord(ev[6]);
        k_heights<<<grid(static_cast<size_t>(n) * 256, tb), tb>>>(n, d_blocks, d_h0);
        k_surface1<<<grid(static_cast<size_t>(n) * 256, 64), 64>>>(d_chunks, n, d_blocks, d_h0, d_h1, d_h2, d_nb, d_bbiomes, d_corners, rows, d_prelim);
        k_surface2<<<grid(static_cast<size_t>(n) * 256, 64), 64>>>(d_chunks, n, d_blocks, d_h0, d_h1, d_h2, d_nb, d_bbiomes, d_corners, rows, d_prelim);
        check(cudaGetLastError(), "surface kernels");
        cudaEventRecord(ev[7]);
        check(cudaMemcpy(terrain_out, d_blocks, static_cast<size_t>(n) * per, cudaMemcpyDeviceToHost), "download terrain");
        if (biomes_out) {
            std::vector<uint8_t> all(static_cast<size_t>(nbk) * qper);
            check(cudaMemcpy(all.data(), d_bbiomes, all.size(), cudaMemcpyDeviceToHost), "download biomes");
            for (int i = 0; i < n; ++i) {
                int b = nb[static_cast<size_t>(i) * 9 + 4];
                std::copy(all.begin() + static_cast<size_t>(b) * qper, all.begin() + static_cast<size_t>(b + 1) * qper, biomes_out + static_cast<size_t>(i) * qper);
            }
        }
        cudaEventRecord(ev[8]);
        check(cudaEventSynchronize(ev[8]), "sync");
        if (tm_) {
            float t[8];
            for (int i = 0; i < 8; ++i) check(cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]), "event timing");
            tm_->h2d_ms += t[0];
            tm_->biome_ms += t[1];
            tm_->noise_ms += t[2];
            tm_->aquifer_ms += t[3];
            tm_->fill_ms += t[4];
            tm_->d2h_ms += t[5] + t[7];
            tm_->surface_ms += t[6];
        }
    }
};

FullTerrainGenerator::FullTerrainGenerator(const IR& ir, const TerrainModel& tm, int max_batch) : impl_(std::make_unique<Impl>(ir, tm, max_batch)) {}
FullTerrainGenerator::~FullTerrainGenerator() = default;
void FullTerrainGenerator::set_seed(int64_t seed) { impl_->set_seed(seed); }

void FullTerrainGenerator::generate(const ChunkXZ* chunks, int count, uint8_t* terrain_out, uint8_t* biomes_out, uint8_t* aquifer_out, TerrainTiming* timing) {
    auto t0 = std::chrono::steady_clock::now();
    for (int off = 0; off < count; off += impl_->max_batch) {
        int n = std::min(impl_->max_batch, count - off);
        impl_->run_batch(chunks + off, n, terrain_out + static_cast<size_t>(off) * impl_->per,
                         biomes_out ? biomes_out + static_cast<size_t>(off) * impl_->qper : nullptr,
                         aquifer_out ? aquifer_out + static_cast<size_t>(off) * impl_->per : nullptr, timing);
    }
    if (timing) timing->wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

uint8_t* alloc_pinned(size_t bytes) {
    void* p = nullptr;
    check(cudaMallocHost(&p, bytes), "pinned host allocation");
    return static_cast<uint8_t*>(p);
}

void free_pinned(void* p) { cudaFreeHost(p); }

}  // namespace mcc::cuda
