// mccuda_fly: first-person flythrough of GPU-generated terrain, generated live.
//
// Every frame, the chunks within the view radius that do not exist yet are generated
// with FullTerrainGenerator (the same bit-exact pipeline as the benchmarks, copy to
// host included), uploaded into a toroidal voxel window on the device, and the frame
// is ray traced from the camera. Blocks are drawn with flat, procedurally varied
// colours (no game assets). Frames go to stdout, each preceded by a 256-byte text
// header with the frame's measured timings; bench/make_flythrough.py adds the overlay
// and encodes the video.
//
//   mccuda_fly --path FILE [--seed S] [--width W --height H --fps F] [--radius R] [--cap N]
//
// Path file: one keyframe per line, "t x z mode y pitch" (mode g: y above the ground
// at the camera, a: absolute y; pitch in degrees, negative looks down).
#include <cuda_runtime.h>
#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <array>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/terrain_model.h"
#include "mccuda_cuda/terrain_gpu.h"

#ifndef MCCUDA_DATA_DIR
#define MCCUDA_DATA_DIR "data"
#endif

using namespace mcc;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int W = 64;            // window side in chunks (toroidal)
constexpr int WB = W * 16;       // window side in blocks
constexpr int CELL = 4;          // empty-space skipping cell, in columns
constexpr int WC = WB / CELL;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// ---- colours (sRGB, converted to linear on upload) -------------------------------------------
const std::map<std::string, std::array<int, 3>> kColours = {
    {"minecraft:stone", {126, 126, 126}}, {"minecraft:water", {38, 84, 172}}, {"minecraft:lava", {222, 98, 22}},
    {"minecraft:terracotta", {152, 94, 67}}, {"minecraft:white_terracotta", {209, 178, 161}}, {"minecraft:orange_terracotta", {161, 83, 37}},
    {"minecraft:yellow_terracotta", {186, 133, 36}}, {"minecraft:brown_terracotta", {77, 51, 36}}, {"minecraft:red_terracotta", {143, 61, 47}},
    {"minecraft:light_gray_terracotta", {135, 107, 98}}, {"minecraft:packed_ice", {141, 180, 250}}, {"minecraft:snow_block", {240, 246, 248}},
    {"minecraft:bedrock", {60, 60, 60}}, {"minecraft:copper_ore", {124, 125, 120}}, {"minecraft:raw_copper_block", {154, 105, 79}},
    {"minecraft:granite", {149, 103, 86}}, {"minecraft:deepslate_iron_ore", {92, 90, 90}}, {"minecraft:raw_iron_block", {166, 136, 107}},
    {"minecraft:tuff", {108, 109, 102}}, {"minecraft:coarse_dirt", {119, 85, 59}}, {"minecraft:grass_block", {88, 128, 56}},
    {"minecraft:dirt", {134, 96, 67}}, {"minecraft:red_sandstone", {181, 98, 31}}, {"minecraft:red_sand", {190, 102, 33}},
    {"minecraft:gravel", {131, 127, 126}}, {"minecraft:ice", {145, 183, 253}}, {"minecraft:powder_snow", {248, 253, 253}},
    {"minecraft:calcite", {223, 224, 220}}, {"minecraft:sandstone", {216, 203, 155}}, {"minecraft:sand", {219, 207, 163}},
    {"minecraft:cinnabar", {158, 52, 46}}, {"minecraft:sulfur", {205, 196, 74}}, {"minecraft:mud", {60, 57, 61}},
    {"minecraft:podzol", {91, 63, 24}}, {"minecraft:mycelium", {111, 99, 105}}, {"minecraft:deepslate", {80, 80, 84}}, {"minecraft:air", {0, 0, 0}},
};

__constant__ float3 c_top[64];   // colour of top faces (and all faces unless c_cap > 0)
__constant__ float3 c_side[64];  // side/bottom colour for blocks with a capped top (grass: dirt)
__constant__ int c_cap[64];      // texels of the top colour at the top edge of side faces
__constant__ int c_air, c_water, c_lava, c_bedrock;

struct World {
    uint8_t* vox;     // [y][z][x] over the window
    uint16_t* hmax;   // per 4x4 column cell: 1 + highest non-air y index, 0 if empty
    int2* slot;       // chunk held by each window slot
    int* born;        // frame the slot's chunk was generated in
    int H, min_y;
};

struct Cam {
    float3 pos, f, r, u;
    float tan_half, aspect;
};

__device__ __forceinline__ int wrapi(int a, int m) {
    int r = a % m;
    return r < 0 ? r + m : r;
}
__device__ __forceinline__ int fdiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

__device__ __forceinline__ bool slot_ok(const World& w, int x, int z) {
    const int cx = x >> 4, cz = z >> 4;
    const int2 s = w.slot[wrapi(cz, W) * W + wrapi(cx, W)];
    return s.x == cx && s.y == cz;
}

__device__ __forceinline__ int voxel(const World& w, int x, int y, int z) {
    const int yi = y - w.min_y;
    if (yi < 0) return c_bedrock;
    if (yi >= w.H || !slot_ok(w, x, z)) return c_air;
    return w.vox[(static_cast<size_t>(yi) * WB + wrapi(z, WB)) * WB + wrapi(x, WB)];
}

__device__ __forceinline__ int cell_top(const World& w, int x, int z) {   // absolute y below which the cell may hold blocks
    if (!slot_ok(w, x, z)) return w.min_y;
    return w.min_y + w.hmax[wrapi(fdiv(z, CELL), WC) * WC + wrapi(fdiv(x, CELL), WC)];
}

__device__ __forceinline__ bool solid(int id) { return id != c_air && id != c_water; }

struct Hit {
    int id;
    int3 p, n;
    float t;
    float water_t;   // where the ray entered water (-1: never)
};

// Amanatides-Woo voxel traversal with empty-space skipping over 4x4 column cells.
// Water is transparent; its entry distance is recorded.
__device__ bool trace(const World& w, float3 o, float3 d, float tmax, bool stop_at_water_free, Hit& h) {
    const float3 inv = make_float3(fabsf(d.x) > 1e-9f ? 1.0f / d.x : 1e30f, fabsf(d.y) > 1e-9f ? 1.0f / d.y : 1e30f, fabsf(d.z) > 1e-9f ? 1.0f / d.z : 1e30f);
    const int3 st = make_int3(d.x > 0 ? 1 : -1, d.y > 0 ? 1 : -1, d.z > 0 ? 1 : -1);
    const float3 td = make_float3(fabsf(inv.x), fabsf(inv.y), fabsf(inv.z));
    float t = 0.0f;
    int3 ip, n = make_int3(0, 0, 0);
    float3 tm;
    auto init = [&](float tt) {
        float3 p = make_float3(o.x + d.x * tt, o.y + d.y * tt, o.z + d.z * tt);
        ip = make_int3(static_cast<int>(floorf(p.x)), static_cast<int>(floorf(p.y)), static_cast<int>(floorf(p.z)));
        tm.x = ((st.x > 0 ? ip.x + 1 : ip.x) - o.x) * inv.x;
        tm.y = ((st.y > 0 ? ip.y + 1 : ip.y) - o.y) * inv.y;
        tm.z = ((st.z > 0 ? ip.z + 1 : ip.z) - o.z) * inv.z;
    };
    init(0.0f);
    h.water_t = -1.0f;
    const int top_y = w.min_y + w.H;
    for (int i = 0; i < 6000 && t < tmax; ++i) {
        if (ip.y >= top_y) {
            if (d.y >= 0.0f) return false;
        } else if (h.water_t < 0.0f) {
            const int ct = cell_top(w, ip.x, ip.z);
            if (ip.y >= ct) {   // skip the rest of this column cell if the ray stays above its blocks
                const int ccx = fdiv(ip.x, CELL), ccz = fdiv(ip.z, CELL);
                const float tx = ((st.x > 0 ? (ccx + 1) * CELL : ccx * CELL) - o.x) * inv.x;
                const float tz = ((st.z > 0 ? (ccz + 1) * CELL : ccz * CELL) - o.z) * inv.z;
                const float te = fminf(tx, tz);
                const float y_exit = o.y + d.y * te;
                if (y_exit >= static_cast<float>(ct) && te > t) {
                    t = te + 1e-4f;
                    n = tx < tz ? make_int3(-st.x, 0, 0) : make_int3(0, 0, -st.z);
                    init(t);
                    continue;
                }
            }
        }
        const int id = voxel(w, ip.x, ip.y, ip.z);
        if (id == c_water) {
            if (stop_at_water_free && h.water_t < 0.0f) h.water_t = t;
        } else if (id != c_air) {
            h.id = id;
            h.p = ip;
            h.n = n;
            h.t = t;
            return true;
        }
        if (tm.x < tm.y && tm.x < tm.z) {
            ip.x += st.x; t = tm.x; tm.x += td.x; n = make_int3(-st.x, 0, 0);
        } else if (tm.y < tm.z) {
            ip.y += st.y; t = tm.y; tm.y += td.y; n = make_int3(0, -st.y, 0);
        } else {
            ip.z += st.z; t = tm.z; tm.z += td.z; n = make_int3(0, 0, -st.z);
        }
    }
    return false;
}

__device__ __forceinline__ uint32_t hash4(int a, int b, int c, int d) {
    uint32_t h = static_cast<uint32_t>(a) * 0x8da6b343u ^ static_cast<uint32_t>(b) * 0xd8163841u ^ static_cast<uint32_t>(c) * 0xcb1ab31fu ^
                 static_cast<uint32_t>(d) * 0x165667b1u;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
    return h;
}

__device__ __forceinline__ float frac(float x) { return x - floorf(x); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 mixf(float3 a, float3 b, float t) { return a * (1.0f - t) + b * t; }
__device__ __forceinline__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ float3 block_colour(int id, int3 ip, int3 n, float3 p) {
    float u, v;
    int face;
    if (n.y != 0) { u = frac(p.x); v = frac(p.z); face = n.y > 0 ? 0 : 1; }
    else if (n.x != 0) { u = frac(p.z); v = frac(p.y); face = 2; }
    else { u = frac(p.x); v = frac(p.y); face = 3; }
    const int tu = min(15, static_cast<int>(u * 16.0f)), tv = min(15, static_cast<int>(v * 16.0f));
    float3 c = c_top[id];
    if (c_cap[id] > 0) {
        if (n.y < 0) c = c_side[id];
        else if (n.y == 0) {
            const int edge = c_cap[id] - static_cast<int>(hash4(ip.x * 31 + tu, ip.y, ip.z * 17 + tu, 7) % 2u);
            if (tv < 16 - edge) c = c_side[id];
        }
    }
    const float r = (hash4(ip.x, ip.y, ip.z, face * 256 + tv * 16 + tu) & 0xffff) / 65535.0f;
    const float b = (hash4(ip.x, ip.y, ip.z, 99) & 0xffff) / 65535.0f;
    return c * (0.86f + 0.16f * r + 0.06f * (b - 0.5f));
}

// Minecraft-style smooth ambient occlusion from the 8 neighbours in front of the face.
__device__ float face_ao(const World& w, int3 ip, int3 n, float3 p) {
    if (n.x == 0 && n.y == 0 && n.z == 0) return 1.0f;
    const int3 f = make_int3(ip.x + n.x, ip.y + n.y, ip.z + n.z);
    int3 U, V;
    float fu, fv;
    if (n.x != 0) { U = make_int3(0, 0, 1); V = make_int3(0, 1, 0); fu = frac(p.z); fv = frac(p.y); }
    else if (n.y != 0) { U = make_int3(1, 0, 0); V = make_int3(0, 0, 1); fu = frac(p.x); fv = frac(p.z); }
    else { U = make_int3(1, 0, 0); V = make_int3(0, 1, 0); fu = frac(p.x); fv = frac(p.y); }
    auto occ = [&](int du, int dv) {
        return solid(voxel(w, f.x + du * U.x + dv * V.x, f.y + du * U.y + dv * V.y, f.z + du * U.z + dv * V.z)) ? 1 : 0;
    };
    const int s_u0 = occ(-1, 0), s_u1 = occ(1, 0), s_v0 = occ(0, -1), s_v1 = occ(0, 1);
    auto corner = [&](int su, int sv, int cu, int cv) {
        const int c = occ(su, sv);
        return (cu && cv) ? 0.0f : (3 - cu - cv - c) / 3.0f;
    };
    const float a00 = corner(-1, -1, s_u0, s_v0), a10 = corner(1, -1, s_u1, s_v0);
    const float a01 = corner(-1, 1, s_u0, s_v1), a11 = corner(1, 1, s_u1, s_v1);
    const float a0 = a00 + (a10 - a00) * fu, a1 = a01 + (a11 - a01) * fu;
    return a0 + (a1 - a0) * fv;
}

__device__ float3 sky(float3 d, float3 L) {
    const float h = fmaxf(d.y, 0.0f);
    const float3 hor = make_float3(0.55f, 0.68f, 0.88f), zen = make_float3(0.11f, 0.26f, 0.62f);
    float3 c = mixf(hor, zen, sqrtf(h));
    if (d.y < 0.0f) c = hor * 0.92f;
    const float s = fmaxf(dot3(d, L), 0.0f);
    c = c + make_float3(1.0f, 0.92f, 0.75f) * (powf(s, 900.0f) * 12.0f + powf(s, 12.0f) * 0.18f);
    return c;
}

__global__ void k_render(World w, Cam cam, float3 L, int width, int height, int frame, float fog, float tmax, uchar3* out) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    float3 acc = make_float3(0, 0, 0);
    const float3 sun_col = make_float3(1.45f, 1.32f, 1.12f), amb_col = make_float3(0.30f, 0.37f, 0.48f);
    for (int s = 0; s < 4; ++s) {   // 2x2 supersampling
        const float sx = (px + 0.25f + 0.5f * (s & 1)) / width * 2.0f - 1.0f;
        const float sy = 1.0f - (py + 0.25f + 0.5f * (s >> 1)) / height * 2.0f;
        float3 d = cam.f + cam.r * (sx * cam.tan_half * cam.aspect) + cam.u * (sy * cam.tan_half);
        d = d * rsqrtf(dot3(d, d));
        Hit h;
        const bool hit = trace(w, cam.pos, d, tmax, true, h);
        float3 col;
        float dist = 1e30f;
        if (hit) {
            dist = h.t;
            const float3 p = cam.pos + d * h.t;
            const float3 nf = make_float3(static_cast<float>(h.n.x), static_cast<float>(h.n.y), static_cast<float>(h.n.z));
            const float3 base = block_colour(h.id, h.p, h.n, p);
            float ndl = fmaxf(dot3(nf, L), 0.0f);
            if (ndl > 0.0f) {
                Hit sh;
                const float3 o2 = p + nf * 0.002f;
                if (trace(w, o2, L, 260.0f, false, sh)) ndl = 0.0f;
            }
            const float ao = face_ao(w, h.p, h.n, p);
            const float face_amb = h.n.y > 0 ? 1.0f : (h.n.y < 0 ? 0.45f : 0.72f);
            col = base * (sun_col * ndl + amb_col * (face_amb * (0.30f + 0.70f * ao)));
            if (h.id == c_lava) col = base * 1.6f;
            const int2 sc = make_int2(h.p.x >> 4, h.p.z >> 4);
            const int age = frame - w.born[wrapi(sc.y, W) * W + wrapi(sc.x, W)];
            if (age >= 0 && age < 10) col = col + make_float3(0.03f, 0.11f, 0.02f) * (1.0f - age / 10.0f);   // just generated: a faint green flash
        } else {
            col = sky(d, L);
        }
        if (h.water_t >= 0.0f) {   // water surface: absorption along the underwater path, then a sky reflection
            const float dw = (hit ? h.t : h.water_t + 40.0f) - h.water_t;
            const float tr = expf(-0.11f * dw);
            const float3 wc = make_float3(0.03f, 0.11f, 0.24f) * (0.55f + 0.6f * fmaxf(L.y, 0.0f));
            col = col * tr + wc * (1.0f - tr);
            const float c = fabsf(d.y);
            const float F = 0.03f + 0.97f * powf(1.0f - c, 5.0f);
            const float3 rd = make_float3(d.x, fabsf(d.y), d.z);
            col = col * (1.0f - F) + sky(rd, L) * F;
            dist = h.water_t;
        }
        if (dist < 1e29f && dist > 0.45f * fog) {   // haze from half the view distance, full at its edge
            const float x = (dist - 0.45f * fog) / (0.5f * fog);
            const float f = 1.0f - expf(-x * x);
            col = mixf(col, sky(make_float3(d.x, fmaxf(d.y, 0.02f), d.z), L), f);
        }
        acc = acc + col;
    }
    acc = acc * 0.25f;
    auto tone = [](float x) {   // ACES-style filmic curve, then sRGB gamma
        x *= 0.9f;
        x = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
        return static_cast<unsigned char>(fminf(255.0f, powf(fminf(fmaxf(x, 0.0f), 1.0f), 1.0f / 2.2f) * 255.0f + 0.5f));
    };
    out[static_cast<size_t>(py) * width + px] = make_uchar3(tone(acc.x), tone(acc.y), tone(acc.z));
}

// New chunks (canonical layout, back to back) into their window slots.
__global__ void k_scatter(World w, const uint8_t* src, const int2* coords, int n) {
    const size_t per = static_cast<size_t>(w.H) * 256;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= per * n) return;
    const int c = static_cast<int>(idx / per), e = static_cast<int>(idx % per);
    const int x = e & 15, z = (e >> 4) & 15, y = e >> 8;
    const int wx = wrapi(coords[c].x * 16 + x, WB), wz = wrapi(coords[c].y * 16 + z, WB);
    w.vox[(static_cast<size_t>(y) * WB + wz) * WB + wx] = src[idx];
}

__global__ void k_cells(World w, const int2* coords, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * 16) return;
    const int c = idx / 16, k = idx % 16;
    const int bx = coords[c].x * 16 + (k & 3) * CELL, bz = coords[c].y * 16 + (k >> 2) * CELL;
    int top = 0;
    for (int y = w.H - 1; y >= 0 && top == 0; --y)
        for (int dz = 0; dz < CELL && top == 0; ++dz)
            for (int dx = 0; dx < CELL; ++dx)
                if (w.vox[(static_cast<size_t>(y) * WB + wrapi(bz + dz, WB)) * WB + wrapi(bx + dx, WB)] != c_air) { top = y + 1; break; }
    w.hmax[wrapi(bz / CELL, WC) * WC + wrapi(bx / CELL, WC)] = static_cast<uint16_t>(top);
}

// ---- host ---------------------------------------------------------------------------------
struct Key {
    double t, x, z, y, pitch;
    char mode;
};

double catmull(double p0, double p1, double p2, double p3, double s) {
    return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s * s * s);
}

struct PathSample {
    double x, z, y, pitch;
    char mode;                 // 'g', 'a', or 'm': between a 'g' and an 'a' keyframe
    double ya, yb, s;          // for 'm': the two keyframe heights and the blend position
    char ma, mb;
};

PathSample sample_path(const std::vector<Key>& k, double t) {
    size_t i = 0;
    while (i + 2 < k.size() && k[i + 1].t <= t) ++i;
    const Key& a = k[i];
    const Key& b = k[std::min(i + 1, k.size() - 1)];
    const Key& p = k[i == 0 ? 0 : i - 1];
    const Key& q = k[std::min(i + 2, k.size() - 1)];
    const double s = b.t > a.t ? std::clamp((t - a.t) / (b.t - a.t), 0.0, 1.0) : 0.0;
    PathSample r;
    r.x = catmull(p.x, a.x, b.x, q.x, s);
    r.z = catmull(p.z, a.z, b.z, q.z, s);
    r.pitch = a.pitch + (b.pitch - a.pitch) * (s * s * (3 - 2 * s));
    if (a.mode == b.mode) {
        r.y = catmull(p.mode == a.mode ? p.y : a.y, a.y, b.y, q.mode == b.mode ? q.y : b.y, s);
        r.mode = a.mode;
    } else {
        r.mode = 'm';
        r.ya = a.y; r.yb = b.y; r.ma = a.mode; r.mb = b.mode;
        r.s = s * s * (3 - 2 * s);
        r.y = 0;
    }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string data = MCCUDA_DATA_DIR, path_file;
        int64_t seed = 123456789;
        int width = 1280, height = 720, fps = 30, radius = 24, cap = 512;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
            if (a == "--path") path_file = next();
            else if (a == "--seed") seed = std::stoll(next());
            else if (a == "--width") width = std::stoi(next());
            else if (a == "--height") height = std::stoi(next());
            else if (a == "--fps") fps = std::stoi(next());
            else if (a == "--radius") radius = std::stoi(next());
            else if (a == "--cap") cap = std::stoi(next());
            else if (a == "--data") data = next();
            else throw std::runtime_error("unknown argument " + a);
        }
        if (radius * 2 + 2 > W) throw std::runtime_error("radius too large for the window");
        std::vector<Key> keys;
        {
            std::ifstream f(path_file);
            if (!f) throw std::runtime_error("cannot open path " + path_file);
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream ss(line);
                Key k;
                ss >> k.t >> k.x >> k.z >> k.mode >> k.y >> k.pitch;
                if (ss) keys.push_back(k);
            }
        }
        if (keys.size() < 2) throw std::runtime_error("path needs at least 2 keyframes");
        _setmode(_fileno(stdout), _O_BINARY);

        IR ir = load_ir(data + "/gamedata/26.3/overworld_ir.json");
        TerrainModel tm = load_terrain_model(data + "/gamedata/26.3/overworld_terrain.json");
        cuda::FullTerrainGenerator gen(ir, tm, cap, 1);
        gen.set_seed(seed);

        // colours by palette index
        float3 top[64] = {}, side[64] = {};
        int capv[64] = {};
        auto lin = [](int c) { return std::pow(c / 255.0f, 2.2f); };
        auto colour = [&](const std::string& n) {
            auto it = kColours.find(n);
            if (it == kColours.end()) throw std::runtime_error("no colour for " + n);
            return make_float3(lin(it->second[0]), lin(it->second[1]), lin(it->second[2]));
        };
        for (size_t i = 0; i < tm.palette.size(); ++i) {
            std::string n = tm.palette[i].name.substr(0, tm.palette[i].name.find('['));
            top[i] = side[i] = colour(n);
            if (n == "minecraft:grass_block") { side[i] = colour("minecraft:dirt"); capv[i] = 3; }
            if (n == "minecraft:podzol" || n == "minecraft:mycelium") { side[i] = colour("minecraft:dirt"); capv[i] = 2; }
        }
        check(cudaMemcpyToSymbol(c_top, top, sizeof top), "colours");
        check(cudaMemcpyToSymbol(c_side, side, sizeof side), "colours");
        check(cudaMemcpyToSymbol(c_cap, capv, sizeof capv), "colours");
        const int bedrock = tm.block("minecraft:bedrock");
        check(cudaMemcpyToSymbol(c_air, &tm.air, sizeof(int)), "ids");
        check(cudaMemcpyToSymbol(c_water, &tm.water, sizeof(int)), "ids");
        check(cudaMemcpyToSymbol(c_lava, &tm.lava, sizeof(int)), "ids");
        check(cudaMemcpyToSymbol(c_bedrock, &bedrock, sizeof(int)), "ids");

        World w{};
        w.H = ir.height;
        w.min_y = ir.min_y;
        const size_t per = static_cast<size_t>(ir.height) * 256;
        check(cudaMalloc(&w.vox, static_cast<size_t>(ir.height) * WB * WB), "malloc");
        check(cudaMemset(w.vox, tm.air, static_cast<size_t>(ir.height) * WB * WB), "memset");
        check(cudaMalloc(&w.hmax, sizeof(uint16_t) * WC * WC), "malloc");
        check(cudaMemset(w.hmax, 0, sizeof(uint16_t) * WC * WC), "memset");
        check(cudaMalloc(&w.slot, sizeof(int2) * W * W), "malloc");
        check(cudaMalloc(&w.born, sizeof(int) * W * W), "malloc");
        std::vector<int2> slot(W * W, make_int2(INT32_MIN, INT32_MIN));
        std::vector<int> born(W * W, -1000);
        uint8_t* host_out = cuda::alloc_pinned(per * cap);
        uint8_t* d_stage = nullptr;
        int2* d_coords = nullptr;
        check(cudaMalloc(&d_stage, per * cap), "malloc");
        check(cudaMalloc(&d_coords, sizeof(int2) * cap), "malloc");
        uchar3* d_frame = nullptr;
        check(cudaMalloc(&d_frame, sizeof(uchar3) * width * height), "malloc");
        std::vector<unsigned char> frame(static_cast<size_t>(width) * height * 3);
        cudaEvent_t r0, r1;
        cudaEventCreate(&r0);
        cudaEventCreate(&r1);

        // surface height per column of generated chunks (ground-following camera)
        std::unordered_map<int64_t, std::vector<int16_t>> heights;
        auto ckey = [](int cx, int cz) { return (static_cast<int64_t>(cx) << 32) ^ static_cast<uint32_t>(cz); };
        auto ground = [&](double x, double z) -> double {
            const int bx = static_cast<int>(std::floor(x)), bz = static_cast<int>(std::floor(z));
            auto it = heights.find(ckey(bx >> 4, bz >> 4));
            if (it == heights.end()) return 64.0;
            return it->second[static_cast<size_t>((bz & 15) * 16 + (bx & 15))];
        };

        const double duration = keys.back().t;
        const int frames = static_cast<int>(std::ceil(duration * fps)) + 1;
        double cam_y = 0.0, yaw = 0.0;
        bool first = true;
        long long total = 0;
        const float3 L = [] {
            float3 l = make_float3(-0.45f, 0.62f, -0.64f);
            float n = std::sqrt(l.x * l.x + l.y * l.y + l.z * l.z);
            return make_float3(l.x / n, l.y / n, l.z / n);
        }();
        std::fprintf(stderr, "mccuda_fly: %d frames %dx%d @ %d fps, radius %d chunks, %zu keyframes\n", frames, width, height, fps, radius, keys.size());
        for (int fr = 0; fr < frames; ++fr) {
            const double t = static_cast<double>(fr) / fps;
            PathSample ps = sample_path(keys, t);
            // generate missing chunks around the camera, nearest first
            const int ccx = static_cast<int>(std::floor(ps.x / 16.0)), ccz = static_cast<int>(std::floor(ps.z / 16.0));
            std::vector<std::pair<int, cuda::ChunkXZ>> need;
            for (int dz = -radius; dz <= radius; ++dz)
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dz * dz > radius * radius) continue;
                    const int cx = ccx + dx, cz = ccz + dz;
                    const int s = ((cz % W + W) % W) * W + ((cx % W + W) % W);
                    if (slot[s].x != cx || slot[s].y != cz) need.push_back({dx * dx + dz * dz, {cx, cz}});
                }
            std::sort(need.begin(), need.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            const int n = std::min(static_cast<int>(need.size()), cap);
            double gen_ms = 0.0, up_ms = 0.0;
            if (n > 0) {
                std::vector<cuda::ChunkXZ> list(static_cast<size_t>(n));
                std::vector<int2> coords(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) { list[i] = need[i].second; coords[i] = make_int2(list[i].x, list[i].z); }
                auto g0 = std::chrono::steady_clock::now();
                gen.generate(list.data(), n, host_out, nullptr, nullptr, nullptr);   // bit-exact TERRAIN step, copy to host included
                auto g1 = std::chrono::steady_clock::now();
                gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
                for (int i = 0; i < n; ++i) {   // ground heights (top non-air block, water included)
                    std::vector<int16_t> hcol(256);
                    const uint8_t* b = host_out + static_cast<size_t>(i) * per;
                    for (int c = 0; c < 256; ++c) {
                        int y = ir.height - 1;
                        while (y > 0 && b[static_cast<size_t>(y) * 256 + c] == tm.air) --y;
                        hcol[static_cast<size_t>(c)] = static_cast<int16_t>(ir.min_y + y + 1);
                    }
                    heights[ckey(list[i].x, list[i].z)] = std::move(hcol);
                    const int s = ((list[i].z % W + W) % W) * W + ((list[i].x % W + W) % W);
                    slot[s] = coords[i];
                    born[s] = fr;
                }
                check(cudaMemcpy(d_stage, host_out, per * n, cudaMemcpyHostToDevice), "upload");
                check(cudaMemcpy(d_coords, coords.data(), sizeof(int2) * n, cudaMemcpyHostToDevice), "upload");
                k_scatter<<<static_cast<unsigned>((per * n + 255) / 256), 256>>>(w, d_stage, d_coords, n);
                k_cells<<<static_cast<unsigned>((n * 16 + 127) / 128), 128>>>(w, d_coords, n);
                check(cudaMemcpy(w.slot, slot.data(), sizeof(int2) * W * W, cudaMemcpyHostToDevice), "upload");
                check(cudaMemcpy(w.born, born.data(), sizeof(int) * W * W, cudaMemcpyHostToDevice), "upload");
                check(cudaDeviceSynchronize(), "upload");
                up_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - g1).count();
                total += n;
            }
            // camera
            const double gy = ground(ps.x, ps.z);
            auto resolve = [&](char m, double y) { return m == 'g' ? gy + y : y; };
            const double target = ps.mode == 'm' ? resolve(ps.ma, ps.ya) * (1 - ps.s) + resolve(ps.mb, ps.yb) * ps.s : resolve(ps.mode, ps.y);
            if (first) cam_y = target;
            else cam_y += (target - cam_y) * std::min(1.0, 6.0 / fps);
            cam_y = std::max(cam_y, gy + 1.62);   // never inside the ground
            PathSample ahead = sample_path(keys, std::min(duration, t + 0.35));
            PathSample behind = sample_path(keys, std::max(0.0, t - 0.05));
            const double vx = ahead.x - behind.x, vz = ahead.z - behind.z;
            if (vx * vx + vz * vz > 1e-6) {
                const double ty = std::atan2(vx, vz);
                double dyaw = ty - yaw;
                while (dyaw > kPi) dyaw -= 2 * kPi;
                while (dyaw < -kPi) dyaw += 2 * kPi;
                yaw = first ? ty : yaw + dyaw * std::min(1.0, 4.0 / fps);
            }
            first = false;
            const double pitch = ps.pitch * kPi / 180.0;
            Cam cam;
            cam.pos = make_float3(static_cast<float>(ps.x), static_cast<float>(cam_y), static_cast<float>(ps.z));
            cam.f = make_float3(static_cast<float>(std::sin(yaw) * std::cos(pitch)), static_cast<float>(std::sin(pitch)), static_cast<float>(std::cos(yaw) * std::cos(pitch)));
            float3 r = make_float3(-cam.f.z, 0.0f, cam.f.x);   // cross(f, up): +x is to the left when facing +z, as in Minecraft
            const float rn = std::sqrt(r.x * r.x + r.z * r.z);
            r = make_float3(r.x / rn, 0.0f, r.z / rn);
            cam.r = r;
            cam.u = make_float3(r.y * cam.f.z - r.z * cam.f.y, r.z * cam.f.x - r.x * cam.f.z, r.x * cam.f.y - r.y * cam.f.x);
            cam.tan_half = std::tan(70.0 * kPi / 360.0);   // Minecraft's default 70 degree vertical field of view
            cam.aspect = static_cast<float>(width) / height;
            cudaEventRecord(r0);
            dim3 blk(16, 8), grd((width + 15) / 16, (height + 7) / 8);
            k_render<<<grd, blk>>>(w, cam, L, width, height, fr, static_cast<float>(radius * 16), static_cast<float>(radius * 16), d_frame);
            cudaEventRecord(r1);
            check(cudaMemcpy(frame.data(), d_frame, frame.size(), cudaMemcpyDeviceToHost), "download frame");
            float render_ms = 0;
            cudaEventElapsedTime(&render_ms, r0, r1);
            char head[256];
            std::memset(head, ' ', sizeof head);
            int len = std::snprintf(head, sizeof head, "frame=%d t=%.4f new=%d gen_ms=%.3f upload_ms=%.3f render_ms=%.3f total=%lld x=%.1f y=%.1f z=%.1f mode=%c",
                                    fr, t, n, gen_ms, up_ms, render_ms, total, ps.x, cam_y, ps.z, ps.mode);
            head[len] = ' ';
            head[255] = '\n';
            std::fwrite(head, 1, sizeof head, stdout);
            std::fwrite(frame.data(), 1, frame.size(), stdout);
        }
        std::fflush(stdout);
        std::fprintf(stderr, "mccuda_fly: done, %lld chunks generated\n", total);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
