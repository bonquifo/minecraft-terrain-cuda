// Device-side (and host-callable) helpers for the GPU TERRAIN pipeline: Xoroshiro
// positional random, BiomeManager zoom, legacy simplex noise for biome temperature.
#pragma once

#include <cstdint>

#include "mccuda/javamath.h"
#include "mccuda/noise_eval.h"

namespace mcc::dev {

struct Xoro {
    uint64_t lo, hi;
};

MCC_HD MCC_FI uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

MCC_HD MCC_FI Xoro xoro_seed(uint64_t lo, uint64_t hi) {
    if ((lo | hi) == 0) { lo = 0x9E3779B97F4A7C15ull; hi = 0x6A09E667F3BCC909ull; }
    return {lo, hi};
}

MCC_HD MCC_FI uint64_t xoro_next(Xoro& s) {
    uint64_t s0 = s.lo, s1 = s.hi;
    uint64_t r = rotl64(s0 + s1, 17) + s0;
    s1 ^= s0;
    s.lo = rotl64(s0, 49) ^ s1 ^ (s1 << 21);
    s.hi = rotl64(s1, 28);
    return r;
}

MCC_HD MCC_FI int32_t xoro_next_int(Xoro& s, int32_t bound) {
    uint64_t bits = static_cast<uint32_t>(static_cast<int32_t>(xoro_next(s)));
    uint64_t m = bits * static_cast<uint64_t>(bound);
    uint64_t frac = m & 0xFFFFFFFFull;
    if (frac < static_cast<uint64_t>(bound)) {
        uint32_t ub = static_cast<uint32_t>(bound);
        uint32_t threshold = (~ub + 1u) % ub;
        while (frac < threshold) {
            bits = static_cast<uint32_t>(static_cast<int32_t>(xoro_next(s)));
            m = bits * static_cast<uint64_t>(bound);
            frac = m & 0xFFFFFFFFull;
        }
    }
    return static_cast<int32_t>(static_cast<int64_t>(m) >> 32);
}

MCC_HD MCC_FI float xoro_next_float(Xoro& s) {
    return static_cast<float>(static_cast<int64_t>(xoro_next(s) >> 40)) * 5.9604645E-8f;
}

MCC_HD MCC_FI double xoro_next_double(Xoro& s) {
    return static_cast<double>(static_cast<int64_t>(xoro_next(s) >> 11)) * static_cast<double>(1.110223E-16f);
}

// Mth.getSeed(x, y, z)
MCC_HD MCC_FI int64_t get_seed(int32_t x, int32_t y, int32_t z) {
    int32_t xm = static_cast<int32_t>(static_cast<uint32_t>(x) * 3129871u);
    uint64_t seed = static_cast<uint64_t>(static_cast<int64_t>(xm)) ^ (static_cast<uint64_t>(static_cast<int64_t>(z)) * 116129781ull) ^
                    static_cast<uint64_t>(static_cast<int64_t>(y));
    seed = seed * seed * 42317861ull + seed * 11ull;
    return static_cast<int64_t>(seed) >> 16;
}

struct Positional {
    uint64_t lo, hi;
    MCC_HD MCC_FI Xoro at(int32_t x, int32_t y, int32_t z) const { return xoro_seed(static_cast<uint64_t>(get_seed(x, y, z)) ^ lo, hi); }
};

// ---- BiomeManager zoom ------------------------------------------------------------
MCC_HD MCC_FI int64_t lcg_next(int64_t rval, int64_t c) {
    uint64_t r = static_cast<uint64_t>(rval);
    r = r * (r * 6364136223846793005ull + 1442695040888963407ull);
    return static_cast<int64_t>(r + static_cast<uint64_t>(c));
}

MCC_HD MCC_FI double fiddle(int64_t rval) {
    int64_t m = (rval >> 24) % 1024;
    if (m < 0) m += 1024;
    return (static_cast<double>(m) / 1024.0 - 0.5) * 0.9;
}

MCC_HD MCC_FI double fiddled_distance(int64_t seed, int xr, int yr, int zr, double dx, double dy, double dz) {
    int64_t r = seed;
    r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
    r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
    double fx = fiddle(r);
    r = lcg_next(r, seed);
    double fy = fiddle(r);
    r = lcg_next(r, seed);
    double fz = fiddle(r);
    double sz = (dz + fz) * (dz + fz);
    double sy = (dy + fy) * (dy + fy);
    double sx = (dx + fx) * (dx + fx);
    return sz + sy + sx;
}

// BiomeManager.getFiddle in two steps: the 10-bit index, and its value.
MCC_HD MCC_FI int fiddle_index(int64_t rval) {
    int64_t m = (rval >> 24) % 1024;
    if (m < 0) m += 1024;
    return static_cast<int>(m);
}

MCC_HD MCC_FI double fiddle_value(int m) { return (static_cast<double>(m) / 1024.0 - 0.5) * 0.9; }

// Squared fiddled offset of one axis, (d + fiddle)^2, for the 8 possible offsets d
// (fraction (a & 3) / 4, minus 1 on the far side) and the 1024 fiddle indices:
// table index ((a & 3) + 4 * far) * 1024 + m. Same expression as fiddled_distance.
MCC_HD MCC_FI double zoom_square(int t) {
    int k = t >> 10, m = t & 1023;
    double f = static_cast<double>(k & 3) / 4.0;
    double d = k < 4 ? f : f - 1.0;
    double v = fiddle_value(m);
    return (d + v) * (d + v);
}

// zoom_quart with the squares read from a zoom_square table (8192 doubles). Distances
// are sums of non-negative doubles, so their bit patterns order like the values and
// the strict minimum is found with integer compares; the FP64 work left is the two
// additions per corner, in vanilla's order (sz + sy) + sx.
MCC_HD MCC_FI void zoom_quart_tab(int64_t seed, const double* sq, int x, int y, int z, int& bx, int& by, int& bz) {
    int ax = x - 2, ay = y - 2, az = z - 2;
    int px = ax >> 2, py = ay >> 2, pz = az >> 2;
    int tx = (ax & 3) << 10, ty = (ay & 3) << 10, tz = (az & 3) << 10;
    int min_i = 0;
    uint64_t min_bits = 0x7FF0000000000000ull;   // +infinity
    for (int i = 0; i < 8; ++i) {
        bool xe = (i & 4) == 0, ye = (i & 2) == 0, ze = (i & 1) == 0;
        int xr = xe ? px : px + 1, yr = ye ? py : py + 1, zr = ze ? pz : pz + 1;
        int64_t r = seed;
        r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
        r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
        int mx = fiddle_index(r);
        r = lcg_next(r, seed);
        int my = fiddle_index(r);
        r = lcg_next(r, seed);
        int mz = fiddle_index(r);
        double sx = sq[tx + (xe ? 0 : 4096) + mx];
        double sy = sq[ty + (ye ? 0 : 4096) + my];
        double sz = sq[tz + (ze ? 0 : 4096) + mz];
        uint64_t d = d2bits(sz + sy + sx);
        if (min_bits > d) { min_i = i; min_bits = d; }
    }
    bx = (min_i & 4) == 0 ? px : px + 1;
    by = (min_i & 2) == 0 ? py : py + 1;
    bz = (min_i & 1) == 0 ? pz : pz + 1;
}

// Returns the quart (bx, by, bz) BiomeManager.getBiome picks for block (x, y, z).
MCC_HD MCC_FI void zoom_quart(int64_t seed, int x, int y, int z, int& bx, int& by, int& bz) {
    int ax = x - 2, ay = y - 2, az = z - 2;
    int px = ax >> 2, py = ay >> 2, pz = az >> 2;
    double fx = static_cast<double>(ax & 3) / 4.0, fy = static_cast<double>(ay & 3) / 4.0, fz = static_cast<double>(az & 3) / 4.0;
    int min_i = 0;
    double min_d = bits2d(0x7FF0000000000000ull);   // Double.POSITIVE_INFINITY
    for (int i = 0; i < 8; ++i) {
        bool xe = (i & 4) == 0, ye = (i & 2) == 0, ze = (i & 1) == 0;
        double d = fiddled_distance(seed, xe ? px : px + 1, ye ? py : py + 1, ze ? pz : pz + 1, xe ? fx : fx - 1.0, ye ? fy : fy - 1.0, ze ? fz : fz - 1.0);
        if (min_d > d) { min_i = i; min_d = d; }
    }
    bx = (min_i & 4) == 0 ? px : px + 1;
    by = (min_i & 2) == 0 ? py : py + 1;
    bz = (min_i & 1) == 0 ? pz : pz + 1;
}

// ---- Legacy 2D simplex noise (biome temperature) -------------------------------------
struct Simplex2D {
    uint8_t perm[256];
};

MCC_HD MCC_FI double simplex_corner(int index, double x, double y, double base) {
    double t0 = base - x * x - y * y - 0.0 * 0.0;
    if (t0 < 0.0) return 0.0;
    t0 *= t0;
    double dot = static_cast<double>(grad_c(index, 0)) * x + static_cast<double>(grad_c(index, 1)) * y;
    dot = dot + static_cast<double>(grad_c(index, 2)) * 0.0;
    return t0 * t0 * dot;
}

MCC_HD MCC_FI float simplex_get(const Simplex2D& s, double xin, double yin) {
    const double sqrt3 = 1.7320508075688772;   // Math.sqrt(3.0)
    const double F2 = 0.5 * (sqrt3 - 1.0);
    const double G2 = (3.0 - sqrt3) / 6.0;
    xin = xin + 0.0;                            // discardNoiseOffset: offsets are 0.0
    yin = yin + 0.0;
    double sk = (xin + yin) * F2;
    int32_t i = mth_floor(xin + sk), j = mth_floor(yin + sk);
    double t = static_cast<double>(jadd(i, j)) * G2;
    double x0 = xin - (static_cast<double>(i) - t);
    double y0 = yin - (static_cast<double>(j) - t);
    int i1 = x0 > y0 ? 1 : 0, j1 = x0 > y0 ? 0 : 1;
    double x1 = x0 - static_cast<double>(i1) + G2;
    double y1 = y0 - static_cast<double>(j1) + G2;
    double x2 = x0 - 1.0 + 2.0 * G2;
    double y2 = y0 - 1.0 + 2.0 * G2;
    int ii = i & 0xFF, jj = j & 0xFF;
    int gi0 = s.perm[(ii + s.perm[jj & 0xFF]) & 0xFF] % 12;
    int gi1 = s.perm[(ii + i1 + s.perm[(jj + j1) & 0xFF]) & 0xFF] % 12;
    int gi2 = s.perm[(ii + 1 + s.perm[(jj + 1) & 0xFF]) & 0xFF] % 12;
    double n0 = simplex_corner(gi0, x0, y0, 0.5);
    double n1 = simplex_corner(gi1, x1, y1, 0.5);
    double n2 = simplex_corner(gi2, x2, y2, 0.5);
    return static_cast<float>(70.0 * (n0 + n1 + n2));
}

}  // namespace mcc::dev
