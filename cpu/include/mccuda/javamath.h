// Java (JLS / java.lang.Math / Mojang Mth) semantics for the operations the
// terrain pipeline uses. Header-only and usable from CUDA device code.
//
// Every function here must produce bit-identical results to the Java code it
// mirrors. That relies on: IEEE-754 binary32/binary64 with round-to-nearest,
// no FMA contraction (MSVC /fp:precise, nvcc -fmad=false), no flush-to-zero,
// and correctly rounded division and sqrt.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__)
#define MCC_HD __host__ __device__
#define MCC_FI __forceinline__
#else
#define MCC_HD
#if defined(_MSC_VER)
#define MCC_FI __forceinline
#else
#define MCC_FI inline __attribute__((always_inline))
#endif
#endif

namespace mcc {

MCC_HD MCC_FI uint32_t f2bits(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u; std::memcpy(&u, &f, 4); return u;
#endif
}

MCC_HD MCC_FI float bits2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f; std::memcpy(&f, &u, 4); return f;
#endif
}

MCC_HD MCC_FI uint64_t d2bits(double d) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint64_t>(__double_as_longlong(d));
#else
    uint64_t u; std::memcpy(&u, &d, 8); return u;
#endif
}

MCC_HD MCC_FI double bits2d(uint64_t u) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(u));
#else
    double d; std::memcpy(&d, &u, 8); return d;
#endif
}

// JLS 5.1.3 narrowing double -> int: NaN -> 0, saturate, otherwise truncate.
MCC_HD MCC_FI int32_t java_d2i(double v) {
    if (v != v) return 0;
    if (v >= 2147483647.0) return 2147483647;
    if (v <= -2147483648.0) return -2147483647 - 1;
    return static_cast<int32_t>(v);
}

// Mth.floor(double) = (int) Math.floor(v)
MCC_HD MCC_FI int32_t mth_floor(double v) { return java_d2i(std::floor(v)); }

// Java int arithmetic wraps; do it in unsigned to avoid C++ UB.
MCC_HD MCC_FI int32_t jadd(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

MCC_HD MCC_FI int32_t jmul(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

// Math.floorDiv / floorMod for int (b > 0 in all our uses, but general).
MCC_HD MCC_FI int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

MCC_HD MCC_FI int32_t floor_mod(int32_t a, int32_t b) { return a - floor_div(a, b) * b; }

// java.lang.Math.min(float, float) / max(float, float), including NaN and signed zero.
MCC_HD MCC_FI float java_fmin(float a, float b) {
    if (a != a) return a;
    if (a == 0.0f && b == 0.0f && f2bits(b) == 0x80000000u) return b;
    return (a <= b) ? a : b;
}

MCC_HD MCC_FI float java_fmax(float a, float b) {
    if (a != a) return a;
    if (a == 0.0f && b == 0.0f && f2bits(a) == 0x80000000u) return b;
    return (a >= b) ? a : b;
}

MCC_HD MCC_FI double java_dmin(double a, double b) {
    if (a != a) return a;
    if (a == 0.0 && b == 0.0 && d2bits(b) == 0x8000000000000000ull) return b;
    return (a <= b) ? a : b;
}

MCC_HD MCC_FI double java_dmax(double a, double b) {
    if (a != a) return a;
    if (a == 0.0 && b == 0.0 && d2bits(a) == 0x8000000000000000ull) return b;
    return (a >= b) ? a : b;
}

// Mth.clamp(double, double, double)
MCC_HD MCC_FI double mth_dclamp(double v, double lo, double hi) { return v < lo ? lo : java_dmin(v, hi); }

// Math.round(float): floor(a + 1/2) computed exactly, then (int) with saturation.
MCC_HD MCC_FI int32_t java_round_f(float a) { return java_d2i(std::floor(static_cast<double>(a) + 0.5)); }

MCC_HD MCC_FI int32_t java_imin(int32_t a, int32_t b) { return a <= b ? a : b; }
MCC_HD MCC_FI int32_t java_imax(int32_t a, int32_t b) { return a >= b ? a : b; }

// java.lang.Math.abs(float): clears the sign bit (also for NaN and -0.0).
MCC_HD MCC_FI float java_fabs(float a) { return bits2f(f2bits(a) & 0x7FFFFFFFu); }

// Mth.clamp(float, float, float) = value < min ? min : Math.min(value, max)
MCC_HD MCC_FI float mth_clamp(float v, float lo, float hi) { return v < lo ? lo : java_fmin(v, hi); }

// Mth.clamp(int, int, int) = Math.min(Math.max(value, min), max)
MCC_HD MCC_FI int32_t mth_iclamp(int32_t v, int32_t lo, int32_t hi) { return java_imin(java_imax(v, lo), hi); }

// Mth.lerp(float alpha, float p0, float p1) = p0 + alpha * (p1 - p0)
MCC_HD MCC_FI float mth_lerp(float a, float p0, float p1) {
    float d = p1 - p0;
    float m = a * d;
    return p0 + m;
}

// Mth.lerp2(a1, a2, x00, x10, x01, x11) = lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11))
MCC_HD MCC_FI float mth_lerp2(float a1, float a2, float x00, float x10, float x01, float x11) {
    float l0 = mth_lerp(a1, x00, x10);
    float l1 = mth_lerp(a1, x01, x11);
    return mth_lerp(a2, l0, l1);
}

// Mth.lerp3(a1, a2, a3, x000, x100, x010, x110, x001, x101, x011, x111)
MCC_HD MCC_FI float mth_lerp3(float a1, float a2, float a3, float x000, float x100, float x010, float x110,
                              float x001, float x101, float x011, float x111) {
    float l0 = mth_lerp2(a1, a2, x000, x100, x010, x110);
    float l1 = mth_lerp2(a1, a2, x001, x101, x011, x111);
    return mth_lerp(a3, l0, l1);
}

// Mth.smoothstep(float x) = x * x * x * (x * (x * 6.0F - 15.0F) + 10.0F)
MCC_HD MCC_FI float mth_smoothstep(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    float t = x * 6.0f;
    t = t - 15.0f;
    t = x * t;
    t = t + 10.0f;
    return x3 * t;
}

// Mth.cube(float) = x * x * x ; Mth.square(float) = x * x
MCC_HD MCC_FI float mth_cube(float x) {
    float x2 = x * x;
    return x2 * x;
}

MCC_HD MCC_FI float mth_square(float x) { return x * x; }

}  // namespace mcc
