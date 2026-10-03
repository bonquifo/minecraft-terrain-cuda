// Per-sample evaluation of vanilla 26.3's gradient noise, mirroring the exact
// arithmetic of each vanilla code path:
//
//   perlin_get           PerlinNoise.get(x, y, z)          (point path; used by
//                         NoiseStack.get, i.e. shifted noises)
//   perlin_volume        PerlinNoise.addToVolume, one sample (unshifted noises)
//   smeared_volume       SmearedPerlinNoise.addToVolume, one sample (BlendedNoise)
//   stack_get            NoiseStack.get
//   stack_add_volume     NoiseStack.addToVolume, one sample
//
// The two Perlin paths associate the gradient dot product differently
// ((gx*x + gy*y) + gz*z  vs  (gx*x + gz*z) + gy*y) and compute coordinates
// differently, so they are kept separate on purpose.
#pragma once

#include "javamath.h"

namespace mcc {

// One octave of gradient noise (GradientNoise state).
struct PerlinOctave {
    double ox, oy, oz;    // random offsets in [0, 256)
    double smear;         // SmearedPerlinNoise.fudgeYScale (0 for plain Perlin)
    uint8_t perm[256];
};

// One NoiseStack layer.
struct StackLayer {
    double frequency;
    float amplitude;
    int32_t octave;       // index into the octave table
};

// GradientNoise.GRADIENT: 12 cube-edge gradients plus 4 repeats. Reference table
// for host code and tests; evaluation uses the register-encoded grad_c below.
static const int8_t kGradTable[16][3] = {
    {1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
    {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}, {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1},
};

// Gradient lookup without a memory access: component c of GRADIENT[h & 15].
MCC_HD MCC_FI int grad_c(int h, int c) {
    // Encoded as 2-bit fields (value + 1), gradient i at bits [2i, 2i+1], per component.
    // gx: 1,-1,1,-1,1,-1,1,-1,0,0,0,0,1,0,-1,0
    // gy: 1,1,-1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1
    // gz: 0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,1,0,-1
    // (checked against kGradTable by the unit tests)
    const uint32_t gx = 0x46552222u, gy = 0x2222550Au, gz = 0x190A0A55u;
    uint32_t tbl = c == 0 ? gx : (c == 1 ? gy : gz);
    return static_cast<int>((tbl >> ((h & 15) * 2)) & 3u) - 1;
}

constexpr double kHalfRoundOff = 0x1.fffffffffffffp+23;   // Math.nextDown(1.6777216E7) = 2^24 - 2^-29

// GradientNoise.wrap
MCC_HD MCC_FI double gn_wrap(double x) {
    if (x >= -kHalfRoundOff && x < kHalfRoundOff) return x;
    double q = x / 3.3554432E7;
    q = q + 0.5;
    q = std::floor(q);
    q = q * 3.3554432E7;
    return x - q;
}

MCC_HD MCC_FI int perm_at(const PerlinOctave& o, int32_t x) { return o.perm[x & 0xFF]; }

// GRADIENT[hash & 15].dot(x, y, z) with float arguments: gx * x + gy * y + gz * z
MCC_HD MCC_FI float grad_dot3(int hash, float x, float y, float z) {
    float a = static_cast<float>(grad_c(hash, 0)) * x;
    float b = static_cast<float>(grad_c(hash, 1)) * y;
    float c = static_cast<float>(grad_c(hash, 2)) * z;
    float s = a + b;
    return s + c;
}

// Gradient.dotXz(x, z) = this.x * x + this.z * z
MCC_HD MCC_FI float grad_dot_xz(int hash, float x, float z) {
    float a = static_cast<float>(grad_c(hash, 0)) * x;
    float c = static_cast<float>(grad_c(hash, 2)) * z;
    return a + c;
}

// PerlinNoise.sampleAndLerp
MCC_HD MCC_FI float perlin_sample_and_lerp(const PerlinOctave& o, int32_t x, int32_t y, int32_t z,
                                           float rx, float ry, float rz, float ory) {
    int x0 = perm_at(o, x);
    int x1 = perm_at(o, jadd(x, 1));
    int xy00 = perm_at(o, jadd(x0, y));
    int xy01 = perm_at(o, jadd(jadd(x0, y), 1));
    int xy10 = perm_at(o, jadd(x1, y));
    int xy11 = perm_at(o, jadd(jadd(x1, y), 1));
    float rx1 = rx - 1.0f, ry1 = ry - 1.0f, rz1 = rz - 1.0f;
    float d000 = grad_dot3(perm_at(o, jadd(xy00, z)), rx, ry, rz);
    float d100 = grad_dot3(perm_at(o, jadd(xy10, z)), rx1, ry, rz);
    float d010 = grad_dot3(perm_at(o, jadd(xy01, z)), rx, ry1, rz);
    float d110 = grad_dot3(perm_at(o, jadd(xy11, z)), rx1, ry1, rz);
    float d001 = grad_dot3(perm_at(o, jadd(jadd(xy00, z), 1)), rx, ry, rz1);
    float d101 = grad_dot3(perm_at(o, jadd(jadd(xy10, z), 1)), rx1, ry, rz1);
    float d011 = grad_dot3(perm_at(o, jadd(jadd(xy01, z), 1)), rx, ry1, rz1);
    float d111 = grad_dot3(perm_at(o, jadd(jadd(xy11, z), 1)), rx1, ry1, rz1);
    float ax = mth_smoothstep(rx);
    float ay = mth_smoothstep(ory);
    float az = mth_smoothstep(rz);
    return mth_lerp3(ax, ay, az, d000, d100, d010, d110, d001, d101, d011, d111);
}

// PerlinNoise.get(double x, double y, double z)
MCC_HD MCC_FI float perlin_get(const PerlinOctave& o, double px, double py, double pz) {
    double x = gn_wrap(px) + o.ox;
    double y = gn_wrap(py) + o.oy;
    double z = gn_wrap(pz) + o.oz;
    int32_t fx = mth_floor(x), fy = mth_floor(y), fz = mth_floor(z);
    float rx = static_cast<float>(x - static_cast<double>(fx));
    float ry = static_cast<float>(y - static_cast<double>(fy));
    float rz = static_cast<float>(z - static_cast<double>(fz));
    return perlin_sample_and_lerp(o, fx, fy, fz, rx, ry, rz, ry);
}

// The 8 lattice gradients used by addToVolume for one (floorX, floorY, floorZ).
struct VolumeCorners {
    float d000xz, d100xz, d010xz, d110xz, d001xz, d101xz, d011xz, d111xz;
    float g000y, g100y, g010y, g110y, g001y, g101y, g011y, g111y;
};

MCC_HD MCC_FI VolumeCorners volume_corners(const PerlinOctave& o, int32_t fx, int32_t fy, int32_t fz, float rx, float rz) {
    VolumeCorners c;
    int x0 = perm_at(o, fx);
    int x1 = perm_at(o, jadd(fx, 1));
    int xy00 = perm_at(o, jadd(x0, fy));
    int xy01 = perm_at(o, jadd(jadd(x0, fy), 1));
    int xy10 = perm_at(o, jadd(x1, fy));
    int xy11 = perm_at(o, jadd(jadd(x1, fy), 1));
    float rx1 = rx - 1.0f, rz1 = rz - 1.0f;
    int h;
    h = perm_at(o, jadd(xy00, fz));           c.d000xz = grad_dot_xz(h, rx, rz);   c.g000y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(xy10, fz));           c.d100xz = grad_dot_xz(h, rx1, rz);  c.g100y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(xy01, fz));           c.d010xz = grad_dot_xz(h, rx, rz);   c.g010y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(xy11, fz));           c.d110xz = grad_dot_xz(h, rx1, rz);  c.g110y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(jadd(xy00, fz), 1));  c.d001xz = grad_dot_xz(h, rx, rz1);  c.g001y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(jadd(xy10, fz), 1));  c.d101xz = grad_dot_xz(h, rx1, rz1); c.g101y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(jadd(xy01, fz), 1));  c.d011xz = grad_dot_xz(h, rx, rz1);  c.g011y = static_cast<float>(grad_c(h, 1));
    h = perm_at(o, jadd(jadd(xy11, fz), 1));  c.d111xz = grad_dot_xz(h, rx1, rz1); c.g111y = static_cast<float>(grad_c(h, 1));
    return c;
}

MCC_HD MCC_FI float volume_lerp(const VolumeCorners& c, float ax, float ay, float az, float ry) {
    float ry1 = ry - 1.0f;
    return mth_lerp3(ax, ay, az,
                     c.d000xz + c.g000y * ry, c.d100xz + c.g100y * ry,
                     c.d010xz + c.g010y * ry1, c.d110xz + c.g110y * ry1,
                     c.d001xz + c.g001y * ry, c.d101xz + c.g101y * ry,
                     c.d011xz + c.g011y * ry1, c.d111xz + c.g111y * ry1);
}

// PerlinNoise.addToVolume: the value added for one sample (before the amplitude multiply).
MCC_HD MCC_FI float perlin_volume(const PerlinOctave& o, int32_t bx, int32_t by, int32_t bz, double xzScale, double yScale) {
    double z = gn_wrap(static_cast<double>(bz) * xzScale) + o.oz;
    int32_t fz = mth_floor(z);
    float rz = static_cast<float>(z - static_cast<double>(fz));
    float az = mth_smoothstep(rz);
    double x = gn_wrap(static_cast<double>(bx) * xzScale) + o.ox;
    int32_t fx = mth_floor(x);
    float rx = static_cast<float>(x - static_cast<double>(fx));
    float ax = mth_smoothstep(rx);
    double y = gn_wrap(static_cast<double>(by) * yScale) + o.oy;
    int32_t fy = mth_floor(y);
    float ry = static_cast<float>(y - static_cast<double>(fy));
    float ay = mth_smoothstep(ry);
    VolumeCorners c = volume_corners(o, fx, fy, fz, rx, rz);
    return volume_lerp(c, ax, ay, az, ry);
}

// SmearedPerlinNoise.computeFudgeY
MCC_HD MCC_FI double smeared_fudge_y(double originalY, double relativeY, double fudgeYScale) {
    double limit = (originalY >= 0.0 && originalY < relativeY) ? originalY : relativeY;
    double q = limit / fudgeYScale;
    q = q + static_cast<double>(1.0E-7f);
    return static_cast<double>(mth_floor(q)) * fudgeYScale;
}

// SmearedPerlinNoise.get(x, y, z) (point path; not used by chunk generation,
// which only uses addToVolume, but part of the L0 reference tests).
MCC_HD MCC_FI float smeared_get(const PerlinOctave& o, double px, double py, double pz) {
    double x = gn_wrap(px) + o.ox;
    double y = gn_wrap(py) + o.oy;
    double z = gn_wrap(pz) + o.oz;
    int32_t fx = mth_floor(x), fy = mth_floor(y), fz = mth_floor(z);
    float rx = static_cast<float>(x - static_cast<double>(fx));
    double relativeY = y - static_cast<double>(fy);
    float rz = static_cast<float>(z - static_cast<double>(fz));
    float fudged = static_cast<float>(relativeY - smeared_fudge_y(py, relativeY, o.smear));
    return perlin_sample_and_lerp(o, fx, fy, fz, rx, fudged, rz, static_cast<float>(relativeY));
}

// SmearedPerlinNoise.addToVolume: the value added for one sample.
MCC_HD MCC_FI float smeared_volume(const PerlinOctave& o, int32_t bx, int32_t by, int32_t bz, double xzScale, double yScale) {
    double z = gn_wrap(static_cast<double>(bz) * xzScale) + o.oz;
    int32_t fz = mth_floor(z);
    float rz = static_cast<float>(z - static_cast<double>(fz));
    float az = mth_smoothstep(rz);
    double x = gn_wrap(static_cast<double>(bx) * xzScale) + o.ox;
    int32_t fx = mth_floor(x);
    float rx = static_cast<float>(x - static_cast<double>(fx));
    float ax = mth_smoothstep(rx);
    double originalY = static_cast<double>(by) * yScale;
    double y = gn_wrap(originalY) + o.oy;
    int32_t fy = mth_floor(y);
    double relativeY = y - static_cast<double>(fy);
    float ay = mth_smoothstep(static_cast<float>(relativeY));
    VolumeCorners c = volume_corners(o, fx, fy, fz, rx, rz);
    float fudged = static_cast<float>(relativeY - smeared_fudge_y(originalY, relativeY, o.smear));
    return volume_lerp(c, ax, ay, az, fudged);
}

// NoiseStack.get(x, y, z) over plain Perlin layers.
MCC_HD MCC_FI float stack_get(const StackLayer* layers, int count, const PerlinOctave* octaves, double x, double y, double z) {
    float value = 0.0f;
    for (int i = 0; i < count; ++i) {
        const StackLayer& L = layers[i];
        double f = L.frequency;
        float n = perlin_get(octaves[L.octave], x * f, y * f, z * f);
        value += L.amplitude * n;
    }
    return value;
}

// NoiseStack.SmearedPerlin.get (L0 tests only).
MCC_HD MCC_FI float stack_get_smeared(const StackLayer* layers, int count, const PerlinOctave* octaves, double x, double y, double z) {
    float value = 0.0f;
    for (int i = 0; i < count; ++i) {
        const StackLayer& L = layers[i];
        double f = L.frequency;
        float n = smeared_get(octaves[L.octave], x * f, y * f, z * f);
        value += L.amplitude * n;
    }
    return value;
}

// NoiseStack.addToVolume for one sample; `acc` is the buffer value before the call.
// smeared selects SmearedPerlinNoise layers (BlendedNoise fbm stacks).
MCC_HD MCC_FI float stack_add_volume(float acc, const StackLayer* layers, int count, const PerlinOctave* octaves, bool smeared,
                                     int32_t bx, int32_t by, int32_t bz, double xzScale, double yScale, float amplitude) {
    for (int i = 0; i < count; ++i) {
        const StackLayer& L = layers[i];
        double f = L.frequency;
        double xs = xzScale * f, ys = yScale * f;
        float amp = amplitude * L.amplitude;
        float v = smeared ? smeared_volume(octaves[L.octave], bx, by, bz, xs, ys)
                          : perlin_volume(octaves[L.octave], bx, by, bz, xs, ys);
        acc = acc + amp * v;
    }
    return acc;
}

}  // namespace mcc
