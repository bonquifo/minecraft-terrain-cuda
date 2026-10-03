// Element-wise semantics of vanilla 26.3 density samplers on the *volume* path
// (DensitySampler.sampleVolume), which is the path chunk generation uses.
// Shared by the CPU interpreter and the generated CUDA code.
#pragma once

#include "mccuda/javamath.h"

namespace mcc {

// ---- Binary ops -----------------------------------------------------------
// MIN/MAX on the volume path are "keep the buffer value unless the other value
// is strictly smaller/larger". `first` is the operand that initialises the
// buffer, which depends on how vanilla compiled the node (see binary_plan()).
MCC_HD MCC_FI float vol_min(float first, float second) { return second < first ? second : first; }
MCC_HD MCC_FI float vol_max(float first, float second) { return second > first ? second : first; }

// ---- Unary ops ------------------------------------------------------------
MCC_HD MCC_FI float un_abs(float x) { return java_fabs(x); }
MCC_HD MCC_FI float un_square(float x) { return mth_square(x); }
MCC_HD MCC_FI float un_cube(float x) { return mth_cube(x); }
MCC_HD MCC_FI float un_sqrt(float x) { return static_cast<float>(std::sqrt(static_cast<double>(x))); }   // Mth.sqrt
MCC_HD MCC_FI float un_leaky(float x, float negative_factor) { return x > 0.0f ? x : x * negative_factor; }
MCC_HD MCC_FI float un_reciprocal(float x) { return 1.0f / x; }
MCC_HD MCC_FI float un_negate(float x) { return -x; }
MCC_HD MCC_FI float un_squeeze(float x) {
    float c = mth_clamp(x, -1.0f, 1.0f);
    float a = c / 2.0f;
    float b = mth_cube(c) / 24.0f;
    return a - b;
}
MCC_HD MCC_FI float un_sign(float x) {                          // Math.signum(float)
    if (x != x || x == 0.0f) return x;
    return x > 0.0f ? 1.0f : -1.0f;
}

// ---- Lerp / choice ops ------------------------------------------------------
MCC_HD MCC_FI bool in_range(float v, float min_inclusive, float max_exclusive) { return v >= min_inclusive && v < max_exclusive; }

// ---- Gradient (GradientFunction samplers) ------------------------------------
struct GradientPlan {
    int mode;            // 0 clamp_to_edge, 1 repeat, 2 mirrored_repeat
    int from;
    int min_c, max_c;    // clamp bounds
    int range;           // to - from
    float from_value;
    float factor;        // (toValue - fromValue) / coordinateRange
};

MCC_HD MCC_FI float gradient_compute(const GradientPlan& g, int32_t c) {
    if (g.mode == 0) {
        int32_t rel = mth_iclamp(c, g.min_c, g.max_c) - g.from;
        return g.from_value + static_cast<float>(rel) * g.factor;
    }
    int32_t rel = jadd(c, -g.from);
    if (g.mode == 1) return g.from_value + static_cast<float>(floor_mod(rel, g.range)) * g.factor;
    int32_t tile = floor_div(rel, g.range);
    int32_t local = rel - tile * g.range;
    return (tile & 1) == 0 ? g.from_value + static_cast<float>(local) * g.factor
                           : g.from_value + static_cast<float>(g.range - local) * g.factor;
}

// ---- InterpolatedFunction.fillCell for one block ------------------------------
// v_abc: corner values with a = x, b = y, c = z offsets. xi/zi: block offset in
// the cell (0..cellXz-1); yi: offset in the cell along y; y0: first y offset the
// output volume covers in this cell (0 for cell-aligned volumes).
MCC_HD MCC_FI float interp_fill(float v000, float v100, float v010, float v110, float v001, float v101, float v011, float v111,
                                int xi, int yi, int zi, int y0, float cell_xz_inv, float cell_y_inv) {
    float alpha_z = static_cast<float>(zi) * cell_xz_inv;
    float v00 = mth_lerp(alpha_z, v000, v001);
    float v01 = mth_lerp(alpha_z, v010, v011);
    float v10 = mth_lerp(alpha_z, v100, v101);
    float v11 = mth_lerp(alpha_z, v110, v111);
    float alpha_x = static_cast<float>(xi) * cell_xz_inv;
    float v_0 = mth_lerp(alpha_x, v00, v10);
    float v_1 = mth_lerp(alpha_x, v01, v11);
    float step = (v_1 - v_0) * cell_y_inv;
    float value = v_0 + step * static_cast<float>(y0);
    for (int y = y0; y < yi; ++y) value += step;
    return value;
}

}  // namespace mcc
