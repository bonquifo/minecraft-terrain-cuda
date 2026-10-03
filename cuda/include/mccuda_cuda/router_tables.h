// Seed-dependent tables consumed by the generated evaluators (host or device).
#pragma once

#include "mccuda/noise_eval.h"

namespace mcc {

struct StackRef2 {
    int first;
    int count;
};

struct BlendedRef {
    StackRef2 min_limit, max_limit, main;
};

constexpr int kMaxStacks = 64;
constexpr int kMaxBlended = 4;
constexpr int kMaxAxisSites = 64;

// A volume-path noise call site whose coordinate setup is tabulated per axis
// (see AxisRec): which layer stack it samples and at which scales.
struct AxisSite {
    int kind;        // 0: stacks[index]; 1, 2, 3: blended[index].main, .min_limit, .max_limit
    int index;
    double xz, y;    // xzScale, yScale passed to NoiseStack.addToVolume
    int smeared;     // SmearedPerlinNoise layers (BlendedNoise)
};

struct RouterTables {
    const PerlinOctave* octaves;   // device or host pointer
    const StackLayer* layers;
    StackRef2 stacks[kMaxStacks];  // indexed like gen::kNoiseKeys
    BlendedRef blended[kMaxBlended];
    int axis_off[kMaxAxisSites];   // first layer record of each axis site in the AxisRec tables
};

}  // namespace mcc
