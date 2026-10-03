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

struct RouterTables {
    const PerlinOctave* octaves;   // device or host pointer
    const StackLayer* layers;
    StackRef2 stacks[kMaxStacks];  // indexed like gen::kNoiseKeys
    BlendedRef blended[kMaxBlended];
};

}  // namespace mcc
