// Per-seed noise state, built the way vanilla's RandomState builds it:
//   RandomState.random     = XoroshiroRandomSource(seed).forkPositional()
//   noise K                = NormalNoise(params(K)).create(random.fromHashOf(K))
//   BlendedNoise fbm set   = createFbmSet(random.fromHashOf("minecraft:terrain"))
// All octaves live in one flat table so the same data can be uploaded to the GPU.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "mccuda/ir.h"
#include "mccuda/noise_eval.h"

namespace mcc {

struct StackRef {
    int first_layer = 0;
    int count = 0;
    bool smeared = false;
};

struct BlendedSet {
    StackRef min_limit, max_limit, main;
};

struct SeedTables {
    int64_t seed = 0;
    std::vector<PerlinOctave> octaves;
    std::vector<StackLayer> layers;
    std::map<std::string, StackRef> noise;           // by noise key
    std::map<int, BlendedSet> blended;               // by IR node id
    std::map<std::string, double> normalization;     // per noise key (diagnostics)

    const StackLayer* layer_ptr(const StackRef& s) const { return layers.data() + s.first_layer; }
};

SeedTables build_seed_tables(const IR& ir, int64_t seed);

// Exposed for unit tests: java.util.stream.DoubleStream.sum() (compensated).
double java_double_stream_sum(const std::vector<double>& values);

}  // namespace mcc
