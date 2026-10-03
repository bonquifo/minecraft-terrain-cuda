#include "mccuda/seed_tables.h"

#include <cmath>
#include <stdexcept>

#include "mccuda/rng.h"

namespace mcc {

namespace {

struct OctaveInfo {
    int index;
    double frequency;
    double amplitude;
};

double amplitude_modifier(const NoiseDef& d, int i) {
    return d.amplitude_modifiers.empty() ? 1.0 : d.amplitude_modifiers[static_cast<size_t>(i)];
}

// NormalNoise.buildOctaves
std::vector<OctaveInfo> build_octaves(int base_octave, double base_amplitude, int octave_count, bool normalize, const NoiseDef& d) {
    double frequency = std::ldexp(1.0, base_octave);                     // Math.pow(2.0, baseOctave), exact
    double amplitude = base_amplitude;
    if (normalize) {
        // Math.pow(0.5, -(n - 1)) / (Math.pow(0.5, -n) - 1.0): exact powers of two.
        double num = std::ldexp(1.0, octave_count - 1);
        double den = std::ldexp(1.0, octave_count) - 1.0;
        amplitude *= num / den;
    }
    std::vector<OctaveInfo> out;
    for (int i = 0; i < octave_count; ++i) {
        double mod = amplitude_modifier(d, i);
        if (mod != 0.0) out.push_back({base_octave + i, frequency, amplitude * mod});
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return out;
}

double estimate_deviation(const std::vector<OctaveInfo>& octaves) {
    double variance = 0.0;
    for (const auto& o : octaves) {
        double layer = 0.2702247831245211 * std::fabs(o.amplitude);
        variance += layer * layer;
    }
    return std::sqrt(variance);
}

double compute_normalization_factor(double target_amplitude, const std::vector<OctaveInfo>& octaves) {
    double input_deviation = estimate_deviation(octaves);
    if (input_deviation == 0.0) return 0.0;
    double input_sum_deviation = input_deviation * std::sqrt(2.0);
    double target_deviation = target_amplitude * 0.3333333333333333;
    return target_deviation / input_sum_deviation;
}

double parity_expected_deviation(int span) { return 0.1 * (1.0 + 1.0 / (span + 1)); }

double compute_parity_normalization_factor(double base_amplitude, int octave_count, const NoiseDef& d) {
    int min_octave = 2147483647, max_octave = -2147483647 - 1;
    for (int i = 0; i < octave_count; ++i) {
        if (amplitude_modifier(d, i) != 0.0) {
            min_octave = std::min(min_octave, i);
            max_octave = std::max(max_octave, i);
        }
    }
    return base_amplitude * 0.5 * 0.3333333333333333 / parity_expected_deviation(max_octave - min_octave);
}

// GradientNoise(RandomSource, 256.0)
PerlinOctave make_octave(XoroshiroRandom& r, double smear) {
    PerlinOctave o{};
    o.ox = r.next_double() * 256.0;
    o.oy = r.next_double() * 256.0;
    o.oz = r.next_double() * 256.0;
    o.smear = smear;
    for (int i = 0; i < 256; ++i) o.perm[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 256; ++i) {
        int offset = r.next_int(256 - i);
        uint8_t tmp = o.perm[i];
        o.perm[i] = o.perm[offset + i];
        o.perm[offset + i] = tmp;
    }
    return o;
}

// NormalNoise(params).create(random)
StackRef create_normal_noise(SeedTables& t, const NoiseDef& d, XoroshiroRandom random, double* normalization_out) {
    bool normalize = d.normalize != "disabled";
    std::vector<OctaveInfo> octaves = build_octaves(d.base_octave, d.base_amplitude, d.octave_count, normalize, d);
    std::vector<double> abs_amps;
    for (const auto& o : octaves) abs_amps.push_back(std::fabs(o.amplitude));
    double target_amplitude = java_double_stream_sum(abs_amps);
    double normalization = compute_normalization_factor(target_amplitude, octaves);
    if (d.normalize == "legacy" && normalization != 0.0) {
        normalization = compute_parity_normalization_factor(d.base_amplitude, d.octave_count, d);
    }
    *normalization_out = normalization;

    XoroshiroPositional first = random.fork_positional();
    XoroshiroPositional second = random.fork_positional();
    StackRef ref;
    ref.first_layer = static_cast<int>(t.layers.size());
    for (const auto& o : octaves) {
        std::string seed = "octave_" + std::to_string(o.index);
        XoroshiroRandom r1 = first.from_hash_of(seed);
        XoroshiroRandom r2 = second.from_hash_of(seed);
        int i1 = static_cast<int>(t.octaves.size());
        t.octaves.push_back(make_octave(r1, 0.0));
        int i2 = static_cast<int>(t.octaves.size());
        t.octaves.push_back(make_octave(r2, 0.0));
        float value_factor = static_cast<float>(normalization * o.amplitude);
        t.layers.push_back({o.frequency, value_factor, i1});
        t.layers.push_back({o.frequency * 1.0181268882175227, value_factor, i2});
    }
    ref.count = static_cast<int>(t.layers.size()) - ref.first_layer;
    return ref;
}

// BlendedNoise.createFbm
StackRef create_fbm(SeedTables& t, XoroshiroRandom& random, int first_octave, double smear_scale_y, double value_factor) {
    if (first_octave > 0) throw std::runtime_error("firstOctave>0");
    int octaves = -first_octave + 1;
    double factor = 1.0;
    value_factor /= std::ldexp(1.0, octaves) - 1.0;                  // Math.pow(2.0, octaves) - 1.0
    StackRef ref;
    ref.first_layer = static_cast<int>(t.layers.size());
    ref.smeared = true;
    for (int i = octaves - 1; i >= 0; --i) {
        int idx = static_cast<int>(t.octaves.size());
        t.octaves.push_back(make_octave(random, smear_scale_y * factor));
        t.layers.push_back({factor, static_cast<float>(value_factor), idx});
        factor /= 2.0;
        value_factor *= 2.0;
    }
    ref.count = static_cast<int>(t.layers.size()) - ref.first_layer;
    return ref;
}

}  // namespace

double java_double_stream_sum(const std::vector<double>& values) {
    double s0 = 0.0, s1 = 0.0, simple = 0.0;
    for (double v : values) {
        double tmp = v - s1;
        double sum = s0;
        double velvel = sum + tmp;
        s1 = (velvel - sum) - tmp;
        s0 = velvel;
        simple += v;
    }
    double result = s0 - s1;
    if (std::isnan(result) && std::isinf(simple)) return simple;
    return result;
}

SeedTables build_seed_tables(const IR& ir, int64_t seed) {
    SeedTables t;
    t.seed = seed;
    XoroshiroPositional random = XoroshiroRandom(seed).fork_positional();

    for (const auto& [key, def] : ir.noises) {
        double normalization = 0.0;
        t.noise[key] = create_normal_noise(t, def, random.from_hash_of(key), &normalization);
        t.normalization[key] = normalization;
    }

    for (const Node& n : ir.nodes) {
        if (n.type != NodeType::BlendedNoise) continue;
        XoroshiroRandom r = random.from_hash_of("minecraft:terrain");
        double y_multiplier = 684.412 * n.y_scale;
        double limit_smear = y_multiplier * n.smear_mult;
        double main_smear = limit_smear / n.y_factor;
        const double limit_factor = static_cast<double>(0.99998474f);
        BlendedSet b;
        b.min_limit = create_fbm(t, r, -15, limit_smear, limit_factor);
        b.max_limit = create_fbm(t, r, -15, limit_smear, limit_factor);
        b.main = create_fbm(t, r, -7, main_smear, 12.75);
        t.blended[n.id] = b;
    }
    return t;
}

}  // namespace mcc
