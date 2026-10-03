// In-memory form of the density-function IR exported by the oracle.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mcc {

enum class NodeType {
    Cache, Constant, Simple, Gradient, Noise, ShiftA, ShiftB, Shift, BlendedNoise, Binary, Unary, Clamp, Lerp,
    RangeChoice, IntervalSelect, Spline, Interpolated, Slice, BlendDensity,
};

enum class BinOp { Add, Sub, Mul, Div, Min, Max };
enum class UnOp { Abs, Square, Cube, Sqrt, HalfNegative, QuarterNegative, Reciprocal, Negate, Squeeze, Log, Sign };
enum class SimpleKind { BlendAlpha, BlendOffset, Beardifier };
enum class Axis { X = 0, Y = 1, Z = 2 };
enum class Tiling { ClampToEdge, Repeat, MirroredRepeat };

struct Spline {
    bool constant = false;
    float k = 0.0f;               // constant value
    int coord = -1;               // coordinate node id (multipoint)
    std::vector<float> locations, derivatives;
    std::vector<Spline> values;
};

struct Node {
    int id = -1;
    NodeType type = NodeType::Constant;
    float range_min = 0.0f, range_max = 0.0f;
    int axes = 0;

    // Generic children / parameters (meaning depends on type).
    int in = -1, l = -1, r = -1;          // single input, binary left/right
    int alpha = -1, first = -1, second = -1;
    int when_in = -1, when_out = -1;
    int shift[3] = {-1, -1, -1};
    bool shift_zero[3] = {true, true, true};
    std::vector<int> functions;
    std::vector<float> thresholds;

    float v = 0.0f;                       // constant
    float fmin = 0.0f, fmax = 0.0f;       // clamp min/max, range_choice min_inclusive/max_exclusive
    BinOp bin = BinOp::Add;
    UnOp un = UnOp::Abs;
    SimpleKind simple = SimpleKind::BlendAlpha;
    Axis axis = Axis::X;
    Tiling tiling = Tiling::ClampToEdge;
    int from = 0, to = 0, coord = 0;      // gradient from/to, slice coordinate
    float from_value = 0.0f, to_value = 0.0f;
    std::string noise;                    // noise key
    double xz_scale = 0.0, y_scale = 0.0; // noise / blended scales
    double xz_factor = 0.0, y_factor = 0.0, smear_mult = 0.0;
    int cache_id = -1;
    int cell_xz = 0, cell_y = 0;
    std::shared_ptr<Spline> spline;
};

struct NoiseDef {
    double base_amplitude = 1.0;
    int base_octave = 0;
    int octave_count = 1;
    std::string normalize;                // "enabled" | "disabled" | "legacy"
    std::vector<double> amplitude_modifiers;
};

struct IR {
    std::string minecraft;
    int min_y = 0, height = 0, sea_level = 0;
    std::vector<Node> nodes;              // index == id; children have smaller ids
    std::map<std::string, int> roots;
    std::map<std::string, NoiseDef> noises;

    const Node& at(int id) const { return nodes.at(static_cast<size_t>(id)); }
    int root(const std::string& name) const { return roots.at(name); }
};

IR load_ir(const std::string& path);

}  // namespace mcc
