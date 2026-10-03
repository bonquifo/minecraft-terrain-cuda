#include "mccuda/ir.h"

#include <fstream>
#include <stdexcept>

#include "mccuda/javamath.h"
#include "nlohmann/json.hpp"

namespace mcc {

namespace {

using json = nlohmann::json;

float jf(const json& j) { return bits2f(static_cast<uint32_t>(j.get<uint64_t>())); }

double jd(const json& j) { return bits2d(std::stoull(j.get<std::string>(), nullptr, 16)); }

Axis parse_axis(const std::string& s) {
    if (s == "x") return Axis::X;
    if (s == "y") return Axis::Y;
    if (s == "z") return Axis::Z;
    throw std::runtime_error("bad axis " + s);
}

BinOp parse_bin(const std::string& s) {
    if (s == "add") return BinOp::Add;
    if (s == "sub") return BinOp::Sub;
    if (s == "mul") return BinOp::Mul;
    if (s == "div") return BinOp::Div;
    if (s == "min") return BinOp::Min;
    if (s == "max") return BinOp::Max;
    throw std::runtime_error("bad binary op " + s);
}

UnOp parse_un(const std::string& s) {
    static const std::map<std::string, UnOp> m = {
        {"abs", UnOp::Abs}, {"square", UnOp::Square}, {"cube", UnOp::Cube}, {"sqrt", UnOp::Sqrt},
        {"half_negative", UnOp::HalfNegative}, {"quarter_negative", UnOp::QuarterNegative},
        {"reciprocal", UnOp::Reciprocal}, {"negate", UnOp::Negate}, {"squeeze", UnOp::Squeeze},
        {"log", UnOp::Log}, {"sign", UnOp::Sign}};
    auto it = m.find(s);
    if (it == m.end()) throw std::runtime_error("bad unary op " + s);
    return it->second;
}

Spline parse_spline(const json& j) {
    Spline s;
    if (j.contains("k")) {
        s.constant = true;
        s.k = jf(j["k"]);
        return s;
    }
    s.coord = j["coord"].get<int>();
    for (const auto& v : j["loc"]) s.locations.push_back(jf(v));
    for (const auto& v : j["der"]) s.derivatives.push_back(jf(v));
    for (const auto& v : j["val"]) s.values.push_back(parse_spline(v));
    return s;
}

}  // namespace

IR load_ir(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open IR " + path);
    json doc = json::parse(in);
    if (doc["format"] != "mccuda-ir/1") throw std::runtime_error("unexpected IR format");
    IR ir;
    ir.minecraft = doc["minecraft"].get<std::string>();
    ir.min_y = doc["min_y"].get<int>();
    ir.height = doc["height"].get<int>();
    ir.sea_level = doc["sea_level"].get<int>();
    for (auto& [k, v] : doc["roots"].items()) ir.roots[k] = v.get<int>();

    for (const auto& j : doc["nodes"]) {
        Node n;
        n.id = j["id"].get<int>();
        if (n.id != static_cast<int>(ir.nodes.size())) throw std::runtime_error("IR node ids must be dense and ordered");
        n.range_min = jf(j["range"][0]);
        n.range_max = jf(j["range"][1]);
        n.axes = j["axes"].get<int>();
        const std::string t = j["t"].get<std::string>();
        if (t == "cache") {
            n.type = NodeType::Cache; n.cache_id = j["cache_id"]; n.in = j["in"];
        } else if (t == "constant") {
            n.type = NodeType::Constant; n.v = jf(j["v"]);
        } else if (t == "simple") {
            n.type = NodeType::Simple;
            std::string k = j["kind"];
            n.simple = k == "blend_alpha" ? SimpleKind::BlendAlpha : k == "blend_offset" ? SimpleKind::BlendOffset : SimpleKind::Beardifier;
            if (k != "blend_alpha" && k != "blend_offset" && k != "beardifier") throw std::runtime_error("bad simple " + k);
        } else if (t == "gradient") {
            n.type = NodeType::Gradient;
            n.axis = parse_axis(j["axis"]);
            std::string tl = j["tiling"];
            n.tiling = tl == "clamp_to_edge" ? Tiling::ClampToEdge : tl == "repeat" ? Tiling::Repeat : Tiling::MirroredRepeat;
            n.from = j["from"]; n.to = j["to"];
            n.from_value = jf(j["from_value"]); n.to_value = jf(j["to_value"]);
        } else if (t == "noise") {
            n.type = NodeType::Noise;
            n.noise = j["noise"];
            n.xz_scale = jd(j["xz_scale"]); n.y_scale = jd(j["y_scale"]);
            n.shift[0] = j["shift_x"]; n.shift[1] = j["shift_y"]; n.shift[2] = j["shift_z"];
            n.shift_zero[0] = j["shift_x_zero"]; n.shift_zero[1] = j["shift_y_zero"]; n.shift_zero[2] = j["shift_z_zero"];
        } else if (t == "shift_a" || t == "shift_b" || t == "shift") {
            n.type = t == "shift_a" ? NodeType::ShiftA : t == "shift_b" ? NodeType::ShiftB : NodeType::Shift;
            n.noise = j["noise"];
        } else if (t == "blended_noise") {
            n.type = NodeType::BlendedNoise;
            n.xz_scale = jd(j["xz_scale"]); n.y_scale = jd(j["y_scale"]);
            n.xz_factor = jd(j["xz_factor"]); n.y_factor = jd(j["y_factor"]);
            n.smear_mult = jd(j["smear_scale_multiplier"]);
        } else if (t == "binary") {
            n.type = NodeType::Binary; n.bin = parse_bin(j["op"]); n.l = j["l"]; n.r = j["r"];
        } else if (t == "unary") {
            n.type = NodeType::Unary; n.un = parse_un(j["op"]); n.in = j["in"];
        } else if (t == "clamp") {
            n.type = NodeType::Clamp; n.in = j["in"]; n.fmin = jf(j["min"]); n.fmax = jf(j["max"]);
        } else if (t == "lerp") {
            n.type = NodeType::Lerp; n.alpha = j["alpha"]; n.first = j["first"]; n.second = j["second"];
        } else if (t == "range_choice") {
            n.type = NodeType::RangeChoice; n.in = j["in"];
            n.fmin = jf(j["min_inclusive"]); n.fmax = jf(j["max_exclusive"]);
            n.when_in = j["when_in"]; n.when_out = j["when_out"];
        } else if (t == "interval_select") {
            n.type = NodeType::IntervalSelect; n.in = j["in"];
            for (const auto& v : j["thresholds"]) n.thresholds.push_back(jf(v));
            for (const auto& v : j["functions"]) n.functions.push_back(v.get<int>());
            if (n.thresholds.size() + 1 != n.functions.size()) throw std::runtime_error("interval_select arity");
        } else if (t == "spline") {
            n.type = NodeType::Spline; n.spline = std::make_shared<Spline>(parse_spline(j["spline"]));
        } else if (t == "interpolated") {
            n.type = NodeType::Interpolated; n.in = j["in"]; n.cell_xz = j["cell_xz"]; n.cell_y = j["cell_y"];
        } else if (t == "slice") {
            n.type = NodeType::Slice; n.axis = parse_axis(j["axis"]); n.coord = j["coord"]; n.in = j["in"];
        } else if (t == "blend_density") {
            n.type = NodeType::BlendDensity; n.in = j["in"];
        } else if (t == "find_top_surface") {
            n.type = NodeType::FindTopSurface; n.density = j["density"]; n.upper_bound = j["upper_bound"];
            n.lower_bound = j["lower_bound"]; n.cell_height = j["cell_height"];
        } else {
            throw std::runtime_error("unsupported IR node type " + t);
        }
        ir.nodes.push_back(std::move(n));
    }

    for (auto& [k, v] : doc["noises"].items()) {
        NoiseDef d;
        d.base_amplitude = jd(v["base_amplitude"]);
        d.base_octave = v["base_octave"];
        d.octave_count = v["octave_count"];
        d.normalize = v["normalize"];
        for (const auto& m : v["amplitude_modifiers"]) d.amplitude_modifiers.push_back(jd(m));
        ir.noises[k] = d;
    }
    return ir;
}

}  // namespace mcc
