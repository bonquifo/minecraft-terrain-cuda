#include "mccuda/volume_eval.h"

#include <algorithm>
#include <stdexcept>

namespace mcc {

namespace {

bool is_const(const IR& ir, int id) { return ir.at(id).type == NodeType::Constant; }

// Mth.binarySearch(from, to, i -> input < locations[i])
int find_interval_start(const std::vector<float>& loc, float input) {
    int from = 0;
    int len = static_cast<int>(loc.size());
    while (len > 0) {
        int half = len / 2;
        int middle = from + half;
        if (input < loc[static_cast<size_t>(middle)]) {
            len = half;
        } else {
            from = middle + 1;
            len -= half + 1;
        }
    }
    return from - 1;
}

float linear_extend(float input, const std::vector<float>& loc, float value, const std::vector<float>& der, int index) {
    float d = der[static_cast<size_t>(index)];
    if (d == 0.0f) return value;
    float dx = input - loc[static_cast<size_t>(index)];
    return value + d * dx;
}

}  // namespace

BinaryPlan binary_plan(const IR& ir, const Node& n) {
    BinaryPlan p{};
    const bool lc = is_const(ir, n.l), rc = is_const(ir, n.r);
    switch (n.bin) {
        case BinOp::Add: p = {BinaryPlan::Add, n.l, n.r}; break;
        case BinOp::Sub: p = {BinaryPlan::Sub, n.l, n.r}; break;
        case BinOp::Mul: p = {BinaryPlan::Mul, n.l, n.r}; break;
        case BinOp::Div:
            if (!lc && rc) {
                p = {BinaryPlan::DivByConst, n.l, n.r};
                p.inv = 1.0f / ir.at(n.r).v;
            } else {
                p = {BinaryPlan::Div, n.l, n.r};
            }
            break;
        case BinOp::Min: {
            const Node &L = ir.at(n.l), &R = ir.at(n.r);
            if (L.range_max < R.range_min) p = {BinaryPlan::PassLeft, n.l, -1};
            else if (R.range_max < L.range_min) p = {BinaryPlan::PassRight, n.r, -1};
            else if (lc) p = {BinaryPlan::Min, n.r, n.l};           // ConstMinSampler(right, leftValue)
            else p = {BinaryPlan::Min, n.l, n.r};                   // ConstMinSampler(left, c) or MinSampler(left, right)
            break;
        }
        case BinOp::Max: {
            const Node &L = ir.at(n.l), &R = ir.at(n.r);
            if (L.range_min > R.range_max) p = {BinaryPlan::PassLeft, n.l, -1};
            else if (R.range_min > L.range_max) p = {BinaryPlan::PassRight, n.r, -1};
            else if (lc) p = {BinaryPlan::Max, n.r, n.l};
            else p = {BinaryPlan::Max, n.l, n.r};
            break;
        }
    }
    return p;
}

GradientPlan gradient_plan(const Node& n) {
    GradientPlan g{};
    g.from = n.from;
    g.range = n.to - n.from;
    g.from_value = n.from_value;
    float dv = n.to_value - n.from_value;
    g.factor = dv / static_cast<float>(g.range);
    g.min_c = std::min(n.from, n.to);
    g.max_c = std::max(n.from, n.to);
    g.mode = n.tiling == Tiling::ClampToEdge ? 0 : (n.tiling == Tiling::Repeat ? 1 : 2);
    return g;
}

VolumeEvaluator::VolumeEvaluator(const IR& ir, const SeedTables& tables) : ir_(ir), t_(tables) {
    bin_.resize(ir.nodes.size());
    grad_.resize(ir.nodes.size());
    int max_cache = -1;
    for (const Node& n : ir.nodes) {
        if (n.type == NodeType::Binary) bin_[static_cast<size_t>(n.id)] = binary_plan(ir, n);
        if (n.type == NodeType::Gradient) grad_[static_cast<size_t>(n.id)] = gradient_plan(n);
        if (n.type == NodeType::Cache) max_cache = std::max(max_cache, n.cache_id);
    }
    caches_.resize(static_cast<size_t>(max_cache + 1));
}

void VolumeEvaluator::clear_caches() {
    for (auto& c : caches_) c.valid = false;
}

void VolumeEvaluator::sample_noise_plain(const StackRef& s, const Volume& v, double xz, double y, float amplitude, float* acc) {
    const StackLayer* layers = t_.layer_ptr(s);
    int index = 0;
    for (int iz = 0; iz < v.sz; ++iz)
        for (int ix = 0; ix < v.sx; ++ix)
            for (int iy = 0; iy < v.sy; ++iy, ++index)
                acc[index] = stack_add_volume(acc[index], layers, s.count, t_.octaves.data(), s.smeared, v.bx(ix), v.by(iy), v.bz(iz), xz, y, amplitude);
}

void VolumeEvaluator::sample(int id, const Volume& v, float* out) {
    const Node& n = ir_.at(id);
    const int size = v.size();
    switch (n.type) {
        case NodeType::Cache: {
            CacheCell& c = caches_[static_cast<size_t>(n.cache_id)];
            if (!c.valid || !(c.vol == v)) {
                c.buf.assign(static_cast<size_t>(size), 0.0f);
                sample(n.in, v, c.buf.data());
                c.vol = v;
                c.valid = true;
            }
            std::copy(c.buf.begin(), c.buf.end(), out);
            return;
        }
        case NodeType::Constant:
            std::fill(out, out + size, n.v);
            return;
        case NodeType::Simple:
            // Empty blender and Beardifier.EMPTY: blend_alpha -> 1, blend_offset -> 0, beardifier -> 0.
            std::fill(out, out + size, n.simple == SimpleKind::BlendAlpha ? 1.0f : 0.0f);
            return;
        case NodeType::Gradient: {
            const GradientPlan& g = grad_[static_cast<size_t>(id)];
            for (int iz = 0; iz < v.sz; ++iz)
                for (int ix = 0; ix < v.sx; ++ix)
                    for (int iy = 0; iy < v.sy; ++iy) {
                        int c = n.axis == Axis::X ? v.bx(ix) : (n.axis == Axis::Y ? v.by(iy) : v.bz(iz));
                        out[v.index(ix, iy, iz)] = gradient_compute(g, c);
                    }
            return;
        }
        case NodeType::Noise: {
            const StackRef& s = t_.noise.at(n.noise);
            if (n.shift_zero[0] && n.shift_zero[1] && n.shift_zero[2]) {
                std::fill(out, out + size, 0.0f);
                sample_noise_plain(s, v, n.xz_scale, n.y_scale, 1.0f, out);
                return;
            }
            std::vector<float> sx(static_cast<size_t>(size)), sy, sz(static_cast<size_t>(size));
            sample(n.shift[0], v, sx.data());
            const bool xyz = !n.shift_zero[1];
            if (xyz) {
                sy.resize(static_cast<size_t>(size));
                sample(n.shift[1], v, sy.data());
            }
            sample(n.shift[2], v, sz.data());
            const StackLayer* layers = t_.layer_ptr(s);
            int index = 0;
            for (int iz = 0; iz < v.sz; ++iz) {
                double base_z = static_cast<double>(v.bz(iz)) * n.xz_scale;
                for (int ix = 0; ix < v.sx; ++ix) {
                    double base_x = static_cast<double>(v.bx(ix)) * n.xz_scale;
                    for (int iy = 0; iy < v.sy; ++iy, ++index) {
                        double nx = base_x + static_cast<double>(sx[static_cast<size_t>(index)]);
                        double ny = static_cast<double>(v.by(iy)) * n.y_scale;
                        if (xyz) ny = ny + static_cast<double>(sy[static_cast<size_t>(index)]);
                        double nz = base_z + static_cast<double>(sz[static_cast<size_t>(index)]);
                        out[index] = stack_get(layers, s.count, t_.octaves.data(), nx, ny, nz);
                    }
                }
            }
            return;
        }
        case NodeType::ShiftA:
        case NodeType::Shift: {
            const StackRef& s = t_.noise.at(n.noise);
            std::fill(out, out + size, 0.0f);
            sample_noise_plain(s, v, 0.25, n.type == NodeType::ShiftA ? 0.0 : 0.25, 1.0f, out);
            for (int i = 0; i < size; ++i) out[i] = out[i] * 4.0f;
            return;
        }
        case NodeType::ShiftB: {
            const StackRef& s = t_.noise.at(n.noise);
            Volume tv{v.sz, v.sx, 1, v.minz, v.minx, 0, v.stz, v.stx, 1};
            std::vector<float> tb(static_cast<size_t>(tv.size()), 0.0f);
            sample_noise_plain(s, tv, 0.25, 0.25, 4.0f, tb.data());
            for (int iz = 0; iz < v.sz; ++iz)
                for (int ix = 0; ix < v.sx; ++ix) {
                    float value = tb[static_cast<size_t>(tv.index(iz, ix, 0))];
                    std::fill(out + v.index(ix, 0, iz), out + v.index(ix, 0, iz) + v.sy, value);
                }
            return;
        }
        case NodeType::BlendedNoise: {
            const BlendedSet& b = t_.blended.at(id);
            double xz_mult = 684.412 * n.xz_scale;
            double y_mult = 684.412 * n.y_scale;
            std::vector<float> first(static_cast<size_t>(size), 0.0f), second(static_cast<size_t>(size), 0.0f);
            std::fill(out, out + size, 0.0f);
            sample_noise_plain(b.main, v, xz_mult / n.xz_factor, y_mult / n.y_factor, 1.0f, out);
            for (int i = 0; i < size; ++i) out[i] = mth_clamp(out[i] + 0.5f, 0.0f, 1.0f);
            sample_noise_plain(b.min_limit, v, xz_mult, y_mult, 1.0f, first.data());
            sample_noise_plain(b.max_limit, v, xz_mult, y_mult, 1.0f, second.data());
            for (int i = 0; i < size; ++i) {
                float a = out[i];
                out[i] = a == 0.0f ? first[static_cast<size_t>(i)] : (a == 1.0f ? second[static_cast<size_t>(i)] : mth_lerp(a, first[static_cast<size_t>(i)], second[static_cast<size_t>(i)]));
            }
            return;
        }
        case NodeType::Binary: {
            const BinaryPlan& p = bin_[static_cast<size_t>(id)];
            if (p.kind == BinaryPlan::PassLeft || p.kind == BinaryPlan::PassRight) {
                sample(p.first, v, out);
                return;
            }
            sample(p.first, v, out);
            std::vector<float> b(static_cast<size_t>(size));
            sample(p.second, v, b.data());
            for (int i = 0; i < size; ++i) {
                float a = out[i], c = b[static_cast<size_t>(i)];
                switch (p.kind) {
                    case BinaryPlan::Add: out[i] = a + c; break;
                    case BinaryPlan::Sub: out[i] = a - c; break;
                    case BinaryPlan::Mul: out[i] = a * c; break;
                    case BinaryPlan::Div: out[i] = a / c; break;
                    case BinaryPlan::DivByConst: out[i] = a * p.inv; break;
                    case BinaryPlan::Min: out[i] = vol_min(a, c); break;
                    case BinaryPlan::Max: out[i] = vol_max(a, c); break;
                    default: break;
                }
            }
            return;
        }
        case NodeType::Unary: {
            sample(n.in, v, out);
            for (int i = 0; i < size; ++i) {
                float x = out[i];
                switch (n.un) {
                    case UnOp::Abs: x = un_abs(x); break;
                    case UnOp::Square: x = un_square(x); break;
                    case UnOp::Cube: x = un_cube(x); break;
                    case UnOp::Sqrt: x = un_sqrt(x); break;
                    case UnOp::HalfNegative: x = un_leaky(x, 0.5f); break;
                    case UnOp::QuarterNegative: x = un_leaky(x, 0.25f); break;
                    case UnOp::Reciprocal: x = un_reciprocal(x); break;
                    case UnOp::Negate: x = un_negate(x); break;
                    case UnOp::Squeeze: x = un_squeeze(x); break;
                    case UnOp::Sign: x = un_sign(x); break;
                    case UnOp::Log: throw std::runtime_error("log unsupported (Math.log is a HotSpot intrinsic; not used by the overworld)");
                }
                out[i] = x;
            }
            return;
        }
        case NodeType::Clamp:
            sample(n.in, v, out);
            for (int i = 0; i < size; ++i) out[i] = mth_clamp(out[i], n.fmin, n.fmax);
            return;
        case NodeType::Lerp: {
            sample(n.alpha, v, out);
            std::vector<float> f(static_cast<size_t>(size)), s(static_cast<size_t>(size));
            sample(n.first, v, f.data());
            sample(n.second, v, s.data());
            for (int i = 0; i < size; ++i) {
                float a = out[i];
                out[i] = a == 0.0f ? f[static_cast<size_t>(i)] : (a == 1.0f ? s[static_cast<size_t>(i)] : mth_lerp(a, f[static_cast<size_t>(i)], s[static_cast<size_t>(i)]));
            }
            return;
        }
        case NodeType::RangeChoice: {
            std::vector<float> in(static_cast<size_t>(size)), wo(static_cast<size_t>(size));
            sample(n.when_in, v, out);
            sample(n.in, v, in.data());
            sample(n.when_out, v, wo.data());
            for (int i = 0; i < size; ++i)
                if (!in_range(in[static_cast<size_t>(i)], n.fmin, n.fmax)) out[i] = wo[static_cast<size_t>(i)];
            return;
        }
        case NodeType::IntervalSelect: {
            sample(n.in, v, out);
            std::vector<std::vector<float>> bufs(n.functions.size(), std::vector<float>(static_cast<size_t>(size)));
            for (size_t k = 0; k < n.functions.size(); ++k) sample(n.functions[k], v, bufs[k].data());
            for (int i = 0; i < size; ++i) {
                float x = out[i];
                size_t sel = n.functions.size() - 1;
                for (size_t k = 0; k < n.thresholds.size(); ++k) {
                    if (x < n.thresholds[k]) { sel = k; break; }
                }
                out[i] = bufs[sel][static_cast<size_t>(i)];
            }
            return;
        }
        case NodeType::Spline: {
            std::vector<std::vector<float>> coords;
            std::vector<int> coord_ids;
            for (int i = 0; i < size; ++i) out[i] = spline_eval(*n.spline, i, coords, coord_ids, v);
            return;
        }
        case NodeType::Interpolated:
            sample_interpolated(n, v, out);
            return;
        case NodeType::Slice:
            sample_slice(n, v, out);
            return;
        case NodeType::BlendDensity:
            sample(n.in, v, out);   // empty blender: identity
            return;
    }
    throw std::runtime_error("unhandled node type");
}

float VolumeEvaluator::spline_eval(const Spline& s, int index, std::vector<std::vector<float>>& coords, std::vector<int>& coord_ids, const Volume& v) {
    if (s.constant) return s.k;
    // BufferSplineInput: each distinct coordinate function is sampled over the whole volume once.
    size_t slot = 0;
    while (slot < coord_ids.size() && coord_ids[slot] != s.coord) ++slot;
    if (slot == coord_ids.size()) {
        coord_ids.push_back(s.coord);
        coords.emplace_back(static_cast<size_t>(v.size()));
        sample(s.coord, v, coords.back().data());
    }
    float input = coords[slot][static_cast<size_t>(index)];
    int start = find_interval_start(s.locations, input);
    int last = static_cast<int>(s.locations.size()) - 1;
    if (start < 0) return linear_extend(input, s.locations, spline_eval(s.values.front(), index, coords, coord_ids, v), s.derivatives, 0);
    if (start == last) return linear_extend(input, s.locations, spline_eval(s.values[static_cast<size_t>(last)], index, coords, coord_ids, v), s.derivatives, last);
    float x1 = s.locations[static_cast<size_t>(start)];
    float x2 = s.locations[static_cast<size_t>(start) + 1];
    float t = (input - x1) / (x2 - x1);
    float d1 = s.derivatives[static_cast<size_t>(start)];
    float d2 = s.derivatives[static_cast<size_t>(start) + 1];
    float y1 = spline_eval(s.values[static_cast<size_t>(start)], index, coords, coord_ids, v);
    float y2 = spline_eval(s.values[static_cast<size_t>(start) + 1], index, coords, coord_ids, v);
    float dx = x2 - x1;
    float dy = y2 - y1;
    float a = d1 * dx - dy;
    float b = -d2 * dx + dy;
    float l1 = mth_lerp(t, y1, y2);
    float w = t * (1.0f - t);
    float l2 = mth_lerp(t, a, b);
    return l1 + w * l2;
}

void VolumeEvaluator::sample_slice(const Node& n, const Volume& v, float* out) {
    const Node& inner = ir_.at(n.in);
    // Slice X over slice Z (or Z over X) compiles to vanilla's XzSampler.
    if (inner.type == NodeType::Slice &&
        ((n.axis == Axis::X && inner.axis == Axis::Z) || (n.axis == Axis::Z && inner.axis == Axis::X))) {
        int x = n.axis == Axis::X ? n.coord : inner.coord;
        int z = n.axis == Axis::X ? inner.coord : n.coord;
        if (v.sx == 1 && v.sz == 1 && v.minx == x && v.minz == z) {
            sample(inner.in, v, out);
            return;
        }
        Volume iv{1, v.sy, 1, x, v.miny, z, v.stx, v.sty, v.stz};
        std::vector<float> ib(static_cast<size_t>(iv.size()));
        sample(inner.in, iv, ib.data());
        for (int y = 0; y < v.sy; ++y) {
            float value = ib[static_cast<size_t>(iv.index(0, y, 0))];
            for (int z2 = 0; z2 < v.sz; ++z2)
                for (int x2 = 0; x2 < v.sx; ++x2) out[v.index(x2, y, z2)] = value;
        }
        return;
    }
    switch (n.axis) {
        case Axis::X: {
            if (v.sx == 1 && v.minx == n.coord) { sample(n.in, v, out); return; }
            Volume iv{1, v.sy, v.sz, n.coord, v.miny, v.minz, v.stx, v.sty, v.stz};
            std::vector<float> ib(static_cast<size_t>(iv.size()));
            sample(n.in, iv, ib.data());
            for (int z = 0; z < v.sz; ++z)
                for (int x = 0; x < v.sx; ++x)
                    for (int y = 0; y < v.sy; ++y) out[v.index(x, y, z)] = ib[static_cast<size_t>(iv.index(0, y, z))];
            return;
        }
        case Axis::Y: {
            if (v.sy == 1 && v.miny == n.coord) { sample(n.in, v, out); return; }
            Volume iv{v.sx, 1, v.sz, v.minx, n.coord, v.minz, v.stx, v.sty, v.stz};
            std::vector<float> ib(static_cast<size_t>(iv.size()));
            sample(n.in, iv, ib.data());
            for (int z = 0; z < v.sz; ++z)
                for (int x = 0; x < v.sx; ++x) {
                    float value = ib[static_cast<size_t>(iv.index(x, 0, z))];
                    std::fill(out + v.index(x, 0, z), out + v.index(x, 0, z) + v.sy, value);
                }
            return;
        }
        case Axis::Z: {
            if (v.sz == 1 && v.minz == n.coord) { sample(n.in, v, out); return; }
            Volume iv{v.sx, v.sy, 1, v.minx, v.miny, n.coord, v.stx, v.sty, v.stz};
            std::vector<float> ib(static_cast<size_t>(iv.size()));
            sample(n.in, iv, ib.data());
            for (int z = 0; z < v.sz; ++z)
                for (int x = 0; x < v.sx; ++x)
                    for (int y = 0; y < v.sy; ++y) out[v.index(x, y, z)] = ib[static_cast<size_t>(iv.index(x, y, 0))];
            return;
        }
    }
}

void VolumeEvaluator::sample_interpolated(const Node& n, const Volume& v, float* out) {
    const int cxz = n.cell_xz, cy = n.cell_y;
    if ((v.stx == cxz || v.sx == 1) && (v.sty == cy || v.sy == 1) && (v.stz == cxz || v.sz == 1) &&
        floor_mod(v.minx, cxz) == 0 && floor_mod(v.miny, cy) == 0 && floor_mod(v.minz, cxz) == 0) {
        sample(n.in, v, out);
        return;
    }
    if (v.stx == 1 && v.sty == 1 && v.stz == 1) {
        sample_interpolated_block_step(n, v, out);
        return;
    }
    Volume bv{v.sx * v.stx, v.sy * v.sty, v.sz * v.stz, v.minx, v.miny, v.minz, 1, 1, 1};
    std::vector<float> bb(static_cast<size_t>(bv.size()));
    sample_interpolated_block_step(n, bv, bb.data());
    for (int z = 0; z < v.sz; ++z)
        for (int x = 0; x < v.sx; ++x)
            for (int y = 0; y < v.sy; ++y) out[v.index(x, y, z)] = bb[static_cast<size_t>(bv.index(x * v.stx, y * v.sty, z * v.stz))];
}

void VolumeEvaluator::sample_interpolated_block_step(const Node& n, const Volume& v, float* out) {
    const int cxz = n.cell_xz, cy = n.cell_y;
    const float inv_xz = 1.0f / static_cast<float>(cxz);
    const float inv_y = 1.0f / static_cast<float>(cy);
    int min_cx = floor_div(v.minx, cxz), min_cy = floor_div(v.miny, cy), min_cz = floor_div(v.minz, cxz);
    int max_cx = floor_div(v.max_bx(), cxz), max_cy = floor_div(v.max_by(), cy), max_cz = floor_div(v.max_bz(), cxz);
    int count_x = max_cx - min_cx + 1, count_y = max_cy - min_cy + 1, count_z = max_cz - min_cz + 1;
    Volume cv{floor_mod(v.max_bx(), cxz) == 0 ? count_x : count_x + 1,
              floor_mod(v.max_by(), cy) == 0 ? count_y : count_y + 1,
              floor_mod(v.max_bz(), cxz) == 0 ? count_z : count_z + 1,
              min_cx * cxz, min_cy * cy, min_cz * cxz, cxz, cy, cxz};
    std::vector<float> cb(static_cast<size_t>(cv.size()));
    sample(n.in, cv, cb.data());

    for (int cell_z = 0; cell_z < count_z; ++cell_z) {
        int next_z = std::min(cell_z + 1, cv.sz - 1);
        for (int cell_x = 0; cell_x < count_x; ++cell_x) {
            int next_x = std::min(cell_x + 1, cv.sx - 1);
            float v000 = cb[static_cast<size_t>(cv.index(cell_x, 0, cell_z))];
            float v100 = cb[static_cast<size_t>(cv.index(next_x, 0, cell_z))];
            float v001 = cb[static_cast<size_t>(cv.index(cell_x, 0, next_z))];
            float v101 = cb[static_cast<size_t>(cv.index(next_x, 0, next_z))];
            for (int cell_y = 0; cell_y < count_y; ++cell_y) {
                int next_y = std::min(cell_y + 1, cv.sy - 1);
                float v010 = cb[static_cast<size_t>(cv.index(cell_x, next_y, cell_z))];
                float v110 = cb[static_cast<size_t>(cv.index(next_x, next_y, cell_z))];
                float v011 = cb[static_cast<size_t>(cv.index(cell_x, next_y, next_z))];
                float v111 = cb[static_cast<size_t>(cv.index(next_x, next_y, next_z))];
                // fillCell
                int ox = cv.bx(cell_x) - v.minx, oy = cv.by(cell_y) - v.miny, oz = cv.bz(cell_z) - v.minz;
                int x0 = std::max(0, -ox), y0 = std::max(0, -oy), z0 = std::max(0, -oz);
                int x1 = std::min(cxz, v.sx - ox) - 1, y1 = std::min(cy, v.sy - oy) - 1, z1 = std::min(cxz, v.sz - oz) - 1;
                for (int z = z0; z <= z1; ++z) {
                    float az = static_cast<float>(z) * inv_xz;
                    float v00 = mth_lerp(az, v000, v001);
                    float v01 = mth_lerp(az, v010, v011);
                    float v10 = mth_lerp(az, v100, v101);
                    float v11 = mth_lerp(az, v110, v111);
                    for (int x = x0; x <= x1; ++x) {
                        float ax = static_cast<float>(x) * inv_xz;
                        float a0 = mth_lerp(ax, v00, v10);
                        float a1 = mth_lerp(ax, v01, v11);
                        float step = (a1 - a0) * inv_y;
                        float value = a0 + step * static_cast<float>(y0);
                        int oi = v.index(ox + x, oy + y0, oz + z);
                        for (int y = y0; y <= y1; ++y) {
                            out[oi++] = value;
                            value += step;
                        }
                    }
                }
                v000 = v010; v100 = v110; v001 = v011; v101 = v111;
            }
        }
    }
}

void cpu_generate_chunk(const IR& ir, VolumeEvaluator& eval, int cx, int cz, float* density, uint8_t* fill) {
    const int h = ir.height;
    Volume v{16, h, 16, cx * 16, ir.min_y, cz * 16, 1, 1, 1};
    std::vector<float> buf(static_cast<size_t>(v.size()));
    eval.clear_caches();
    eval.sample(ir.root("final_density"), v, buf.data());
    for (int z = 0; z < 16; ++z)
        for (int x = 0; x < 16; ++x)
            for (int y = 0; y < h; ++y) {
                float d = buf[static_cast<size_t>(v.index(x, y, z))];
                size_t o = static_cast<size_t>((y * 16 + z) * 16 + x);
                if (density) density[o] = d;
                if (fill) fill[o] = core_fill(d, ir.min_y + y, ir.sea_level);
            }
}

}  // namespace mcc
