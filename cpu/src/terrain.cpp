#include "mccuda/terrain.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace mcc {

namespace {

constexpr int kWayBelowMinY = -2032 << 4;   // DimensionType.WAY_BELOW_MIN_Y

const double kSqrt3 = std::sqrt(3.0);
const double kF2 = 0.5 * (kSqrt3 - 1.0);
const double kG2 = (3.0 - kSqrt3) / 6.0;

Simplex2D make_simplex(LegacyRandom& r) {
    Simplex2D s{};
    // GradientNoise(random, 0.0): offsets nextDouble() * 0.0 (draws are still consumed).
    s.ox = r.next_double() * 0.0;
    s.oy = r.next_double() * 0.0;
    double unused = r.next_double() * 0.0;
    (void)unused;
    for (int i = 0; i < 256; ++i) s.perm[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 256; ++i) {
        int offset = r.next_int(256 - i);
        std::swap(s.perm[i], s.perm[offset + i]);
    }
    return s;
}

double simplex_corner(int index, double x, double y, double z, double base) {
    double t0 = base - x * x - y * y - z * z;
    if (t0 < 0.0) return 0.0;
    t0 *= t0;
    double dot = static_cast<double>(kGradTable[index][0]) * x + static_cast<double>(kGradTable[index][1]) * y;
    dot = dot + static_cast<double>(kGradTable[index][2]) * z;
    return t0 * t0 * dot;
}

}  // namespace

float Simplex2D::get(double px, double py) const {
    double xin = px + ox, yin = py + oy;
    double s = (xin + yin) * kF2;
    int32_t i = mth_floor(xin + s), j = mth_floor(yin + s);
    double t = static_cast<double>(jadd(i, j)) * kG2;
    double X0 = static_cast<double>(i) - t, Y0 = static_cast<double>(j) - t;
    double x0 = xin - X0, y0 = yin - Y0;
    int i1, j1;
    if (x0 > y0) { i1 = 1; j1 = 0; } else { i1 = 0; j1 = 1; }
    double x1 = x0 - static_cast<double>(i1) + kG2;
    double y1 = y0 - static_cast<double>(j1) + kG2;
    double x2 = x0 - 1.0 + 2.0 * kG2;
    double y2 = y0 - 1.0 + 2.0 * kG2;
    int ii = i & 0xFF, jj = j & 0xFF;
    auto P = [&](int v) { return static_cast<int>(perm[v & 0xFF]); };
    int gi0 = P(ii + P(jj)) % 12;
    int gi1 = P(ii + i1 + P(jj + j1)) % 12;
    int gi2 = P(ii + 1 + P(jj + 1)) % 12;
    double n0 = simplex_corner(gi0, x0, y0, 0.0, 0.5);
    double n1 = simplex_corner(gi1, x1, y1, 0.0, 0.5);
    double n2 = simplex_corner(gi2, x2, y2, 0.0, 0.5);
    return static_cast<float>(70.0 * (n0 + n1 + n2));
}

BiomeTemperature::BiomeTemperature() {
    LegacyRandom r1(1234);
    temperature = make_simplex(r1);
    LegacyRandom r3(3456);
    for (int i = 0; i < 3; ++i) frozen[i] = make_simplex(r3);
    frozen_amp[0] = 0.14285715f;
    frozen_amp[1] = 0.2857143f;
    frozen_amp[2] = 0.5714286f;
    LegacyRandom r2(2345);
    biome_info = make_simplex(r2);
}

float BiomeTemperature::frozen_get(double x, double y) const {
    const double freq[3] = {1.0, 0.5, 0.25};
    float v = 0.0f;
    for (int i = 0; i < 3; ++i) v += frozen_amp[i] * frozen[i].get(x * freq[i], y * freq[i]);
    return v;
}

float BiomeTemperature::height_adjusted(const BiomeInfo& b, int x, int y, int z, int sea_level) const {
    float adjusted = b.temperature;
    if (b.frozen_modifier) {
        double large = static_cast<double>(frozen_get(static_cast<double>(x) * 0.05, static_cast<double>(z) * 0.05) * 7.0f);
        double edge = static_cast<double>(biome_info.get(static_cast<double>(x) * 0.2, static_cast<double>(z) * 0.2));
        double patches = large + edge;
        if (patches < 0.3) {
            double small = static_cast<double>(biome_info.get(static_cast<double>(x) * 0.09, static_cast<double>(z) * 0.09));
            if (small < 0.8) adjusted = 0.2f;
        }
    }
    int snow_level = sea_level + 17;
    if (y > snow_level) {
        float v = temperature.get(static_cast<double>(static_cast<float>(x) / 8.0f), static_cast<double>(static_cast<float>(z) / 8.0f)) * 8.0f;
        float t = v + static_cast<float>(y);
        t = t - static_cast<float>(snow_level);
        t = t * 0.05f;
        t = t / 40.0f;
        return adjusted - t;
    }
    return adjusted;
}

TerrainSeed::TerrainSeed(const IR& ir_, const TerrainModel& tm_, int64_t seed_)
    : ir(ir_), tm(tm_), tables(build_seed_tables(ir_, seed_)), tree(tm_.params), seed(seed_), zoom_seed(biome_zoom_seed(seed_)),
      random(XoroshiroRandom(seed_).fork_positional()),
      aquifer_random(random.from_hash_of("minecraft:aquifer").fork_positional()),
      ore_random(random.from_hash_of("minecraft:ore").fork_positional()) {
    for (const Condition& c : tm.conds)
        if (c.type == CondType::VerticalGradient && !gradient_random.count(c.random_name))
            gradient_random.emplace(c.random_name, random.from_hash_of(c.random_name).fork_positional());

    // MaterialSystem.generateBands(noiseRandom.fromHashOf("minecraft:clay_bands"))
    XoroshiroRandom r = random.from_hash_of("minecraft:clay_bands");
    const int T = tm.terracotta[0], WHITE = tm.terracotta[1], ORANGE = tm.terracotta[2], YELLOW = tm.terracotta[3], BROWN = tm.terracotta[4],
              RED = tm.terracotta[5], LIGHT_GRAY = tm.terracotta[6];
    std::fill(clay_bands, clay_bands + 192, T);
    for (int i = 0; i < 192; ++i) {
        i += r.next_int(5) + 1;
        if (i < 192) clay_bands[i] = ORANGE;
    }
    auto make_bands = [&](int base_width, int state) {
        int count = r.next_int_between_inclusive(6, 15);
        for (int i = 0; i < count; ++i) {
            int width = base_width + r.next_int(3);
            int start = r.next_int(192);
            for (int p = 0; start + p < 192 && p < width; ++p) clay_bands[start + p] = state;
        }
    };
    make_bands(1, YELLOW);
    make_bands(2, BROWN);
    make_bands(1, RED);
    int white_count = r.next_int_between_inclusive(9, 15);
    int i = 0;
    for (int start = 0; i < white_count && start < 192; start += r.next_int(16) + 4) {
        clay_bands[start] = WHITE;
        if (start - 1 > 0 && r.next_boolean()) clay_bands[start - 1] = LIGHT_GRAY;
        if (start + 1 < 192 && r.next_boolean()) clay_bands[start + 1] = LIGHT_GRAY;
        ++i;
    }
}

// ---- biomes ----------------------------------------------------------------------------
void cpu_chunk_biomes(TerrainSeed& ts, VolumeEvaluator& eval, int cx, int cz, uint8_t* out) {
    const IR& ir = ts.ir;
    const int qh = ir.height / 4;
    Volume v{4, qh, 4, cx * 16, ir.min_y, cz * 16, 4, 4, 4};
    static const char* roots[6] = {"climate/temperature", "climate/vegetation", "climate/continents", "climate/erosion", "climate/depth", "climate/ridges"};
    std::vector<float> buf[6];
    eval.clear_caches();
    for (int k = 0; k < 6; ++k) {
        buf[k].resize(static_cast<size_t>(v.size()));
        eval.sample(ir.root(roots[k]), v, buf[k].data());
    }
    int last = -1;   // R-tree candidate, reset per chunk
    for (int sy = 0; sy < qh / 4; ++sy)
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y)
                for (int z = 0; z < 4; ++z) {
                    int qy = sy * 4 + y;
                    int idx = v.index(x, qy, z);
                    int64_t target[7];
                    for (int k = 0; k < 6; ++k) target[k] = quantize_coord(buf[k][static_cast<size_t>(idx)]);
                    target[6] = 0;
                    last = ts.tree.search(target, last);
                    out[(qy * 4 + z) * 4 + x] = static_cast<uint8_t>(ts.tree.leaf_biome(last));
                }
}

namespace {

struct FluidStatus {
    int level;
    int type;
    int at(int y, int air) const { return y < level ? type : air; }
    bool operator==(const FluidStatus& o) const { return level == o.level && type == o.type; }
};

// ---- NoiseBasedAquifer ------------------------------------------------------------------
class Aquifer {
public:
    Aquifer(TerrainSeed& ts, VolumeEvaluator& eval, const Volume& vol) : ts_(ts), ev_(eval), tm_(ts.tm) {
        const IR& ir = ts.ir;
        barrier_ = ir.root("aquifer/barrier");
        floodedness_ = ir.root("aquifer/fluid_level_floodedness");
        spread_ = ir.root("aquifer/fluid_level_spread");
        lava_ = ir.root("aquifer/lava");
        exclusion_ = ir.root("aquifer/exclusion");
        surface_level_ = ir.root("aquifer/surface_level");
        min_grid_x_ = grid_x(vol.minx - 5);
        int max_grid_x = grid_x(vol.max_bx() - 5) + 1;
        grid_size_x_ = max_grid_x - min_grid_x_ + 1;
        min_grid_y_ = grid_y(vol.miny + 1) - 1;
        int max_grid_y = grid_y(vol.max_by() + 1) + 1;
        int grid_size_y = max_grid_y - min_grid_y_ + 1;
        min_grid_z_ = grid_z(vol.minz - 5);
        int max_grid_z = grid_z(vol.max_bz() - 5) + 1;
        grid_size_z_ = max_grid_z - min_grid_z_ + 1;
        size_t total = static_cast<size_t>(grid_size_x_) * grid_size_y * grid_size_z_;
        status_.assign(total, FluidStatus{0, -1});
        has_status_.assign(total, false);
        location_.assign(total, INT64_MAX);
        int max_adjusted = adjust(max_surface_level(from_grid_x(min_grid_x_, 0), from_grid_z(min_grid_z_, 0), from_grid_x(max_grid_x, 9), from_grid_z(max_grid_z, 9)));
        int skip_grid_y = grid_y(max_adjusted + 12) - -1;
        skip_above_y_ = from_grid_y(skip_grid_y, 11) - 1;
    }

    // computeSubstance; returns -1 for "null" (solid: use the default block).
    int compute_substance(int x, int y, int z, double density) {
        if (density > 0.0) return -1;
        FluidStatus global = global_fluid(x, y, z);
        if (y > skip_above_y_) return global.at(y, tm_.air);
        if (global.at(y, tm_.air) == tm_.lava) return tm_.lava;
        int xa = grid_x(x - 5), ya = grid_y(y + 1), za = grid_z(z - 5);
        int d1 = INT_MAX, d2 = INT_MAX, d3 = INT_MAX, d4 = INT_MAX;
        int i1 = 0, i2 = 0, i3 = 0, i4 = 0;
        for (int ox = 0; ox <= 1; ++ox)
            for (int oy = -1; oy <= 1; ++oy)
                for (int oz = 0; oz <= 1; ++oz) {
                    int gx = xa + ox, gy = ya + oy, gz = za + oz;
                    int index = get_index(gx, gy, gz);
                    int64_t loc = location_[static_cast<size_t>(index)];
                    int lx, ly, lz;
                    if (loc != INT64_MAX) {
                        unpack(loc, lx, ly, lz);
                    } else {
                        XoroshiroRandom r = ts_.aquifer_random.at(gx, gy, gz);
                        lx = from_grid_x(gx, r.next_int(10));
                        ly = from_grid_y(gy, r.next_int(9));
                        lz = from_grid_z(gz, r.next_int(10));
                        location_[static_cast<size_t>(index)] = pack(lx, ly, lz);
                        unpack(location_[static_cast<size_t>(index)], lx, ly, lz);
                    }
                    int dx = lx - x, dy = ly - y, dz = lz - z;
                    int nd = dx * dx + dy * dy + dz * dz;
                    if (d1 >= nd) { i4 = i3; i3 = i2; i2 = i1; i1 = index; d4 = d3; d3 = d2; d2 = d1; d1 = nd; }
                    else if (d2 >= nd) { i4 = i3; i3 = i2; i2 = index; d4 = d3; d3 = d2; d2 = nd; }
                    else if (d3 >= nd) { i4 = i3; i3 = index; d4 = d3; d3 = nd; }
                    else if (d4 >= nd) { i4 = index; d4 = nd; }
                }
        FluidStatus s1 = aquifer_status(i1);
        double sim12 = similarity(d1, d2);
        int fluid = s1.at(y, tm_.air);
        if (sim12 <= 0.0) return fluid;
        if (fluid == tm_.water && global_fluid(x, y - 1, z).at(y - 1, tm_.air) == tm_.lava) return fluid;
        double barrier_value = std::nan("");
        bool barrier_set = false;
        FluidStatus s2 = aquifer_status(i2);
        double b12 = sim12 * pressure(x, y, z, barrier_value, barrier_set, s1, s2);
        if (density + b12 > 0.0) return -1;
        FluidStatus s3 = aquifer_status(i3);
        double sim13 = similarity(d1, d3);
        if (sim13 > 0.0) {
            double b13 = sim12 * sim13 * pressure(x, y, z, barrier_value, barrier_set, s1, s3);
            if (density + b13 > 0.0) return -1;
        }
        double sim23 = similarity(d2, d3);
        if (sim23 > 0.0) {
            double b23 = sim12 * sim23 * pressure(x, y, z, barrier_value, barrier_set, s2, s3);
            if (density + b23 > 0.0) return -1;
        }
        (void)i4;
        return fluid;
    }

private:
    static int grid_x(int b) { return b >> 4; }
    static int grid_z(int b) { return b >> 4; }
    static int grid_y(int b) { return floor_div(b, 12); }
    static int from_grid_x(int g, int o) { return (g << 4) + o; }
    static int from_grid_z(int g, int o) { return (g << 4) + o; }
    static int from_grid_y(int g, int o) { return g * 12 + o; }
    static int adjust(int level) { return level + 8; }
    static double similarity(int a, int b) { return 1.0 - static_cast<double>(b - a) / 25.0; }

    // BlockPos.asLong / getX / getY / getZ (26/12/26 bit packing)
    static int64_t pack(int x, int y, int z) {
        uint64_t v = ((static_cast<uint64_t>(static_cast<int64_t>(x)) & 0x3FFFFFFull) << 38) |
                     ((static_cast<uint64_t>(static_cast<int64_t>(z)) & 0x3FFFFFFull) << 12) | (static_cast<uint64_t>(static_cast<int64_t>(y)) & 0xFFFull);
        return static_cast<int64_t>(v);
    }
    static void unpack(int64_t v, int& x, int& y, int& z) {
        x = static_cast<int>(v >> 38);
        y = static_cast<int>((v << 52) >> 52);
        z = static_cast<int>((v << 26) >> 38);
    }

    FluidStatus global_fluid(int x, int y, int z) const {
        (void)x; (void)z;
        int lava_level = std::min(-54, tm_.sea_level);
        return y < lava_level ? FluidStatus{-54, tm_.lava} : FluidStatus{tm_.sea_level, tm_.default_fluid};
    }

    int get_index(int gx, int gy, int gz) const {
        int x = gx - min_grid_x_, y = gy - min_grid_y_, z = gz - min_grid_z_;
        return (y * grid_size_z_ + z) * grid_size_x_ + x;
    }

    int max_surface_level(int min_bx, int min_bz, int max_bx, int max_bz) {
        int min_qx = min_bx >> 2, max_qx = max_bx >> 2, min_qz = min_bz >> 2, max_qz = max_bz >> 2;
        Volume v{max_qx - min_qx + 1, 1, max_qz - min_qz + 1, min_qx << 2, 0, min_qz << 2, 4, 1, 4};
        std::vector<float> b(static_cast<size_t>(v.size()));
        ev_.sample(surface_level_, v, b.data());
        int max_y = INT_MIN;
        for (int z = 0; z < v.sz; ++z)
            for (int x = 0; x < v.sx; ++x) {
                int level = mth_floor(static_cast<double>(b[static_cast<size_t>(v.index(x, 0, z))]));
                surface_cache_[key(v.bx(x), v.bz(z))] = level;
                if (level > max_y) max_y = level;
            }
        return max_y;
    }

    static int64_t key(int x, int z) { return (static_cast<int64_t>(z) << 32) | static_cast<uint32_t>(x); }

    int surface_level(int bx, int bz) {
        int qx = (bx >> 2) << 2, qz = (bz >> 2) << 2;
        auto it = surface_cache_.find(key(qx, qz));
        if (it != surface_cache_.end()) return it->second;
        int level = mth_floor(static_cast<double>(ev_.point(surface_level_, qx, 0, qz)));
        surface_cache_[key(qx, qz)] = level;
        return level;
    }

    FluidStatus aquifer_status(int index) {
        if (has_status_[static_cast<size_t>(index)]) return status_[static_cast<size_t>(index)];
        int x, y, z;
        unpack(location_[static_cast<size_t>(index)], x, y, z);
        FluidStatus s = compute_fluid(x, y, z);
        status_[static_cast<size_t>(index)] = s;
        has_status_[static_cast<size_t>(index)] = true;
        return s;
    }

    FluidStatus compute_fluid(int x, int y, int z) {
        static const int offsets[13][2] = {{0, 0}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {-3, 0}, {-2, 0}, {-1, 0}, {1, 0}, {-2, 1}, {-1, 1}, {0, 1}, {1, 1}};
        FluidStatus global = global_fluid(x, y, z);
        int lowest = INT_MAX;
        int top = y + 12, bottom = y - 12;
        bool center_under_global = false;
        for (const auto& o : offsets) {
            int sx = x + (o[0] << 4), sz = z + (o[1] << 4);
            int level = surface_level(sx, sz);
            int adjusted = adjust(level);
            bool start = o[0] == 0 && o[1] == 0;
            if (start && bottom > adjusted) return global;
            bool pokes_above = top > adjusted;
            if (pokes_above || start) {
                FluidStatus at_surface = global_fluid(sx, adjusted, sz);
                if (at_surface.at(adjusted, tm_.air) != tm_.air) {
                    if (start) center_under_global = true;
                    if (pokes_above) return at_surface;
                }
            }
            lowest = std::min(lowest, level);
        }
        int level = compute_surface_level(x, y, z, global, lowest, center_under_global);
        return FluidStatus{level, compute_fluid_type(x, y, z, global, level)};
    }

    int compute_surface_level(int x, int y, int z, const FluidStatus& global, int lowest, bool center_under_global) {
        double partially, fully;
        if (ev_.point(exclusion_, x, y, z) > 0.0f) {
            partially = -1.0;
            fully = -1.0;
        } else {
            int below = adjust(lowest) - y;
            double factor = 0.0;
            if (center_under_global) {
                double f = (static_cast<double>(below) - 0.0) / (64.0 - 0.0);      // Mth.clampedMap(below, 0, 64, 1, 0)
                factor = f < 0.0 ? 1.0 : (f > 1.0 ? 0.0 : 1.0 + f * (0.0 - 1.0));
            }
            double noise = mth_dclamp(static_cast<double>(ev_.point(floodedness_, x, y, z)), -1.0, 1.0);
            double fully_t = -0.3 + ((factor - 1.0) / (0.0 - 1.0)) * (0.8 - -0.3);      // Mth.map(factor, 1, 0, -0.3, 0.8)
            double partial_t = -0.8 + ((factor - 1.0) / (0.0 - 1.0)) * (0.4 - -0.8);    // Mth.map(factor, 1, 0, -0.8, 0.4)
            partially = noise - partial_t;
            fully = noise - fully_t;
        }
        if (fully > 0.0) return global.level;
        if (partially > 0.0) return randomized_level(x, y, z, lowest);
        return kWayBelowMinY;
    }

    int randomized_level(int x, int y, int z, int lowest) {
        int cx = floor_div(x, 16), cy = floor_div(y, 40), cz = floor_div(z, 16);
        int middle = cy * 40 + 20;
        double spread = static_cast<double>(ev_.point(spread_, cx, cy, cz) * 10.0f);
        int q = mth_floor(spread / 3.0) * 3;                                         // Mth.quantize(spread, 3)
        return std::min(lowest, middle + q);
    }

    int compute_fluid_type(int x, int y, int z, const FluidStatus& global, int level) {
        int type = global.type;
        if (level <= -10 && level != kWayBelowMinY && global.type != tm_.lava) {
            int cx = floor_div(x, 64), cy = floor_div(y, 40), cz = floor_div(z, 64);
            double lava = static_cast<double>(ev_.point(lava_, cx, cy, cz));
            if (std::fabs(lava) > 0.3) type = tm_.lava;
        }
        return type;
    }

    double pressure(int x, int y, int z, double& barrier_value, bool& barrier_set, const FluidStatus& a, const FluidStatus& b) {
        int t1 = a.at(y, tm_.air), t2 = b.at(y, tm_.air);
        bool lava_water = (t1 == tm_.lava && t2 == tm_.water) || (t1 == tm_.water && t2 == tm_.lava);
        if (lava_water) return 2.0;
        int diff = std::abs(a.level - b.level);
        if (diff == 0) return 0.0;
        double average = 0.5 * static_cast<double>(a.level + b.level);
        double above = static_cast<double>(y) + 0.5 - average;
        double base = static_cast<double>(diff) / 2.0;
        double towards_middle = base - std::fabs(above);
        double gradient;
        if (above > 0.0) {
            double center = 0.0 + towards_middle;
            gradient = center > 0.0 ? center / 1.5 : center / 2.5;
        } else {
            double center = 3.0 + towards_middle;
            gradient = center > 0.0 ? center / 3.0 : center / 10.0;
        }
        double noise;
        if (!(gradient < -2.0) && !(gradient > 2.0)) {
            if (!barrier_set) {
                barrier_value = static_cast<double>(ev_.point(barrier_, x, y, z));
                barrier_set = true;
            }
            noise = barrier_value;
        } else {
            noise = 0.0;
        }
        return 2.0 * (noise + gradient);
    }

    TerrainSeed& ts_;
    VolumeEvaluator& ev_;
    const TerrainModel& tm_;
    int barrier_, floodedness_, spread_, lava_, exclusion_, surface_level_;
    int min_grid_x_, min_grid_y_, min_grid_z_, grid_size_x_, grid_size_z_;
    int skip_above_y_ = 0;
    std::vector<FluidStatus> status_;
    std::vector<bool> has_status_;
    std::vector<int64_t> location_;
    std::unordered_map<int64_t, int> surface_cache_;
};

// ---- Material system ---------------------------------------------------------------------
class Surface {
public:
    Surface(TerrainSeed& ts, VolumeEvaluator& ev, int cx, int cz, std::vector<uint16_t>& blocks, int* heightmap,
            const std::function<int(int, int, int)>& quart_biome)
        : ts_(ts), ev_(ev), tm_(ts.tm), ir_(ts.ir), cx_(cx), cz_(cz), blocks_(blocks), hm_(heightmap), quart_biome_(quart_biome) {
        min_y_ = ir_.min_y;
        max_y_ = ir_.min_y + ir_.height - 1;
        const auto& n = ts.tables.noise;
        surface_ = &n.at("minecraft:surface");
        surface_secondary_ = &n.at("minecraft:surface_secondary");
        clay_offset_ = &n.at("minecraft:clay_bands_offset");
        badlands_pillar_ = &n.at("minecraft:badlands_pillar");
        badlands_roof_ = &n.at("minecraft:badlands_pillar_roof");
        badlands_surface_ = &n.at("minecraft:badlands_surface");
        iceberg_pillar_ = &n.at("minecraft:iceberg_pillar");
        iceberg_roof_ = &n.at("minecraft:iceberg_pillar_roof");
        iceberg_surface_ = &n.at("minecraft:iceberg_surface");
        // Preliminary surface level: chunk_surface_level over (16, 1, 16) at y = 0.
        Volume pv{16, 1, 16, cx * 16, 0, cz * 16, 1, 1, 1};
        prelim_.resize(256);
        ev_.sample(ir_.root("surface/chunk_surface_level"), pv, prelim_.data());
    }

    void build() {
        const int minx = cx_ * 16, minz = cz_ * 16;
        for (int x = 0; x < 16; ++x) {
            for (int z = 0; z < 16; ++z) {
                lx_ = x; lz_ = z;
                int bx = minx + x, bz = minz + z;
                int starting_height = H(x, z) + 1;
                int surface_biome = biome_at(bx, starting_height, bz);
                if (surface_biome == tm_.biome_eroded_badlands) eroded_badlands(bx, bz, starting_height);
                int height = H(x, z) + 1;
                int grad_x = H(std::min(x + 1, 15), z) - H(std::max(x - 1, 0), z);
                int grad_z = H(x, std::min(z + 1, 15)) - H(x, std::max(z - 1, 0));
                update_xz(bx, bz, grad_x, grad_z);
                int stone_above = 0;
                int water_height = INT_MIN;
                int next_ceiling = INT_MAX;
                const int end_y = min_y_;
                for (int y = height; y >= end_y; --y) {
                    int old = get(y);
                    if (tm_.palette[static_cast<size_t>(old)].air) {
                        stone_above = 0;
                        water_height = INT_MIN;
                    } else if (tm_.palette[static_cast<size_t>(old)].fluid) {
                        if (water_height == INT_MIN) water_height = y + 1;
                    } else {
                        if (next_ceiling >= y) {
                            next_ceiling = kWayBelowMinY;
                            for (int la = y - 1; la >= end_y - 1; --la) {
                                if (!is_stone(get(la))) { next_ceiling = la + 1; break; }
                            }
                        }
                        ++stone_above;
                        int stone_below = y - next_ceiling + 1;
                        stone_above_ = stone_above; stone_below_ = stone_below; water_height_ = water_height; by_ = y;
                        biome_cached_ = false;
                        if (y >= min_y_ && y <= max_y_) {
                            int state = apply(tm_.root_rule);
                            if (state >= 0) set(y, state);
                        }
                    }
                }
                if (surface_biome == tm_.biome_frozen_ocean || surface_biome == tm_.biome_deep_frozen_ocean)
                    frozen_ocean(min_surface_level(), surface_biome, bx, bz, starting_height);
            }
        }
    }

private:
    // ChunkAccess.getHeight(WORLD_SURFACE_WG, x, z) = heightmap.getFirstAvailable(x, z) - 1
    int H(int x, int z) const { return hm_[x * 16 + z] - 1; }
    size_t idx(int y) const { return static_cast<size_t>(((y - min_y_) * 16 + lz_) * 16 + lx_); }
    int get(int y) const { return (y < min_y_ || y > max_y_) ? tm_.air : blocks_[idx(y)]; }
    bool is_stone(int s) const { return !tm_.palette[static_cast<size_t>(s)].air && !tm_.palette[static_cast<size_t>(s)].fluid; }

    // protoChunk.setBlockState + Heightmap.update(WORLD_SURFACE_WG, NOT_AIR)
    void set(int y, int state) {
        if (y < min_y_ || y > max_y_) return;
        blocks_[idx(y)] = static_cast<uint16_t>(state);
        int& h = hm_[lx_ * 16 + lz_];
        int first = h;
        if (y <= first - 2) return;
        if (!tm_.palette[static_cast<size_t>(state)].air) {
            if (y >= first) h = y + 1;
        } else if (first - 1 == y) {
            for (int yy = y - 1; yy >= min_y_; --yy)
                if (!tm_.palette[static_cast<size_t>(blocks_[idx(yy)])].air) { h = yy + 1; return; }
            h = min_y_;
        }
    }

    int biome_at(int x, int y, int z) const { return zoomed_biome(ts_.zoom_seed, x, y, z, quart_biome_); }

    double noise_get(const StackRef* s, double x, double y, double z) const {
        return static_cast<double>(stack_get(ts_.tables.layer_ptr(*s), s->count, ts_.tables.octaves.data(), x, y, z));
    }

    void update_xz(int bx, int bz, int gx, int gz) {
        bx_ = bx; bz_ = bz; grad_x_ = gx; grad_z_ = gz;
        double n = noise_get(surface_, bx, 0.0, bz);
        double r = ts_.random.at(bx, 0, bz).next_double();
        surface_depth_ = java_d2i(n * 2.75 + 3.0 + r * 0.25);
        secondary_set_ = false;
        min_surface_set_ = false;
    }

    double surface_secondary() {
        if (!secondary_set_) { secondary_ = noise_get(surface_secondary_, bx_, 0.0, bz_); secondary_set_ = true; }
        return secondary_;
    }

    int min_surface_level() {
        if (!min_surface_set_) {
            float p = prelim_[static_cast<size_t>(lz_ * 16 + lx_)];   // index (x + z * 16) of the (16,1,16) volume
            min_surface_ = mth_floor(static_cast<double>(p)) + surface_depth_ - 8;
            min_surface_set_ = true;
        }
        return min_surface_;
    }

    int biome() {
        if (!biome_cached_) { biome_ = biome_at(bx_, by_, bz_); biome_cached_ = true; }
        return biome_;
    }

    float temperature(int biome, int x, int y, int z) const {
        return ts_.temp.height_adjusted(tm_.biomes[static_cast<size_t>(biome)], x, y, z, tm_.sea_level);
    }

    int band(int x, int y, int z) const {
        int offset = java_round_f(static_cast<float>(noise_get(clay_offset_, x, 0.0, z)) * 4.0f);
        return ts_.clay_bands[(y + offset + 192) % 192];
    }

    bool test(int ci) {
        const Condition& c = tm_.conds[static_cast<size_t>(ci)];
        switch (c.type) {
            case CondType::Biome: {
                int b = biome();
                return (c.biome_mask[b >> 6] >> (b & 63)) & 1ull;
            }
            case CondType::NoiseThreshold: {
                const StackRef& s = ts_.tables.noise.at(c.noise);
                double v = c.is_3d ? noise_get(&s, bx_, by_, bz_) : noise_get(&s, bx_, 0.0, bz_);
                return v >= c.min && v <= c.max;
            }
            case CondType::VerticalGradient: {
                if (by_ <= c.true_at_and_below) return true;
                if (by_ >= c.false_at_and_above) return false;
                double t = (static_cast<double>(by_) - c.true_at_and_below) / static_cast<double>(c.false_at_and_above - c.true_at_and_below);
                double probability = 1.0 + t * (0.0 - 1.0);                        // Mth.map(y, true, false, 1, 0)
                XoroshiroRandom r = ts_.gradient_random.at(c.random_name).at(bx_, by_, bz_);
                return static_cast<double>(r.next_float()) < probability;
            }
            case CondType::YAbove:
                return by_ + (c.add_stone_depth ? stone_above_ : 0) >= c.anchor + surface_depth_ * c.surface_depth_multiplier;
            case CondType::Water:
                return water_height_ == INT_MIN ||
                       by_ + (c.add_stone_depth ? stone_above_ : 0) >= water_height_ + c.offset + surface_depth_ * c.surface_depth_multiplier;
            case CondType::Temperature:
                return !(temperature(biome(), bx_, by_, bz_) >= 0.15f);
            case CondType::Steep:
                return grad_x_ <= -4 || grad_z_ >= 4;
            case CondType::Not:
                return !test(c.child);
            case CondType::Hole:
                return surface_depth_ <= 0;
            case CondType::AbovePreliminarySurface:
                return by_ >= min_surface_level();
            case CondType::StoneDepth: {
                int stone = c.ceiling ? stone_below_ : stone_above_;
                int sd = c.add_surface_depth ? surface_depth_ : 0;
                int secondary = 0;
                if (c.secondary_depth_range != 0) {
                    double t = (surface_secondary() - -1.0) / (1.0 - -1.0);       // Mth.map(sec, -1, 1, 0, range)
                    secondary = java_d2i(0.0 + t * (static_cast<double>(c.secondary_depth_range) - 0.0));
                }
                return stone <= 1 + c.offset + sd + secondary;
            }
        }
        return false;
    }

    float ore_buffer(const std::string& root, std::map<std::string, std::vector<float>>& cache) {
        auto it = cache.find(root);
        if (it == cache.end()) {
            Volume v{16, ir_.height, 16, cx_ * 16, min_y_, cz_ * 16, 1, 1, 1};
            std::vector<float> b(static_cast<size_t>(v.size()));
            ev_.sample(ir_.root(root), v, b.data());
            it = cache.emplace(root, std::move(b)).first;
        }
        Volume v{16, ir_.height, 16, cx_ * 16, min_y_, cz_ * 16, 1, 1, 1};
        return it->second[static_cast<size_t>(v.index(lx_, by_ - min_y_, lz_))];
    }

    int apply(int ri) {
        const Rule& r = tm_.rules[static_cast<size_t>(ri)];
        switch (r.type) {
            case RuleType::Block: return r.state;
            case RuleType::Sequence:
                for (int sub : r.rules) {
                    int s = apply(sub);
                    if (s >= 0) return s;
                }
                return -1;
            case RuleType::Condition: return test(r.cond) ? apply(r.then) : -1;
            case RuleType::Bandlands: return band(bx_, by_, bz_);
            case RuleType::OreVein: {
                float density = ore_buffer(r.density, ore_cache_);
                if (density <= 0.0f) return -1;
                XoroshiroRandom rnd = ts_.ore_random.at(bx_, by_, bz_);
                if (rnd.next_float() > density) return -1;
                float richness = ore_buffer(r.richness, ore_cache_);
                if (rnd.next_float() < richness && ev_.point(ir_.root(r.filler_gap), bx_, by_, bz_) < 0.0f)
                    return rnd.next_float() < r.raw_ore_chance ? r.raw_ore : r.ore;
                return r.filler;
            }
        }
        return -1;
    }

    void eroded_badlands(int bx, int bz, int height) {
        double a = std::fabs(noise_get(badlands_surface_, bx, 0.0, bz) * 8.25);
        double b = static_cast<double>(static_cast<float>(noise_get(badlands_pillar_, bx * 0.2, 0.0, bz * 0.2)) * 15.0f);
        double pillar_buffer = java_dmin(a, b);
        if (pillar_buffer <= 0.0) return;
        double pillar_floor = std::fabs(noise_get(badlands_roof_, bx * 0.75, 0.0, bz * 0.75) * 1.5);
        double top = 64.0 + java_dmin(pillar_buffer * pillar_buffer * 2.5, std::ceil(pillar_floor * 50.0) + 24.0);
        int start_y = mth_floor(top);
        if (height > start_y) return;
        for (int y = start_y; y >= min_y_; --y) {
            int old = get(y);
            if (old == tm_.default_block) break;
            if (tm_.palette[static_cast<size_t>(old)].water) return;
        }
        for (int y = start_y; y >= min_y_ && tm_.palette[static_cast<size_t>(get(y))].air; --y) set(y, tm_.default_block);
    }

    void frozen_ocean(int min_surface, int biome, int bx, int bz, int height) {
        double a = std::fabs(noise_get(iceberg_surface_, bx, 0.0, bz) * 8.25);
        double b = static_cast<double>(static_cast<float>(noise_get(iceberg_pillar_, bx * 1.28, 0.0, bz * 1.28)) * 15.0f);
        double iceberg = java_dmin(a, b);
        if (iceberg <= 1.8) return;
        double roof = std::fabs(noise_get(iceberg_roof_, bx * 1.17, 0.0, bz * 1.17) * 1.5);
        double top = java_dmin(iceberg * iceberg * 1.2, std::ceil(roof * 40.0) + 14.0);
        const int sea = tm_.sea_level;
        if (temperature(biome, bx, sea, bz) > 0.1f) top -= 2.0;
        if (top <= 2.0) return;
        double bottom = static_cast<double>(sea) - top - 7.0;
        top += static_cast<double>(sea);
        double ext_top = top;
        XoroshiroRandom r = ts_.random.at(bx, 0, bz);
        int max_snow = 2 + r.next_int(4);
        int min_snow_height = sea + 18 + r.next_int(10);
        int snow = 0;
        int itop = java_d2i(ext_top), ibottom = java_d2i(bottom);
        for (int y = std::max(height, itop + 1); y >= min_surface; --y) {
            bool place = false;
            if (tm_.palette[static_cast<size_t>(get(y))].air && y < itop && r.next_double() > 0.01) place = true;
            else if (get(y) == tm_.water && y > ibottom && y < sea && r.next_double() > 0.15) place = true;
            if (place) {
                if (snow <= max_snow && y > min_snow_height) {
                    set(y, tm_.snow_block);
                    ++snow;
                } else {
                    set(y, tm_.packed_ice);
                }
            }
        }
    }

    TerrainSeed& ts_;
    VolumeEvaluator& ev_;
    const TerrainModel& tm_;
    const IR& ir_;
    int cx_, cz_;
    std::vector<uint16_t>& blocks_;
    int* hm_;
    const std::function<int(int, int, int)>& quart_biome_;
    int min_y_, max_y_;
    const StackRef *surface_, *surface_secondary_, *clay_offset_, *badlands_pillar_, *badlands_roof_, *badlands_surface_, *iceberg_pillar_, *iceberg_roof_,
        *iceberg_surface_;
    std::vector<float> prelim_;
    std::map<std::string, std::vector<float>> ore_cache_;
    // per-column / per-block context
    int lx_ = 0, lz_ = 0, bx_ = 0, by_ = 0, bz_ = 0, grad_x_ = 0, grad_z_ = 0, surface_depth_ = 0;
    int stone_above_ = 0, stone_below_ = 0, water_height_ = 0;
    bool secondary_set_ = false, min_surface_set_ = false, biome_cached_ = false;
    double secondary_ = 0;
    int min_surface_ = 0, biome_ = 0;
};

}  // namespace

void cpu_generate_terrain(TerrainSeed& ts, int cx, int cz, ChunkTerrain& out) {
    const IR& ir = ts.ir;
    const TerrainModel& tm = ts.tm;
    const int h = ir.height;
    const int qh = h / 4;
    const size_t per = static_cast<size_t>(16 * 16 * h);
    VolumeEvaluator eval(ir, ts.tables);

    // Biomes of the 3x3 neighbourhood (each chunk with a fresh R-tree candidate).
    std::vector<std::vector<uint8_t>> nb(9, std::vector<uint8_t>(static_cast<size_t>(16 * qh)));
    for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) cpu_chunk_biomes(ts, eval, cx + dx, cz + dz, nb[static_cast<size_t>((dz + 1) * 3 + dx + 1)].data());
    out.biomes = nb[4];
    const int qmin = ir.min_y >> 2;
    std::function<int(int, int, int)> quart_biome = [&](int qx, int qy, int qz) {
        int ccx = qx >> 2, ccz = qz >> 2;
        int dx = ccx - cx, dz = ccz - cz;
        if (dx < -1 || dx > 1 || dz < -1 || dz > 1) throw std::runtime_error("biome lookup outside the 3x3 neighbourhood");
        int cqy = std::min(std::max(qy, qmin), qmin + qh - 1) - qmin;
        return static_cast<int>(nb[static_cast<size_t>((dz + 1) * 3 + dx + 1)][static_cast<size_t>((cqy * 4 + (qz & 3)) * 4 + (qx & 3))]);
    };

    // Final density over the chunk volume, then doFill with the aquifer (z, x, y descending).
    Volume v{16, h, 16, cx * 16, ir.min_y, cz * 16, 1, 1, 1};
    eval.clear_caches();
    Aquifer aquifer(ts, eval, v);
    std::vector<float> dens(per);
    eval.sample(ir.root("final_density"), v, dens.data());
    out.density.assign(per, 0.0f);
    out.aquifer.assign(per, static_cast<uint16_t>(tm.air));
    int hm[256];
    for (int& x : hm) x = ir.min_y;
    for (int z = 0; z < 16; ++z)
        for (int x = 0; x < 16; ++x)
            for (int y = h - 1; y >= 0; --y) {
                float d = dens[static_cast<size_t>(v.index(x, y, z))];
                size_t o = static_cast<size_t>((y * 16 + z) * 16 + x);
                out.density[o] = d;
                int s = aquifer.compute_substance(v.bx(x), v.by(y), v.bz(z), static_cast<double>(d));
                if (s < 0) s = tm.default_block;
                out.aquifer[o] = static_cast<uint16_t>(s);
                if (!tm.palette[static_cast<size_t>(s)].air) {
                    int& hh = hm[x * 16 + z];
                    int by = ir.min_y + y;
                    if (!(by <= hh - 2) && by >= hh) hh = by + 1;
                }
            }

    out.terrain = out.aquifer;
    Surface surface(ts, eval, cx, cz, out.terrain, hm, quart_biome);
    surface.build();
}

}  // namespace mcc
