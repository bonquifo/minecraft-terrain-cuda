// CPU reference port: evaluates the IR over block volumes with the same
// structure as vanilla's DensitySampler.sampleVolume implementations (same
// buffer orders, interpolation cell volumes, slices and caches). This is the
// debugging baseline the GPU kernels are compared against.
#pragma once

#include <cstdint>
#include <vector>

#include "mccuda/df_ops.h"
#include "mccuda/ir.h"
#include "mccuda/seed_tables.h"

namespace mcc {

// DensityVolume
struct Volume {
    int sx, sy, sz;
    int minx, miny, minz;
    int stx = 1, sty = 1, stz = 1;

    int size() const { return sx * sy * sz; }
    int index(int x, int y, int z) const { return y + (x + z * sx) * sy; }
    int bx(int i) const { return minx + i * stx; }
    int by(int i) const { return miny + i * sty; }
    int bz(int i) const { return minz + i * stz; }
    int max_bx() const { return minx + sx * stx - 1; }
    int max_by() const { return miny + sy * sty - 1; }
    int max_bz() const { return minz + sz * stz - 1; }
    bool operator==(const Volume& o) const {
        return sx == o.sx && sy == o.sy && sz == o.sz && minx == o.minx && miny == o.miny && minz == o.minz &&
               stx == o.stx && sty == o.sty && stz == o.stz;
    }
};

// How vanilla's BinaryFunction.compileSampler resolved a node.
struct BinaryPlan {
    enum Kind { Add, Sub, Mul, Div, DivByConst, Min, Max, PassLeft, PassRight } kind;
    int first = -1, second = -1;   // operand nodes; for MIN/MAX `first` initialises the buffer
    float inv = 0.0f;              // DivByConst: 1.0F / constant
};

BinaryPlan binary_plan(const IR& ir, const Node& n);
GradientPlan gradient_plan(const Node& n);

class VolumeEvaluator {
public:
    VolumeEvaluator(const IR& ir, const SeedTables& tables);
    void sample(int node, const Volume& v, float* out);
    void clear_caches();

private:
    void sample_noise_plain(const StackRef& s, const Volume& v, double xz, double y, float amplitude, float* acc);
    void sample_interpolated(const Node& n, const Volume& v, float* out);
    void sample_interpolated_block_step(const Node& n, const Volume& v, float* out);
    void sample_slice(const Node& n, const Volume& v, float* out);
    float spline_eval(const Spline& s, int index, std::vector<std::vector<float>>& coords, std::vector<int>& coord_ids, const Volume& v);

    const IR& ir_;
    const SeedTables& t_;
    std::vector<BinaryPlan> bin_;
    std::vector<GradientPlan> grad_;
    struct CacheCell {
        bool valid = false;
        Volume vol{};
        std::vector<float> buf;
    };
    std::vector<CacheCell> caches_;
};

// Canonical chunk output layout: index = ((y - minY) * 16 + z) * 16 + x.
enum BlockId : uint8_t { kAir = 0, kStone = 1, kWater = 2, kLava = 3 };

// Vanilla doFill substance for the core configuration (aquifers disabled):
// density > 0 -> default block, else the global fluid picker.
MCC_HD MCC_FI uint8_t core_fill(float density, int y, int sea_level) {
    if (static_cast<double>(density) > 0.0) return kStone;
    int lava_level = sea_level < -54 ? sea_level : -54;     // Math.min(-54, seaLevel)
    if (y < lava_level) return y < -54 ? kLava : kAir;
    return y < sea_level ? kWater : kAir;
}

// Generates one chunk with the CPU reference port.
void cpu_generate_chunk(const IR& ir, VolumeEvaluator& eval, int cx, int cz, float* density, uint8_t* fill);

}  // namespace mcc
