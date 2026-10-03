// Level-0 tests: Java-semantics helpers, RNG streams, per-seed noise tables and
// noise samples, all compared bit-for-bit against the oracle dumps.
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "check.h"
#include "mccuda/df_ops.h"
#include "mccuda/noise_eval.h"
#include "mccuda/rng.h"
#include "mccuda/seed_tables.h"
#include "mccuda/volume_eval.h"
#include "nlohmann/json.hpp"
#include "tests.h"

namespace mcc::test {

using json = nlohmann::json;

namespace {

int64_t jl(const json& j) { return std::stoll(j.get<std::string>()); }
uint64_t jhex(const json& j) { return std::stoull(j.get<std::string>(), nullptr, 16); }

std::string hex16(const uint8_t* h) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 16; ++i) { s += d[h[i] >> 4]; s += d[h[i] & 15]; }
    return s;
}

}  // namespace

bool run_unit() {
    std::printf("[unit] Java semantics, gradient table, MD5, host FP contraction\n");
    Check c("unit");
    for (int h = 0; h < 16; ++h)
        for (int k = 0; k < 3; ++k) c.expect(grad_c(h, k) == kGradTable[h][k], "grad_c vs table");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    c.expect(java_d2i(nan) == 0, "d2i NaN");
    c.expect(java_d2i(1e10) == 2147483647, "d2i +sat");
    c.expect(java_d2i(-1e10) == -2147483647 - 1, "d2i -sat");
    c.expect(java_d2i(-0.5) == 0, "d2i trunc");
    c.expect(java_d2i(2147483647.5) == 2147483647, "d2i near max");
    c.expect(mth_floor(-0.5) == -1, "floor -0.5");
    c.expect(mth_floor(-1e-300) == -1, "floor tiny negative");
    c.expect(mth_floor(3e9) == 2147483647, "floor saturate");
    c.expect(floor_div(-1, 4) == -1 && floor_mod(-1, 4) == 3 && floor_div(-8, 8) == -1 && floor_mod(7, 4) == 3, "floorDiv/Mod");

    const float nz = -0.0f, pz = 0.0f, fnan = std::numeric_limits<float>::quiet_NaN();
    c.expect(f2bits(java_fmin(pz, nz)) == 0x80000000u, "min(+0,-0)");
    c.expect(f2bits(java_fmin(nz, pz)) == 0x80000000u, "min(-0,+0)");
    c.expect(f2bits(java_fmax(nz, pz)) == 0x00000000u, "max(-0,+0)");
    c.expect(f2bits(java_fmax(pz, nz)) == 0x00000000u, "max(+0,-0)");
    c.expect(std::isnan(java_fmin(fnan, 1.0f)) && std::isnan(java_fmin(1.0f, fnan)), "min NaN");
    c.expect(f2bits(java_fabs(nz)) == 0u, "abs(-0)");
    c.expect(f2bits(vol_min(pz, nz)) == 0x00000000u, "volume min keeps first on tie");

    uint8_t h[16];
    md5(reinterpret_cast<const uint8_t*>(""), 0, h);
    c.expect(hex16(h) == "d41d8cd98f00b204e9800998ecf8427e", "md5 empty");
    const char* fox = "The quick brown fox jumps over the lazy dog";
    md5(reinterpret_cast<const uint8_t*>(fox), std::strlen(fox), h);
    c.expect(hex16(h) == "9e107d9d372bb6826bd81d3542a419d6", "md5 fox");
    std::string big(1000, 'a');
    md5(reinterpret_cast<const uint8_t*>(big.data()), big.size(), h);
    c.expect(hex16(h) == "cabe45dcc9ae5b66ba86600cca6b8ba8", "md5 1000a");

    c.expect(f2bits(rng_float_unit()) == 0x33800000u, "float unit = 2^-24");
    c.expect(d2bits(rng_double_unit()) == 0x3CA0000000000000ull, "double unit = 2^-53");
    c.expect(d2bits(kHalfRoundOff) == 0x416FFFFFFFFFFFFFull, "HALF_ROUND_OFF = nextDown(2^24)");

    // Host FMA contraction probe (same construction as tests/env/hello_sm120.cu).
    volatile double a = 1.0 + 0x1p-52, b = 1.0 - 0x1p-53, cc = -1.0;
    double r = a * b + cc;
    c.expect(r == 0.0, "host a*b+c not contracted");
    volatile float fa = 1.0f + 0x1p-23f, fb = 1.0f - 0x1p-24f, fc = -1.0f;
    float fr = fa * fb + fc;
    c.expect(fr == 0.0f, "host float a*b+c not contracted");
    c.print();
    return c.ok();
}

bool run_rng(const std::string& path) {
    std::printf("[rng] %s\n", path.c_str());
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    json j = json::parse(in);
    Check ch("seedFromHashOf (MD5)"), cx("Xoroshiro128++ source"), cl("legacy LCG source");
    for (const auto& e : j["hash"]) {
        Seed128 s = seed_from_hash_of(e["input"].get<std::string>());
        ++ch.total;
        if (static_cast<int64_t>(s.lo) != jl(e["lo"]) || static_cast<int64_t>(s.hi) != jl(e["hi"])) ch.fail("hash of '%s'", e["input"].get<std::string>().c_str());
    }
    std::vector<int> bounds;
    for (const auto& b : j["bounds"]) bounds.push_back(b.get<int>());
    const std::vector<std::string> hash_inputs = [&] {
        std::vector<std::string> v;
        for (const auto& e : j["hash"]) v.push_back(e["input"]);
        return v;
    }();
    for (const auto& e : j["xoroshiro"]) {
        int64_t seed = jl(e["seed"]);
        Seed128 up = upgrade_seed_to_128bit(seed);
        cx.total += 3;
        if (static_cast<int64_t>(up.lo) != jl(e["upgraded_lo"]) || static_cast<int64_t>(up.hi) != jl(e["upgraded_hi"])) cx.fail("upgrade seed %lld", static_cast<long long>(seed));
        if (static_cast<int64_t>(mix_stafford13(static_cast<uint64_t>(seed))) != jl(e["stafford13"])) cx.fail("stafford13 %lld", static_cast<long long>(seed));
        XoroshiroRandom r(seed);
        for (const auto& v : e["next_long"]) { ++cx.total; if (r.next_long() != jl(v)) cx.fail("next_long seed %lld", static_cast<long long>(seed)); }
        r = XoroshiroRandom(seed);
        size_t k = 0;
        for (int b : bounds) for (int i = 0; i < 4; ++i, ++k) { ++cx.total; if (r.next_int(b) != e["next_int_bounded"][k].get<int>()) cx.fail("next_int(%d)", b); }
        r = XoroshiroRandom(seed);
        for (const auto& v : e["next_double"]) { ++cx.total; if (d2bits(r.next_double()) != jhex(v)) cx.fail("next_double seed %lld", static_cast<long long>(seed)); }
        r = XoroshiroRandom(seed);
        for (const auto& v : e["next_float"]) { ++cx.total; if (f2bits(r.next_float()) != v.get<uint32_t>()) cx.fail("next_float"); }
        XoroshiroPositional pos = XoroshiroRandom(seed).fork_positional();
        for (size_t i = 0; i < hash_inputs.size(); ++i) { ++cx.total; if (pos.from_hash_of(hash_inputs[i]).next_long() != jl(e["positional_hash"][i])) cx.fail("positional fromHashOf"); }
        const int at[4][3] = {{0, 0, 0}, {1, -64, -1}, {-29999984, 319, 29999984}, {123456, 7, -654321}};
        for (int i = 0; i < 4; ++i) { ++cx.total; if (pos.at(at[i][0], at[i][1], at[i][2]).next_long() != jl(e["positional_at"][i])) cx.fail("positional at"); }
    }
    for (const auto& e : j["legacy"]) {
        int64_t seed = jl(e["seed"]);
        LegacyRandom r(seed);
        for (const auto& v : e["next_long"]) { ++cl.total; if (r.next_long() != jl(v)) cl.fail("next_long"); }
        r = LegacyRandom(seed);
        size_t k = 0;
        for (int b : bounds) for (int i = 0; i < 4; ++i, ++k) { ++cl.total; if (r.next_int(b) != e["next_int_bounded"][k].get<int>()) cl.fail("next_int(%d)", b); }
        r = LegacyRandom(seed);
        for (const auto& v : e["next_double"]) { ++cl.total; if (d2bits(r.next_double()) != jhex(v)) cl.fail("next_double"); }
        r = LegacyRandom(seed);
        for (const auto& v : e["next_float"]) { ++cl.total; if (f2bits(r.next_float()) != v.get<uint32_t>()) cl.fail("next_float"); }
        r = LegacyRandom(seed);
        r.consume_count(262);
        ++cl.total;
        if (r.next_long() != jl(e["after_consume_262"])) cl.fail("consumeCount(262)");
    }
    ch.print(); cx.print(); cl.print();
    return ch.ok() && cx.ok() && cl.ok();
}

namespace {

void compare_stack(Check& c, const SeedTables& t, const StackRef& s, const json& expected, const std::string& label) {
    const auto& layers = expected["layers"];
    ++c.total;
    if (static_cast<int>(layers.size()) != s.count) { c.fail("%s layer count %d vs %d", label.c_str(), s.count, static_cast<int>(layers.size())); return; }
    for (int i = 0; i < s.count; ++i) {
        const StackLayer& L = t.layers[static_cast<size_t>(s.first_layer + i)];
        const PerlinOctave& o = t.octaves[static_cast<size_t>(L.octave)];
        const json& e = layers[static_cast<size_t>(i)];
        c.total += 6;
        if (d2bits(L.frequency) != jhex(e["frequency"])) c.fail("%s layer %d frequency", label.c_str(), i);
        if (f2bits(L.amplitude) != e["amplitude"].get<uint32_t>()) c.fail("%s layer %d amplitude %08x vs %08x", label.c_str(), i, f2bits(L.amplitude), e["amplitude"].get<uint32_t>());
        if (d2bits(o.ox) != jhex(e["offset"][0]) || d2bits(o.oy) != jhex(e["offset"][1]) || d2bits(o.oz) != jhex(e["offset"][2])) c.fail("%s layer %d offsets", label.c_str(), i);
        std::string perms = e["perms"];
        bool pok = true;
        for (int k = 0; k < 256; ++k) pok &= std::stoi(perms.substr(static_cast<size_t>(2 * k), 2), nullptr, 16) == o.perm[k];
        if (!pok) c.fail("%s layer %d permutation", label.c_str(), i);
        bool smeared = e["class"] == "SmearedPerlinNoise";
        if (smeared != s.smeared) c.fail("%s layer %d class", label.c_str(), i);
        if (smeared && d2bits(o.smear) != jhex(e["smear"])) c.fail("%s layer %d smear", label.c_str(), i);
    }
    const StackLayer* lp = t.layer_ptr(s);
    for (const auto& p : expected["points"]) {
        double x = bits2d(jhex(p[0])), y = bits2d(jhex(p[1])), z = bits2d(jhex(p[2]));
        float v = s.smeared ? stack_get_smeared(lp, s.count, t.octaves.data(), x, y, z) : stack_get(lp, s.count, t.octaves.data(), x, y, z);
        ++c.total;
        if (f2bits(v) != p[3].get<uint32_t>()) c.fail("%s get(%.17g, %.17g, %.17g) %08x vs %08x", label.c_str(), x, y, z, f2bits(v), p[3].get<uint32_t>());
    }
    for (const auto& vol : expected["volumes"]) {
        const auto& d = vol["volume"];
        Volume v{d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]};
        double xz = bits2d(jhex(vol["xz_scale"])), ys = bits2d(jhex(vol["y_scale"]));
        float amp = bits2f(vol["amplitude"].get<uint32_t>());
        int index = 0;
        for (int iz = 0; iz < v.sz; ++iz)
            for (int ix = 0; ix < v.sx; ++ix)
                for (int iy = 0; iy < v.sy; ++iy, ++index) {
                    float got = stack_add_volume(0.0f, lp, s.count, t.octaves.data(), s.smeared, v.bx(ix), v.by(iy), v.bz(iz), xz, ys, amp);
                    ++c.total;
                    if (f2bits(got) != vol["values"][static_cast<size_t>(index)].get<uint32_t>()) c.fail("%s addToVolume at (%d,%d,%d) scale %g/%g", label.c_str(), v.bx(ix), v.by(iy), v.bz(iz), xz, ys);
                }
    }
}

}  // namespace

bool run_noise(const IR& ir, const std::string& dir, bool quick) {
    std::printf("[noise] %s\n", dir.c_str());
    Check c("noise tables + samples");
    int files = 0;
    for (int idx = 0; idx < 64; ++idx) {
        std::ifstream in(dir + "/s" + std::to_string(idx) + ".json");
        if (!in) continue;
        json j = json::parse(in);
        int64_t seed = jl(j["seed"]);
        if (quick && files >= 2) break;
        SeedTables t = build_seed_tables(ir, seed);
        for (auto& [key, val] : j["noises"].items()) compare_stack(c, t, t.noise.at(key), val, key);
        if (t.blended.size() != j["blended"].size()) c.fail("blended noise count");
        for (auto& [key, val] : j["blended"].items()) {
            const BlendedSet& b = t.blended.begin()->second;
            compare_stack(c, t, b.min_limit, val["min_limit"], "blended.min_limit");
            compare_stack(c, t, b.max_limit, val["max_limit"], "blended.max_limit");
            compare_stack(c, t, b.main, val["main"], "blended.main");
        }
        ++files;
    }
    std::printf("  seeds checked: %d\n", files);
    c.print();
    return c.ok() && files > 0;
}

}  // namespace mcc::test
