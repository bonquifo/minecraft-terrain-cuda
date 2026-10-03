// Vanilla 26.3 random sources, reimplemented:
//   Xoroshiro128PlusPlus / XoroshiroRandomSource (+ positional factory)
//   LegacyRandomSource (java.util.Random LCG, BitRandomSource defaults)
//   RandomSupport: Stafford13 mixing, 128-bit seed upgrade, MD5 seedFromHashOf
#pragma once

#include <cstdint>
#include <string>

namespace mcc {

struct Seed128 {
    uint64_t lo, hi;
};

uint64_t mix_stafford13(uint64_t z);
Seed128 upgrade_seed_to_128bit(int64_t legacy_seed);
Seed128 seed_from_hash_of(const std::string& utf8);
void md5(const uint8_t* data, size_t len, uint8_t out[16]);

class Xoroshiro128pp {
public:
    Xoroshiro128pp(uint64_t lo, uint64_t hi);
    uint64_t next_long();

private:
    uint64_t lo_, hi_;
};

class XoroshiroPositional;

// XoroshiroRandomSource
class XoroshiroRandom {
public:
    explicit XoroshiroRandom(int64_t seed);          // upgradeSeedTo128bit (mixed)
    explicit XoroshiroRandom(Seed128 seed);          // raw 128-bit seed
    XoroshiroRandom(uint64_t lo, uint64_t hi);

    int64_t next_long() { return static_cast<int64_t>(gen_.next_long()); }
    int32_t next_int() { return static_cast<int32_t>(gen_.next_long()); }
    int32_t next_int(int32_t bound);
    double next_double();
    float next_float();
    void consume_count(int rounds);
    XoroshiroPositional fork_positional();

private:
    Xoroshiro128pp gen_;
};

// XoroshiroRandomSource.XoroshiroPositionalRandomFactory
class XoroshiroPositional {
public:
    XoroshiroPositional(uint64_t lo, uint64_t hi) : lo_(lo), hi_(hi) {}
    XoroshiroRandom from_hash_of(const std::string& name) const;
    XoroshiroRandom at(int32_t x, int32_t y, int32_t z) const;
    uint64_t lo() const { return lo_; }
    uint64_t hi() const { return hi_; }

private:
    uint64_t lo_, hi_;
};

// Mth.getSeed(x, y, z)
int64_t mth_get_seed(int32_t x, int32_t y, int32_t z);

// LegacyRandomSource (48-bit LCG) with BitRandomSource's derived methods.
class LegacyRandom {
public:
    explicit LegacyRandom(int64_t seed);
    int32_t next(int bits);
    int32_t next_int() { return next(32); }
    int32_t next_int(int32_t bound);
    int64_t next_long();
    double next_double();
    float next_float();
    void consume_count(int rounds);

private:
    uint64_t seed_;
};

// Bit patterns of vanilla's float literals used as RNG scale factors
// (5.9604645E-8F and 1.110223E-16F); checked against the oracle.
float rng_float_unit();
double rng_double_unit();

}  // namespace mcc
