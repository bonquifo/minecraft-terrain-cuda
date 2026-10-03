#include "mccuda/rng.h"

#include <cstring>

namespace mcc {

namespace {

constexpr uint64_t kGoldenRatio64 = 0x9E3779B97F4A7C15ull;   // -7046029254386353131L
constexpr uint64_t kSilverRatio64 = 0x6A09E667F3BCC909ull;   //  7640891576956012809L

uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

}  // namespace

float rng_float_unit() { return 5.9604645E-8f; }
double rng_double_unit() { return static_cast<double>(1.110223E-16f); }

uint64_t mix_stafford13(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;   // -4658895280553007687L
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;   // -7723592293110705685L
    return z ^ (z >> 31);
}

Seed128 upgrade_seed_to_128bit(int64_t legacy_seed) {
    uint64_t lo = static_cast<uint64_t>(legacy_seed) ^ kSilverRatio64;
    uint64_t hi = lo + kGoldenRatio64;
    return {mix_stafford13(lo), mix_stafford13(hi)};
}

Seed128 seed_from_hash_of(const std::string& utf8) {
    uint8_t h[16];
    md5(reinterpret_cast<const uint8_t*>(utf8.data()), utf8.size(), h);
    uint64_t lo = 0, hi = 0;
    for (int i = 0; i < 8; ++i) lo = (lo << 8) | h[i];        // Longs.fromBytes: big-endian
    for (int i = 8; i < 16; ++i) hi = (hi << 8) | h[i];
    return {lo, hi};
}

Xoroshiro128pp::Xoroshiro128pp(uint64_t lo, uint64_t hi) : lo_(lo), hi_(hi) {
    if ((lo_ | hi_) == 0) {
        lo_ = kGoldenRatio64;
        hi_ = kSilverRatio64;
    }
}

uint64_t Xoroshiro128pp::next_long() {
    uint64_t s0 = lo_, s1 = hi_;
    uint64_t result = rotl(s0 + s1, 17) + s0;
    s1 ^= s0;
    lo_ = rotl(s0, 49) ^ s1 ^ (s1 << 21);
    hi_ = rotl(s1, 28);
    return result;
}

XoroshiroRandom::XoroshiroRandom(int64_t seed)
    : gen_(upgrade_seed_to_128bit(seed).lo, upgrade_seed_to_128bit(seed).hi) {}

XoroshiroRandom::XoroshiroRandom(Seed128 seed) : gen_(seed.lo, seed.hi) {}

XoroshiroRandom::XoroshiroRandom(uint64_t lo, uint64_t hi) : gen_(lo, hi) {}

int32_t XoroshiroRandom::next_int(int32_t bound) {
    // Lemire's method with rejection, exactly as XoroshiroRandomSource.nextInt(int).
    uint64_t random_bits = static_cast<uint32_t>(next_int());
    uint64_t multiplied = random_bits * static_cast<uint64_t>(bound);
    uint64_t fractional = multiplied & 0xFFFFFFFFull;
    if (fractional < static_cast<uint64_t>(bound)) {
        uint32_t ubound = static_cast<uint32_t>(bound);
        uint32_t threshold = (~ubound + 1u) % ubound;             // Integer.remainderUnsigned(-bound, bound)
        while (fractional < threshold) {
            random_bits = static_cast<uint32_t>(next_int());
            multiplied = random_bits * static_cast<uint64_t>(bound);
            fractional = multiplied & 0xFFFFFFFFull;
        }
    }
    return static_cast<int32_t>(static_cast<int64_t>(multiplied) >> 32);
}

double XoroshiroRandom::next_double() {
    uint64_t bits = gen_.next_long() >> (64 - 53);
    return static_cast<double>(static_cast<int64_t>(bits)) * rng_double_unit();
}

float XoroshiroRandom::next_float() {
    uint64_t bits = gen_.next_long() >> (64 - 24);
    return static_cast<float>(static_cast<int64_t>(bits)) * rng_float_unit();
}

void XoroshiroRandom::consume_count(int rounds) {
    for (int i = 0; i < rounds; ++i) gen_.next_long();
}

XoroshiroPositional XoroshiroRandom::fork_positional() {
    uint64_t lo = gen_.next_long();
    uint64_t hi = gen_.next_long();
    return XoroshiroPositional(lo, hi);
}

XoroshiroRandom XoroshiroPositional::from_hash_of(const std::string& name) const {
    Seed128 s = seed_from_hash_of(name);
    return XoroshiroRandom(Seed128{s.lo ^ lo_, s.hi ^ hi_});
}

int64_t mth_get_seed(int32_t x, int32_t y, int32_t z) {
    // long seed = x * 3129871 ^ z * 116129781L ^ y;  (x * 3129871 is an int multiply)
    int32_t xm = static_cast<int32_t>(static_cast<uint32_t>(x) * 3129871u);
    uint64_t seed = static_cast<uint64_t>(static_cast<int64_t>(xm)) ^
                    (static_cast<uint64_t>(static_cast<int64_t>(z)) * 116129781ull) ^
                    static_cast<uint64_t>(static_cast<int64_t>(y));
    seed = seed * seed * 42317861ull + seed * 11ull;
    return static_cast<int64_t>(seed) >> 16;
}

XoroshiroRandom XoroshiroPositional::at(int32_t x, int32_t y, int32_t z) const {
    uint64_t positional = static_cast<uint64_t>(mth_get_seed(x, y, z));
    return XoroshiroRandom(positional ^ lo_, hi_);
}

LegacyRandom::LegacyRandom(int64_t seed) : seed_((static_cast<uint64_t>(seed) ^ 25214903917ull) & 0xFFFFFFFFFFFFull) {}

int32_t LegacyRandom::next(int bits) {
    seed_ = (seed_ * 25214903917ull + 11ull) & 0xFFFFFFFFFFFFull;
    return static_cast<int32_t>(static_cast<int64_t>(seed_) >> (48 - bits));
}

int32_t LegacyRandom::next_int(int32_t bound) {
    if ((bound & (bound - 1)) == 0) {
        return static_cast<int32_t>((static_cast<int64_t>(bound) * next(31)) >> 31);
    }
    int32_t sample, modulo;
    do {
        sample = next(31);
        modulo = sample % bound;
    } while (static_cast<int32_t>(static_cast<uint32_t>(sample) - static_cast<uint32_t>(modulo) + static_cast<uint32_t>(bound - 1)) < 0);
    return modulo;
}

int64_t LegacyRandom::next_long() {
    int32_t upper = next(32);
    int32_t lower = next(32);
    return static_cast<int64_t>((static_cast<uint64_t>(static_cast<int64_t>(upper)) << 32) + static_cast<uint64_t>(static_cast<int64_t>(lower)));
}

double LegacyRandom::next_double() {
    int32_t upper = next(26);
    int32_t lower = next(27);
    int64_t combined = (static_cast<int64_t>(upper) << 27) + lower;
    return static_cast<double>(combined) * rng_double_unit();
}

float LegacyRandom::next_float() { return static_cast<float>(next(24)) * rng_float_unit(); }

void LegacyRandom::consume_count(int rounds) {
    for (int i = 0; i < rounds; ++i) next_int();
}

// RFC 1321 MD5.
void md5(const uint8_t* data, size_t len, uint8_t out[16]) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    };
    static const int R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                              5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                              4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    uint32_t h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;

    size_t padded = ((len + 8) / 64 + 1) * 64;
    uint8_t* msg = new uint8_t[padded]();
    std::memcpy(msg, data, len);
    msg[len] = 0x80;
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i) msg[padded - 8 + i] = static_cast<uint8_t>(bit_len >> (8 * i));

    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[16];
        for (int i = 0; i < 16; ++i) {
            w[i] = static_cast<uint32_t>(msg[off + 4 * i]) | (static_cast<uint32_t>(msg[off + 4 * i + 1]) << 8) |
                   (static_cast<uint32_t>(msg[off + 4 * i + 2]) << 16) | (static_cast<uint32_t>(msg[off + 4 * i + 3]) << 24);
        }
        uint32_t a = h0, b = h1, c = h2, d = h3;
        for (int i = 0; i < 64; ++i) {
            uint32_t f;
            int g;
            if (i < 16) { f = (b & c) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
            else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
            else { f = c ^ (b | ~d); g = (7 * i) % 16; }
            uint32_t tmp = d;
            d = c;
            c = b;
            uint32_t x = a + f + K[i] + w[g];
            b = b + ((x << R[i]) | (x >> (32 - R[i])));
            a = tmp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d;
    }
    delete[] msg;
    uint32_t hs[4] = {h0, h1, h2, h3};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<uint8_t>(hs[i] >> (8 * j));
}

}  // namespace mcc
