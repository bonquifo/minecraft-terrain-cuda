#include "mccuda/biome.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "mccuda/javamath.h"

namespace mcc {

// ---- SHA-256 (FIPS 180-4) --------------------------------------------------------
namespace {

uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t padded = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> m(padded, 0);
    std::memcpy(m.data(), msg, len);
    m[len] = 0x80;
    uint64_t bits = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i) m[padded - 1 - i] = static_cast<uint8_t>(bits >> (8 * i));
    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(m[off + 4 * i]) << 24) | (static_cast<uint32_t>(m[off + 4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(m[off + 4 * i + 2]) << 8) | m[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<uint8_t>(h[i] >> (24 - 8 * j));
}

}  // namespace

uint64_t sha256_first_long_le(int64_t value) {
    uint8_t in[8], out[32];
    for (int i = 0; i < 8; ++i) in[i] = static_cast<uint8_t>(static_cast<uint64_t>(value) >> (8 * i));   // putLong: little-endian
    sha256(in, 8, out);
    uint64_t r = 0;
    for (int i = 7; i >= 0; --i) r = (r << 8) | out[i];                                                 // asLong: little-endian
    return r;
}

int64_t biome_zoom_seed(int64_t world_seed) { return static_cast<int64_t>(sha256_first_long_le(world_seed)); }

int64_t quantize_coord(float coord) {
    float v = coord * 10000.0f;
    // JLS f2l: NaN -> 0, saturate, truncate.
    if (v != v) return 0;
    if (v >= 9.2233720368547758e18f) return std::numeric_limits<int64_t>::max();
    if (v <= -9.2233720368547758e18f) return std::numeric_limits<int64_t>::min();
    return static_cast<int64_t>(v);
}

// ---- R-tree ------------------------------------------------------------------------
namespace {

int64_t center(int64_t mn, int64_t mx) { return (mn + mx) / 2; }   // Java long division truncates toward zero
int64_t labs64(int64_t v) { return v < 0 ? -v : v; }

int64_t param_distance(int64_t mn, int64_t mx, int64_t target) {
    int64_t above = target - mx;
    int64_t below = mn - target;
    return above > 0 ? above : std::max<int64_t>(below, 0);
}

}  // namespace

BiomeTree::BiomeTree(const std::vector<BiomeParam>& params) {
    std::vector<int> leaves;
    for (size_t i = 0; i < params.size(); ++i) {
        Node n;
        for (int d = 0; d < 7; ++d) { n.min[d] = params[i].min[d]; n.max[d] = params[i].max[d]; }
        n.leaf = static_cast<int>(i);
        nodes_.push_back(n);
        leaf_node_.push_back(static_cast<int>(nodes_.size()) - 1);
        leaf_biome_.push_back(params[i].biome);
        leaves.push_back(static_cast<int>(nodes_.size()) - 1);
    }
    root_ = build(leaves);

    // Flatten (node indices are already stable).
    flat_.resize(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        FlatNode& f = flat_[i];
        std::memcpy(f.min, nodes_[i].min, sizeof f.min);
        std::memcpy(f.max, nodes_[i].max, sizeof f.max);
        f.leaf = nodes_[i].leaf;
        f.first_child = static_cast<int32_t>(flat_children_.size());
        f.child_count = static_cast<int32_t>(nodes_[i].children.size());
        for (int c : nodes_[i].children) flat_children_.push_back(c);
    }
}

int BiomeTree::build(std::vector<int> children) {
    const int per_node = 19;
    if (children.empty()) throw std::runtime_error("empty R-tree node");
    if (children.size() == 1) return children[0];
    auto make_subtree = [&](const std::vector<int>& kids) {
        Node n;
        for (int d = 0; d < 7; ++d) {
            n.min[d] = nodes_[static_cast<size_t>(kids[0])].min[d];
            n.max[d] = nodes_[static_cast<size_t>(kids[0])].max[d];
            for (size_t k = 1; k < kids.size(); ++k) {
                n.min[d] = std::min(n.min[d], nodes_[static_cast<size_t>(kids[k])].min[d]);
                n.max[d] = std::max(n.max[d], nodes_[static_cast<size_t>(kids[k])].max[d]);
            }
        }
        n.children = kids;
        nodes_.push_back(n);
        return static_cast<int>(nodes_.size()) - 1;
    };
    // Comparator: center in `dimension` (abs if requested), then the following dimensions cyclically.
    auto sort_by = [&](std::vector<int>& v, int dimension, bool absolute) {
        std::stable_sort(v.begin(), v.end(), [&](int a, int b) {
            for (int k = 0; k < 7; ++k) {
                int d = (dimension + k) % 7;
                int64_t ca = center(nodes_[static_cast<size_t>(a)].min[d], nodes_[static_cast<size_t>(a)].max[d]);
                int64_t cb = center(nodes_[static_cast<size_t>(b)].min[d], nodes_[static_cast<size_t>(b)].max[d]);
                if (absolute) { ca = labs64(ca); cb = labs64(cb); }
                if (ca != cb) return ca < cb;
            }
            return false;
        });
    };
    if (static_cast<int>(children.size()) <= per_node) {
        std::stable_sort(children.begin(), children.end(), [&](int a, int b) {
            int64_t ma = 0, mb = 0;
            for (int d = 0; d < 7; ++d) {
                ma += labs64(center(nodes_[static_cast<size_t>(a)].min[d], nodes_[static_cast<size_t>(a)].max[d]));
                mb += labs64(center(nodes_[static_cast<size_t>(b)].min[d], nodes_[static_cast<size_t>(b)].max[d]));
            }
            return ma < mb;
        });
        return make_subtree(children);
    }
    // (int) Math.pow(19, Math.floor(Math.log(n - 0.01) / Math.log(19)))
    double lg = std::floor(std::log(static_cast<double>(children.size()) - 0.01) / std::log(static_cast<double>(per_node)));
    int expected = static_cast<int>(std::pow(static_cast<double>(per_node), lg));

    int64_t min_cost = std::numeric_limits<int64_t>::max();
    int min_dim = -1;
    std::vector<std::vector<int>> min_buckets;
    for (int d = 0; d < 7; ++d) {
        sort_by(children, d, false);
        std::vector<std::vector<int>> buckets;
        std::vector<int> cur;
        for (int c : children) {
            cur.push_back(c);
            if (static_cast<int>(cur.size()) >= expected) { buckets.push_back(cur); cur.clear(); }
        }
        if (!cur.empty()) buckets.push_back(cur);
        int64_t total = 0;
        for (const auto& b : buckets) {
            for (int dd = 0; dd < 7; ++dd) {
                int64_t mn = nodes_[static_cast<size_t>(b[0])].min[dd], mx = nodes_[static_cast<size_t>(b[0])].max[dd];
                for (size_t k = 1; k < b.size(); ++k) {
                    mn = std::min(mn, nodes_[static_cast<size_t>(b[k])].min[dd]);
                    mx = std::max(mx, nodes_[static_cast<size_t>(b[k])].max[dd]);
                }
                total += labs64(mx - mn);
            }
        }
        if (min_cost > total) { min_cost = total; min_dim = d; min_buckets = buckets; }
    }
    // Buckets become subtrees (for their bounds), sorted by |center| in the chosen dimension.
    std::vector<int> bucket_nodes;
    for (const auto& b : min_buckets) bucket_nodes.push_back(make_subtree(b));
    sort_by(bucket_nodes, min_dim, true);
    std::vector<int> built;
    for (int bn : bucket_nodes) built.push_back(build(nodes_[static_cast<size_t>(bn)].children));
    return make_subtree(built);
}

int64_t BiomeTree::distance(int node, const int64_t target[7]) const {
    const Node& n = nodes_[static_cast<size_t>(node)];
    int64_t d = 0;
    for (int i = 0; i < 7; ++i) {
        int64_t p = param_distance(n.min[i], n.max[i], target[i]);
        d += p * p;
    }
    return d;
}

int64_t BiomeTree::leaf_distance(int leaf, const int64_t target[7]) const { return distance(leaf_node_[static_cast<size_t>(leaf)], target); }

int BiomeTree::search_node(int node, const int64_t target[7], int candidate) const {
    const Node& n = nodes_[static_cast<size_t>(node)];
    if (n.leaf >= 0) return node;
    int64_t min_distance = candidate < 0 ? std::numeric_limits<int64_t>::max() : distance(candidate, target);
    int closest = candidate;
    for (int child : n.children) {
        int64_t child_distance = distance(child, target);
        if (min_distance > child_distance) {
            int leaf = search_node(child, target, closest);
            int64_t leaf_distance = child == leaf ? child_distance : distance(leaf, target);
            if (min_distance > leaf_distance) {
                min_distance = leaf_distance;
                closest = leaf;
            }
        }
    }
    return closest;
}

int BiomeTree::search(const int64_t target[7], int candidate_leaf) const {
    int cand = candidate_leaf < 0 ? -1 : leaf_node_[static_cast<size_t>(candidate_leaf)];
    int node = search_node(root_, target, cand);
    return nodes_[static_cast<size_t>(node)].leaf;
}

// ---- BiomeManager zoom ---------------------------------------------------------------
namespace {

int64_t lcg_next(int64_t rval, int64_t c) {
    uint64_t r = static_cast<uint64_t>(rval);
    r = r * (r * 6364136223846793005ull + 1442695040888963407ull);
    return static_cast<int64_t>(r + static_cast<uint64_t>(c));
}

double fiddle(int64_t rval) {
    int64_t q = rval >> 24;
    int64_t m = q % 1024;
    if (m < 0) m += 1024;                                     // Math.floorMod
    double uniform = static_cast<double>(m) / 1024.0;
    return (uniform - 0.5) * 0.9;
}

double fiddled_distance(int64_t seed, int xr, int yr, int zr, double dx, double dy, double dz) {
    int64_t r = seed;
    r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
    r = lcg_next(r, xr); r = lcg_next(r, yr); r = lcg_next(r, zr);
    double fx = fiddle(r);
    r = lcg_next(r, seed);
    double fy = fiddle(r);
    r = lcg_next(r, seed);
    double fz = fiddle(r);
    double sz = (dz + fz) * (dz + fz);
    double sy = (dy + fy) * (dy + fy);
    double sx = (dx + fx) * (dx + fx);
    return sz + sy + sx;
}

}  // namespace

int zoomed_biome(int64_t zoom_seed, int x, int y, int z, const std::function<int(int, int, int)>& quart_biome) {
    int ax = x - 2, ay = y - 2, az = z - 2;
    int px = ax >> 2, py = ay >> 2, pz = az >> 2;
    double fx = static_cast<double>(ax & 3) / 4.0, fy = static_cast<double>(ay & 3) / 4.0, fz = static_cast<double>(az & 3) / 4.0;
    int min_i = 0;
    double min_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 8; ++i) {
        bool xe = (i & 4) == 0, ye = (i & 2) == 0, ze = (i & 1) == 0;
        int cx = xe ? px : px + 1, cy = ye ? py : py + 1, cz = ze ? pz : pz + 1;
        double dx = xe ? fx : fx - 1.0, dy = ye ? fy : fy - 1.0, dz = ze ? fz : fz - 1.0;
        double d = fiddled_distance(zoom_seed, cx, cy, cz, dx, dy, dz);
        if (min_d > d) { min_i = i; min_d = d; }
    }
    int bx = (min_i & 4) == 0 ? px : px + 1;
    int by = (min_i & 2) == 0 ? py : py + 1;
    int bz = (min_i & 1) == 0 ? pz : pz + 1;
    return quart_biome(bx, by, bz);
}

}  // namespace mcc
