// Biome placement, reimplemented:
//   Climate.ParameterList R-tree (exact vanilla build and search, including the
//   last-result candidate that decides exact ties), chunk biome fill from the
//   climate density functions, and BiomeManager's zoomed per-block lookup.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "mccuda/terrain_model.h"

namespace mcc {

uint64_t sha256_first_long_le(int64_t value);   // Guava Hashing.sha256().hashLong(v).asLong()
int64_t biome_zoom_seed(int64_t world_seed);     // BiomeManager.obfuscateSeed

// Climate.quantizeCoord: (long)(coord * 10000.0F)
int64_t quantize_coord(float coord);

class BiomeTree {
public:
    explicit BiomeTree(const std::vector<BiomeParam>& params);
    // Climate.RTree.search with the given candidate leaf (-1 = none). Returns the leaf index.
    int search(const int64_t target[7], int candidate) const;
    int leaf_biome(int leaf) const { return leaf_biome_[static_cast<size_t>(leaf)]; }
    int64_t leaf_distance(int leaf, const int64_t target[7]) const;

    // Flattened tree for device use: nodes[i] = {param space, first child, child count, leaf index (-1)}.
    struct FlatNode {
        int64_t min[7], max[7];
        int32_t first_child, child_count, leaf;
    };
    const std::vector<FlatNode>& flat() const { return flat_; }
    const std::vector<int32_t>& flat_children() const { return flat_children_; }
    int root() const { return root_; }

private:
    struct Node {
        int64_t min[7], max[7];
        std::vector<int> children;   // node indices
        int leaf = -1;               // leaf index for leaves
    };
    int build(std::vector<int> children);
    int64_t distance(int node, const int64_t target[7]) const;
    int search_node(int node, const int64_t target[7], int candidate_node) const;

    std::vector<Node> nodes_;
    std::vector<int> leaf_node_;     // leaf index -> node index
    std::vector<int> leaf_biome_;
    int root_ = -1;
    std::vector<FlatNode> flat_;
    std::vector<int32_t> flat_children_;
};

// BiomeManager.getBiome(x, y, z) over an accessor of stored quart biomes.
int zoomed_biome(int64_t zoom_seed, int x, int y, int z, const std::function<int(int, int, int)>& quart_biome);

}  // namespace mcc
