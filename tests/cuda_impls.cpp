#include "mccuda_cuda/generator.h"
#include "mccuda_cuda/terrain_gpu.h"
#include "tests.h"

namespace mcc::test {

std::vector<Impl> cuda_implementations(const IR&) {
    std::vector<Impl> v;
    v.push_back({"gen-host", true, [](const IR& ir, const SeedTables& t, const std::vector<std::pair<int, int>>& chunks, std::vector<float>& d, std::vector<uint8_t>& f) {
                     const size_t per = static_cast<size_t>(16 * 16 * ir.height);
                     d.assign(per * chunks.size(), 0.0f);
                     f.assign(per * chunks.size(), 0);
                     for (size_t i = 0; i < chunks.size(); ++i)
                         cuda::generated_host_chunk(ir, t, chunks[i].first, chunks[i].second, d.data() + i * per, f.data() + i * per);
                 }});
    for (const auto& ver : cuda::versions()) {
        int id = ver.id;
        v.push_back({ver.name, true, [id](const IR& ir, const SeedTables& t, const std::vector<std::pair<int, int>>& chunks, std::vector<float>& d, std::vector<uint8_t>& f) {
                         const size_t per = static_cast<size_t>(16 * 16 * ir.height);
                         d.assign(per * chunks.size(), 0.0f);
                         f.assign(per * chunks.size(), 0);
                         std::vector<cuda::ChunkXZ> c;
                         for (auto& p : chunks) c.push_back({p.first, p.second});
                         cuda::TerrainGenerator gen(ir, id, 256);
                         gen.set_seed(t);
                         gen.generate(c.data(), static_cast<int>(c.size()), d.data(), f.data());
                     }});
    }
    return v;
}

std::vector<TerrainImpl> cuda_terrain_implementations() {
    std::vector<TerrainImpl> v;
    v.push_back({"cuda-terrain", [](const IR& ir, const TerrainModel& tm, int64_t seed, const std::vector<std::pair<int, int>>& chunks, std::vector<uint8_t>& b,
                                    std::vector<uint16_t>& a, std::vector<uint16_t>& t) {
                     const size_t per = static_cast<size_t>(16 * 16 * ir.height), qper = static_cast<size_t>(16 * ir.height / 4);
                     std::vector<cuda::ChunkXZ> c;
                     for (auto& p : chunks) c.push_back({p.first, p.second});
                     std::vector<uint8_t> t8(per * c.size()), a8(per * c.size()), b8(qper * c.size());
                     cuda::FullTerrainGenerator gen(ir, tm, 256);
                     gen.set_seed(seed);
                     gen.generate(c.data(), static_cast<int>(c.size()), t8.data(), b8.data(), a8.data());
                     b.insert(b.end(), b8.begin(), b8.end());
                     a.insert(a.end(), a8.begin(), a8.end());
                     t.insert(t.end(), t8.begin(), t8.end());
                 }});
    return v;
}

}  // namespace mcc::test
