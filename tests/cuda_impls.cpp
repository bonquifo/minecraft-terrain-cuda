#include "mccuda_cuda/generator.h"
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

std::vector<TerrainImpl> cuda_terrain_implementations() { return {}; }

}  // namespace mcc::test
