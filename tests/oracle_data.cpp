#include "oracle_data.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace mcc::test {

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::streamsize n = in.tellg();
    in.seekg(0);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    in.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

static std::vector<float> as_floats(const std::vector<uint8_t>& b) {
    std::vector<float> f(b.size() / 4);
    std::memcpy(f.data(), b.data(), f.size() * 4);
    return f;
}

std::vector<float> OracleSet::density(const OracleSeed& s) const {
    auto f = as_floats(read_file(dir + "/density/s" + std::to_string(s.index) + ".f32"));
    if (f.size() != s.chunks.size() * chunk_floats()) throw std::runtime_error("density size mismatch");
    return f;
}

std::vector<uint8_t> OracleSet::fill(const OracleSeed& s) const {
    auto b = read_file(dir + "/fill/s" + std::to_string(s.index) + ".u8");
    if (b.size() != s.chunks.size() * chunk_floats()) throw std::runtime_error("fill size mismatch");
    return b;
}

std::vector<float> OracleSet::node(const std::string& dump, const std::string& file) const {
    return as_floats(read_file(dir + "/nodes/" + dump + "/" + file));
}

OracleSet load_oracle_set(const std::string& dir) {
    std::ifstream in(dir + "/manifest.json");
    if (!in) throw std::runtime_error("no oracle manifest in " + dir + " (run oracle/regen.ps1)");
    nlohmann::json m = nlohmann::json::parse(in);
    if (m["format"] != "mccuda-oracle/1") throw std::runtime_error("unexpected oracle format");
    OracleSet set;
    set.dir = dir;
    set.min_y = m["min_y"];
    set.height = m["height"];
    for (const auto& n : m["named_functions"]) set.named.emplace_back(n["name"], n["file"]);
    for (const auto& s : m["seeds"]) {
        OracleSeed os;
        os.index = s["index"];
        os.seed = std::stoll(s["seed"].get<std::string>());
        os.quick = s["quick"];
        for (const auto& c : s["chunks"]) os.chunks.push_back({c[0].get<int>(), c[1].get<int>(), c[2].get<std::string>()});
        for (const auto& d : s["node_dumps"]) os.node_dumps.push_back(d);
        set.seeds.push_back(std::move(os));
    }
    return set;
}

}  // namespace mcc::test
