// Device-side deduplication of 64-bit position keys (CUB radix sort + run marking).
// Neighbouring chunks need many of the same positions (surface-level cells,
// corner columns); deduplicating evaluates each distinct position once, for any
// chunk list, without a host round trip: the unique count stays on the device.
#pragma once

#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mcc::cuda {

namespace dedupe_detail {

__global__ void k_iota(int n, int* v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = i;
}

__global__ void k_heads(int n, const uint64_t* sorted, int* flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = (i == 0 || sorted[i] != sorted[i - 1]) ? 1 : 0;
}

__global__ void k_scatter(int n, const uint64_t* sorted, const int* sorted_idx, const int* flags, const int* incl, uint64_t* unique, int* map,
                          int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int u = incl[i] - 1;
    if (flags[i]) unique[u] = sorted[i];
    map[sorted_idx[i]] = u;
    if (i == n - 1) *count = incl[i];
}

}  // namespace dedupe_detail

class Dedupe {
public:
    // Capacity in keys; key_bits = number of low bits that may be set.
    Dedupe(int capacity, int key_bits) : cap_(capacity), bits_(key_bits) {
        chk(cudaMalloc(&sorted_, sizeof(uint64_t) * cap_), "malloc");
        chk(cudaMalloc(&idx_, sizeof(int) * cap_), "malloc");
        chk(cudaMalloc(&sorted_idx_, sizeof(int) * cap_), "malloc");
        chk(cudaMalloc(&flags_, sizeof(int) * cap_), "malloc");
        chk(cudaMalloc(&incl_, sizeof(int) * cap_), "malloc");
        size_t a = 0, b = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, a, static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
                                        static_cast<const int*>(nullptr), static_cast<int*>(nullptr), cap_, 0, bits_);
        cub::DeviceScan::InclusiveSum(nullptr, b, static_cast<const int*>(nullptr), static_cast<int*>(nullptr), cap_);
        temp_bytes_ = a > b ? a : b;
        chk(cudaMalloc(&temp_, temp_bytes_), "malloc");
    }
    ~Dedupe() {
        void* p[] = {sorted_, idx_, sorted_idx_, flags_, incl_, temp_};
        for (void* q : p) cudaFree(q);
    }
    Dedupe(const Dedupe&) = delete;
    Dedupe& operator=(const Dedupe&) = delete;

    // unique[0 .. *count) = distinct keys in increasing order; map[i] = index of keys[i] in unique.
    void run(const uint64_t* keys, int n, uint64_t* unique, int* map, int* count, cudaStream_t s = 0) {
        if (n > cap_) throw std::runtime_error("dedupe capacity exceeded");
        if (n == 0) { chk(cudaMemsetAsync(count, 0, sizeof(int), s), "memset"); return; }
        const int tb = 256, g = (n + tb - 1) / tb;
        dedupe_detail::k_iota<<<g, tb, 0, s>>>(n, idx_);
        size_t t = temp_bytes_;
        chk(cub::DeviceRadixSort::SortPairs(temp_, t, keys, sorted_, idx_, sorted_idx_, n, 0, bits_, s), "radix sort");
        dedupe_detail::k_heads<<<g, tb, 0, s>>>(n, sorted_, flags_);
        t = temp_bytes_;
        chk(cub::DeviceScan::InclusiveSum(temp_, t, flags_, incl_, n, s), "scan");
        dedupe_detail::k_scatter<<<g, tb, 0, s>>>(n, sorted_, sorted_idx_, flags_, incl_, unique, map, count);
        chk(cudaGetLastError(), "dedupe kernels");
    }

private:
    static void chk(cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in dedupe ") + what + ": " + cudaGetErrorString(e));
    }
    int cap_, bits_;
    uint64_t* sorted_ = nullptr;
    int *idx_ = nullptr, *sorted_idx_ = nullptr, *flags_ = nullptr, *incl_ = nullptr;
    void* temp_ = nullptr;
    size_t temp_bytes_ = 0;
};

}  // namespace mcc::cuda
