// Minimal copy/compute overlap probe: a ~20 ms busy kernel on one stream and a large
// pinned device-to-host copy on another. Prints how much the two overlapped.
//   nvcc -O2 -o overlap_probe overlap_probe.cu && overlap_probe
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

__global__ void spin(long long cycles, int* out) {
    long long start = clock64();
    while (clock64() - start < cycles) {
    }
    if (threadIdx.x == 0) out[blockIdx.x] = 1;
}

int main() {
    const size_t bytes = size_t{400} << 20;
    char *dev = nullptr, *host = nullptr;
    int* flag = nullptr;
    cudaMalloc(&dev, bytes);
    cudaMallocHost(&host, bytes);
    cudaMalloc(&flag, sizeof(int) * 1024);
    cudaStream_t a, b;
    cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
    cudaEvent_t k0, k1, c0, c1;
    cudaEventCreate(&k0); cudaEventCreate(&k1); cudaEventCreate(&c0); cudaEventCreate(&c1);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    const long long cycles = static_cast<long long>(clock_khz) * 20;   // ~20 ms at the base clock
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        cudaEventRecord(k0, a);
        spin<<<170, 128, 0, a>>>(cycles, flag);
        cudaEventRecord(k1, a);
        cudaEventRecord(c0, b);
        cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, b);
        cudaEventRecord(c1, b);
        cudaDeviceSynchronize();
        double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        float tk = 0, tc = 0, gap = 0;
        cudaEventElapsedTime(&tk, k0, k1);
        cudaEventElapsedTime(&tc, c0, c1);
        cudaEventElapsedTime(&gap, k0, c1);
        std::printf("rep %d: kernel %.2f ms, copy %.2f ms (%.1f GB/s), kernel start -> copy end %.2f ms, wall %.2f ms -> %s\n", rep, tk, tc,
                    bytes / (tc * 1e6), gap, wall, wall < (tk + tc) * 0.9 ? "OVERLAPPED" : "SERIALIZED");
    }
    std::printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
