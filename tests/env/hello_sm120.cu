// Phase 0 environment probe for the RTX 5090 (sm_120).
//
//   1. "hello world": device printf from an sm_120 kernel.
//   2. FMA-contraction check: a*b + c with inputs chosen so the fused and unfused
//      results differ. Built with -fmad=false the device must match the host.
//   3. Subnormal check: float/double subnormals must survive (no flush-to-zero).
//   4. Throughput microbenchmark: dependent mul+add chains in FP64, FP32, INT32
//      and INT64, to measure how much the consumer FP64 rate matters.
//
// Build (from a VS 2022 x64 developer prompt):
//   nvcc -O3 -arch=sm_120 -fmad=false -o hello_sm120.exe hello_sm120.cu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return 1; } } while (0)

static uint64_t bits(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
static uint32_t bitsf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

__global__ void hello() {
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) printf("hello from sm_%d%d block %d\n", __CUDA_ARCH__ / 100, (__CUDA_ARCH__ / 10) % 10, blockIdx.x);
#endif
}

__global__ void contraction(const double* in, double* out, const float* inf, float* outf) {
    out[0] = in[0] * in[1] + in[2];      // fused only if the compiler contracts
    out[1] = in[3] * in[4];              // double subnormal product
    outf[0] = inf[0] * inf[1];           // float subnormal product
}

// ILP independent chains per thread, ITERS dependent steps each: x = x*a + b.
template <typename T>
__global__ void chain(T* out, T a, T b, int iters) {
    constexpr int ILP = 8;
    T x[ILP];
#pragma unroll
    for (int i = 0; i < ILP; ++i) x[i] = (T)(threadIdx.x + i);
    for (int k = 0; k < iters; ++k) {
#pragma unroll
        for (int i = 0; i < ILP; ++i) x[i] = x[i] * a + b;
    }
    T s = 0;
#pragma unroll
    for (int i = 0; i < ILP; ++i) s += x[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <typename T>
static int bench(const char* name, T a, T b, int blocks, int threads, int iters, double* rate) {
    T* d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(T) * blocks * threads));
    chain<T><<<blocks, threads>>>(d, a, b, 16);               // warm-up
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(e0);
        chain<T><<<blocks, threads>>>(d, a, b, iters);
        cudaEventRecord(e1);
        CHECK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double pairs = (double)blocks * threads * 8.0 * iters;      // one mul + one add each
    *rate = pairs / (best * 1e-3) / 1e9;
    std::printf("  %-6s %9.1f G mul+add pairs/s  (best of 5: %.3f ms)\n", name, *rate, best);
    cudaFree(d);
    return 0;
}

int main() {
    int dev = 0;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    int clk = 0, ratio = 0, rtv = 0, drv = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, dev);
    cudaRuntimeGetVersion(&rtv); cudaDriverGetVersion(&drv);
    std::printf("device: %s  cc %d.%d  SMs %d  clock %d MHz  mem %.1f GiB  FP32:FP64 ratio %d\n",
                p.name, p.major, p.minor, p.multiProcessorCount, clk / 1000,
                p.totalGlobalMem / 1073741824.0, ratio);
    std::printf("runtime %d.%d  driver %d.%d  smem/block opt-in %zu KiB  regs/SM %d  maxThreads/SM %d\n",
                rtv / 1000, (rtv % 1000) / 10, drv / 1000, (drv % 1000) / 10,
                p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor);

    hello<<<2, 32>>>();
    CHECK(cudaDeviceSynchronize());

    // a = 1 + 2^-52, b = 1 - 2^-53, c = -1: unfused a*b rounds to 1.0 so the sum is 0;
    // fused it is 2^-53 - 2^-105.
    volatile double ha = 1.0 + 0x1p-52, hb = 1.0 - 0x1p-53, hc = -1.0;
    volatile double hs1 = 0x1p-540, hs2 = 0x1p-500;       // product 2^-1040 is subnormal
    volatile float hf1 = 0x1p-70f, hf2 = 0x1p-60f;        // product 2^-130 is subnormal
    double hin[5] = {ha, hb, hc, hs1, hs2}, hout[2];
    float hinf[2] = {hf1, hf2}, houtf[1];
    double host_fma = ha * hb + hc;
    double host_sub = hs1 * hs2;
    float host_subf = hf1 * hf2;
    double *din, *dout; float *dinf, *doutf;
    CHECK(cudaMalloc(&din, sizeof hin)); CHECK(cudaMalloc(&dout, sizeof hout));
    CHECK(cudaMalloc(&dinf, sizeof hinf)); CHECK(cudaMalloc(&doutf, sizeof houtf));
    CHECK(cudaMemcpy(din, hin, sizeof hin, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dinf, hinf, sizeof hinf, cudaMemcpyHostToDevice));
    contraction<<<1, 1>>>(din, dout, dinf, doutf);
    CHECK(cudaMemcpy(hout, dout, sizeof hout, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(houtf, doutf, sizeof houtf, cudaMemcpyDeviceToHost));
    bool ok_fma = bits(hout[0]) == bits(host_fma);
    bool ok_sub = bits(hout[1]) == bits(host_sub) && hout[1] != 0.0;
    bool ok_subf = bitsf(houtf[0]) == bitsf(host_subf) && houtf[0] != 0.0f;
    std::printf("a*b+c     host %.17g (0x%016llx)  device %.17g (0x%016llx)  %s\n",
                host_fma, (unsigned long long)bits(host_fma), hout[0], (unsigned long long)bits(hout[0]),
                ok_fma ? "MATCH" : "MISMATCH (contracted to FMA?)");
    std::printf("subnormal double 0x%016llx vs host 0x%016llx %s | float 0x%08x vs host 0x%08x %s\n",
                (unsigned long long)bits(hout[1]), (unsigned long long)bits(host_sub), ok_sub ? "MATCH" : "MISMATCH",
                bitsf(houtf[0]), bitsf(host_subf), ok_subf ? "MATCH" : "MISMATCH");

    std::printf("throughput (dependent x = x*a + b chains, 8 chains/thread):\n");
    int blocks = p.multiProcessorCount * 6, threads = 256;
    double r64, r32, ri32, ri64;
    if (bench<double>("fp64", 0.999999, 1e-7, blocks, threads, 1 << 12, &r64)) return 1;
    if (bench<float>("fp32", 0.999999f, 1e-7f, blocks, threads, 1 << 14, &r32)) return 1;
    if (bench<uint32_t>("int32", 1664525u, 1013904223u, blocks, threads, 1 << 14, &ri32)) return 1;
    if (bench<uint64_t>("int64", 6364136223846793005ull, 1442695040888963407ull, blocks, threads, 1 << 12, &ri64)) return 1;
    std::printf("  ratios: fp32/fp64 %.1fx  int32/fp64 %.1fx  int64/fp64 %.2fx\n", r32 / r64, ri32 / r64, ri64 / r64);

    bool ok = ok_fma && ok_sub && ok_subf;
    std::printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
