#include <cuda_runtime.h>
#include <cstdio>

namespace mcc::cuda {
void print_device() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);
}
}  // namespace mcc::cuda
