// Toolchain smoke test: a kernel compiled to a gfx1100 code object and launched through the HIP module API.
#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy(float * __restrict__ y, const float * __restrict__ x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
