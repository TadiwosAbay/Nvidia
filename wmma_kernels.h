#ifndef WMMA_KERNELS_H
#define WMMA_KERNELS_H

#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace nvcuda;

using binary16_t = half;
using bfloat16_t = nv_bfloat16;

template <typename input_t, typename return_t>
__global__ void wmma_ker(input_t *A, input_t *B, return_t *C, bool init = false);

// Optional: Generic template implementation if needed
template <typename input_t, typename return_t>
__global__ void wmma_ker(input_t *A, input_t *B, return_t *C, bool init) {
    // Generic implementation can go here (if needed)
}

#endif // WMMA_KERNELS_H
