#include "wmma_kernels.h"

// Specialization for binary16_t
template <>
__global__ void wmma_ker<binary16_t, float>(binary16_t *A, binary16_t *B, float *C, bool init) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, binary16_t, wmma::row_major> A_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, binary16_t, wmma::col_major> B_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_fragment;

    wmma::load_matrix_sync(A_fragment, A, 16);
    wmma::load_matrix_sync(B_fragment, B, 16);
    if (init)
        wmma::fill_fragment(C_fragment, 0.0f);
    else
        wmma::load_matrix_sync(C_fragment, C, 16, wmma::mem_col_major);

    wmma::mma_sync(C_fragment, A_fragment, B_fragment, C_fragment);
    wmma::store_matrix_sync(C, C_fragment, 16, wmma::mem_col_major);
}

// Specialization for binary32_t (float)
// template <>
// __global__ void wmma_ker<binary32_t, float>(binary32_t *A, binary32_t *B, float *C, bool init) {
//     // Similar implementation for binary32_t
// }

// Specialization for binary64_t (double)
// template <>
// __global__ void wmma_ker<binary64_t, double>(binary64_t *A, binary64_t *B, double *C, bool init) {
//     // Similar implementation for binary64_t
// }
