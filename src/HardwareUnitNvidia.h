#ifndef HARDWARE_UNIT_NVIDIA_H
#define HARDWARE_UNIT_NVIDIA_H

#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "HardwareUnit.h"

using namespace nvcuda;

using binary16_t = half;
using bfloat16_t = nv_bfloat16;
using binary16 = IEEEFloatFormat<binary16_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<bfloat16_t, 8, 127>;

template <> std::string get_type_name<bfloat16_t>() {return "bfloat16";}
template <> std::string get_type_name<binary16_t>() {return "binary16 (half)";}

#include "fp_utils.h"

// Overload << operator for half type (binary16_t)
std::ostream& operator<<(std::ostream& os, const half& h_value) {
    float f_value = __half2float(h_value);
    os << f_value;
    return os;
}

// Overload << operator for bfloat16_t type
std::ostream& operator<<(std::ostream& os, const nv_bfloat16& bf_value) {
    float f_value = __bfloat162float(bf_value);
    os << f_value;
    return os;
}

/* std::ostream& operator<<(std::ostream& os, const half& h_value) {
    unsigned short us_value = *(unsigned short *)(&h_value);
    for (int i = 15; i >= 0; i--) {
        os << ((us_value >> i) & 1);
        if (i == 15 || i == 10)
            os << " ";
    }
    return os;
} */

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */

template <typename output_t>
__global__ void wmma_ker(binary16_t *A, binary16_t *B, output_t *C, bool init = false) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, binary16_t, wmma::row_major> A_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, binary16_t, wmma::col_major> B_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, output_t> C_fragment;

    wmma::load_matrix_sync(A_fragment, A, 16);
    wmma::load_matrix_sync(B_fragment, B, 16);
    if (init)
        wmma::fill_fragment(C_fragment, 0.0f);
    else
        wmma::load_matrix_sync(C_fragment, C, 16, wmma::mem_col_major);

    wmma::mma_sync(C_fragment, A_fragment, B_fragment, C_fragment);
    wmma::store_matrix_sync(C, C_fragment, 16, wmma::mem_col_major);
}

template <typename InputFormat, typename OutputFormat>
class HardwareUnitNvidia : public HardwareUnit<InputFormat, OutputFormat> {

    using input_t = typename InputFormat::storageFormat;
    using output_t = typename OutputFormat::storageFormat;

    input_t *A_d, *B_d;
    output_t *C_d;

    public:
        HardwareUnitNvidia(size_t M, size_t N, size_t K) :
            HardwareUnit<InputFormat, OutputFormat>(M, N, K) {
            cudaMalloc(&A_d, M * K * sizeof(input_t));
            cudaMalloc(&B_d, K * N * sizeof(input_t));
            cudaMalloc(&C_d, M * N * sizeof(output_t));
        }

        ~HardwareUnitNvidia() {
            cudaFree(C_d);
            cudaFree(B_d);
            cudaFree(A_d);
        }

    private:
        void run_mfma_kernel() {

            // Copy input from host to device.
            cudaMemcpy(A_d, this->A.data(), this->A_size * sizeof(input_t), cudaMemcpyHostToDevice);
            cudaMemcpy(B_d, this->B.data(), this->B_size * sizeof(input_t), cudaMemcpyHostToDevice);
            cudaMemcpy(C_d, this->C.data(), this->C_size * sizeof(output_t), cudaMemcpyHostToDevice);

            wmma_ker<<<1,32>>>(A_d, B_d, C_d);

            // Copy result from device to host.
            cudaMemcpy(this->C.data(), C_d, this->C_size * sizeof(output_t), cudaMemcpyDeviceToHost);
        }
};

#endif // HARDWARE_UNIT_NVIDIA_H