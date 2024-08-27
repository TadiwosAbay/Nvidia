#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

using binary16_t = _Float16;
using bfloat16_t = hip_bfloat16;

#define HIP_CHECK(command)                                    \
{                                                             \
  hipError_t stat = (command);                                \
  if(stat != hipSuccess) {                                    \
    std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
    " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(-1);                                                 \
  }                                                           \
}

bool gpuArchCheck(const std::string arch) {
  hipDeviceProp_t hipProps;
  int deviceId=0;
  HIP_CHECK(hipGetDevice(&deviceId));
  HIP_CHECK(hipGetDeviceProperties(&hipProps, deviceId));
  return (std::string(hipProps.gcnArchName).find(arch)
          !=std::string::npos);
}

/* @brief Compute C += A * B using an MFMA operation.
 *
 * The matrices A, B, and C are stored by columns. The function only requires
 * the leading dimension of these three matrices, as the other is constrained
 * by the size of the intrinsics operation.
 *
 * The function uses a single wavefront (64 threads) in a 16 x 4 layout. The
 * kernel should be called with a <<<1, dim3(16, 4)>>> layout, which implies
 * 0 <= ThreadIdx.x <= 15 and 0 <= ThreadIdx.y <= 3.
 *
 * The matrix A has leading dimension LDA.
 * The matrix B has leading dimension LDB.
 * The matrix C has leading dimension LDC.
 *
 * ./matrix_calculator.py --architecture cdna2 --instruction v_mfma_f32_16x16x16f16 --detail-instruction
 */
template <typename input_t, typename output_t>
__global__ void run_mfma(const input_t *A, const input_t *B, output_t *C,
                         const size_t LDA, const size_t LDB, const size_t LDC) {

    using input_lane = __attribute__((__vector_size__(4 * sizeof(input_t)))) input_t;
    using output_lane = __attribute__((__vector_size__(4 * sizeof(output_t)))) output_t;

    output_lane C_VGPR; // Stored by rows.

    // Declare and populate VGPRs.
    input_lane A_VGPR; // Stored by columns.
    input_lane B_VGPR; // Stored by rows.
    for(int i = 0; i < 4; ++i){
        const int a_idx =  threadIdx.x * LDA      // consecutive threads cover 16 consecutive rows
                        + i                      // consecutive registers take consecutive columns
                        + threadIdx.y * 4;       // groups of 16 lanes skip 4 columns
        A_VGPR[i] = A[a_idx];

        const int b_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                        + i * LDB                // consecutive registers take consecutive rows
                        + threadIdx.y * LDB * 4; // groups of 16 lanes skip 4 rows
        B_VGPR[i] = B[b_idx];

        const int c_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                        + i * LDC                // consecutive registers take consecutive rows of 16 floats
                        + threadIdx.y * 4 * LDC; // groups of 16 lanes skip 4 rows
        C_VGPR[i] = C[c_idx];
    }

    C_VGPR = __builtin_amdgcn_mfma_f32_16x16x16f16(A_VGPR, B_VGPR, C_VGPR, 0, 0, 0);

    for(int i = 0; i < 4; ++i){
        const int c_idx =  threadIdx.x            // consecutive threads cover 16 consecutive columns
                        + i * LDC                // consecutive registers take consecutive rows of 16 floats
                        + threadIdx.y * 4 * LDC; // groups of 16 lanes skip 4 rows
        C[c_idx] = C_VGPR[i];
    }

}

MFMAWrapper::mfma_wrapper (size_t M, size_t N, size_t K) :
                           M(M), N(N), K(K), LDA(K), LDB(N), LDC(N),
                           A_size(M * K), B_size(K * N), C_size(M * N),
                           A(A_size), B(B_size), C(C_size) {
    HIP_CHECK(hipMalloc(&A_d, A_size * sizeof(*A_d)));
    HIP_CHECK(hipMalloc(&B_d, B_size * sizeof(*B_d)));
    HIP_CHECK(hipMalloc(&C_d, C_size * sizeof(*C_d)));
}

MFMAWrapper::~mfma_wrapper () {
    HIP_CHECK(hipFree(C_d));
    HIP_CHECK(hipFree(B_d));
    HIP_CHECK(hipFree(A_d));
}

void MFMAWrapper::reset_host_matrices() {
    A.assign(A.size(), 0);
    B.assign(B.size(), 0);
    C.assign(C.size(), 0);
}

void MFMAWrapper::run_mfma_kernel() {

    // Populate device buffers
    HIP_CHECK(hipMemcpy(A_d, A.data(), A_size * sizeof(input_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B.data(), B_size * sizeof(input_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(C_d, C.data(), C_size * sizeof(output_t), hipMemcpyHostToDevice));

    run_mfma<<<1, dim3(16, 4)>>>(A_d, M, B_d, N, C_d, N);
    HIP_CHECK(hipGetLastError());

    // Copy result back to host
    HIP_CHECK(hipMemcpy(C.data(), C_d, C_size * sizeof(output_t), hipMemcpyDeviceToHost));
}

// Following code comes from main().
/*
    if (!gpuArchCheck("gfx90a") && !gpuArchCheck("gfx908")) {
        std::cout << "mfma_f32_16x16x16f16 instruction only available on gfx908 or later."
                << std::endl;
        exit(-1);
    }
*/
