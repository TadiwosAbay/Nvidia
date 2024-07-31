#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

using binary16_t = half;
using bfloat16_t = nv_bfloat16;

/* Compute C += A*B, where A, B, and C are 16x16x16 matrices.
   The matrix C is initialized to 0 when `init` is true. */
//template <typename input_t, typename return_t>
__global__ void wmma_ker(input_t *A, input_t *B, return_t *C, bool init = false) {

    // Declare fragments.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, input_t, wmma::row_major> A_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, input_t, wmma::col_major> B_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, return_t> C_fragment;

    // Load input matrices and initialize output (if required).
    wmma::load_matrix_sync(A_fragment, A, 16);
    wmma::load_matrix_sync(B_fragment, B, 16);
    if (init)
        wmma::fill_fragment(C_fragment, 0.0f);
    else
        wmma::load_matrix_sync(C_fragment, C, 16, wmma::mem_col_major);

    // Multiply
    wmma::mma_sync(C_fragment, A_fragment, B_fragment, C_fragment);

    // Store the output
    wmma::store_matrix_sync(C, C_fragment, 16, wmma::mem_col_major);
}



// CUDA kernel for matrix multiplication with float (FP32) inputs
//__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  //  int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    //if (row < M && col < N) {
      //  float value = 0;
        //for (int i = 0; i < K; ++i) {
          //  value += A[row * K + i] * B[i * N + col];
        //}
        //C[row * N + col] += value;
    //}
//}


template <typename input_t, typename return_t>
MFMAWrapper<input_t, return_t>::MFMAWrapper(size_t M, size_t N, size_t K) :
                         M(M), N(N), K(K), LDA(K), LDB(N), LDC(N),
                         A_size(M * K), B_size(K * N), C_size(M * N),
                         A(A_size), B(B_size), C(C_size) {
    cudaMalloc(&A_d, 16 * 16 * sizeof(input_t));
    cudaMalloc(&B_d, 16 * 16 * sizeof(input_t));
    cudaMalloc(&C_d, 16 * 16 * sizeof(return_t));
}

template <typename input_t, typename return_t>
MFMAWrapper<input_t, return_t>::~MFMAWrapper() {
    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);
}

template <typename input_t, typename return_t>
void MFMAWrapper<input_t, return_t>::run_mfma_kernel() {

    // Copy input from host to device.
    cudaMemcpy(A_d, A.data(), A_size * sizeof(input_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), B_size * sizeof(input_t), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C.data(), C_size * sizeof(return_t), cudaMemcpyHostToDevice);

    // Perform matrix multiplication.
   //if (std::is_same<input_t, float>::value)
   if (true) {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Perform matrix multiplication using the FP32 kernel.
        matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);
    } else {
        // Perform matrix multiplication using WMMA.
         
        //wmma_ker<<<1, 32>>>(A_d, B_d, C_d);
    }
    //wmma_ker<<<1,32>>>(A_d, B_d, C_d);

    // Copy result from device to host.
    cudaMemcpy(C.data(), C_d, C_size * sizeof(return_t), cudaMemcpyDeviceToHost);
}
