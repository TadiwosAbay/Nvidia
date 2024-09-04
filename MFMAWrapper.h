#ifndef MFMA_H
#define MFMA_H

template <typename InputFormat, typename OutputFormat>
class MFMAWrapper {

    private:
        size_t M, N, K, A_size, B_size, C_size;

        using input_t = typename InputFormat::storageFormat;
        using output_t = typename OutputFormat::storageFormat;

        input_t *A_d, *B_d;
        output_t *C_d;

        /*
         * Support for subnormals.
         */
        bool produces_subnormals_from_subnormals() {
            reset_host_matrices();
            A[0] = InputFormat::constant(0.5);
            B[0] = InputFormat::maxSubnormal();
            run_mfma_kernel();
            print_matrix(A, M, N, true);
            print_matrix(B, M, N, false);
            print_matrix(C, M, N, true);
            return InputFormat::isSubnormal(C[0]) ? true : false;
        }

        bool produces_subnormals_from_normals() {
            reset_host_matrices();
            A[0] = InputFormat::constant(0.5);
            B[0] = InputFormat::minNormal();
            run_mfma_kernel();
            return InputFormat::isSubnormal(C[0]) ? true : false;
        }

        bool produces_normals_from_subnormals() {
            reset_host_matrices();
            A[0] = InputFormat::constant(2);
            B[0] = InputFormat::midwaySubnormal();
            run_mfma_kernel();
            return InputFormat::isNormal(C[0]) ? true : false;
        }

        bool keeps_subnormals_in_accumulator() {
            reset_host_matrices();
            C[0] = OutputFormat::minSubnormal();
            run_mfma_kernel();
            return OutputFormat::isSubnormal(C[0]) ? true : false;
        }

        /*
         * Accuracy of multiplications in dot product.
         */
        bool multiplications_are_exact() {
            reset_host_matrices();
            A[0] = InputFormat::belowOne();
            B[0] = InputFormat::belowOne();
            run_mfma_kernel();
            return true;
        }

        /*
         * Accuracy of partial products.
         */
        bool partial_products_are_exact() {
            return true;
        }

        /*
         * Extra bits and rounding modes.
         */
        bool has_one_extra_bit() {
            return true;
        }


        /*
         * Size of the FMA.
         */


        /*
         * Accumulation order.
         */

    public:
        std::vector<input_t> A, B;
        std::vector<output_t> C;

    //MFMAWrapper(size_t M, size_t N, size_t K);

    MFMAWrapper(size_t M, size_t N, size_t K)
        : M(M), N(N), K(K), A_size(M * K), B_size(K * N), C_size(M * N),
          A(A_size), B(B_size), C(C_size) {
        cudaMalloc(&A_d, A_size * sizeof(input_t));
        cudaMalloc(&B_d, B_size * sizeof(input_t));
        cudaMalloc(&C_d, C_size * sizeof(output_t));
    }

  //  ~MFMAWrapper();

    ~MFMAWrapper() {
        cudaFree(C_d);
        cudaFree(B_d);
        cudaFree(A_d);
    }

    /* Set the entries of host arrays to zero. */
    void reset_host_matrices() {
        A.assign(A.size(), 0);
        B.assign(B.size(), 0);
        C.assign(C.size(), 0);
    };

    void print_matrix(const std::vector<input_t>& A,
                  const size_t rows,
                  const size_t cols,
                  const bool bycols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      std::cout << std::setw(6);
      auto next_element = bycols ? A[j*cols+i] : A[i*rows+j];
      std::cout << next_element << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
    /* Run MFMA kernel on device. */
   // void run_mfma_kernel();
    void run_mfma_kernel() {
        cudaMemcpy(A_d, A.data(), A_size * sizeof(input_t), cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B.data(), B_size * sizeof(input_t), cudaMemcpyHostToDevice);
        cudaMemcpy(C_d, C.data(), C_size * sizeof(output_t), cudaMemcpyHostToDevice);

        // Call the WMMA kernel
        wmma_ker<<<1, 32>>>(A_d, B_d, C_d);

        cudaMemcpy(C.data(), C_d, C_size * sizeof(output_t), cudaMemcpyDeviceToHost);
    }
    void run_test() {
        if (produces_subnormals_from_subnormals()) {
            std::cout << "Produces subnormals from subnormals." << std::endl;
        } else {
            std::cout << "Does not produce subnormals from subnormals." << std::endl;
        }
        if (produces_normals_from_subnormals()) {
            std::cout << "Produces normals from subnormals." << std::endl;
        } else {
            std::cout << "Does not produce normals from subnormals." << std::endl;
        }
        if (produces_subnormals_from_normals()) {
            std::cout << "Produces subnormals from normals." << std::endl;
        } else {
            std::cout << "Does not produce subnormals from normals." << std::endl;
        }
        if (keeps_subnormals_in_accumulator()) {
            std::cout << "Subnormals in accumulator are kept." << std::endl;
        } else {
            std::cout << "Subnormals in accumulator are lost." << std::endl;
        }
    }

};

#endif // MFMA_H
