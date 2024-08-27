// fp_utils.h
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

        bool produces_subnormals_from_subnormals() {
            reset_host_matrices();
            A[0] = InputFormat::constant(0.5);
            B[0] = InputFormat::maxSubnormal();
            run_mfma_kernel();
            return InputFormat::isSubnormal(C[0]) ? true : false;
        }

        bool keeps_subnormals_in_accumulator() {
            reset_host_matrices();
            C[0] = OutputFormat::minSubnormal();
            run_mfma_kernel();
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

        bool partial_products_are_exact() {
            
        }

        bool has_one_extra_bit() {
            return true;
        }

    public:
        std::vector<input_t> A, B;
        std::vector<output_t> C;

    MFMAWrapper(size_t M, size_t N, size_t K);

    ~MFMAWrapper();

    /* Set the entries of host arrays to zero. */
    void reset_host_matrices() {
        A.assign(A.size(), 0);
        B.assign(B.size(), 0);
        C.assign(C.size(), 0);
    };

    /* Run MFMA kernel on device. */
    void run_mfma_kernel();

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
        if (accumulates_subnormals()) {
            std::cout << "Accumulates subnormals." << std::endl;
        } else {
            std::cout << "Does not accumulate subnormals." << std::endl;
        }
    }

};

#endif // MFMA_H