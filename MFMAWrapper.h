template <typename input_t, typename return_t>
class MFMAWrapper {

    private:
        size_t M, N, K, A_size, B_size, C_size;

        input_t *A_d, *B_d;
        return_t *C_d;

        bool produces_normals_from_subnormals() {
            reset_host_matrices();
            return true;
        }

        bool produces_subnormals_from_normals() {
            return true;
        }

        bool accumulates_subnormals() {
            return true;
        }

        bool has_one_extra_bit() {
            return true;
        }

    public:
        std::vector<input_t> A, B;
        std::vector<return_t> C;

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
