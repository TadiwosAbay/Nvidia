template <typename input_t, typename return_t>
class MFMAWrapper {
    private:
        size_t M, N, K,
            LDA, LDB, LDC,
            A_size, B_size, C_size;

        input_t *A_d, *B_d;
        return_t *C_d;

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
};
