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
            if (InputFormat::precision * 2 > OutputFormat::precision) {
                return false; // Not enough precision to represent the result.
            } else {
                reset_host_matrices();
                A[0] = InputFormat::beforeOne();
                B[0] = InputFormat::beforeOne();
                run_mfma_kernel();
                return double(C[0]) == double(A[0]) * double(B[0]);
            }
        }

        /*  ADDED code here
         * Extra bits and rounding modes.
         */
        bool has_one_extra_bit() {
            reset_host_matrices();
            A[0] = InputFormat::one();
            B[0] = InputFormat::one();
            C[0] = -InputFormat::unitRoundoff();
            run_mfma_kernel();
            std::cout << "C[0] = " << C[0] << std::endl;
            return OutputFormat::isOne(C[0]) ? false : true;
        }

        bool round_mode() {
            reset_host_matrices();
            A[0] = InputFormat::one();
            A[1] = InputFormat::one();
            A[2] = InputFormat::one();
            A[3] = InputFormat::one();
            B[0] = InputFormat::one();
            B[1] = InputFormat::half_ulp;
            B[2] = InputFormat::half_ulp;
            B[3] = InputFormat::half_ulp;
            run_mfma_kernel();
            std::cout << "C[0] = " << C[0] << std::endl;
            pos_result=C[0];
            pos_upper=OutputFormat::one()+ OutputFormat::four()*(half_ulp);
            pos_lower=OutputFormat::one()+ OutputFormat::two()*(half_ulp);
            return OutputFormat::isOne(C[0]) ? false : true;

            reset_host_matrices();
            A[0] = InputFormat::minus_one();
            A[1] = InputFormat::minus_one();
            A[2] = InputFormat::minus_one();
            A[3] = InputFormat::minus_one();
            B[0] = InputFormat::one();
            B[1] = InputFormat::half_ulp;
            B[2] = InputFormat::half_ulp;
            B[3] = InputFormat::half_ulp;
            run_mfma_kernel();
            neg_result=C[0];
            neg_toNeg=OutputFormat::minus_one()+ OutputFormat::minus_four()*(OutputFormat::half_ulp);
            neg_toPos=OutputFormat::minus_one()+ OutputFormat::minus_two()*(OutputFormat::half_ulp);

            if(pos_result==pos_upper && neg_result==neg_toPos)  //round to infinity
                return 1;
            if(pos_result==pos_upper && neg_result==neg_toNeg) //RTN-TE
                return 2;

            if(pos_result==pos_lower && neg_result==neg_toPos)  //round-to-zero
                return 3;
            if(pos_result==pos_lower && neg_result==neg_toNeg) //round to minus infinity
                return 4;
            
        }


        /*
         * Size of the FMA.
         */
        size_t fma_size() {
            reset_host_matrices();
            auto a_value = InputFormat::ulp();
            auto b_value = InputFormat::constant(OutputFormat::unitRoundoff() / float(InputFormat::ulp()));
            A[0] = a_value;
            B[0] = b_value;
            size_t size = 0;
            for (size_t i = 1; i < N; i++) {
                if (i > 1) {
                    A[i-1] = InputFormat::constant(0);
                    B[i-1] = InputFormat::constant(0);
                }
                C[0] = InputFormat::constant(1);
                A[i] = a_value;
                B[i] = b_value;
                run_mfma_kernel();
                if (OutputFormat::isOne(C[0])) {
                    size = i;
                    break;
                }
            }
            return size;
        }

        /*
         * Accumulation order.
         */

    public:
        std::vector<input_t> A, B;
        std::vector<output_t> C;

    MFMAWrapper(size_t M, size_t N, size_t K);

    ~MFMAWrapper();

    /* Run MFMA kernel on device. */
    void run_mfma_kernel();

    /* Set the entries of host arrays to zero. */
    void reset_host_matrices() {
        A.assign(A.size(), 0);
        B.assign(B.size(), 0);
        C.assign(C.size(), 0);
    };

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
        if (multiplications_are_exact()) {
            std::cout << "Multiplications are exact." << std::endl;
        } else {
            std::cout << "Multiplications are not exact." << std::endl;
        }
        if(has_one_extra_bit()) {
            std::cout << "Accumulator has one extra bit." << std::endl;
        }
        else {
            std::cout << "Accumulator does not have extra bits." << std::endl;
        }

        if(round_mode()==1) {
            std::cout << "The rounding mode is round to positive infinity." << std::endl;
        }
        else  if(round_mode()==2) {
           std::cout << "The rounding mode is RTN-TE." << std::endl;
        }
        else  if(round_mode()==3) {
           std::cout << "The rounding mode is round to zero." << std::endl;
        }

        else  if(round_mode()==4) {
            std::cout << "The rounding mode is round to minus infinity." << std::endl;
        }
        std::cout << "The size of the FMA is " << fma_size() << std::endl;
        
    }

};

#endif // MFMA_H
