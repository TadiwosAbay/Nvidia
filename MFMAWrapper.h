#ifndef MFMA_H
#define MFMA_H

enum class RoundingMode : uint8_t {
    roundNotFaithful = 0,
    roundToNearestEven = 1,
    roundToNearestZero = 2,
    roundToNearestAway = 3,
    roundUp = 4,
    roundDown = 5,
    roundToZero = 6,
    roundUnknown = 7
};

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
            A[0] = InputFormat::constant(1.0);
            B[0] = InputFormat::constant(1.0);
            C[0] = -InputFormat::unitRoundoff();
            run_mfma_kernel();
            return (C[0]) == OutputFormat::constant(1.0) ? false : true;
        }

        RoundingMode detect_rounding_mode() {
            // Positive values.
            reset_host_matrices();
            A[0] = InputFormat::minNormal();
            A[1] = InputFormat::constant(1.0);
            B[0] = InputFormat::machineEpsilon() * InputFormat::constant(3.0);
            B[1] = InputFormat::constant(2.0);
            run_mfma_kernel();
            std::cout.precision(20);
            auto roundingCandidate = OutputFormat::constant(2.0) + OutputFormat::constant(4.0) * OutputFormat::machineEpsilon();
            bool positive_rounds_down = (C[0] == OutputFormat::constant(2.0));
            if (!positive_rounds_down && C[0] != roundingCandidate) // Not rounding to either rounding candidate.
                return RoundingMode::roundNotFaithful;

            // Negative values.
            C[0] = OutputFormat::constant(0.0);
            B[0] = -B[0];
            B[1] = -B[1];
            run_mfma_kernel();
            bool negative_rounds_up = (C[0] == -OutputFormat::constant(2.0));
            if (!negative_rounds_up && C[0] != -roundingCandidate) // Not rounding to either rounding candidate.
                return RoundingMode::roundNotFaithful;

            // Determine rounding mode.
            if (positive_rounds_down && negative_rounds_up) {
                return RoundingMode::roundToZero;
            } else if (positive_rounds_down) { // Negative also rounds down.
                return RoundingMode::roundDown;
            } else if (negative_rounds_up) {   // Positive also rounds up.
                return RoundingMode::roundUp;
            } else { // A round-to-nearest mode – check tie-breaking rule.
                //Check the tie-breaking rule.
                C[0] = OutputFormat::constant(0.0);
                B[0] = InputFormat::machineEpsilon() * InputFormat::constant(2.0);
                B[1] = InputFormat::constant(2.0);
                run_mfma_kernel();
                bool positive_tie_rounds_down = (C[0] == OutputFormat::constant(2.0));
                if (!positive_tie_rounds_down && C[0] != roundingCandidate) // Not rounding to either rounding candidate.
                    return RoundingMode::roundNotFaithful;

                C[0] = OutputFormat::constant(0.0);
                B[0] = -B[0];
                B[1] = -B[1];
                run_mfma_kernel();
                bool negative_tie_rounds_up = (C[0] == -OutputFormat::constant(2.0));
                if (!positive_tie_rounds_down && C[0] != roundingCandidate) // Not rounding to either rounding candidate.
                    return RoundingMode::roundNotFaithful;

                if (!positive_tie_rounds_down && !negative_tie_rounds_up)
                    return RoundingMode::roundToNearestAway;

                C[0] = OutputFormat::constant(0.0);
                A[1] = InputFormat::minNormal();
                B[0] = InputFormat::machineEpsilon() * InputFormat::constant(2.0);
                B[1] = InputFormat::constant(2.0) + InputFormat::machineEpsilon();
                run_mfma_kernel();
                bool positive_tie_rounds_to_odd = (C[0] == OutputFormat::constant(2.0) + 
                        OutputFormat::constant(2.0) * OutputFormat::machineEpsilon());

                if (positive_tie_rounds_to_odd)
                    return RoundingMode::roundToNearestZero;
                else
                    return RoundingMode::roundToNearestEven;
            }
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
                if (C[0] == OutputFormat::constant(1.0)) {
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
        switch (detect_rounding_mode()) {
            case RoundingMode::roundToNearestEven:
                std::cout << "The rounding mode is round to nearest even." << std::endl;
                break;
            case RoundingMode::roundToNearestZero:
                std::cout << "The rounding mode is round to nearest zero." << std::endl;
                break;
            case RoundingMode::roundToNearestAway:
                std::cout << "The rounding mode is round to nearest away." << std::endl;
                break;
            case RoundingMode::roundUp:
                std::cout << "The rounding mode is round up." << std::endl;
                break;
            case RoundingMode::roundDown:
                std::cout << "The rounding mode is round down." << std::endl;
                break;
            case RoundingMode::roundToZero:
                std::cout << "The rounding mode is round to zero." << std::endl;
                break;
        }
        std::cout << "The size of the FMA is " << fma_size() << std::endl;
        
    }

};

#endif // MFMA_H
