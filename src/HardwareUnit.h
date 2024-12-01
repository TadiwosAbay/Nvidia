#ifndef HARDWARE_UNIT_H
#define HARDWARE_UNIT_H

#include <vector>
#include "Features.h"
#include "fp_utils.h"

template <typename InputFormat, typename OutputFormat>
class HardwareUnit {

    public:
        size_t M, N, K, A_size, B_size, C_size;

        using input_t = typename InputFormat::storageFormat;
        using output_t = typename OutputFormat::storageFormat;

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
            if (InputFormat::getPrecision() * 2 > OutputFormat::getPrecision()) {
                return false; // Not enough precision to represent the result.
            } else {
                reset_host_matrices();
                A[0] = InputFormat::beforeOne();
                B[0] = InputFormat::beforeOne();
                run_mfma_kernel();
                return double(C[0]) == double(A[0]) * double(B[0]);
            }
        }

        /*
         * Rounding mode and extra bits to the right of the accumulator.
         */
        bool has_one_extra_bit() {
            reset_host_matrices();
            A[0] = InputFormat::constant(1.0);
            B[0] = InputFormat::constant(1.0);
            C[0] = -OutputFormat::unitRoundoff();
            run_mfma_kernel();
            return (C[0]) == OutputFormat::constant(1.0) ? false : true;
        }

        RoundingMode detect_rounding_mode() {
            // Positive values.
            reset_host_matrices();
            A[0] = InputFormat::minNormal();
            A[1] = InputFormat::constant(1.0);
            B[0] = InputFormat::constant(3.0)*InputFormat::constant(OutputFormat::unitRoundoff() / OutputFormat::constant(InputFormat::minNormal()));
            B[1] = InputFormat::constant(2.0);
            run_mfma_kernel();
            std::cout.precision(20);
            auto roundingCandidate = OutputFormat::constant(2.0) + OutputFormat::constant(4.0) * OutputFormat::unitRoundoff();
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
                // FIX: we shouldn't be using the input format in B, but the output format.
                // This will only work for binary16 -> barinary32.
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
         * Extra bits to the right of the accumulator.
         */
        size_t count_extra_bits(RoundingMode rounding_mode) {
            if (!has_one_extra_bit())
                return 0;

            reset_host_matrices();
            B[0] = InputFormat::constant(1.0);
            B[1] = InputFormat::constant(1.0);
            // A[0] = 2^emax
            // A[1] = 2^(emax - output_precision - 3 + input_precision)
            // ulp(A[1]) will add the third bit that is shifted out
            A[0] = InputFormat::constant(ldexp(1.0, InputFormat::getEmax()));
            input_t tmp = InputFormat::constant(ldexp(1.0, InputFormat::getEmax() - OutputFormat::getPrecision() - 3 + InputFormat::getPrecision()));
            input_t test_1, test_2;
            switch (rounding_mode) {
                case RoundingMode::roundToNearestEven:
                    test_1 = tmp + InputFormat::constant(3) * InputFormat::ulp(tmp);
                    test_2 = tmp + InputFormat::constant(5) * InputFormat::ulp(tmp);
                    break;
                case RoundingMode::roundToZero:
                    test_1 = tmp + InputFormat::constant(1) * InputFormat::ulp(tmp);
                    test_2 = tmp + InputFormat::constant(2) * InputFormat::ulp(tmp);
                    break;
                default:
                    std::cerr << "Requested rounding mode cannot be simulated." << std::endl;
                    exit(1);
            }

            output_t low_precision_result = OutputFormat::constant(A[0]) - OutputFormat::constant(tmp);
            output_t high_precision_result = low_precision_result - OutputFormat::ulp(low_precision_result);

            // Run first computation.
            A[1] = -test_1;
            run_mfma_kernel();
            output_t result_1 = C[0];

            // Run second computation.
            C[0] = 0;
            A[1] = -test_2;
            run_mfma_kernel();
            output_t result_2 = C[0];

            if (result_1 == low_precision_result && result_2 == low_precision_result) {
                return 1;
            } else if (result_1 == low_precision_result && result_2 == high_precision_result) {
                return 2;
            } else if (result_1 == high_precision_result && result_2 == high_precision_result) {
                return 3;
            } else {
                std::cerr << "Could not determine the number of extra bits." << std::endl;
                exit(1);
            }
            
        }

        //###################################################################################################################################################################
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
            return size == 0 ? K : size;
        }

        /*
         * Accumulation order.
         */
        bool sum_starts_from_largest() {
/*             reset_host_matrices();
            A[0] = InputFormat::constant(1.0);
            B[0] = InputFormat::constant(1.0);
            A[1] = InputFormat::constant(1.0);
            B[1] = InputFormat::constant(1.0);
            run_mfma_kernel();
            return C[0] == OutputFormat::constant(2.0); */
            return false;
        }

    public:
        std::vector<input_t> A, B;
        std::vector<output_t> C;

        HardwareUnit(size_t M, size_t N, size_t K) :
                    M(M), N(N), K(K),
                    A_size(M * K), B_size(K * N), C_size(M * N),
                    A(A_size), B(B_size), C(C_size) {};

        /* Run MFMA kernel on device. */
        void virtual run_mfma_kernel() = 0;

        /* Set the entries of host arrays to zero. */
        void reset_host_matrices() {
            A.assign(A.size(), 0);
            B.assign(B.size(), 0);
            C.assign(C.size(), 0);
        }

        Features run_tests() {
            RoundingMode rounding_mode;
            size_t number_of_extra_bits;
            if (has_one_extra_bit()) {
                rounding_mode = detect_rounding_mode();
                number_of_extra_bits = count_extra_bits(rounding_mode);
            } else {
                rounding_mode = RoundingMode::roundNotFaithful;
                number_of_extra_bits = 0;
            }
            
            std::cout << "Have detected " << count_extra_bits(detect_rounding_mode()) << " extra bits." << std::endl;
            Features features (produces_subnormals_from_subnormals(), produces_subnormals_from_normals(),
                        produces_normals_from_subnormals(), keeps_subnormals_in_accumulator(),
                        multiplications_are_exact(), rounding_mode, number_of_extra_bits, fma_size());
            return features;
        }
};

#endif // HARDWARE_UNIT_H
