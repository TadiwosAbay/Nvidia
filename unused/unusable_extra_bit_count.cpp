       bool has_extra_bits_to_the_right() {
            // Reset the matrices to start with clean values
            reset_host_matrices();

            // Set A[0] = 1.0
            A[0] = InputFormat::constant(1.0);

            // Set A[1] through A[4] to 1/4 ULP
            for (int i = 1; i <= 4; ++i) {
                A[i] = OutputFormat::ulp() / OutputFormat::constant(4.0);
            }

            // Set B[0] through B[4] to 1.0
            for (int i = 0; i <= 4; ++i) {
                B[i] = InputFormat::constant(1.0);
            }

            // Run the matrix multiplication kernel
            run_mfma_kernel();

            // Check the value in C[0] to see if the accumulator has extra precision.
            // If extra precision exists, the result should be slightly more than 1.0
            auto expected_result = OutputFormat::constant(1.0) + OutputFormat::ulp();
            bool has_extra_precision = (C[0] == expected_result);

            // Return the result
            return has_extra_precision;
        }


        bool Extra_bits_with_two_nums() {
                reset_host_matrices();
    
            // Set A[0] * B[0] to -(1 + 2 * ulp)
            A[0] = -InputFormat::constant(1.0) - InputFormat::ulp() ;//* InputFormat::constant(2.0);
            B[0] = InputFormat::constant(1.0);
    
            // Set A[1] * B[1] to 1 + 2^2
            A[1] = InputFormat::constant(1.0);
            B[1] = InputFormat::constant(1.0) + InputFormat::constant(4.0);
    

                run_mfma_kernel();
    
            // Evaluate the result
            auto expected_output = -(0.5 + 2 * OutputFormat::ulp()) + (1.0 + 4.0);
            bool result = (C[0] == expected_output);

            return result;
        }

        bool check_extra_bits_with_three_bits() {
            reset_host_matrices();
            auto ulp = InputFormat::machineEpsilon();
            auto ulp_2 = InputFormat::machineEpsilon() * InputFormat::constant(2);
            auto ulp_4 = InputFormat::constant(4) * InputFormat::machineEpsilon();

            auto last_3_bits = ulp + ulp_2 + ulp_4;

            // Test Case 1
            A[0] = InputFormat::constant(1) + (InputFormat::constant(2^-2));
            B[0] = InputFormat::constant(1.0);
            A[1] = InputFormat::constant(1) + (ulp + ulp_2);
            B[1] = InputFormat::constant(-1.0);

            run_mfma_kernel();
            // bool test1_result = (C[0] & last_3_bits);

            // Test Case 2
            reset_host_matrices();
            A[0] = InputFormat::constant(1) + (InputFormat::constant(2^-2));
            B[0] = InputFormat::constant(1.0);
            A[1] = InputFormat::constant(1) + ulp;
            B[1] = InputFormat::constant(-1.0);

            run_mfma_kernel();
            // bool test2_result = (C[0] & last_3_bits);

            // if (last_3_bits == test1_result) {
            //     std::cout << "has one extra bit" << std::endl;
            // } else if (last_3_bits == test2_result) {
            //     std::cout << "has three extra bits" << std::endl;
            // } else {
            //     std::cout << "has two extra bits" << std::endl;
            // }

            // Additional RTN-TE Test
            reset_host_matrices();
            A[0] = InputFormat::constant(1) + (InputFormat::constant(2^-2));
            B[0] = InputFormat::constant(1.0);
            A[1] = InputFormat::constant(1) + (ulp_4 + ulp);
            B[1] = InputFormat::constant(-1.0);

            run_mfma_kernel();
            // bool test3_result = (C[0] & last_3_bits);

            reset_host_matrices();
            A[0] = InputFormat::constant(1) + (InputFormat::constant(2^-2));
            B[0] = InputFormat::constant(1.0);
            A[1] = InputFormat::constant(1) + (ulp_4 + ulp_2);
            B[1] = InputFormat::constant(-1.0);

            run_mfma_kernel();
            // bool test4_result = (C[0] & last_3_bits);

            // if (last_3_bits == test3_result) {
            //     std::cout << "has three extra bits" << std::endl;
            // } else if (last_3_bits == test4_result) {
            //     std::cout << "has two extra bits" << std::endl;
            // } else {
            //     std::cout << "has one extra bit" << std::endl;
            // }
            return true;
        }