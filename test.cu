#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "MFMAWrapper.h"
#include "nvidia_specific_code.cpp"
#include "fp_utils.h"
#include "utils.cpp"

#include <string>

template <typename T> std::string get_type_name() {return "unknown type";}
template <> std::string get_type_name<bfloat16_t>() {return "bfloat16";}
template <> std::string get_type_name<binary16_t>() {return "binary16 (half)";}
template <> std::string get_type_name<binary32_t>() {return "binary32 (single)";}
template <> std::string get_type_name<binary64_t>() {return "binary64 (double)";}

template <typename InputFormat, typename OutputFormat>
void run_tests(){
    // Possible matrix sizes are here:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes
    // Indeed, there is no 4 x 4 x 4. That worked in our first paper, but probably just because most of
    // the matrix was empty anyway. Good spot!
    constexpr size_t M = 4;
    constexpr size_t N = 4;
    constexpr size_t K = 4;
    auto mw = MFMAWrapper<typename InputFormat::type, typename OutputFormat::type>(M, N, K);
    //std::cout << InputFormat << std::endl;

     // Print the InputFormat type
    std::cout << "Input  Format: " << get_type_name<typename InputFormat::type>() << std::endl;
    std::cout << "Output Format: " << get_type_name<typename OutputFormat::type>() << std::endl;

    //auto mw = MFMAWrapper<typename InputFormat::type, float>(M, N, K);

    // Test 1: Normal input
    mw.reset_host_matrices();
    const auto normal_input = InputFormat::constant(4);
    const auto normal_input2= InputFormat::minNormal();
    //const auto normal_input2 = InputFormat::minimumNormal();
    mw.A[0] = InputFormat::constant(1) / normal_input;
    mw.B[0] = normal_input2;
    mw.run_mfma_kernel();
    //print_matrix<InputFormat>(mw.A, M, N, true);
    //print_matrix<InputFormat>(mw.B, M, N, false);
    print_matrix<InputFormat>(mw.C, M, N, true);

    // Test 2: Subnormal input
    mw.reset_host_matrices();
    std::cout << "Subnormal input\n";
    const auto input_normal = InputFormat::constant(4);
    const auto subnormal_input = InputFormat::midwaySubnormal();
    mw.A[0] = input_normal;
    mw.B[0] = subnormal_input;
    mw.run_mfma_kernel();
    //print_matrix<InputFormat>(mw.B, M, N, false);
    print_matrix<InputFormat>(mw.C, M, N, true);

    std::cout << input_normal << std::endl;

    // Test 3: Extra bit
    mw.reset_host_matrices();
    std::cout << "Extra bit---20th bit is the extra?\n";
    const auto one = InputFormat::constant(1);
    auto extra_bit = InputFormat::extra_bit();
    //print_matrix<InputFormat>(mw.A, M, N, true);
    //print_matrix<InputFormat>(mw.B, M, N, false);
    //print_matrix<InputFormat>(mw.C, M, N, true);
    mw.A[0] = one;
    mw.B[0] = one;
    mw.C[0] = static_cast<decltype(extra_bit)>(-1)*extra_bit;
    mw.run_mfma_kernel();
    //print_matrix<InputFormat>(mw.A, M, N, true);
    //print_matrix<InputFormat>(mw.B, M, N, false);
    print_matrix<InputFormat>(mw.C, M, N, true);

    // Test 4: Rounding Mode
    std::cout << "Rounding Mode\n";
    const auto half_ulp = InputFormat::extra_bit(); // This part may need to be generalized based on format
    mw.reset_host_matrices();
    mw.A[0] = one;
    mw.A[1] = one;
    mw.A[2] = one;
    mw.A[3] = one;
    mw.B[0] = one;
    mw.B[1] = half_ulp;
    mw.B[2] = half_ulp;
    mw.B[3] = half_ulp;
    mw.run_mfma_kernel();
    print_matrix<InputFormat>(mw.C, M, N, true);

}
