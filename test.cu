#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "MFMAWrapper.h"
#include "nvidia_specific_code.cpp"
#include "fp_utils.h"
#include "utils.cpp"

#include <typeinfo>
#include <string>


template <typename T>
std::string get_type_name() {
    if (typeid(T) == typeid(__half)) {
        return "binary16_t (half)";
    } else if (typeid(T) == typeid(nv_bfloat16)) {
        return "bfloat16_t";
    } else if (typeid(T) == typeid(float)) {
        return "float";
    } else if (typeid(T) == typeid(double)) {
        return "double";
    }
    return "unknown type";
}

template <typename InputFormat, typename OutputFormat>
void run_tests(){
    constexpr size_t M = 4;
    constexpr size_t N = 4;
    constexpr size_t K = 4;
    auto mw = MFMAWrapper<typename InputFormat::type, typename OutputFormat::type>(M, N, K);
    //std::cout << InputFormat << std::endl;

     // Print the InputFormat type
    std::cout << "InputFormat: " << get_type_name<typename InputFormat::type>() << std::endl;

    //auto mw = MFMAWrapper<typename InputFormat::type, float>(M, N, K);

    // Test 1: Normal input
    mw.reset_host_matrices();
    const auto normal_input = InputFormat::four();
    const auto normal_input2 = InputFormat::minimumNormal();
    mw.A[0] = InputFormat::one() / normal_input;
    mw.B[0] = normal_input2;
    mw.run_mfma_kernel();
    print_matrix(mw.A, M, N, true);
    print_matrix(mw.B, M, N, false);
    print_matrix(mw.C, M, N, true);

    // Test 2: Subnormal input
    mw.reset_host_matrices();
    std::cout << "Subnormal input\n";
    const auto input_normal = InputFormat::four();
    const auto subnormal_input = InputFormat::largeSubnormal();
    mw.A[0] = input_normal;
    mw.B[0] = subnormal_input;
    mw.run_mfma_kernel();
    print_matrix(mw.B, M, N, false);
    print_matrix(mw.C, M, N, true);

    // Test 3: Extra bit
    mw.reset_host_matrices();
    std::cout << "Extra bit---20th bit is the extra?\n";
    const auto one = InputFormat::one();
    auto extra_bit = InputFormat::minSubnormal();
    mw.A[0] = one;
    mw.B[0] = one;
    mw.C[0] = extra_bit;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);

    // Test 4: Rounding Mode
    std::cout << "Rounding Mode\n";
    const auto half_ulp = ldexpf(1.0f, -20); // This part may need to be generalized based on format
    mw.reset_host_matrices();
    mw.A[0] = one;
    mw.A[1] = one;
    mw.A[2] = one;
    mw.A[3] = one;
    mw.B[0] = one;
    mw.B[4] = half_ulp;
    mw.B[8] = half_ulp;
    mw.B[12] = half_ulp;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);

}
