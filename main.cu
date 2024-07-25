#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "MFMAWrapper.h"
#include "nvidia_specific_code.cpp"
#include "fp_utils.cpp"
#include "utils.cpp"

int main(){

    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t K = 16;

    // Initialization.
    auto mw = MFMAWrapper<binary16_t, float>(M, N, K);

    const binary16_t minnormal_input = __float2half(1.0 / (1 << 14)); // ldexp(1., -24)
    const binary16_t minsubnormal_input = __float2half(1.0 / (1 << 24)); // ldexp(1., -24)
    const binary16_t one_input = __float2half(1.0);
    const binary16_t four_input = __float2half(4.0);

    // Test case
    mw.reset_host_matrices();
    /* size_t i = 0;
    for (auto & item : mw.A)
        item = __float2half(i++);
    for (auto & item : mw.B)
        item = __float2half(i++);
    for (auto & item : mw.C)
        item = __float2half(i++); */

    mw.A[0] = minnormal_input;
    mw.B[0] = one_input;
    mw.C[0] = 1.0;
    //print_matrix(mw.A, M, K, true);
    //print_matrix(mw.B, M, K, true);
    //print_matrix(mw.C, M, N, true);
    mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);

    const binary16_t a21=static_cast<binary16_t>(1.0);
    const binary16_t b21=static_cast<binary16_t>(1.0);
    float c21_float = -1.0f * (1.0f / (1 << 11));
    binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    std::cout << "c21 without any change: " << c21 << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    mw.reset_host_matrices();

    mw.A[0]=a21;
    mw.B[0]=b21;
    mw.C[0]=c21;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);


    float expected2= a21 + c21;
    std::cout<<"Expected Result: "<<static_cast<float>(expected2)<<std::endl;
    std::cout<<"GPU result at C[0]: "<<static_cast<float>(mw.C[0])<<std::endl;




    return 0;
}
