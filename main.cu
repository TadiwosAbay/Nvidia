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
    print_matrix(mw.A, M, K, true);
    print_matrix(mw.B, M, K, true);
    print_matrix(mw.C, M, N, true);
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);

    return 0;
}
