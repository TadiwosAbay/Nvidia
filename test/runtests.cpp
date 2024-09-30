#include <iostream>

#include "utils.cpp"
#include "HardwareUnitSimulator.h"

using binary16 = IEEEFloatFormat<binary32_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<binary32_t, 8, 127>;

int main() {
    Features features(true, true, true, true, true, true, RoundingMode::roundToNearestEven, 8);
    const size_t M = 16, N = 16, K = 16;
    HardwareUnitSimulator<binary16, binary32> hw(M, N, K, features);

    // Visual test of correctness of the simulator.
    /* for (size_t i = 0; i < hw.A.size(); i++) {
        hw.A[i] = i;
    }
    for (size_t i = 0; i < hw.B.size(); i++) {
        hw.B[i] = i;
    }
    print_matrix<>(hw.A, 16, 16, false);
    print_matrix<>(hw.B, 16, 16, true);
    hw.run_mfma_kernel();
    print_matrix<>(hw.C, 16, 16, true); */

    Features detected_features = hw.run_tests();

    return 0;
}