#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

#include "HardwareUnitNvidia.h"
#include "fp_utils.h"
#include "test.cpp"

int main(){

    // Possible matrix sizes are here:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes
    // Indeed, there is no 4 x 4 x 4. That worked in our first paper, but probably just because most of
    // the matrix was empty anyway. Good spot!
    constexpr size_t M = 16, N = 16, K = 16;
    auto mw = HardwareUnitNvidia<binary16,binary32>(M, N, K);
    /* run_tests<binary16, binary32>(); */
    mw.run_tests();

    return 0;
}
