#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "MFMAWrapper.h"
#include "nvidia_specific_code.cpp"
//#include "fp_utils.cpp"
#include "utils.cpp"


void run_tests(){
    mw.reset_host_matrices();
    const float normal_input=ldexpf(-1.0f, -120);
    const float normal_input2=ldexpf(-1.0f, -7);
    mw.A[0]=normal_input;
    mw.B[0]=normal_input2;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);


    mw.reset_host_matrices();
    printf("subnormal input\n");
    const float input_normal=static_cast<float>(8);
    const float subnormal_input= binary16::minSubnormal();
    mw.A[0]=input_normal;
    mw.B[0]=subnormal_input;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);


    mw.reset_host_matrices();
    printf("Extra bit---20nd bit is the extra?\n");
    const float one=ldexpf(1.0f, 0);
    float extra_bit = ldexpf(-1.0f, -20);
    
    mw.A[0]=one;
    mw.B[0]=one;
    mw.C[0]=extra_bit;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);

    printf("Rounding Mode\n");
    const float half_ulp =ldexpf(1.0f, -20);
    mw.reset_host_matrices();
    
    
    mw.A[0]=one;
    mw.A[1]=one;
    mw.A[2]=one;
    mw.A[3]=one;
    mw.B[0]=one;
    mw.B[4]=half_ulp;
    mw.B[8]=half_ulp;
    mw.B[12]=half_ulp;
    
    mw.run_mfma_kernel();
   
    print_matrix(mw.C, M, N, true);

}
