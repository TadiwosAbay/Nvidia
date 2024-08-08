#include <iostream>
#include <iomanip>
//#include <vector>
#include <random>
#include <cmath>
#include "test.cu"

//#include "MFMAWrapper.h"
//#include "nvidia_specific_code.cpp"
#include "fp_utils.h"
//#include "utils.cpp"

int main(){

    //constexpr size_t M = 4;
    //constexpr size_t N = 4;
    //constexpr size_t K = 4;

    // Initialization.
    run_tests<binary64, binary64>();
    //auto mw = MFMAWrapper<float, float>(M, N, K);
    //auto mw_float = MFMAWrapper<float, float>(M, N, K);

   
    // Subnormal input support
    //const float a11 = 8.0f;
    //const binary16_t a11= __float2half(ldexpf(1.0f, 2));
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -24));

    //mw.reset_host_matrices();

    //mw.A[0]=a11;
    //mw.A[1]=a11;
    //mw.B[0]=b11;
    //mw.B[1]=b11;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, false);


    //float expected= a11*b11+a11*b11;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected)<<std::endl;


               //Extra Bit 
    //mw.reset_host_matrices();
    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 24));
    //binary16_t c21 = __float2half(c21_float);
    //mw.A[0]=a21;
    //mw.B[0]=b21;
    //mw.C[0]=c21;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);




                 // Rounding Mode
    //mw.reset_host_matrices();
    //const binary16_t a41 = __float2half(1);
    //const binary16_t b41 = __float2half(ldexpf(1.0f, -11));
    
    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    //mw.A[0]=a41;
    //mw.A[1]=a41;
    //mw.A[2]=a41;
    //mw.A[3]=a41;
    //mw.B[0]=a41;
    //mw.B[1]=b41;
    //mw.B[2]=b41;
    //mw.B[3]=b41;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, false);




    //printf("subnormal output\n");
    //mw.reset_host_matrices();
    //const float normal_input=ldexpf(-1.0f, -120);
    //const float normal_input2=ldexpf(-1.0f, -7);
    //mw.A[0]=normal_input;
    //mw.B[0]=normal_input2;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);


    //mw.reset_host_matrices();
    //printf("subnormal input\n");
    //const float input_normal=static_cast<float>(8);
    //const float subnormal_input=ldexpf(1.0f, -127);
    //mw.A[0]=input_normal;
    //mw.B[0]=subnormal_input;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);


    //mw.reset_host_matrices();
    //printf("Extra bit---20nd bit is the extra?\n");
    //const float one=ldexpf(1.0f, 0);
    //float extra_bit = ldexpf(-1.0f, -20);
    
    //mw.A[0]=one;
    //mw.B[0]=one;
    //mw.C[0]=extra_bit;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);

    //printf("Rounding Mode\n");
    //const float half_ulp =ldexpf(1.0f, -20);
    //mw.reset_host_matrices();
    
    
    //mw.A[0]=one;
    //mw.A[1]=one;
    //mw.A[2]=one;
    //mw.A[3]=one;
    //mw.B[0]=one;
    //mw.B[4]=half_ulp;
    //mw.B[8]=half_ulp;
    //mw.B[12]=half_ulp;
    
    //mw.run_mfma_kernel();
   
    //print_matrix(mw.C, M, N, true);



//////////////////////////////////////////////////////////////////////////////////////////////////
  


    return 0;
}
