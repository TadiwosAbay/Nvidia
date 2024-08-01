#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

#include "MFMAWrapper.h"
#include "nvidia_specific_code.cpp"
//#include "fp_utils.cpp"
#include "utils.cpp"

int main(){

    constexpr size_t M = 16;
    constexpr size_t N = 16;
    constexpr size_t K = 16;

    // Initialization.
    auto mw = MFMAWrapper<float, float>(M, N, K);
    //auto mw_float = MFMAWrapper<float, float>(M, N, K);

    //const binary16_t minnormal_input = __float2half(1.0 / (1 << 14)); // ldexp(1., -24)
    //const binary16_t minsubnormal_input = __float2half(1.0 / (1 << 24)); // ldexp(1., -24)
    //const binary16_t one_input = __float2half(1.0);
    //const binary16_t four_input = __float2half(4.0);

    // Test case
    mw.reset_host_matrices();
    /* size_t i = 0;
    //for (auto & item : mw.A)
      //  item = __float2half(i++);
    //for (auto & item : mw.B)
      //  item = __float2half(i++);
    //for (auto & item : mw.C)
       // item = __float2half(i++); */

    //mw.A[0] = minnormal_input;
    //mw.B[0] = one_input;
    //mw.C[0] = 1.0;
    //print_matrix(mw.A, M, K, true);
    //print_matrix(mw.B, M, K, true);
    //print_matrix(mw.C, M, N, true);
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);
//////////////////////////////////////////////////////////////////////////////////////////////////
    //const binary16_t a11 = __float2half(8);
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -15));

    //const binary16_t b11 = __float2half(1.0f/(1<<15));

    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a11;
    //mw.A[1]=a11;
    //mw.B[0]=b11;
    //mw.B[1]=b11;
    //mw.run_mfma_kernel();
    //print_matrix(mw.A, M, N, false);
    //print_matrix(mw.B, M, N, false);
    //print_matrix(mw.C, M, N, false);


    //float expected= a11*b11+a11*b11;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected)<<std::endl;

////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
    //const binary16_t a13 = __float2half(ldexpf(1.0f, -8));
    //const binary16_t b13 = __float2half(ldexpf(1.0f, -7));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a13;
    //mw.B[0]=b13;
    //mw.C[0]=c21;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);


    //float expected3= a13*b13;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected3)<<std::endl;

////////////////////////////////////////////







    //const binary16_t a21 = __float2half(1.0f);
    //const binary16_t b21 = __float2half(1.0f);

    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a21;
    //mw.B[0]=b21;
    //mw.C[0]=c21;
    //mw.run_mfma_kernel();
    //print_matrix(mw.C, M, N, true);


    //float expected2= a21 + c21;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected2)<<std::endl;
    //std::cout<<"GPU result at C[0]: "<<static_cast<float>(mw.C[0])<<std::endl;





    //////////////////////////////////////////////////////////////////////////////////////////////////
    //const binary16_t a41 = __float2half(1);
    //const binary16_t b41 = __float2half(ldexpf(1.0f, -11));
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -15));

    //const binary16_t b11 = __float2half(1.0f/(1<<15));

    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a41;
    //mw.A[1]=a41;
    //mw.A[2]=a41;
    //mw.A[3]=a41;
    //mw.B[0]=a41;
    //mw.B[1]=b41;
    //mw.B[2]=b41;
    //mw.B[3]=b41;
    //mw.run_mfma_kernel();
    //print_matrix(mw.A, M, N, false);
    //print_matrix(mw.B, M, N, false);
    //print_matrix(mw.C, M, N, false);


    //float expected4= a41*a41+a41*b41+a41*b41+a41*b41;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected4)<<std::endl;

////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////////////////////////////
    //const binary16_t a61 = __float2half(-1.0f);
    //const binary16_t a51 = __float2half(1);
    //const binary16_t b41 = __float2half(ldexpf(1.0f, -11));
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -15));

    //const binary16_t b11 = __float2half(1.0f/(1<<15));

    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a61;
    //mw.A[1]=a61;
    //mw.A[2]=a61;
    //mw.A[3]=a61;
    //mw.B[0]=a51;
    //mw.B[1]=b41;
    //mw.B[2]=b41;
    //mw.B[3]=b41;
    //mw.run_mfma_kernel();
    //print_matrix(mw.A, M, N, false);
    //print_matrix(mw.B, M, N, false);
    //print_matrix(mw.C, M, N, false);


    //float expected5= a61*a51+a61*b41+a61*b41+a61*b41;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected5)<<std::endl;

////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    //const float a11 = 8.0f;
    //const binary16_t a11= __float2half(ldexpf(1.0f, 2));
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -24));

    //const binary16_t b11 = __float2half(1.0f/(1<<15));

    //const binary16_t a21=static_cast<binary16_t>(1.0);
    //const binary16_t b21=static_cast<binary16_t>(1.0);
    //float c21_float = -1.0f * (1.0f / (1 << 11));
    //binary16_t c21 = __float2half(c21_float);

    // Display the values (Note: __half cannot be directly printed)
  //std::cout << "a11: " << __half2float(a11) << std::endl;
  //const float16_t c21=static_cast<float16_t>(-1*(1/(1<<2)));

    //std::cout << "c21 without any change: " << __half2float(c21) << std::endl;
  //std::cout << "c21: " << __half2float(c21) << std::endl;
    //mw.reset_host_matrices();

    //mw.A[0]=a11;
    //mw.A[1]=a11;
    //mw.B[0]=b11;
    //mw.B[1]=b11;
    //mw.run_mfma_kernel();
    //print_matrix(mw.A, M, N, false);
    //print_matrix(mw.B, M, N, false);
    //print_matrix(mw.C, M, N, false);


    //float expected= a11*b11+a11*b11;
    //std::cout<<"Expected Result: "<<static_cast<float>(expected)<<std::endl;

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

    //mw.reset_host_matrices();
    //const binary16_t a41 = __float2half(1);
    //const binary16_t b41 = __float2half(ldexpf(1.0f, -11));
    //const binary16_t b11 = __float2half(ldexpf(1.0f, -15));

    //const binary16_t b11 = __float2half(1.0f/(1<<15));

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
    //print_matrix(mw.A, M, N, false);
    //print_matrix(mw.B, M, N, false);
    //print_matrix(mw.C, M, N, false);
    mw.reset_host_matrices();
    const float a21=static_cast<float>(1.0);
    const float b21=static_cast<float>(1.0);
    float c21 = -1.0f * (1.0f / (1 << 22));
    //binary16_t c21 = __float2half(c21_float);
    mw.A[0]=a21;
    mw.B[0]=b21;
    mw.C[0]=c21;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);

    mw.reset_host_matrices();
    const float a1=static_cast<float>(8);
    const float b1=(1.0f / (1 << 127));
    //float c21 = -1.0f * (1.0f / (1 << ));
    //binary16_t c21 = __float2half(c21_float);
    mw.A[0]=a1;
    mw.B[0]=b1;
    //mw.C[0]=c21;
    mw.run_mfma_kernel();
    print_matrix(mw.C, M, N, true);



//////////////////////////////////////////////////////////////////////////////////////////////////
  


    return 0;
}
