/* Print the elements of the m x n matrix A. The elements are assumed to be
   stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
#include <cuda_fp16.h>   // For __half (binary16)
#include <cuda_bf16.h>
#include "fp_utils.h"

template <typename InputFormat, typename float_type>
void print_matrix(const std::vector<float_type>& A,
                  const size_t rows,
                  const size_t cols,
                  const bool bycols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << std::setw(6);
            auto next_element = bycols ? A[j * cols + i] : A[i * rows + j];
            
            if constexpr (std::is_same_v<InputFormat, binary16>) {
                std::cout << __half2float(next_element);  // Convert __half to float
            } else if constexpr (std::is_same_v<InputFormat, bfloat16>) {
                std::cout << __half2float(next_element);  // Convert bfloat16 to float
               //__bfloat162float(next_element)
            } else {
                std::cout << next_element;  // For other types, print directly
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}