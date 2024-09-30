/* Print the elements of the m x n matrix A. The elements are assumed to be
   stored by columns if `bycols` is `true` and by rows if `bycols` is false. */
#include <iostream>
#include <vector>
#include "fp_utils.h"

template <typename float_type>
void print_matrix(const std::vector<float_type>& A,
                  const size_t rows,
                  const size_t cols,
                  const bool bycols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout.precision(20);
            auto next_element = bycols ? A[j * cols + i] : A[i * rows + j];
            std::cout << convert<float_type, binary64_t>(next_element) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
