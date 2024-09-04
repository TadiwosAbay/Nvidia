#ifndef CUSTOM_PRINT_H
#define CUSTOM_PRINT_H

#include <iostream>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Overload << operator for half type (binary16_t)
std::ostream& operator<<(std::ostream& os, const half& h_value) {
    float f_value = __half2float(h_value);
    os << f_value;
    return os;
}

// Overload << operator for bfloat16_t type
std::ostream& operator<<(std::ostream& os, const nv_bfloat16& bf_value) {
    float f_value = __bfloat162float(bf_value);
    os << f_value;
    return os;
}

#endif // CUSTOM_PRINT_H
