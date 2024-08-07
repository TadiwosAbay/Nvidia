// fp_utils.h
#ifndef FP_UTILS_H
#define FP_UTILS_H

#include <cmath>
#include <limits>

using binary32_t = float;
using binary64_t = double;

template <int Exponent, typename T>
constexpr T constexpr_ldexp(T x) {
    return x * (1 << Exponent);
}

template <typename storage_format, int Precision, int Emax>
class IEEEFloatFormat {
public:
    static constexpr int precision = Precision;
    static constexpr int emax = Emax;
    using type = storage_format;

    static constexpr type minSubnormal() {
        return static_cast<type>(constexpr_ldexp<2 - emax - precision>(1));
    }

    static constexpr type largestSubnormal() {
        return (type(1) - std::numeric_limits<type>::epsilon()) * minNormal();
    }
    
    static constexpr type minNormal() {
        return type(1) / (type(1) << (emax - precision));
    }

    static constexpr type one() {
        return type(1);
    }

    static constexpr type two() {
        return type(2);
    }

    static constexpr type four() {
        return type(4);
    }

    static constexpr type signedZero() {
        return type(-0.0);
    }
};

using binary16 = IEEEFloatFormat<binary16_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<binary16_t, 8, 127>;
using binary32 = IEEEFloatFormat<binary32_t, 24, 127>;
using binary64 = IEEEFloatFormat<binary64_t, 53, 1023>;

#endif // FP_UTILS_H
