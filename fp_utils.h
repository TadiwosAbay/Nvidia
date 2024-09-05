// fp_utils.h
#ifndef FP_UTILS_H
#define FP_UTILS_H

#include <cmath>
#include <limits>
//#include "nvidia_specific_code.h"

using binary32_t = float;
using binary64_t = double;

template <typename input_t, typename output_t>
output_t convert(input_t x) {
    return static_cast<output_t>(x);
}

template <size_t Exponent, typename T>
constexpr T constexpr_ldexp(T x) {
    return x * (1 << Exponent);
}

/*
 * All constants are computed in binary64 arithetmic and converted to the
 * target precision by the `convert` function.
 */
template <typename storage_format, size_t Precision, size_t Emax>
class IEEEFloatFormat {
public:
    static constexpr int precision = Precision;
    static constexpr int emax = Emax;
    using storageFormat = storage_format;

    static bool isNormal(storage_format x) {
        return (x >= minNormal() && x <= maxNormal()) ||
            (x >= -maxNormal() && x <= -minNormal());
    }

    static bool isOne( storage_format x) {
        return x==one();
    }

    static bool isSubnormal(storage_format x) {
        return x >= -maxSubnormal() && x <= maxSubnormal() && x != convert<binary64_t, storage_format>(0);
    }

    static constexpr storage_format unitRoundoff() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, -precision));
    }

    static constexpr storage_format machineEpsilon() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, 1 - precision));
    }

    static constexpr storage_format ulp(storage_format x = 1) {
        auto arg_exponent = ilogb(binary64_t(x));
        return convert<binary64_t, storage_format>
            (ldexp(1.0, arg_exponent + 1 - precision));
    }

    static constexpr storage_format binary_power(int n) {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, n));
    }

    static constexpr storage_format beforeOne() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0 - ldexp(1.0, -precision), 1));
    }

    static constexpr storage_format minSubnormal() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, 2 - emax - precision));
    }

    // This is MinNormal() / 2.
    static constexpr storage_format midwaySubnormal() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, -emax));
    }

    static constexpr storage_format maxSubnormal() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0 - ldexp(1.0, 1 - precision), 1 - emax));
    }

    static constexpr storage_format minNormal() {
        return convert<binary64_t, storage_format>
            (ldexp(1.0, 1 - emax));
    }

    static constexpr storage_format maxNormal() {
        return convert<binary64_t, storage_format>
            (ldexp(2.0 - ldexp(1.0, -precision), emax));
    }

    static constexpr storage_format extra_bit() {
         if constexpr (std::is_same_v<storage_format, __half>) {
        return __float2half(1.0f) / __float2half(ldexpf(1.0f, precision));
    } else {
            return static_cast<storage_format>(std::ldexp(1.0, -precision));
        }
    }

    static constexpr storage_format constant(binary64_t x) {
        return convert<binary64_t, storage_format>(x);
    }

     
    static constexpr storage_format one() {
        return storage_format(1);
    }

    static constexpr storage_format two() {
        return storage_format(2);
    }

    static constexpr storage_format four() {
        return storage_format(4);
    }

    static constexpr storage_format signedZero() {
        return storage_format(-0.0);
    }
};

using binary16 = IEEEFloatFormat<binary16_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<binary16_t, 8, 127>;
using binary32 = IEEEFloatFormat<binary32_t, 24, 127>;
using binary64 = IEEEFloatFormat<binary64_t, 53, 1023>;

#endif // FP_UTILS_H
