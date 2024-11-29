// fp_utils.h
#ifndef FP_UTILS_H
#define FP_UTILS_H

#include <cmath>
#include <limits>

template <typename input_t, typename output_t>
output_t convert(input_t x) {
    return static_cast<output_t>(x);
}

template <size_t Exponent, typename T>
constexpr T constexpr_ldexp(T x) {
    return x * (1 << Exponent);
}

using extended_float_t = double;
/*
 * All constants are computed in binary64 arithetmic and converted to the
 * target precision by the `convert` function.
 */
template <typename storage_format, size_t Precision, size_t Emax>
class IEEEFloatFormat {
private:
    static constexpr int precision = Precision;
    static constexpr int emax = Emax;

public:

    using storageFormat = storage_format;

    static size_t getPrecision() {
        return precision;
    }

    static int getEmax() {
        return emax;
    }

    static int getEmin() {
        return 1 - emax;
    }

    static bool isNormal(storage_format x) {
        return (x >= minNormal() && x <= maxNormal()) ||
            (x >= -maxNormal() && x <= -minNormal());
    }

    static bool isSubnormal(storage_format x) {
        return x >= -maxSubnormal() && x <= maxSubnormal() && x != convert<extended_float_t, storage_format>(0);
    }

    static constexpr storage_format unitRoundoff() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, -precision));
    }

    static constexpr storage_format machineEpsilon() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, 1 - precision));
    }

    static constexpr storage_format ulp(storage_format x = 1) {
        auto arg_exponent = ilogb(extended_float_t(x));
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, arg_exponent + 1 - precision));
    }

    static constexpr storage_format beforeOne() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0 - ldexp(1.0, -precision), 1));
    }

    static constexpr storage_format minSubnormal() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, 2 - emax - precision));
    }

    // This is MinNormal() / 2.
    static constexpr storage_format midwaySubnormal() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, -emax));
    }

    static constexpr storage_format maxSubnormal() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0 - ldexp(1.0, 1 - precision), 1 - emax));
    }

    static constexpr storage_format minNormal() {
        return convert<extended_float_t, storage_format>
            (ldexp(1.0, 1 - emax));
    }

    static constexpr storage_format maxNormal() {
        return convert<extended_float_t, storage_format>
            (ldexp(2.0 - ldexp(1.0, -precision), emax));
    }

    static constexpr storage_format constant(extended_float_t x) {
        return convert<extended_float_t, storage_format>(x);
    }

};

using binary32_t = float;
using binary64_t = double;

using binary32 = IEEEFloatFormat<binary32_t, 24, 127>;
using binary64 = IEEEFloatFormat<binary64_t, 53, 1023>;

template <typename T> std::string get_type_name() {return "unknown storageFormat";}
template <> std::string get_type_name<binary32_t>() {return "binary32 (single)";}
template <> std::string get_type_name<binary64_t>() {return "binary64 (double)";}

#endif // FP_UTILS_H
