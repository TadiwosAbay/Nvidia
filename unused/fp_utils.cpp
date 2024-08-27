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
    using storageFormat = storage_format;

    static constexpr storageFormat minSubnormal() {
        return static_cast<storageFormat>(constexpr_ldexp<2 - emax - precision>(1));
    }

    static constexpr storageFormat largestSubnormal() {
        return (storageFormat(1) - std::numeric_limits<storageFormat>::epsilon()) * minNormal();
    }
    static constexpr storageFormat minNormal() {
        return storageFormat(1) / (storageFormat(1) << (emax - precision));
    }
    static constexpr storageFormat minimumNormal() {
        return storageFormat(1) / (storageFormat(1) << (emax-1));
    }
    static constexpr storageFormat largeSubnormal() {
        return storageFormat(1) / (storageFormat(1) << (emax));
    }

    static constexpr storageFormat one() {
        return storageFormat(1);
    }

    static constexpr storageFormat two() {
        return storageFormat(2);
    }

    static constexpr storageFormat four() {
        return storageFormat(4);
    }

    static constexpr storageFormat signedZero() {
        return storageFormat(-0.0);
    }
};

using binary16 = IEEEFloatFormat<binary16_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<binary16_t,  8, 127>;
using binary32 = IEEEFloatFormat<binary32_t, 24, 127>;
using binary64 = IEEEFloatFormat<binary64_t, 53, 1023>;

/* <template typename input_type>
std::ostream& operator<<(std::ostream& output_stream,
                         const input_type& value) {
    output_stream << static_cast<float>(value);
    return output_stream;
} */
