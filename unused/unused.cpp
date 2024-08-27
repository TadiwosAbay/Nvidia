/* using binary16 = IEEEFloatFormat<binary16_t, 11, 15>;
using bfloat16 = IEEEFloatFormat<binary16_t, 8, 127>;
using binary32 = IEEEFloatFormat<binary32_t, 24, 127>;
using binary64 = IEEEFloatFormat<binary64_t, 53, 1023>;

template <typename T> size_t return_emax() {return 0}
template <> size_t return_emax<binary16_t> () {return 15}
template <> size_t return_emax<bfloat16_t> () {return 127}
template <> size_t return_emax<binary32_t> () {return 127}
template <> size_t return_emax<binary64_t> () {return 1023}

template <typename T> size_t return_precision() {return 0}
template <> size_t return_precision<binary16_t> () {return 11}
template <> size_t return_precision<bfloat16_t> () {return 8}
template <> size_t return_precision<binary32_t> () {return 24}
template <> size_t return_precision<binary64_t> () {return 53} */