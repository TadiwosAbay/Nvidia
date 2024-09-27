#ifndef FEATURES_H
#define FEATURES_H
#include <cstddef>
#include <cstdint>
#include <iostream>

enum class RoundingMode : uint8_t {
    roundNotFaithful = 0,
    roundToNearestEven = 1,
    roundToNearestZero = 2,
    roundToNearestAway = 3,
    roundUp = 4,
    roundDown = 5,
    roundToZero = 6,
    roundUnknown = 7
};

class Features {
    private:
        bool subnormals_from_subnormals;
        bool subnormals_from_normals;
        bool normals_from_subnormals;
        bool subnormals_in_accumulator;
        bool multiplications_exact;
        bool extra_bit;
        RoundingMode rounding_mode;
        size_t fma_size;

    public:
        Features(bool subnormals_from_subnormals, bool subnormals_from_normals, bool normals_from_subnormals,
               bool subnormals_in_accumulator, bool multiplications_exact,
               bool extra_bit, RoundingMode rounding_mode, size_t fma_size) : 
            subnormals_from_subnormals(subnormals_from_subnormals),
            subnormals_from_normals(subnormals_from_normals),
            normals_from_subnormals(normals_from_subnormals),
            subnormals_in_accumulator(subnormals_in_accumulator),
            multiplications_exact(multiplications_exact),
            extra_bit(extra_bit),
            rounding_mode(rounding_mode),
            fma_size(fma_size) {};

    void print_report() {
        if (subnormals_from_subnormals) {
            std::cout << "Produces subnormals from subnormals." << std::endl;
        } else {
            std::cout << "Does not produce subnormals from subnormals." << std::endl;
        }
        if (subnormals_from_normals) {
            std::cout << "Produces subnormals from normals." << std::endl;
        } else {
            std::cout << "Does not produce subnormals from normals." << std::endl;
        }
        if (normals_from_subnormals) {
            std::cout << "Produces normals from subnormals." << std::endl;
        } else {
            std::cout << "Does not produce normals from subnormals." << std::endl;
        }
        if (subnormals_in_accumulator) {
            std::cout << "Subnormals in accumulator are kept." << std::endl;
        } else {
            std::cout << "Subnormals in accumulator are lost." << std::endl;
        }
        if (multiplications_exact) {
            std::cout << "Multiplications are exact." << std::endl;
        } else {
            std::cout << "Multiplications are not exact." << std::endl;
        }
        if (extra_bit) {
            std::cout << "Accumulator has one extra bit." << std::endl;
        } else {
            std::cout << "Accumulator does not have extra bits." << std::endl;
        }
        switch (rounding_mode) {
            case RoundingMode::roundToNearestEven:
                std::cout << "The rounding mode is round to nearest even." << std::endl;
                break;
            case RoundingMode::roundToNearestZero:
                std::cout << "The rounding mode is round to nearest zero." << std::endl;
                break;
            case RoundingMode::roundToNearestAway:
                std::cout << "The rounding mode is round to nearest away." << std::endl;
                break;
            case RoundingMode::roundUp:
                std::cout << "The rounding mode is round up." << std::endl;
                break;
            case RoundingMode::roundDown:
                std::cout << "The rounding mode is round down." << std::endl;
                break;
            case RoundingMode::roundToZero:
                std::cout << "The rounding mode is round to zero." << std::endl;
                break;
        }
        std::cout << "FMA size: " << fma_size << std::endl;
    }

};
#endif // FEATURES_H