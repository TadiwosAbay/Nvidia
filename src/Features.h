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
        bool subnormalsFromSubnormals;
        bool subnormalsFromNormals;
        bool normalsFromSubnormals;
        bool subnormals_in_accumulator;
        bool multiplicationsAreExact;
        bool extra_bit;
        bool extra_bits_three_bit;
        RoundingMode rounding_mode;
        size_t FmaSize;

    public:

        bool getSubnormalsFromSubnormals() {
            return subnormalsFromSubnormals;
        }

        bool getSubnormalsFromNormals() {
            return subnormalsFromNormals;
        }

        bool getNormalsFromSubnormals() {
            return normalsFromSubnormals;
        }

        RoundingMode getRoundingMode() {
        return rounding_mode;
        }

        size_t getFmaSize() {
            return FmaSize;
        }

        bool hasExtraBitsThreeBit() const {
            return extra_bits_three_bit;
        }

        Features(bool subnormalsFromSubnormals, bool subnormalsFromNormals, bool normalsFromSubnormals,
               bool subnormals_in_accumulator, bool multiplicationsAreExact,
               bool extra_bit, bool extra_bits_three_bit, RoundingMode rounding_mode, size_t FmaSize) : 
            subnormalsFromSubnormals(subnormalsFromSubnormals),
            subnormalsFromNormals(subnormalsFromNormals),
            normalsFromSubnormals(normalsFromSubnormals),
            subnormals_in_accumulator(subnormals_in_accumulator),
            multiplicationsAreExact(multiplicationsAreExact),
            extra_bit(extra_bit),
            extra_bits_three_bit(extra_bits_three_bit),
            rounding_mode(rounding_mode),
            FmaSize(FmaSize) {};

        void print_report() {
            if (subnormalsFromSubnormals) {
                std::cout << "Produces subnormals from subnormals." << std::endl;
            } else {
                std::cout << "Does not produce subnormals from subnormals." << std::endl;
            }
            if (subnormalsFromNormals) {
                std::cout << "Produces subnormals from normals." << std::endl;
            } else {
                std::cout << "Does not produce subnormals from normals." << std::endl;
            }
            if (normalsFromSubnormals) {
                std::cout << "Produces normals from subnormals." << std::endl;
            } else {
                std::cout << "Does not produce normals from subnormals." << std::endl;
            }
            if (subnormals_in_accumulator) {
                std::cout << "Subnormals in accumulator are kept." << std::endl;
            } else {
                std::cout << "Subnormals in accumulator are lost." << std::endl;
            }
            if (multiplicationsAreExact) {
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
            std::cout << "FMA size: " << FmaSize << std::endl;
        }

        friend bool operator==(const Features& lhs, const Features& rhs) {
            return (lhs.subnormalsFromSubnormals == rhs.subnormalsFromSubnormals) && 
                   (lhs.subnormalsFromNormals == rhs.subnormalsFromNormals) &&
                   (lhs.normalsFromSubnormals == rhs.normalsFromSubnormals) &&
                   (lhs.subnormals_in_accumulator == rhs.subnormals_in_accumulator) &&
                   (lhs.multiplicationsAreExact == rhs.multiplicationsAreExact) &&
                   (lhs.extra_bit == rhs.extra_bit) &&
                   (lhs.extra_bits_three_bit == rhs.extra_bits_three_bit) &&
                   (lhs.rounding_mode == rhs.rounding_mode) &&
                   (lhs.FmaSize == rhs.FmaSize);
        }

};
#endif // FEATURES_H
