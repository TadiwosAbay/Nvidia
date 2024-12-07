#ifndef FEATURES_H
#define FEATURES_H
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

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
        RoundingMode rounding_mode;
        size_t number_of_extra_bits;
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

        size_t getNumberOfExtraBits() {
            return number_of_extra_bits;
        }

        size_t getFmaSize() {
            return FmaSize;
        }

        Features(bool subnormalsFromSubnormals, bool subnormalsFromNormals, bool normalsFromSubnormals,
               bool subnormals_in_accumulator, bool multiplicationsAreExact,
               RoundingMode rounding_mode, size_t number_of_extra_bits, size_t FmaSize) : 
            subnormalsFromSubnormals(subnormalsFromSubnormals),
            subnormalsFromNormals(subnormalsFromNormals),
            normalsFromSubnormals(normalsFromSubnormals),
            subnormals_in_accumulator(subnormals_in_accumulator),
            multiplicationsAreExact(multiplicationsAreExact),
            rounding_mode(rounding_mode),
            number_of_extra_bits(number_of_extra_bits),
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
            std::cout << "Number of extra bits in accumulator: " << number_of_extra_bits << std::endl;
            switch (rounding_mode) {
                case RoundingMode::roundNotFaithful:
                    std::cout << "Rounding is not faithful." << std::endl;
                    break;
                case RoundingMode::roundToNearestEven:
                    std::cout << "The rounding mode is round-to-nearest ties-to-even." << std::endl;
                    break;
                case RoundingMode::roundToNearestZero:
                    std::cout << "The rounding mode is round-to-nearest ties-to-zero." << std::endl;
                    break;
                case RoundingMode::roundToNearestAway:
                    std::cout << "The rounding mode is round-to-nearest ties-to-away." << std::endl;
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
                case RoundingMode::roundUnknown:
                    std::cout << "The rounding mode cannot be determined." << std::endl;
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
                   (lhs.rounding_mode == rhs.rounding_mode) &&
                   (lhs.number_of_extra_bits == rhs.number_of_extra_bits) &&
                   (lhs.FmaSize == rhs.FmaSize);
        }

};
#endif // FEATURES_H
