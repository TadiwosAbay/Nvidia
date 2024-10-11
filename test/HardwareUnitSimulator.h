#ifndef HARDWARE_UNIT_SIMULATOR_H
#define HARDWARE_UNIT_SIMULATOR_H

#include "fp_utils.h"
#include "Features.h"
#include "HardwareUnit.h"

extern "C" {
    #include "cpfloat.h"
}

template <typename InputFormat, typename OutputFormat>
class HardwareUnitSimulator : public HardwareUnit<InputFormat, OutputFormat> {

    using input_t = typename InputFormat::storageFormat;
    using output_t = typename OutputFormat::storageFormat;
    using internal_t = double;

    std::vector<double> Ad, Bd, Cd, partial_products;

    optstruct *fpopts_input_t, *fpopts_output_t, *fpopts_accumulator_t;

    Features features;

    size_t num_extra_bits = 1;

    public:
        HardwareUnitSimulator(size_t M, size_t N, size_t K, Features features) :
                              HardwareUnit<InputFormat, OutputFormat>(M, N, K),
                              features(features) {
            partial_products.resize(features.getFmaSize());
            Ad.resize(this->A_size);
            Bd.resize(this->B_size);
            Cd.resize(this->C_size);

            // Allocate memory for CPFloat formats.
            fpopts_input_t = init_optstruct();
            fpopts_accumulator_t = init_optstruct();
            fpopts_output_t = init_optstruct();

            // Initialize CPFloat formats.
            fpopts_input_t->p = InputFormat::getPrecision();
            fpopts_input_t->emin = InputFormat::getEmin();
            fpopts_input_t->emax = InputFormat::getEmax();
            fpopts_input_t->explim = CPFLOAT_EXPRANGE_STOR;
            fpopts_input_t->infinity = CPFLOAT_INF_USE;
            fpopts_input_t->round = CPFLOAT_RND_NE;
            fpopts_input_t->saturation = CPFLOAT_SAT_NO;
            if (features.getNormalsFromSubnormals() || features.getSubnormalsFromSubnormals())
                fpopts_input_t->subnormal = CPFLOAT_SUBN_USE;
            else
                fpopts_input_t->subnormal = CPFLOAT_SUBN_RND; // I should implement FTZ!
            fpopts_input_t->subnormal = CPFLOAT_SUBN_USE;
            fpopts_input_t->flip = CPFLOAT_SOFTERR_NO;

            // Initialize CPFloat formats.
            fpopts_accumulator_t->p = OutputFormat::getPrecision();
            fpopts_accumulator_t->emin = OutputFormat::getEmin();
            fpopts_accumulator_t->emax = OutputFormat::getEmax();
            fpopts_accumulator_t->explim = CPFLOAT_EXPRANGE_STOR;
            fpopts_accumulator_t->infinity = CPFLOAT_INF_USE;
            fpopts_accumulator_t->round = CPFLOAT_RND_TZ;
            fpopts_accumulator_t->saturation = CPFLOAT_SAT_NO;
            fpopts_accumulator_t->subnormal = CPFLOAT_SUBN_USE;
            fpopts_accumulator_t->flip = CPFLOAT_SOFTERR_NO;

            // Initialize CPFloat formats.
            fpopts_output_t->p = OutputFormat::getPrecision();
            fpopts_output_t->emin = OutputFormat::getEmin();
            fpopts_output_t->emax = OutputFormat::getEmax();
            fpopts_output_t->explim = CPFLOAT_EXPRANGE_STOR;
            fpopts_output_t->infinity = CPFLOAT_INF_USE;
            switch (features.getRoundingMode()) {
                case RoundingMode::roundToNearestEven:
                    fpopts_output_t->round = CPFLOAT_RND_NE;
                    break;
                case RoundingMode::roundDown:
                    fpopts_output_t->round = CPFLOAT_RND_TN;
                    break;
                case RoundingMode::roundUp:
                    fpopts_output_t->round = CPFLOAT_RND_TP;
                    break;
                case RoundingMode::roundToZero:
                    fpopts_output_t->round = CPFLOAT_RND_TZ;
                    break;
                case RoundingMode::roundToNearestAway:
                    fpopts_output_t->round = CPFLOAT_RND_NA;
                    break;
                case RoundingMode::roundToNearestZero:
                    fpopts_output_t->round = CPFLOAT_RND_NZ;
                    break;
            }
            fpopts_output_t->saturation = CPFLOAT_SAT_NO;
            if (features.getSubnormalsFromNormals() || features.getSubnormalsFromSubnormals())
                fpopts_output_t->subnormal = CPFLOAT_SUBN_USE;
            else
                fpopts_output_t->subnormal = CPFLOAT_SUBN_RND;  // I should implement FTZ!
            fpopts_output_t->flip = CPFLOAT_SOFTERR_NO;
        }

        ~HardwareUnitSimulator() {
            free_optstruct(fpopts_input_t);
            free_optstruct(fpopts_accumulator_t);
            free_optstruct(fpopts_output_t);
        }

    public:
        void run_mfma_kernel() {
            // Copy input matrices to internal format.
            for (size_t i = 0; i < this->A_size; i++)
                Ad[i] = double(this->A[i]);
            for (size_t i = 0; i < this->B_size; i++)
                Bd[i] = this->B[i];
            for (size_t i = 0; i < this->C_size; i++)
                Cd[i] = this->C[i];

            // Round to input format using round to nearest even.
            cpf_round(this->Ad.data(), this->Ad.data(), this->A_size, fpopts_input_t);
            cpf_round(this->Bd.data(), this->Bd.data(), this->B_size, fpopts_input_t);
            cpf_round(this->Cd.data(), this->Cd.data(), this->C_size, fpopts_input_t);

            int accumulator_precision = OutputFormat::getPrecision() + num_extra_bits;

            // Compute elements one by one.
            for (size_t i = 0; i < this->M; i++) {
                for (size_t j = 0; j < this->N; j++) {
                    for (size_t block = 0; block < this->K; block += features.getFmaSize()) {

                        // Compute partial products exactly.
                        size_t number_of_elements = std::min(features.getFmaSize(), this->K - block);
                        for (size_t k = 0; k < number_of_elements; k++)
                            partial_products[k] = Ad[i * this->K + (block + k)] * Bd[j * this->K + (block + k)];

                        // Find maximum-magnitude value.
                        int max_exp = INT_MIN;
                        for (size_t k = 0; k < number_of_elements; k++) {
                            int exp = std::ilogb(partial_products[k]);
                            max_exp = std::max(max_exp, exp);
                        }

                        // Simulate alignment of the sgniﬁcands.
                        for (size_t k = 0; k < number_of_elements; k++) {
                            int curr_exp = std::ilogb(partial_products[k]);
                            fpopts_accumulator_t->precision = accumulator_precision - (max_exp - curr_exp);
                            cpf_round(&partial_products[k], &partial_products[k], 1, fpopts_accumulator_t);
                        }

                        // Accumulate partial products.
                        double accumulator = 0;
                        for (size_t k = 0; k < number_of_elements; k++)
                            accumulator += partial_products[k];

                        // Round to output format.
                        cpf_round(&accumulator, &accumulator, 1, fpopts_output_t);

                        // Keep accumulation.
                        this->Cd[i * this->N + j] += accumulator;
                        cpf_round(&Cd[i * this->N + j], &Cd[i * this->N + j], 1, fpopts_output_t);
                    }
                }
            }

            for (size_t i = 0; i < this->C_size; i++)
                this->C[i] = Cd[i];
        }
};

#endif // HARDWARE_UNIT_SIMULATOR_H
