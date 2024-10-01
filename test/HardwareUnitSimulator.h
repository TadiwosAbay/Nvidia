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

    std::vector<double> partial_products;

    optstruct *fpopts_input_t, *fpopts_output_t;

    Features features;

    public:
        HardwareUnitSimulator(size_t M, size_t N, size_t K, Features features) :
                              HardwareUnit<InputFormat, OutputFormat>(M, N, K),
                              features(features) {
            partial_products.resize(K);
            fpopts_input_t = init_optstruct();
            fpopts_output_t = init_optstruct();
        }

        ~HardwareUnitSimulator() {
            free_optstruct(fpopts_input_t);
            free_optstruct(fpopts_output_t);
        }

    private:
        void run_mfma_kernel() {
            // Round to input format using round to nearest even.

            for (size_t i = 0; i < this->M; i++) {
                for (size_t j = 0; j < this->N; j++) {
                    // Compute partial products.
                    for (size_t k = 0; k < this->K; k++)
                        partial_products[k] = double(this->A[i * this->K + k]) * double(this->B[j * this->K + k]);
                    // Accumulate partial products.
                    double accumulator = 0;
                    for (size_t k = 0; k < this->K; k++) {
                        accumulator += partial_products[k];
                    }
                    this->C[i * this->N + j] += accumulator;
                }
            }
        }
};

#endif // HARDWARE_UNIT_SIMULATOR_H
