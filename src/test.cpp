#include <iostream>
#include <string>

#include "fp_utils.h"

template <typename InputFormat, typename OutputFormat>
void run_tests(){

    std::cout << "Input  Format: " << get_type_name<typename InputFormat::storageFormat>() << std::endl;
    std::cout << "Output Format: " << get_type_name<typename OutputFormat::storageFormat>() << std::endl;

}