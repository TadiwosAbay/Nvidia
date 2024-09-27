template <typename T> std::string get_type_name() {return "unknown storageFormat";}
template <> std::string get_type_name<bfloat16_t>() {return "bfloat16";}
template <> std::string get_type_name<binary16_t>() {return "binary16 (half)";}
template <> std::string get_type_name<binary32_t>() {return "binary32 (single)";}
template <> std::string get_type_name<binary64_t>() {return "binary64 (double)";}

template <typename InputFormat, typename OutputFormat>
void run_tests(){

    std::cout << "Input  Format: " << get_type_name<typename InputFormat::storageFormat>() << std::endl;
    std::cout << "Output Format: " << get_type_name<typename OutputFormat::storageFormat>() << std::endl;

}
