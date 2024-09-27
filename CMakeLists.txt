cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(nvidia_test LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# Add executable
add_executable(nvidia_test main.cu)
set_property(TARGET nvidia_test PROPERTY CUDA_ARCHITECTURES native)

# Link with CUDA runtime library
target_link_libraries(nvidia_test PRIVATE cudart)
