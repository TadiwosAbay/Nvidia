cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nvidia_test LANGUAGES CXX)
include(CheckLanguage)

# Set C++ standard.
set(CMAKE_CXX_STANDARD 20)

# Set compiler flags.
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g -Og")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -flto")

# Build test suite for NVIDIA GPUs, if CUDA is available.
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    add_executable(nvidia_test src/main.cu)
    set_property(TARGET nvidia_test PROPERTY CUDA_ARCHITECTURES native)

    target_link_libraries(nvidia_test PRIVATE cudart)
endif()

# Build the tests.
option (BUILD_TESTS "Build tests" True)
if (BUILD_TESTS)
    add_custom_target(CPFloat
        COMMAND make lib
        WORKING_DIRECTORY ../deps/cpfloat)
    add_executable(runtests test/runtests.cpp)
    add_dependencies(runtests CPFloat)
    target_include_directories(runtests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(runtests PRIVATE ${PROJECT_SOURCE_DIR}/deps/cpfloat/build/include)
    target_link_libraries(runtests PRIVATE cpfloat)
    target_link_directories(runtests PRIVATE ${PROJECT_SOURCE_DIR}/deps/cpfloat/build/lib)
    set_property(TARGET runtests PROPERTY CXX_STANDARD 20)
endif()