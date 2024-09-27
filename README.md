# Accelerator PARANOIA

A suite of tests to assess the numerical behaviour of hardware accelerators for matrix multiplication.

## Compilation

The recommended system to build this project is `CMake`. The NVIDIA test suite can be built with:

```shell
mkdir build
cmake -B build
cd build && make
```

This will produce the executable `build/nvidia_test`.