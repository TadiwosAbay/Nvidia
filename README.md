# Accelerator PARANOIA

A suite of tests to assess the numerical behaviour of hardware accelerators for matrix multiplication.

## Build

The recommended system to build this project is `CMake`. The commands:
```shell
cmake -S. -Bbuild
cmake --build build
```
will build the hardware test suite for all supported hardware for which a compiler is available.

This will produce the executable `build/nvidia_test`.

## Tests

Tests are build by default in the `build/` directory. To disable them, you should turn off the `CMake` variable `BUILD_TESTS`. This can be achieved, for example, with the commands:
```shell
cmake -S. -Bbuild -DBUILD_TESTS=False
cmake --build build
```