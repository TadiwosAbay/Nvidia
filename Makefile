CXX=nvcc
CXXFLAGS=-arch=sm_80 -std=c++20

all: nvidia_test

nvidia_test: src/main.cu
	$(CXX) -o $@ $(CXXFLAGS) $<

.PHONY: clean
clean:
	rm -f nvidia_test
