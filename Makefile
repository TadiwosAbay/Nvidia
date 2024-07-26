CXX=nvcc
CXXFLAGS=-arch=sm_70 -std=c++17

all: nvidia_test

nvidia_test: main.cu
	$(CXX) -o $@ $(CXXFLAGS) $<

.PHONY: clean
clean:
	rm -f nvidia_test
