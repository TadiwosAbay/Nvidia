CXX=nvcc
CXXFLAGS=-arch=native -std=c++20

all: nvidia_test

testing_suite: src/main.cu src/*
	$(CXX) -o $@ $(CXXFLAGS) $<

run: testing_suite
	./$<

.PHONY: deps
deps:
	cd deps/cpfloat && make lib

nvidia_test: test/runtests.cpp test/* src/* deps/cpfloat/build/include/cpfloat.h deps/cpfloat/build/lib/libcpfloat.so
	$(CXX) -o $@ $(CXXFLAGS) -I src/ -I deps/cpfloat/build/include/ -L deps/cpfloat/build/lib/ $< -l:libcpfloat.a

test: nvidia_test
	./$<

.PHONY: clean
clean:
	rm -f testing_suite nvidia_test
