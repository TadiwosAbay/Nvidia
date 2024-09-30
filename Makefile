CXX=nvcc
CXXFLAGS=-arch=native -std=c++20

all: nvidia_test

testing_suite: src/main.cu src/*
	$(CXX) -o $@ $(CXXFLAGS) $<

run: testing_suite
	./$<

nvidia_test: test/runtests.cpp test/* src/*
	$(CXX) -o $@ $(CXXFLAGS) -I src/ $<

test: nvidia_test
	./$<

.PHONY: clean
clean:
	rm -f testing_suite nvidia_test
