nvcc=/usr/local/cuda/bin/nvcc
ldflags=-lgtest
cxxflags=-std=c++11 -g
build_dir=build

test: Makefile test.cu
	mkdir -p build
	$(nvcc) $(cxxflags) $(ldflags) -o $(build_dir)/test test.cu

dotest: Makefile test
	$(build_dir)/test

clean: Makefile
	rm -rf build
