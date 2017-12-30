nvcc=/usr/local/cuda/bin/nvcc
ldflags=-lgtest
cxxflags=-std=c++11
build_dir=build

test: test.cu
	mkdir build
	$(nvcc) $(cxxflags) $(ldflags) -o $(build_dir)/test test.cu

dotest: test
	$(build_dir)/test

clean:
	rm -rf build
