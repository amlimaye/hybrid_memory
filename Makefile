nvcc=/usr/local/cuda/bin/nvcc
ldflags=-lgtest
cxxflags=-std=c++11 -g
build_dir=build
src_dir=src

test: Makefile $(src_dir)/test.cu
	mkdir -p $(build_dir)
	$(nvcc) $(cxxflags) $(ldflags) -o $(build_dir)/test $(src_dir)/test.cu

dotest: Makefile test
	$(build_dir)/test

clean: Makefile
	rm -rf $(build_dir)
