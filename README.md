# hybrid_memory
Utility for easily managing C++ memory that switches between host and device address spaces.

Header-only class that wraps CUDA's `cudaMalloc` and `cudaMemcpy` functions to allow for easy switching between host and device memory. Tests are written using the GoogleTest library.
