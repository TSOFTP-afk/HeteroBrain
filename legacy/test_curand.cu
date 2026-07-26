#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
__global__ void k() { printf("hi\n"); }
int main() { k<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
