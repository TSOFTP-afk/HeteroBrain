// 最小测试：只包含 cuda_runtime，不包含任何其他东西
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_kernel() {
    printf("hello from GPU\n");
}

int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("done\n");
    return 0;
}
