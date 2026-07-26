// 测试：包含 cuda_runtime.h + 宏 + enum class
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#define N_SENSORY_NEURONS     2000
#define N_ASSOCIATION_NEURONS 6000
#define N_MOTOR_NEURONS       2000
#define N_TOTAL_NEURONS       (N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS + N_MOTOR_NEURONS)
#define EXCITATORY_RATIO      0.8f
#define LIF_REFRACTORY        2
#define STDP_W_MIN            0.0f
#define STDP_W_MAX            1.0f
#define THREADS_PER_BLOCK     256

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) {} } while(0)
#define CUDA_CHECK_LAST() do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) {} } while(0)

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
    INHIBITORY = 1,
};

int main() { return 0; }
