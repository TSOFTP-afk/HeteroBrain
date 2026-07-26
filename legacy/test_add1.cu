// 逐步添加：先加 struct NeuronState
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
#define CUDA_CHECK_LAST() do {} while(0)

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
    INHIBITORY = 1,
};

enum class BrainRegion : unsigned char {
    SENSORY     = 0,
    ASSOCIATION = 1,
    MOTOR       = 2,
};

struct NeuronState {
    float membrane_potential;
    float synaptic_current;
    int   last_spike_time;
    int   refractory_remaining;
    NeuronType type;
    BrainRegion region;
    float fire_rate;
    unsigned char padding[2];
};

// static_assert temporarily disabled to check actual size
// static_assert(sizeof(NeuronState) == 24, "size mismatch");

int main() {
    printf("NeuronState size: %zu\n", sizeof(NeuronState));
    printf("NeuronType size: %zu\n", sizeof(NeuronType));
    printf("BrainRegion size: %zu\n", sizeof(BrainRegion));
    return 0;
}
