// 直接展开 types.h + config.h 的内容
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

// types.h 内容开始
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

static_assert(sizeof(NeuronState) == 24, "size mismatch");

struct Synapse {
    int   pre_idx;
    int   post_idx;
    float weight;
    float delay;
    float last_pre_spike;
    float last_post_spike;
};

static_assert(sizeof(Synapse) == 24, "size mismatch");

struct NeuromodulatorState {
    float dopamine;
    float serotonin;
    float energy;
    float arousal;
};

struct NetworkStats {
    int   total_spikes;
    int   excitatory_spikes;
    int   inhibitory_spikes;
    float mean_fire_rate;
    float mean_weight;
    int   active_synapses;
    float dopamine_level;
    float serotonin_level;
};

__host__ __device__ inline int region_start(BrainRegion region) {
    switch (region) {
        case BrainRegion::SENSORY:     return 0;
        case BrainRegion::ASSOCIATION: return N_SENSORY_NEURONS;
        case BrainRegion::MOTOR:       return N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS;
        default:                       return 0;
    }
}

__host__ __device__ inline int region_size(BrainRegion region) {
    switch (region) {
        case BrainRegion::SENSORY:     return N_SENSORY_NEURONS;
        case BrainRegion::ASSOCIATION: return N_ASSOCIATION_NEURONS;
        case BrainRegion::MOTOR:       return N_MOTOR_NEURONS;
        default:                       return 0;
    }
}
// types.h 内容结束

int main() { return 0; }
