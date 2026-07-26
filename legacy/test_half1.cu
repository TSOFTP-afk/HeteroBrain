// 二分法测试：只包含 config.h 的前半部分宏
#define N_SENSORY_NEURONS     2000
#define N_ASSOCIATION_NEURONS 6000
#define N_MOTOR_NEURONS       2000
#define N_TOTAL_NEURONS       (N_SENSORY_NEURONS + N_ASSOCIATION_NEURONS + N_MOTOR_NEURONS)
#define SYNAPSES_PER_NEURON   100
#define N_TOTAL_SYNAPSES      (N_TOTAL_NEURONS * SYNAPSES_PER_NEURON)
#define EXCITATORY_RATIO      0.8f
#define LIF_BETA              0.95f
#define LIF_THRESHOLD         1.0f
#define LIF_RESET             0.0f
#define LIF_REST              0.0f
#define LIF_REFRACTORY        2

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
    INHIBITORY = 1,
};

int main() { return 0; }
