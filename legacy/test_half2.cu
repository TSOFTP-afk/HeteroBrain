// 二分法测试：后半部分宏
#define STDP_A_PLUS           0.01f
#define STDP_A_MINUS          0.01f
#define STDP_TAU_PLUS         20.0f
#define STDP_TAU_MINUS        20.0f
#define STDP_W_MIN            0.0f
#define STDP_W_MAX            1.0f
#define TIME_STEP_MS          1.0f
#define DEFAULT_TIME_STEPS    500
#define N_NEUROMODULATORS     2
#define MOD_DOPAMINE_DECAY    0.9f
#define MOD_DOPAMINE_RATE     0.1f
#define MOD_SEROTONIN_DECAY   0.9f
#define MOD_SEROTONIN_RATE    0.1f
#define THREADS_PER_BLOCK     256

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
    INHIBITORY = 1,
};

int main() { return 0; }
