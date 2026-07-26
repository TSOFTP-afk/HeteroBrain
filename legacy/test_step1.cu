// 步骤1：在 enum class 前加 cuda_runtime.h
#include <cuda_runtime.h>

enum class NeuronType : unsigned char {
    EXCITATORY = 0,
    INHIBITORY = 1,
};

int main() { return 0; }
