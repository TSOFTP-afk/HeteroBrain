// src/snn/embodied_motor.cu
// CUDA 版本: 从 GPU d_motor_spike_flags 读出动作
#include "embodied_motor.h"
#include <cuda_runtime.h>

namespace stage2e {

MotorReadout read_motor_output(const bool* d_motor_spike_flags) {
    // 拷贝到 host
    bool h_spike_flags[5000];
    cudaMemcpy(h_spike_flags, d_motor_spike_flags, 5000 * sizeof(bool),
               cudaMemcpyDeviceToHost);
    // 50组 × 100神经元 = 5000
    return compute_readout(h_spike_flags, 5000, 50, 100);
}

} // namespace stage2e
