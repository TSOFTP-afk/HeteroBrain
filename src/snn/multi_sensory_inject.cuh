// src/snn/multi_sensory_inject.cuh
#ifndef SNN_MULTI_SENSORY_INJECT_CUH
#define SNN_MULTI_SENSORY_INJECT_CUH

#include "thalamic_gate.cuh"
#include "config.h"

namespace stage2e {

// 多模态感知注入: 50柱信号 → L4层 input_current
// 每柱按信号强度×丘脑门控激活K个L4神经元
void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states);

} // namespace stage2e

#endif // SNN_MULTI_SENSORY_INJECT_CUH
