// src/snn/multi_sensory_inject.cuh
#ifndef SNN_MULTI_SENSORY_INJECT_CUH
#define SNN_MULTI_SENSORY_INJECT_CUH

#include "thalamic_gate.cuh"
#include "config.h"

namespace stage2e {

// 2026-08-01 感知注入独立增益: 感知是"弱引导", 远弱于输入编码的 POP_CODING_GAIN(80)
//   根因: POP_CODING_GAIN 对 5000 L4 神经元注入过强 → PCA Oja 输入分布突变 → W 发散
//   取值: 10.0 = POP_CODING_GAIN 的 1/8, 维持感知可分辨又不破坏 PCA 统计稳定性
#define SENSORY_INJECT_GAIN 10.0f

// 多模态感知注入: 50柱信号 → L4层 input_current
// 每柱按信号强度×丘脑门控激活K个L4神经元
void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states);

} // namespace stage2e

#endif // SNN_MULTI_SENSORY_INJECT_CUH
