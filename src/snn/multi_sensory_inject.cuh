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

// =============================================================================
// Phase 3a-G (A): 事件→联合皮层直通注入 (2026-08-06)
// =============================================================================
// 根因修复: 事件调制对联合皮层传导 <2.3% (被文本流淹没) → readout 平均状态拟合器。
// 事件类型 k → 联合皮层固定子区域 [k*REGION, (k+1)*REGION) 注入电流,
// 与文本流 (感觉神经元路径) 并行互不覆盖; rate 携带事件信息后 readout 才有可学信号。
// 时序: scheduler.step() 在 delay_inject (清零 input_current) 之后、lif_adex 之前调用;
//   持续注入由 scheduler 按 EVENT_CORTEX_HOLD_STEPS 步控制 (每次调用注入一步)。
// event_type ∈ [0, EVT_COUNT), gain = 调用方算好的电流增益 (强度已缩放)
void launch_event_cortex_inject(int event_type, float gain, float* d_input_current);

} // namespace stage2e

#endif // SNN_MULTI_SENSORY_INJECT_CUH
