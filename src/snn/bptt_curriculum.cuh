#ifndef SNN_STAGE2E_BPTT_CURRICULUM_CUH
#define SNN_STAGE2E_BPTT_CURRICULUM_CUH

// =============================================================================
// Stage 2e 课程训练调质监督 (Phase 3a-D3)
// =============================================================================
// 调质 readout 层: spike → 6 维调质预测, 与解码器同构 (N×6 权重矩阵)
// 课程损失: MSE(pred_mod, target_mod) + w_pad·MSE(pred_pad, target_pad)
// 反传: 调质误差经 readout 权重注入 BPTT 最终步梯度 (复用现有反向循环)
// =============================================================================

#include "memory_allocator.cuh"

namespace stage2e {

// Kernel 0: readout 权重初始化 W[i*6+m] = (rand-0.5)*2*init_scale
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed);

// Kernel 1: readout 前向 logits[m] = Σ_i W[i*6+m]·spike[i] → d_curriculum_logits[6]
void launch_curriculum_readout_forward(PersistentBuffers& buf);

// Kernel 2: error[m] = logits[m] - target[m]; loss = 0.5·Σ error² → d_curriculum_error[6]
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             float* out_loss);

// Kernel 3: dL_dS_direct[i] = Σ_m W[i*6+m]·error[m] → 写入 dL_dS_direct (调用方缓冲)
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct);

// Kernel 4: readout 权重更新 W[i*6+m] -= lr·error[m]·spike[i]
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr);

} // namespace stage2e

#endif // SNN_STAGE2E_BPTT_CURRICULUM_CUH
