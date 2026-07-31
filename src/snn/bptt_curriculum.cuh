#ifndef SNN_STAGE2E_BPTT_CURRICULUM_CUH
#define SNN_STAGE2E_BPTT_CURRICULUM_CUH

// =============================================================================
// Stage 2e 课程训练调质 + 工具调用监督 (Phase 3a-D3)
// =============================================================================
// 发育期双任务监督 (知识框架: SNN 只学决策, 知识内容交给 TF):
//   1. 调质 readout (N×6):  spike → 6 维调质预测, 监督 = 目标调质 (MSE)
//   2. 工具 readout  (N×7):  spike → 7 类工具注意力 (6 工具 + 1 不调用),
//                            监督 = 目标工具索引 (CE/softmax)
// 总损失: L = w_mod·MSE(pred_mod, target_mod) + w_tool·CE(pred_tool, target_tool)
// 反传:   两路误差经各自 readout 权重合并注入 BPTT 最终步梯度
//         dL/dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                         + Σ_t w_tool·W_tool[i,t]·err_tool[t]
// =============================================================================

#include "memory_allocator.cuh"

namespace stage2e {

// 工具类数: 0-5 = 6 类工具索引, 6 = 不调用 (纯内部推理)
static const int CURRICULUM_N_TOOL = 7;

// Kernel 0: readout 权重初始化 (调质 N×6 + 工具 N×7, 小随机值对称破缺)
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed);

// Kernel 1: readout 前向 (调质 logits[6] + 工具 logits[7])
void launch_curriculum_readout_forward(PersistentBuffers& buf);

// Kernel 2: 误差 + 损失 (调质 MSE + 工具 CE)
//   mod_error[6], tool_error[7] 写入缓冲; out_loss = w_mod·MSE + w_tool·CE
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             int target_tool,
                             float w_mod, float w_tool,
                             float* out_loss);

// Kernel 3: 两路误差经 readout 权重合并反传到神经元 dL/dS_direct
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct,
                                float w_mod, float w_tool);

// Kernel 4: readout 权重更新 (调质 + 工具, SGD)
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr,
                                      float w_mod, float w_tool);

} // namespace stage2e

#endif // SNN_STAGE2E_BPTT_CURRICULUM_CUH
