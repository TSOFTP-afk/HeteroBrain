#ifndef SNN_STAGE2E_BPTT_CURRICULUM_CUH
#define SNN_STAGE2E_BPTT_CURRICULUM_CUH

// =============================================================================
// Stage 2e 课程训练调质 + 工具调用监督 + PAD 情感 readout (Phase 3a-D3)
// =============================================================================
// 发育期多任务监督 (知识框架: SNN 只学决策, 知识内容交给 TF):
//   1. 调质 readout (N×6):  spike → 6 维调质预测, 监督 = 目标调质 (MSE)
//   2. 工具 readout  (N×7):  spike → 7 类工具注意力 (6 工具 + 1 不调用),
//                            监督 = 目标工具索引 (CE/softmax)
//   3. PAD readout   (N×3):  spike → 3 维 PAD 情感预测 [Pleasure, Arousal,
//                            Dominance], 监督 = 目标 PAD (MSE)  (2026-08-02 Task 5)
// 总损失: L = w_mod·MSE(pred_mod, target_mod)
//           + w_pad·MSE(pred_pad, target_pad)
//           + w_tool·CE(pred_tool, target_tool)
// 反传:   三路误差经各自 readout 权重合并注入 BPTT 最终步梯度 / N3F eligibility
//         dL/dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                         + Σ_p w_pad·W_pad[i,p]·err_pad[p]
//                         + Σ_t w_tool·W_tool[i,t]·err_tool[t]
// =============================================================================

#include "memory_allocator.cuh"

namespace stage2e {

// 工具类数: 0-5 = 6 类工具索引, 6 = 不调用 (纯内部推理)
static const int CURRICULUM_N_TOOL = 7;

// =============================================================================
// 调质通道顺序契约 (2026-08-03 固化, 防止跨链路语义错位):
//   readout 调质头 / 课程模拟器 conc / target_mod 统一使用 GENE_MAP 列顺序:
//     [DA, ACh, NE, 5HT, GABA, Oxy]  (索引 0-5)
//   通道索引常量定义见 mod_simulator.h: MOD_CH_DA / MOD_CH_ACH / MOD_CH_NE /
//     MOD_CH_5HT / MOD_CH_GABA / MOD_CH_OXY (GENE_MAP 顺序).
//   ⚠️ personality_profiles.h 的 baseline_mod 是不同顺序:
//     [DA, 5HT, NE, ACh, GABA, Oxy] — 跨接口 (main.cpp base_conc /
//     test_event_scheduler.cpp base_signal) 使用前必须按 {0, 3, 2, 1, 4, 5} 重排!
//   modulatory_kernels.cu 按 personality 顺序直接解包 stage_baseline (与定义一致).
//   取 readout/模拟器通道时一律用 MOD_CH_* 常量, 勿用裸索引.
// =============================================================================

// 2026-08-01 spec §7.8 修复: readout 权重裁剪阈值
//   若某些神经元 spike_rates[i] 系统性偏高, 无裁剪的 SGD 更新会发散
//   clip 到 [-10, 10] 防长时间训练数值不稳定 (logits 量级 ~ 60K×0.01 ≈ 600)
//   用宏而非 static const float: 后者在 CUDA device code 中不可直接引用
#define CURRICULUM_READOUT_WEIGHT_CLIP (10.0f)

// Kernel 0: readout 权重初始化 (调质 N×6 + 工具 N×7 + PAD N×3, 小随机值对称破缺)
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed);

// Kernel 0b: PAD readout 权重专用初始化 (仅旧 checkpoint 缺失 PAD 节时调用,
//   不触碰 mod/tool 权重 — 避免重随机化已加载的 readout 头)
void launch_curriculum_pad_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed);

// Kernel 1: readout 前向 (调质 logits[6] + 工具 logits[7])
void launch_curriculum_readout_forward(PersistentBuffers& buf);

// Kernel 1c: PAD readout 前向 (累计帧: 窗口平均发放率 → logits_pad[3])
void launch_curriculum_pad_forward(PersistentBuffers& buf);

// Kernel 2: 误差 + 损失 (调质 MSE + PAD MSE + 工具 CE)
//   mod_error[6], pad_error[3], tool_error[7] 写入缓冲;
//   out_loss = w_mod·MSE_mod + w_pad·MSE_pad + w_tool·CE
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             int target_tool,
                             const float target_pad[3],
                             float w_mod, float w_tool, float w_pad,
                             float* out_loss);

// Kernel 3: 三路误差经 readout 权重合并反传到神经元 dL/dS_direct
//   w_pad 默认 0.0f: BPTT 路径 (bptt_trainer.backward_curriculum, 不可修改文件)
//   以 4 参调用 → PAD 项数学上无贡献 (安全默认); N3F eligibility 路径显式传 w_pad
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct,
                                float w_mod, float w_tool, float w_pad = 0.0f);

// Kernel 4: readout 权重更新 (调质 + 工具 + PAD, SGD)
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr,
                                      float w_mod, float w_tool, float w_pad);

// Kernel 5: 课程窗口内逐帧累计 spike 平均发放率
//   accum[i] += (spike_flags[i] ? 1.0f : 0.0f) / window_size
//   窗口开始先 launch_curriculum_accum_clear 清零
void launch_curriculum_accum_clear(PersistentBuffers& buf);
void launch_curriculum_accumulate(PersistentBuffers& buf, int window_size);

// Kernel 6 (N3F): readout 前向 — 当前帧 bool spike (在线推理/教学信号用)
//   与累计版本区别: 输入为当前步 spike_flags, 非窗口累计率
void launch_curriculum_readout_forward_frame(PersistentBuffers& buf);

// Kernel 6c (N3F): PAD readout 前向 — 当前帧 bool spike → logits_pad[3]
void launch_curriculum_pad_forward_frame(PersistentBuffers& buf);

// Kernel 7 (N3F): 课程误差 → 神经元级 eligibility (教学信号注入)
//   neuron_elig[i] = λ·elig[i] - g·(Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                                 + Σ_p w_pad·W_pad[i,p]·err_pad[p]
//                                 + Σ_t w_tool·W_tool[i,t]·err_tool[t])
//   与 decode_eligibility_update_kernel 同构, 误差源换成课程监督
//   之后 STDP kernel 读取 neuron_eligibility[post] 调制突触证据 (三因子闭环)
void launch_curriculum_eligibility_update(PersistentBuffers& buf,
                                          float decay_factor, float gain,
                                          float w_mod, float w_tool, float w_pad);

// Kernel 8 (N3F, 2026-08-01 spec §7.1): 具身奖励 → 神经元级 eligibility
//   neuron_elig[i] = λ·elig[i] + g·reward   (reward ∈ [-1,1])
//   生物对应: DA 奖赏广播 (uniform), reward>0 强化 / reward<0 削弱
//   第三因子来源从"课程误差"切换为"沙盒 feedback" (内稳态改善/社会反馈/任务结果)
void launch_embodied_eligibility_update(PersistentBuffers& buf,
                                        float reward, float decay_factor, float gain);

} // namespace stage2e

#endif // SNN_STAGE2E_BPTT_CURRICULUM_CUH
