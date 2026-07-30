#ifndef SNN_STAGE2E_MODULATORY_KERNELS_CUH
#define SNN_STAGE2E_MODULATORY_KERNELS_CUH

// =============================================================================
// Stage 2e 调质系统 + DA价值函数 + 字节选择性统计 (P2)
// Phase 3a 扩充: 6 维调质向量 (DA/ACh/NE/5HT + GABA/催产素) + AffectiveState readout
// =============================================================================
// 对应设计文档 §3.1-§3.2:
//   - modulatory_kernel: 6 种调质浓度动力学 (每100步)
//   - da_value_function: TD error + novelty + pred_succ (每100步)
//   - byte_histogram_kernel: 字节选择性统计 (每注入步)
//
// Phase 3a 新增 (2026-07-30):
//   - GABA 全局浓度变量 (复用现有 GABA_A/B 受体, 抗焦虑反馈)
//   - 催产素浓度 + 受体 (共情/社交联结, 慢变量 τ=500ms)
//   - AffectiveState readout: 6 维调质 → PAD 情感模型 + LLM 调制信号
//   - 详见 docs/snn-emotion-and-workspace-direction.md §3.3, §4.1
//
// 调质浓度已分配: d_da/ach/ne/ht5/gaba/oxytocin_concentration (60K × 4B × 6 = 1.44MB)
// 价值函数已分配: d_w_value (200), d_w_pred (200×200), d_pred_fr (200),
//                  d_subcolumn_fr (200), d_baseline_fr (200), d_subcol_fr_prev (200)
// 字节直方图: d_byte_histogram (256)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 调质浓度动力学 (每100步)
// 输入: d_da/ach/ne/ht5/gaba/oxytocin_concentration (当前浓度)
//       reward_signal, novelty, pred_succ, kl_divergence (host 传入)
//       prediction_error_norm: 在线解码预测误差 L2 范数 (||error||, ∈ [0, ~1.414])
//       empathy_signal: 共情驱动信号 [0,1] (用户情绪宣泄时上升, Phase 3a)
// 输出: 更新后的 6 维调质浓度
//
// DA: δ(t) = R(t) + γ·V(s') - V(s); DA半衰期5步
//    预测误差耦合: DA = DA_BASE + DA_GAIN × (1 - ||error||) + TD 驱动
// ACh: 基线0.2, 惊奇+Δ, 注意力+Δ
// NE: 基线0.05, KL散度触发脉冲
// 5HT: 基线0.1, 预测误差持续负时上升
// GABA: 基线0.15, NE 过高时上升 (抗焦虑负反馈, Phase 3a)
// 催产素: 基线0.05, 共情信号驱动上升 (Phase 3a)
//
// DA 释放区域: [0, N_TOTAL_NEURONS_2E) = [0, 60000), 包含联合皮层 + 前额叶 + 运动皮层
void launch_modulatory(MemoryAllocator* alloc, int step,
                       float reward_signal, float novelty,
                       float pred_succ, float kl_divergence,
                       float da_delta,
                       float prediction_error_norm,
                       float empathy_signal = 0.0f);  // Phase 3a

// DA价值函数更新 (每100步)
void launch_da_value_function(MemoryAllocator* alloc, int step,
                              float reward, float* out_v_s, float* out_v_sp);

// 字节选择性直方图更新 (每注入步)
void launch_byte_histogram(MemoryAllocator* alloc, uint8_t current_byte);
void get_byte_histogram(MemoryAllocator* alloc, int* out_hist);

// 调质系统统计 (host 端读取, Phase 3a 扩充到 6 维)
struct ModulatoryStats {
    float da_mean;
    float ach_mean;
    float ne_mean;
    float ht5_mean;
    float gaba_mean;        // Phase 3a
    float oxytocin_mean;    // Phase 3a
    float v_s;
    float v_sp;
    float da_delta;
    float novelty;
    float pred_succ;
};

struct ModulatoryRuntimeState {
    float v_s;
    float v_sp;
};

ModulatoryStats get_modulatory_stats(MemoryAllocator* alloc);
ModulatoryRuntimeState export_modulatory_runtime_state();
void import_modulatory_runtime_state(const ModulatoryRuntimeState& state);

// =============================================================================
// Phase 3a: AffectiveState readout — 6 维调质 → 情感状态 + LLM 调制信号
// =============================================================================
// 详见 docs/snn-emotion-and-workspace-direction.md §3.2, §6.7
//
// PAD 情感模型映射 (Pleasure-Arousal-Dominance):
//   P (愉悦) = +DA + 内啡肽(暂未建模) - 5HT低 - GABA低
//   A (唤醒) = +NE - GABA - 5HT
//   D (主导) = +DA - 催产素 (催产素高→更顺从/共情)
//
// LLM 调制信号:
//   temperature_delta: DA↑→+0.3, 5HT↑→-0.3, GABA↑→-0.1 (冷静)
//   top_p_delta:       NE↑→-0.2 (更聚焦)
//   repetition_delta:  NE↑→+0.1 (避免重复跑题)
//   empathy_level:     催产素 [0,1] (注入 system prompt)
// =============================================================================

struct AffectiveState {
    // 6 维调质快照 (来自浓度均值)
    float dopamine;         // [0, 2]
    float serotonin;        // [0, 2]
    float norepinephrine;   // [0, 2]
    float acetylcholine;    // [0, 2]
    float gaba;             // [0, 2]
    float oxytocin;         // [0, 2]

    // PAD 情感模型 (映射自 6 维调质)
    float pleasure;         // [-1, 1]  愉悦度
    float arousal;          // [-1, 1]  唤醒度
    float dominance;        // [-1, 1]  主导度

    // LLM 生成调制信号 (delta, 叠加到 LLM 默认参数上)
    float temperature_delta;    // [-0.5, +0.5]
    float top_p_delta;          // [-0.3, 0]
    float repetition_delta;     // [0, +0.2]
    float empathy_level;        // [0, 1]  共情强度 (注入 system prompt)

    // 元信息
    int   step;                 // 当前训练步
    float confidence;           // 状态置信度 [0,1] (基于调质浓度方差)
};

// 从当前 6 维调质浓度提取 AffectiveState (host 端, 每 100 步或每轮对话调用)
AffectiveState get_affective_state(MemoryAllocator* alloc, int step);

// 设置外部共情驱动信号 (host 端, 由 LLM/用户反馈触发, 影响下一轮 launch_modulatory)
//   empathy_signal ∈ [0, 1]: 0=中性, 1=强共情
//   内部缓存, 由 launch_modulatory 读取后清零
void set_empathy_signal(float empathy_signal);

// 设置外部事件驱动的 6 维调质增量 (host 端, 由事件调度器触发) [Phase 3a-C1]
//   modulator_delta[6]: [DA, ACh, NE, 5HT, GABA, Oxy] 增量
//   duration_steps: 事件持续步数 (0=单次脉冲, >0=plateau 型每 100 步递减)
//   内部缓存 h_event_signal[6], 由 launch_modulatory 读取后清零
//   优先级: h_event_signal > h_empathy_signal (empathy 作为 Oxy 通道 fallback)
void set_event_signal(const float modulator_delta[6], int duration_steps);

} // namespace stage2e

#endif // SNN_STAGE2E_MODULATORY_KERNELS_CUH
