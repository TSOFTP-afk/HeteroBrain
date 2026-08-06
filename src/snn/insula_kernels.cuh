#ifndef SNN_STAGE2E_INSULA_KERNELS_CUH
#define SNN_STAGE2E_INSULA_KERNELS_CUH

// =============================================================================
// Phase 3a-H (M4): 脑岛内感受模块 (2026-08-06 生物拟真 spec)
// =============================================================================
// 生物学: 脑岛整合内感受 (饥饿/温度/舒适/疲劳/疼痛) → 主观情绪
//   ("先有身体反应, 才有情绪" — 身体状态是情感的锚点)。
// 实现: 独立小模块 (不并入 60K 主网络, 不破坏 N_TOTAL_NEURONS_2E 契约):
//   - 输入: 内感受 5 维强度 dims[5] (由 15 柱 one-hot 在 host 端换算, 见
//     scheduler.set_insula_sensory): [hunger, temp偏离, comfort, fatigue, pain]
//   - 脑岛: N_INSULA_NEURONS 个简单 LIF, 5 组 × INSULA_GROUP_SIZE,
//     每组由对应维度强度持续注入 (内感受是持续身体状态, 非脉冲事件)
//   - 输出: 5 组发放率 → 调质调制 (不适→NE↑, 舒适→Oxy↑), 无学习权重
// =============================================================================

#include "config.h"
#include "memory_allocator.cuh"

namespace stage2e {

// 内感受注入: 5 维强度 → 对应组注入电流 (每步调用, 内感受持续状态)
//   dims[i] ∈ [0,1]: 组 i 的强度; 注入电流 = dims[i] * INSULA_INJECT_GAIN
void launch_insula_inject(const float dims[5], MemoryAllocator* alloc);

// 脑岛前向一步: LIF 积分 + 发放 (每 SNN 步调用, scheduler.step 内)
void launch_insula_forward(MemoryAllocator* alloc);

// 读取脑岛输出 (host): out[5] = 5 组窗口累计发放率 [0,1]
//   window_steps = 累计窗口长度 (mod_update_interval), 归一化 = cnt/(window_steps*group_size)
//   reset = true 时读取后清零窗口累计 (调制窗口语义, 防跨窗口膨胀)
void read_insula_output(MemoryAllocator* alloc, float out[5], int window_steps,
                        bool reset = true);

// 初始化脑岛 (network_init 调用): 状态清零
void init_insula(MemoryAllocator* alloc);

} // namespace stage2e

#endif // SNN_STAGE2E_INSULA_KERNELS_CUH
