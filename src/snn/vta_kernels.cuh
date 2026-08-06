#ifndef SNN_STAGE2E_VTA_KERNELS_CUH
#define SNN_STAGE2E_VTA_KERNELS_CUH

// =============================================================================
// Phase 3a-I (M2): VTA-DA 奖赏预测误差 (RPE) 神经化 (2026-08-06 生物拟真 spec)
// =============================================================================
// 生物学: VTA DA 神经元发放 = RPE — 奖赏 burst (正 RPE) / 预期落空 dip (负 RPE)。
// 实现: 独立小模块 (不并入 60K 主网络, 不破坏 N_TOTAL_NEURONS_2E 契约):
//   - 输入: 2 维 RPE 强度 rpe[2] (正组/负组, 由 launch_modulatory 在调制窗口
//     缓存: 正 = max(0, da_delta)·VTA_POS_GAIN + 奖赏, 负 = prediction_error_norm)
//   - VTA: N_VTA_NEURONS 个简单 LIF, 正/负各 VTA_GROUP_SIZE, 持续注入 (RPE 是
//     调制窗口级状态, 非脉冲)
//   - 输出: 正/负组窗口累计发放率 → r_vta = pos - neg ∈ [-1,1] →
//     STDP 第三因子 DA 项叠加 (M_ij += VTA_STDP_GAIN·r_vta·da_receptor)
//   - 无学习权重 → 无 checkpoint section。
// =============================================================================

#include "config.h"
#include "memory_allocator.cuh"

namespace stage2e {

// RPE 注入: rpe[2] 强度 → 对应组注入电流 (每步调用, 调制窗口级持续状态)
//   rpe[i] ∈ [0,1]: 组 i 的强度; 注入电流 = rpe[i] * VTA_INJECT_GAIN
void launch_vta_inject(const float rpe[2], MemoryAllocator* alloc);

// VTA 前向一步: LIF 积分 + 发放 + 窗口累计 (每 SNN 步调用, scheduler.step 内)
void launch_vta_forward(MemoryAllocator* alloc);

// 读取 VTA 输出 (host): out[2] = 正/负组窗口累计发放率 [0,1]
//   window_steps = 累计窗口长度 (mod_update_interval), 归一化 = cnt/(window_steps*group_size)
//   reset = true 时读取后清零窗口累计 (调制窗口语义, 防跨窗口膨胀)
void read_vta_output(MemoryAllocator* alloc, float out[2], int window_steps,
                     bool reset = true);

// 初始化 VTA (network_init 调用): 状态清零
void init_vta(MemoryAllocator* alloc);

} // namespace stage2e

#endif // SNN_STAGE2E_VTA_KERNELS_CUH
