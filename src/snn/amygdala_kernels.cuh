#ifndef SNN_STAGE2E_AMYGDALA_KERNELS_CUH
#define SNN_STAGE2E_AMYGDALA_KERNELS_CUH

// =============================================================================
// Phase 3a-F (M1): 杏仁核情感学习核心 (2026-08-06 生物拟真 spec)
// =============================================================================
// 生物学: 杏仁核 LA→BA 回路 = 情感条件反射中枢 (CS-US 关联经 STDP 学习)
//   LA (外侧核) 接收感觉/事件输入, LA→BA 突触经 STDP 学习"刺激→情感反应"关联,
//   BA (基底核) 输出驱动情感反应 (威胁→恐惧/回避, 奖赏→趋近)。
// 实现: 独立小模块 (不并入 60K 主网络, 不破坏 N_TOTAL_NEURONS_2E 契约):
//   - LA: N_AMYGDALA_LA 个简单 LIF, 按事件类型分组 (11 组), 事件注入激活对应组
//   - LA→BA: 全连接权重 W[N_LA × N_BA] (STDP 学习, 1MB)
//   - BA: N_AMYGDALA_BA 个简单 LIF, 前 AMYGDALA_BA_NEG_GROUP 负性 / 后正性
//   - 输出: BA 正/负性组发放率 → 调制调质 (负性→NE↑/皮质醇↑, 正性→DA↑)
// 初始化偏置: 负性事件 LA 组 → 负性 BA 组初始权重略高 (基础关联, STDP 强化)
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "event_types.h"

namespace stage2e {

// 事件→LA 组映射 (固定, 无学习): 事件类型 k 激活 LA 组 [k*group, (k+1)*group)
//   AMYGDALA_EVT_GROUP = N_AMYGDALA_LA / EVT_COUNT
//   __host__ __device__: 供 device kernel (init 偏置) 与 host launcher 共用
__host__ __device__ inline int amygdala_event_group_size() {
    return N_AMYGDALA_LA / EVT_COUNT;
}

// 事件正负性分类 (用于 BA 正负分组初始偏置):
//   负性: food_bland / threat_physical / threat_social / criticism / social_loss
//   正性: food_tasty / praise / social_bond / achievement / novelty / question
__host__ __device__ inline bool amygdala_event_is_negative(int event_type) {
    switch (static_cast<EventType>(event_type)) {
        case EVT_FOOD_BLAND:
        case EVT_THREAT_PHYSICAL:
        case EVT_THREAT_SOCIAL:
        case EVT_CRITICISM:
        case EVT_SOCIAL_LOSS:
            return true;
        default:
            return false;
    }
}

// 事件应激量 [0,1] (HPA 皮质醇 + 杏仁核负性输出用):
//   threat/criticism/social_loss → 高应激; praise/achievement/social_bond → 安抚 (0)
inline float event_stress_level(int event_type, int intensity) {
    if (!amygdala_event_is_negative(event_type)) return 0.0f;
    // 强度归一化 [0,50] → [0.3, 1.0]: 负性事件必带基础应激, 强度越大应激越高
    float mag = (float)(intensity < 0 ? -intensity : intensity);
    if (mag > 50.0f) mag = 50.0f;
    return 0.3f + 0.7f * (mag / 50.0f);
}

// ==================== host 接口 ====================

// 事件注入: 事件类型 → LA 对应组注入电流 (intensity 强度缩放)
//   main.cpp / engine 在事件 dispatch 时调用 (与 set_event_signal 并列)
void launch_amygdala_event_inject(MemoryAllocator* alloc, int event_type, float intensity);

// 杏仁核前向一步: LA 积分+发放 → BA 输入累积 → BA 积分+发放
//   每 SNN 步调用 (scheduler.step 内, 快时间尺度旁)
void launch_amygdala_forward(MemoryAllocator* alloc);

// 杏仁核 STDP 学习一步: LA→BA 权重更新 (pre-post 配对)
//   每 SNN 步调用 (与 forward 配对)
void launch_amygdala_stdp(MemoryAllocator* alloc);

// 读取 BA 输出 (host): 负性/正性组归一化发放率 [0,1]
//   out_neg/out_pos 可为 nullptr (只取其一)
void read_amygdala_ba_output(MemoryAllocator* alloc, float* out_neg, float* out_pos);

// 初始化杏仁核 (network_init 调用): 权重随机小值 + 正负偏置, 状态清零
void init_amygdala(MemoryAllocator* alloc, uint32_t seed);

} // namespace stage2e

#endif // SNN_STAGE2E_AMYGDALA_KERNELS_CUH
