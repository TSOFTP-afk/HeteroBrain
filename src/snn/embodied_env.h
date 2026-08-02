// src/snn/embodied_env.h
#ifndef SNN_EMBODIED_ENV_H
#define SNN_EMBODIED_ENV_H

#include "embodied_body.h"
#include "embodied_motor.h"
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1/D2: 世界沙盒 v1 (极简化, spec §6.1/§6.2)
// =============================================================================
// 2026-08-01 spec 修订:
//   - 移除妈妈 agent 状态机 (mom_present/countdown/is_fed/is_held/mom_speaking/
//     mom_visible/mom_fatigue) — 生理+照料模拟无认知价值
//   - 移除喂食/吸吮/换尿布逻辑 — 生理细节下沉
//   - 保留内稳态状态向量 (感受) + 行为→奖励直接映射 (过程)
//
// 行为效果映射 (v1, 无 agent):
//   cry      → P(求助有效)=σ(4·hunger + 2·cry - 2) → 有效则 hunger↓
//   approach → 趋近温暖源 → temperature 向 0.5 收敛
//   avoid    → 远离威胁 → threat_level↓ + arousal↓
//   interact → 与对象互动 → comfort↑ (社交满足)
//   gaze     → 注意选择 → 无直接状态效果 (影响感知/好奇)
//
// 奖励 (compute_reward): Δcomfort + Δhunger·0.5 + Δarousal·0.2 - fatigue·0.1
// 教师信号 (get_teacher_signal): 基因硬编码 (spec §3.1)
// =============================================================================
class EmbodiedEnvironment {
public:
    BodyState body;
    float last_reward;             // 上次计算的reward (日志用)

    // 2026-08-01 修复 (行为状态依赖): 记录本步 cry 连续强度, 供 compute_reward 加哭泣代价
    float last_cry_intensity_ = 0.0f;

    // === 抽象环境信号 (v1, 非 agent) ===
    float warm_source;             // 温暖源强度 [0,1] (approach 改善温度的前提)
    float threat_level;            // 威胁强度 [0,1] (avoid 的驱动力)
    float social_presence;         // 社交对象在场 [0,1] (v2 起: 同理心/依恋载体)
    float novelty_level;           // 新奇刺激 [0,1] (SEEKING 驱动力, 衰减)
    float light_level;             // 光照 [0,1]

    // 初始化场景
    void init(const std::string& scene_id = "hunger_feeding");

    // 环境步进 (每100 SNN步调用):
    //   1. 应用行为效果映射 (motor → 内感态改善)
    //   2. 身体状态演化
    //   3. 环境信号演化 (威胁/新奇衰减等)
    void step_env(const MotorReadout& motor);

    // 计算DA奖励 (需要传入step_env之前的body状态)
    float compute_reward(const BodyState& prev) const;

    // 获取教师信号 (基因硬编码, -1=无教师)
    int get_teacher_signal() const;

    // 计算感知信号 (50柱抽象信号, spec §6.2 observe 接口)
    void compute_sensory_signals(float out[50]) const;

    // 获取当前body状态 (供compute_reward之前保存)
    BodyState get_body_state() const { return body; }

    // 判定 cry 求助是否有效 (v1 无 agent, 概率映射)
    float compute_help_prob() const;

private:
    void apply_action_effects(const MotorReadout& motor);
};

} // namespace stage2e

#endif // SNN_EMBODIED_ENV_H
