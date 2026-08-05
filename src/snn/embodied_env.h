// src/snn/embodied_env.h
#ifndef SNN_EMBODIED_ENV_H
#define SNN_EMBODIED_ENV_H

#include "embodied_body.h"
#include "embodied_motor.h"
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1/D2: 世界沙盒 v1 (极简化, spec §6.1/§6.2) — 真实成人稳态
// =============================================================================
// 2026-08-01 spec 修订: 移除妈妈 agent 状态机, 保留内稳态向量 + 行为→奖励映射
// 2026-08-04 成人化重构 (启蒙期 = 生理基线验证期):
//   - 移除婴儿语义: 尿布/喂食/婴儿哭喊 (cry 成人化为"呼救")
//   - 生理硬编码: 饥饿→觅食(approach) / 冷→趋暖 / 热→避热 / 威胁→回避
//     / 疼痛→呼救 / 疲劳→静息 / 女性生理周期(经期腹痛+PMS)
//   - 行为→状态映射 (v1, 无 agent):
//     cry      → P(呼救有效)=σ(4·hunger + 2·cry - 2) → 有效则 hunger↓ + pain↓ (获照料/止痛)
//     approach → 趋近温暖源 (temp 向 0.5 收敛) + 觅食 (food_source 在场 → hunger↓)
//     avoid    → 远离威胁 (threat↓ + pain↓) + 避热 (高温 → temp↓)
//     interact → 与对象互动 → comfort↑ (社交满足)
//     gaze     → 注意选择 → 低活动, 触发疲劳恢复 (静息)
//   - 奖励 (compute_reward): Δcomfort + Δhunger·0.5 + Δarousal·0.2 - fatigue·0.1
//     + Δpain 缓解 + Δfatigue 恢复 + 疼痛惩罚 - 哭泣代价
// =============================================================================
class EmbodiedEnvironment {
public:
    BodyState body;
    float last_reward;             // 上次计算的reward (日志用)

    // 记录本步 cry 连续强度 (供 compute_reward 加哭泣代价)
    float last_cry_intensity_ = 0.0f;

    // === 抽象环境信号 (v1, 非 agent) ===
    float warm_source;             // 温暖源强度 [0,1] (approach 趋暖的前提)
    float food_source;             // 食物源强度 [0,1] (approach 觅食的前提, 2026-08-04)
    float threat_level;            // 威胁强度 [0,1] (avoid 的驱动力)
    float social_presence;         // 社交对象在场 [0,1] (v2 起: 同理心/依恋载体)
    float novelty_level;           // 新奇刺激 [0,1] (SEEKING 驱动力, 衰减)
    float light_level;             // 光照 [0,1]

    // 初始化场景 (女性学生/青少年稳态场景; stage=课程阶段, 决定生理周期参数)
    void init(const std::string& scene_id = "hunger_feeding", int stage = 1);

    // 环境步进 (每100 SNN步调用):
    //   1. 应用行为效果映射 (motor → 内感态改善)
    //   2. 身体状态演化 (含女性生理周期)
    //   3. 环境信号演化 (威胁/新奇衰减等)
    //   4. 伤害事件 (威胁 → 疼痛)
    void step_env(const MotorReadout& motor);

    // 计算DA奖励 (需要传入step_env之前的body状态)
    float compute_reward(const BodyState& prev) const;

    // 获取教师信号 (基因硬编码, -1=无教师; 成人反射链)
    int get_teacher_signal() const;

    // 计算感知信号 (50柱抽象信号, spec §6.2 observe 接口)
    void compute_sensory_signals(float out[50]) const;

    // 获取当前body状态 (供compute_reward之前保存)
    BodyState get_body_state() const { return body; }

    // 判定 cry 呼救是否有效 (v1 无 agent, 概率映射)
    float compute_help_prob() const;

private:
    void apply_action_effects(const MotorReadout& motor);
};

} // namespace stage2e

#endif // SNN_EMBODIED_ENV_H
