// src/snn/embodied_env.h
#ifndef SNN_EMBODIED_ENV_H
#define SNN_EMBODIED_ENV_H

#include "embodied_body.h"
#include "embodied_motor.h"
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1: 概率响应环境
// 妈妈响应模型: P(来)=σ(2.0*hunger + 1.5*arousal - 1.0*mom_fatigue - 0.5)
// 响应延迟: 5-20环境步 (0.5-2秒)
// 动作效果: cry→触发响应, suck→喂养, hand/foot→消耗能量
// DA奖励: Δcomfort + Δhunger*0.5 + Δarousal*0.2 - fatigue*0.1
// =============================================================================
class EmbodiedEnvironment {
public:
    BodyState body;
    int   mom_response_countdown;  // 妈妈响应剩余环境步 (0=不在路上)
    bool  mom_present;             // 妈妈是否在场
    float mom_response_prob;       // 当前步响应概率 (日志用)
    float last_reward;             // 上次计算的reward (日志用)
    float noise_intensity;         // 环境噪音强度 [0,1]
    float light_level;             // 光照强度 [0,1]
    bool  mom_speaking;            // 妈妈在说话
    bool  mom_visible;             // 妈妈脸可见

    // 初始化场景
    void init(const std::string& scene_id = "hunger_feeding");

    // 环境步进 (每100 SNN步调用)
    void step_env(const MotorReadout& motor);

    // 计算DA奖励 (需要传入step_env之前的body状态)
    float compute_reward(const BodyState& prev) const;

    // 获取教师信号 (-1=无教师)
    int get_teacher_signal() const;

    // 计算感知信号 (50柱)
    void compute_sensory_signals(float out[50]) const;

    // 获取当前body状态 (供compute_reward之前保存)
    BodyState get_body_state() const { return body; }

    // 妈妈响应概率 (public 供测试访问)
    float compute_mom_response_prob() const;

private:
    void apply_mom_effects(const MotorReadout& motor);
};

} // namespace stage2e

#endif // SNN_EMBODIED_ENV_H
