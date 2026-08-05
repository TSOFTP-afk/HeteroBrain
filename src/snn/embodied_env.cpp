// src/snn/embodied_env.cpp
#include "embodied_env.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace stage2e {

void EmbodiedEnvironment::init(const std::string& scene_id, int stage) {
    body.init_scene(scene_id.c_str());
    // 2026-08-04 (学生标准): 按课程阶段设定生理周期参数 (启蒙/初中 24, 高中/成年 28)
    body.configure_stage(stage);
    last_reward = 0.0f;
    // 默认资源在场 (中性环境)
    warm_source = 0.5f;
    food_source = 0.5f;
    threat_level = 0.0f;
    social_presence = 0.0f;
    novelty_level = 0.0f;
    light_level = 0.5f;

    // 场景资源/压力源配置 (成人稳态场景)
    if (scene_id == "hunger_feeding") {
        food_source = 0.8f;   // 食物源在场, 需要 approach 才能进食
    } else if (scene_id == "warmth_safety") {
        warm_source = 1.0f;   // 温暖源在场, 需要 approach 才能获益 (冷环境)
    } else if (scene_id == "thermal_comfort") {
        warm_source = 0.0f;   // 热环境无温暖源
        food_source = 0.0f;   // 专注温度应激
    } else if (scene_id == "startle_recover") {
        threat_level = 1.0f;  // 威胁在场, 需要 avoid
    } else if (scene_id == "menstrual_cycle") {
        food_source = 0.0f;   // 专注生理周期验证
        warm_source = 0.0f;
    }
}

// cry 呼救有效性: P = σ(2·cry_intensity + 2·pain - 2)
//   疼痛越重 + 呼救越强 → 越可能获救 (成人语义: 呼救=求医/止痛, 与饥饿无关)
float EmbodiedEnvironment::compute_help_prob() const {
    float x = 2.0f * 0.5f /*默认求助强度占位*/ + 2.0f * body.pain - 2.0f;
    return 1.0f / (1.0f + expf(-x));
}

// 行为效果映射 (v1, 无 agent, 成人稳态语义, spec §6.1 + 2026-08-04 成人化):
//   cry      → P(呼救有效) → pain↓ (求医/止痛; 不缓解饥饿 — 饥饿须觅食)
//   approach → 趋暖 (temp 向 0.5 收敛) + 觅食 (food_source → hunger↓)
//   avoid    → 避威胁 (threat↓ + pain↓) + 避热 (高温 → temp↓)
//   interact → 社交满足 → comfort↑
//   gaze     → 低活动 → 疲劳恢复 (静息)
// 2026-08-01 修复 (行为状态依赖): cry 救助改为仅当**实际采样的动作是 cry** 时判定.
void EmbodiedEnvironment::apply_action_effects(const MotorReadout& motor) {
    // cry: 呼救 (仅当采样动作 = cry 时判定; 成人: 表达疼痛/求助 → 止痛)
    if (motor.action_sampled == ACT_CRY) {
        float p_help = 1.0f / (1.0f + expf(-(2.0f * motor.cry_intensity
                                             + 2.0f * body.pain - 2.0f)));
        if ((float)rand() / (float)RAND_MAX < p_help) {
            // 呼救有效: 获得止痛/照料 → 疼痛缓解 (饥饿不缓解, 须 approach 觅食)
            body.pain = std::max(0.0f, body.pain - 0.2f);
        }
    }

    // approach: 趋近温暖源 (温度改善) + 觅食 (食物源, 饥饿下降)
    if (motor.approach_strength > 0.1f) {
        // 趋暖 (warm_source 在场)
        if (warm_source > 0.1f) {
            float gain = 0.1f * motor.approach_strength * warm_source;
            if (body.temperature < 0.5f) {
                body.temperature = std::min(0.5f, body.temperature + gain);
            } else if (body.temperature > 0.5f) {
                body.temperature = std::max(0.5f, body.temperature - gain);
            }
        }
        // 觅食 (food_source 在场 + 有食欲)
        if (food_source > 0.1f && body.hunger > 0.1f) {
            body.hunger = std::max(0.0f,
                body.hunger - 0.12f * motor.approach_strength * food_source);
        }
    }

    // avoid: 远离威胁 (威胁衰减 + 安全感 + 疼痛缓解) + 避热 (高温环境)
    if (motor.avoid_strength > 0.1f) {
        if (threat_level > 0.05f) {
            threat_level = std::max(0.0f, threat_level - 0.15f * motor.avoid_strength);
        }
        body.pain = std::max(0.0f, body.pain - 0.2f * motor.avoid_strength);
        // 避热: 离开高温环境 → 体温向理想收敛 (2026-08-04 温度双向调节)
        if (body.temperature > 0.5f) {
            body.temperature = std::max(0.5f,
                body.temperature - 0.10f * motor.avoid_strength);
        }
    }

    // interact: 与对象互动 → 舒适度提升 (社交满足)
    if ((social_presence > 0.1f || novelty_level > 0.1f) && motor.interact_intensity > 0.1f) {
        float social_gain = 0.08f * motor.interact_intensity;
        body.comfort = std::min(1.0f, body.comfort + social_gain);
    }

    // 静息恢复: 低活动 (gaze/呼救) → 疲劳下降 (睡眠压力缓解, 2026-08-04)
    //   生物对应: 注意/静息时不消耗体力, 睡眠压力开始恢复
    if (motor.action_sampled == ACT_GAZE || motor.action_sampled == ACT_CRY) {
        body.fatigue = std::max(0.0f, body.fatigue - 0.05f);
    }
}

void EmbodiedEnvironment::step_env(const MotorReadout& motor) {
    // 1. 行为效果映射 (行为 → 内感态改善)
    last_cry_intensity_ = motor.cry_intensity;
    apply_action_effects(motor);

    // 2. 身体状态演化 (成人稳态: hunger 累积/温度收敛/疲劳累积/疼痛自愈/生理周期)
    body.step(1.0f);

    // 3. 环境信号演化
    //    - 威胁: 缓慢衰减 (被成功回避后更明显)
    threat_level = std::max(0.0f, threat_level * 0.97f);
    //    - 新奇: 衰减 (刺激熟悉化, SEEKING 消退)
    novelty_level = std::max(0.0f, novelty_level * 0.95f);
    //    - 社交在场: v1 默认无 (v2 起引入同伴 agent)
    social_presence = std::max(0.0f, social_presence * 0.98f);

    // 4. 伤害事件 (spec §3.1 疼痛回避): 威胁→身体接触伤害
    //    威胁越强越持久, 每环境步受伤概率越高; 受伤 → pain 累积.
    //    闭环: 威胁 → 疼痛 → (教师引导 avoid) → 威胁下降 → 停止受伤.
    if (threat_level > 0.3f) {
        float hurt_prob = 0.45f * (threat_level - 0.3f) / 0.7f;  // threat=1 → 45%/环境步
        if ((float)rand() / (float)RAND_MAX < hurt_prob) {
            body.pain = std::min(1.0f, body.pain + 0.5f);
        }
    }
}

float EmbodiedEnvironment::compute_reward(const BodyState& prev) const {
    float reward = 0.0f;
    reward += (body.comfort - prev.comfort) * 1.0f;
    reward += (prev.hunger - body.hunger) * 0.5f;
    reward += (body.arousal_value() - prev.arousal_value()) * 0.2f;
    reward -= body.fatigue * 0.1f;
    // 2026-08-04: 疼痛缓解 → 奖励; 持续疼痛 → 惩罚
    reward += (prev.pain - body.pain) * 1.0f;
    reward -= body.pain * 0.2f;
    // 2026-08-04: 疲劳缓解 (静息/睡眠) → 奖励
    reward += (prev.fatigue - body.fatigue) * 0.5f;
    // 2026-08-01 修复 (行为状态依赖): 哭泣代价 — 无需求时呼救 = 社交/能量浪费
    //   代价 = cry_intensity × (1 - hunger) × 0.2: 需求低时哭得越凶代价越大
    reward -= last_cry_intensity_ * (1.0f - body.hunger) * 0.2f;
    return reward;
}

// 基因硬编码教师信号 (spec §3.1, 2026-08-04 成人反射链):
//   pain > 0.4         → ACT_CRY       (疼痛 → 呼救, 求援/止痛)
//   hunger > 0.6       → ACT_APPROACH  (饥饿 → 觅食, 趋近食物源)
//   fatigue > 0.7      → -1            (疲劳 → 静息)
//   threat_level > 0.5 → ACT_AVOID     (威胁 → 回避)
//   temperature < 0.35 → ACT_APPROACH  (冷 → 趋暖)
//   temperature > 0.65 → ACT_AVOID     (热 → 避热)
//   novelty_level > 0.5 → ACT_INTERACT (新奇 → 探索, SEEKING)
// 优先级: 生理内驱 (疼痛/饥饿/疲劳) > 环境威胁 > 温度 (生理) > 新奇 (探索)
int EmbodiedEnvironment::get_teacher_signal() const {
    if (body.pain > 0.4f) return ACT_CRY;
    if (body.hunger > 0.6f) return ACT_APPROACH;
    if (body.fatigue > 0.7f) return -1;  // 静息优先
    if (threat_level > 0.5f) return ACT_AVOID;
    if (body.temperature < 0.35f) return ACT_APPROACH;
    if (body.temperature > 0.65f) return ACT_AVOID;
    if (novelty_level > 0.5f) return ACT_INTERACT;
    return -1;  // 无教师信号
}

void EmbodiedEnvironment::compute_sensory_signals(float out[50]) const {
    std::memset(out, 0, 50 * sizeof(float));

    // 触觉-温暖源 (柱0-4): approach 趋暖的目标信号
    for (int i = 0; i < 5; ++i) out[i] = warm_source;

    // 触觉-接触/互动 (柱5-9): 社交在场
    for (int i = 5; i < 10; ++i) out[i] = social_presence;

    // 听觉-威胁 (柱10-14): avoid 的驱动力
    for (int i = 10; i < 15; ++i) out[i] = threat_level;

    // 听觉-同伴/资源声 (柱15-19): 社交/资源信号
    for (int i = 15; i < 20; ++i) out[i] = social_presence;

    // 视觉-光 (柱20-24)
    for (int i = 20; i < 25; ++i) out[i] = light_level;

    // 视觉-新奇 (柱25-29): SEEKING 驱动力
    for (int i = 25; i < 30; ++i) out[i] = novelty_level;

    // 嗅觉-食物源 (柱30-34): 觅食目标信号 (2026-08-04: 环境食物源, 非饥饿派生)
    for (int i = 30; i < 35; ++i) out[i] = food_source;

    // 内感态 (柱35-49): hunger/temp/comfort/fatigue/pain (含经期腹痛/PMS 间接)
    float interoception[15];
    body.encode_interoception(interoception);
    for (int i = 0; i < 15; ++i) out[35 + i] = interoception[i];
}

} // namespace stage2e
