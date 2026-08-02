// src/snn/embodied_env.cpp
#include "embodied_env.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace stage2e {

void EmbodiedEnvironment::init(const std::string& scene_id) {
    body.init_scene(scene_id.c_str());
    last_reward = 0.0f;
    warm_source = 0.5f;       // 默认有温暖源
    threat_level = 0.0f;
    social_presence = 0.0f;
    novelty_level = 0.0f;
    light_level = 0.5f;

    if (scene_id == "warmth_safety") {
        warm_source = 1.0f;   // 温暖源在场, 需要 approach 才能获益
    } else if (scene_id == "startle_recover") {
        threat_level = 1.0f;  // 威胁在场, 需要 avoid
    }
}

// cry 求助有效性: P = σ(4·hunger + 2·cry_intensity - 2)
//   高饥饿 + 强求助 → 高概率有效 (基因保底: 婴儿哭喊总能引来照护)
float EmbodiedEnvironment::compute_help_prob() const {
    float x = 4.0f * body.hunger + 2.0f * 0.5f /*默认求助强度占位*/ - 2.0f;
    return 1.0f / (1.0f + expf(-x));
}

// 行为效果映射 (v1, 无 agent, spec §6.1):
//   cry      → P(有效) → hunger↓ (求助成功=获得照护)
//   approach → 趋近温暖源 → temperature 向 0.5 收敛
//   avoid    → 远离威胁 → threat↓ + arousal↓
//   interact → 与对象互动 → comfort↑ (社交满足)
//   gaze     → 注意选择 → 无直接状态效果
// 2026-08-01 修复 (行为状态依赖):
//   cry 救助改为仅当**实际采样的动作是 cry** 时判定概率.
//   根因: 原实现每次 env 步都按 motor.cry_intensity 判定救助, 网络无需"选择"哭,
//   饥饿被自动机制持续缓解, 永远到不了教师阈值, 行为无法学会"饿才哭"(20K 实测 cry 恒 0.37).
//   改为采样动作驱动后, 只有真正 cry 才可能获救, hunger 才有机会上升并触发教师引导.
void EmbodiedEnvironment::apply_action_effects(const MotorReadout& motor) {
    // cry: 求助 (仅当采样动作 = cry 时判定, 让"选择哭"成为必要前提)
    if (motor.action_sampled == ACT_CRY) {
        float p_help = 1.0f / (1.0f + expf(-(4.0f * body.hunger + 2.0f * motor.cry_intensity - 2.0f)));
        if ((float)rand() / (float)RAND_MAX < p_help) {
            // 求助成功: 饥饿大幅下降 (替代原喂食机制, 无 agent 模拟)
            float relief = 0.30f * (0.5f + motor.cry_intensity);
            body.hunger = std::max(0.0f, body.hunger - relief);
        }
    }

    // approach: 趋近温暖源 (温度改善的前提是 warm_source 在场)
    if (warm_source > 0.1f && motor.approach_strength > 0.1f) {
        float gain = 0.1f * motor.approach_strength * warm_source;
        // 向理想温度 0.5 收敛
        if (body.temperature < 0.5f) {
            body.temperature = std::min(0.5f, body.temperature + gain);
        } else if (body.temperature > 0.5f) {
            body.temperature = std::max(0.5f, body.temperature - gain);
        }
    }

    // avoid: 远离威胁 (威胁衰减 + 安全感)
    if (threat_level > 0.05f && motor.avoid_strength > 0.1f) {
        threat_level = std::max(0.0f, threat_level - 0.15f * motor.avoid_strength);
    }

    // interact: 与对象互动 → 舒适度提升 (社交满足)
    if ((social_presence > 0.1f || novelty_level > 0.1f) && motor.interact_intensity > 0.1f) {
        float social_gain = 0.08f * motor.interact_intensity;
        body.comfort = std::min(1.0f, body.comfort + social_gain);
    }
}

void EmbodiedEnvironment::step_env(const MotorReadout& motor) {
    // 1. 行为效果映射 (行为 → 内感态改善)
    //    记录本步 cry 强度 (供 compute_reward 加哭泣代价, 见 compute_reward)
    last_cry_intensity_ = motor.cry_intensity;
    apply_action_effects(motor);

    // 2. 身体状态演化 (自发压力源: hunger 累积/温度收敛/疲劳累积)
    body.step(1.0f);

    // 3. 环境信号演化
    //    - 威胁: 缓慢衰减 (被成功回避后更明显)
    threat_level = std::max(0.0f, threat_level * 0.97f);
    //    - 新奇: 衰减 (刺激熟悉化, SEEKING 消退)
    novelty_level = std::max(0.0f, novelty_level * 0.95f);
    //    - 社交在场: v1 默认无 (v2 起引入同伴 agent)
    social_presence = std::max(0.0f, social_presence * 0.98f);
}

float EmbodiedEnvironment::compute_reward(const BodyState& prev) const {
    float reward = 0.0f;
    reward += (body.comfort - prev.comfort) * 1.0f;
    reward += (prev.hunger - body.hunger) * 0.5f;
    reward += (body.arousal_value() - prev.arousal_value()) * 0.2f;
    reward -= body.fatigue * 0.1f;
    // 2026-08-01 修复 (行为状态依赖): 加入哭泣代价 — 不饿时哭喊 = 社交/能量浪费
    //   根因: 原 reward 只奖励"饥饿下降", 不惩罚"不饿还哭" → REINFORCE 学不到
    //   "饱了就别哭", cry 恒高 0.37 与 hunger 无关.
    //   代价 = cry_intensity × (1 - hunger) × 0.2: 饿时 (hunger=1) 代价 0,
    //   饱时 (hunger=0) 哭得越凶代价越大 → 负奖励压制不必要的哭喊.
    //   需要 last_cry_intensity_ (step_env 记录): 与本步 cry 对应的连续强度.
    reward -= last_cry_intensity_ * (1.0f - body.hunger) * 0.2f;
    return reward;
}

// 基因硬编码教师信号 (spec §3.1):
//   hunger > 0.6        → ACT_CRY       (饥饿求助, 已有)
//   temperature 偏离    → ACT_APPROACH  (寒冷求助, 趋暖)
//   threat_level > 0.5  → ACT_AVOID     (威胁回避)
//   novelty_level > 0.5 → ACT_INTERACT  (新奇探索, SEEKING)
//   fatigue > 0.7       → -1            (静息, 无动作)
// 2026-08-01 修复: CRY 阈值 0.6 → 0.35. 原因: hunger 每沙盒步被 cry 自动机制
//   降至 ~0.6 以下, 原阈值导致教师信号恒为 -1, 行为学习无引导 (20K 实测行为恒均匀)
int EmbodiedEnvironment::get_teacher_signal() const {
    if (body.hunger > 0.35f) return ACT_CRY;
    if (body.fatigue > 0.7f) return -1;  // 静息优先
    if (threat_level > 0.5f) return ACT_AVOID;
    if (novelty_level > 0.5f) return ACT_INTERACT;
    if (fabsf(body.temperature - 0.5f) > 0.2f) return ACT_APPROACH;
    return -1;  // 无教师信号
}

void EmbodiedEnvironment::compute_sensory_signals(float out[50]) const {
    std::memset(out, 0, 50 * sizeof(float));

    // 触觉-温暖源 (柱0-4): approach 的目标信号
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

    // 嗅觉-资源气味 (柱30-34): 与饥饿相关的食物信号
    float food_scent = (1.0f - body.hunger) * 0.5f + 0.2f;
    for (int i = 30; i < 35; ++i) out[i] = food_scent;

    // 内感态 (柱35-49)
    float interoception[15];
    body.encode_interoception(interoception);
    for (int i = 0; i < 15; ++i) out[35 + i] = interoception[i];
}

} // namespace stage2e
