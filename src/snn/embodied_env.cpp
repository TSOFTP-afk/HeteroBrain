// src/snn/embodied_env.cpp
#include "embodied_env.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace stage2e {

void EmbodiedEnvironment::init(const std::string& scene_id) {
    body.init_scene(scene_id.c_str());
    mom_response_countdown = 0;
    mom_present = false;
    mom_response_prob = 0.0f;
    last_reward = 0.0f;
    noise_intensity = 0.0f;
    light_level = 0.5f;
    mom_speaking = false;
    mom_visible = false;

    // 场景特定初始化
    if (scene_id == "startle_recover") {
        noise_intensity = 1.0f;
    }
}

float EmbodiedEnvironment::compute_mom_response_prob() const {
    // D1: hunger为主驱动 (基因保底), 阈值-1.5确保低饥饿时低响应
    float x = 4.0f * body.hunger
            + 1.0f * body.arousal
            - 1.0f * body.mom_fatigue
            - 1.5f;
    return 1.0f / (1.0f + expf(-x));
}

void EmbodiedEnvironment::apply_mom_effects(const MotorReadout& motor) {
    if (!mom_present) return;

    // 喂养: 如果饥饿>0.3则喂奶
    if (body.hunger > 0.3f) {
        body.is_fed = true;
        body.hunger -= 0.3f * motor.suck_strength;
        body.hunger = std::max(0.0f, body.hunger);
    } else {
        body.is_fed = false;
    }

    // 换尿布: 如果尿布脏>0.7
    if (body.diaper_dirty > 0.7f) {
        body.diaper_dirty = 0.0f;
    }

    // 妈妈在场时设置感知
    mom_speaking = true;
    mom_visible = true;

    // 妈妈离开: 饥饿<0.2且尿布干净
    if (body.hunger < 0.2f && body.diaper_dirty < 0.3f) {
        mom_present = false;
        body.is_held = false;
        body.is_fed = false;
        mom_speaking = false;
        mom_visible = false;
    }
}

void EmbodiedEnvironment::step_env(const MotorReadout& motor) {
    // 1. 妈妈响应判定 (概率)
    if (!mom_present && mom_response_countdown <= 0) {
        float p = compute_mom_response_prob();
        mom_response_prob = p;
        float p_cry = p * (0.5f + motor.cry_intensity);
        if ((float)rand() / (float)RAND_MAX < p_cry) {
            mom_response_countdown = 5 + rand() % 16;
        }
    }

    // 2. 响应倒计时
    if (mom_response_countdown > 0) {
        mom_response_countdown--;
        if (mom_response_countdown == 0) {
            mom_present = true;
            body.is_held = true;
            body.mom_fatigue += 0.05f;
        }
    }

    // 3. 妈妈在场时的效果
    apply_mom_effects(motor);

    // 4. 身体状态演化
    body.step(1.0f);

    // 5. 动作能量消耗
    body.fatigue += motor.limb_movement * 0.002f;
    body.fatigue = std::min(1.0f, body.fatigue);

    // 6. 噪音衰减 (startle场景)
    if (noise_intensity > 0) {
        noise_intensity *= 0.95f;
        if (noise_intensity < 0.01f) noise_intensity = 0.0f;
    }
}

float EmbodiedEnvironment::compute_reward(const BodyState& prev) const {
    float reward = 0.0f;
    reward += (body.comfort - prev.comfort) * 1.0f;
    reward += (prev.hunger - body.hunger) * 0.5f;
    reward += (body.arousal - prev.arousal) * 0.2f;
    reward -= body.fatigue * 0.1f;
    return reward;
}

int EmbodiedEnvironment::get_teacher_signal() const {
    // 基因硬编码教师信号
    if (body.hunger > 0.6f) return ACT_CRY;
    if (body.is_fed && body.hunger > 0.3f) return ACT_SUCK;
    if (body.fatigue > 0.7f) return ACT_GAZE;
    return -1;  // 无教师信号
}

void EmbodiedEnvironment::compute_sensory_signals(float out[50]) const {
    std::memset(out, 0, 50 * sizeof(float));

    // 触觉-被抱 (柱0-4)
    if (body.is_held) {
        for (int i = 0; i < 5; ++i) out[i] = 1.0f;
    }
    // 触觉-抚摸 (柱5-9): 被抱时有抚摸
    if (body.is_held) {
        for (int i = 5; i < 10; ++i) out[i] = 0.7f;
    }

    // 听觉-妈妈声 (柱10-14)
    if (mom_present && mom_speaking) {
        for (int i = 10; i < 15; ++i) out[i] = 0.8f;
    }
    // 听觉-噪音 (柱15-19)
    if (noise_intensity > 0) {
        for (int i = 15; i < 20; ++i) out[i] = noise_intensity;
    }

    // 视觉-光 (柱20-24)
    for (int i = 20; i < 25; ++i) out[i] = light_level;
    // 视觉-人脸 (柱25-29)
    if (mom_visible) {
        for (int i = 25; i < 30; ++i) out[i] = 0.9f;
    }

    // 嗅觉-奶味 (柱30-32)
    if (body.is_fed) {
        for (int i = 30; i < 33; ++i) out[i] = 0.8f;
    }
    // 嗅觉-妈妈味 (柱33-34)
    if (mom_present) {
        out[33] = 0.7f;
        out[34] = 0.7f;
    }

    // 内感态 (柱35-49)
    float interoception[15];
    body.encode_interoception(interoception);
    for (int i = 0; i < 15; ++i) out[35 + i] = interoception[i];
}

} // namespace stage2e
