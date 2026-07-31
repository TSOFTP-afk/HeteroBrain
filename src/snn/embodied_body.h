// src/snn/embodied_body.h
#ifndef SNN_EMBODIED_BODY_H
#define SNN_EMBODIED_BODY_H

#include <cmath>
#include <algorithm>
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1: 虚拟婴儿身体状态
// 5维内感态 (hunger/temperature/comfort/fatigue/arousal) + 环境衍生状态
// 每环境步 (100 SNN步) 演化一次
// =============================================================================
struct BodyState {
    // 核心内感态 (5维, ∈[0,1])
    float hunger;       // 饥饿度: 每环境步+0.001, 喂养时-0.3
    float temperature;  // 体感温度: 理想0.5, 向ambient收敛
    float comfort;      // 舒适度: f(diaper, holding, position)
    float fatigue;      // 疲劳度: 每步+0.0005, 睡眠归零
    float arousal;      // 唤醒度: f(hunger, comfort, fatigue)

    // 环境衍生状态
    float ambient_temp;     // 环境温度 [0,1]
    float diaper_dirty;     // 尿布脏度 [0,1], 累积, 换尿布归零
    bool  is_held;          // 是否被抱
    bool  is_fed;           // 是否在喂养
    float mom_fatigue;      // 妈妈疲劳度 [0,1], 影响响应概率

    // 初始化为新生儿默认状态
    void init_default() {
        hunger = 0.3f;
        temperature = 0.5f;
        comfort = 0.7f;
        fatigue = 0.0f;
        arousal = 0.3f;
        ambient_temp = 0.5f;
        diaper_dirty = 0.0f;
        is_held = false;
        is_fed = false;
        mom_fatigue = 0.0f;
    }

    // 初始化为指定场景
    void init_scene(const char* scene_id) {
        init_default();
        std::string s(scene_id);
        if (s == "hunger_feeding") {
            hunger = 0.8f;
        } else if (s == "warmth_safety") {
            temperature = 0.2f;
            ambient_temp = 0.2f;
        } else if (s == "startle_recover") {
            arousal = 0.9f;
        } else if (s == "sleep_wake") {
            fatigue = 0.9f;
        } else if (s == "discomfort_change") {
            diaper_dirty = 0.9f;
            comfort = 0.2f;
        }
    }

    // 演化方程 (每环境步调用一次)
    void step(float dt) {
        // dt = 1.0 (一个环境步)
        // 饥饿: 基础速率 + 唤醒加成
        hunger += 0.001f * (1.0f + arousal * 0.3f);
        if (is_fed) hunger -= 0.3f;
        hunger = std::max(0.0f, std::min(1.0f, hunger));

        // 温度: 向ambient指数收敛
        temperature += 0.01f * (ambient_temp - temperature);

        // 尿布: 饥饿时排泄加快
        diaper_dirty += 0.0003f * (hunger > 0.5f ? 1.5f : 1.0f);
        diaper_dirty = std::max(0.0f, std::min(1.0f, diaper_dirty));

        // 舒适度: 受尿布/温度/抱影响
        comfort = std::max(0.0f, std::min(1.0f,
            0.7f
            - diaper_dirty * 0.5f
            - std::fabs(temperature - 0.5f) * 0.6f
            + (is_held ? 0.2f : 0.0f)
        ));

        // 疲劳: 持续累积
        fatigue += 0.0005f;
        fatigue = std::max(0.0f, std::min(1.0f, fatigue));

        // 唤醒: 综合驱动
        arousal = std::max(0.0f, std::min(1.0f,
            0.3f
            + hunger * 0.4f
            + (1.0f - comfort) * 0.3f
            - fatigue * 0.2f
        ));

        // 妈妈疲劳: 每步微增
        mom_fatigue += 0.0002f;
        mom_fatigue = std::max(0.0f, std::min(1.0f, mom_fatigue));
    }

    // 内感态信号编码 (15柱: 柱35-49, 每维3柱 low/mid/high)
    void encode_interoception(float out[15]) const {
        // hunger: 柱35-37
        out[0] = (hunger < 0.3f) ? 1.0f : 0.0f;
        out[1] = (hunger >= 0.3f && hunger < 0.7f) ? 1.0f : 0.0f;
        out[2] = (hunger >= 0.7f) ? 1.0f : 0.0f;
        // temperature: 柱38-40
        out[3] = (temperature < 0.4f) ? 1.0f : 0.0f;
        out[4] = (temperature >= 0.4f && temperature < 0.6f) ? 1.0f : 0.0f;
        out[5] = (temperature >= 0.6f) ? 1.0f : 0.0f;
        // comfort: 柱41-43
        out[6] = (comfort < 0.3f) ? 1.0f : 0.0f;
        out[7] = (comfort >= 0.3f && comfort < 0.7f) ? 1.0f : 0.0f;
        out[8] = (comfort >= 0.7f) ? 1.0f : 0.0f;
        // fatigue: 柱44-46
        out[9] = (fatigue < 0.3f) ? 1.0f : 0.0f;
        out[10] = (fatigue >= 0.3f && fatigue < 0.7f) ? 1.0f : 0.0f;
        out[11] = (fatigue >= 0.7f) ? 1.0f : 0.0f;
        // arousal: 柱47-49
        out[12] = (arousal < 0.3f) ? 1.0f : 0.0f;
        out[13] = (arousal >= 0.3f && arousal < 0.7f) ? 1.0f : 0.0f;
        out[14] = (arousal >= 0.7f) ? 1.0f : 0.0f;
    }

    // 衍生计算
    float discomfort() const {
        return hunger * 0.4f + (1.0f - comfort) * 0.4f + fatigue * 0.2f;
    }

    float distress() const {
        return discomfort() * (0.5f + arousal * 0.5f);
    }
};

} // namespace stage2e

#endif // SNN_EMBODIED_BODY_H
