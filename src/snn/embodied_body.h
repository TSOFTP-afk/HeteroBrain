// src/snn/embodied_body.h
#ifndef SNN_EMBODIED_BODY_H
#define SNN_EMBODIED_BODY_H

#include <cmath>
#include <algorithm>
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1/D2: 虚拟身体内感态状态向量
// 4维核心内感态 (hunger/temperature/comfort/fatigue) + 环境衍生状态
// 2026-08-01 spec §1.4/§6.1 (沙盒 v1 极简化):
//   去掉 agent 相关字段 (is_held/is_fed/mom_fatigue) — 妈妈 agent 模拟移除
//   保留内稳态状态向量 (感受) + 行为→奖励映射 (过程)
// 每环境步 (100 SNN步) 演化一次
// =============================================================================
struct BodyState {
    // 核心内感态 (4维, ∈[0,1])
    float hunger;       // 饥饿度: 自发累积, 行为 (cry 求助成功) 时下降
    float temperature;  // 体感温度: 理想0.5, 向 ambient 收敛, 行为 (approach 趋暖) 时改善
    float comfort;      // 舒适度: f(diaper, temperature), 行为 (interact) 时提升
    float fatigue;      // 疲劳度: 每步+0.0005, 静息归零

    // 环境衍生状态
    float ambient_temp;     // 环境温度 [0,1]
    float diaper_dirty;     // 尿布脏度 [0,1], 累积 (舒适度相关, 非 agent)

    // 初始化为新生儿默认状态
    void init_default() {
        hunger = 0.3f;
        temperature = 0.5f;
        comfort = 0.7f;
        fatigue = 0.0f;
        ambient_temp = 0.5f;
        diaper_dirty = 0.0f;
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
            // 场景已迁移: 威胁信号由 env 层提供 (v1 无此场景专用字段)
        } else if (s == "sleep_wake") {
            fatigue = 0.9f;
        } else if (s == "discomfort_change") {
            diaper_dirty = 0.9f;
            comfort = 0.2f;
        }
    }

    // 演化方程 (每环境步调用一次)
    // 2026-08-01: 移除 is_fed/mom_fatigue 分支 — 喂食由 env 行为映射处理
    // 2026-08-01 修复 (行为状态依赖): hunger 累积率 0.001 → 0.01.
    //   根因: 原 0.001/环境步, 20K SNN 步 (200 环境步) 仅累积 +0.2, 饥饿被 cry 救助
    //   机制一次性降到 0 后从不再升到 0.35 教师阈值 → 教师信号恒 -1, 行为学习无引导
    //   (20K 实测 cry 恒 0.37, 与 hunger 无关, corr=-0.91). 提高后 hunger 可在 20K 内
    //   循环 3-5 次 (0→0.35 需 35 环境步), 教师反复触发, 行为学会"饿才哭".
    void step(float dt) {
        // dt = 1.0 (一个环境步)
        // 饥饿: 基础速率 + 唤醒加成 (自发累积, 压力源)
        hunger += 0.01f * (1.0f + arousal_value() * 0.3f);
        hunger = std::max(0.0f, std::min(1.0f, hunger));

        // 温度: 向 ambient 指数收敛
        temperature += 0.01f * (ambient_temp - temperature);

        // 尿布: 饥饿时排泄加快
        diaper_dirty += 0.0003f * (hunger > 0.5f ? 1.5f : 1.0f);
        diaper_dirty = std::max(0.0f, std::min(1.0f, diaper_dirty));

        // 舒适度: 受尿布/温度影响
        comfort = std::max(0.0f, std::min(1.0f,
            0.7f
            - diaper_dirty * 0.5f
            - std::fabs(temperature - 0.5f) * 0.6f
        ));

        // 疲劳: 持续累积
        fatigue += 0.0005f;
        fatigue = std::max(0.0f, std::min(1.0f, fatigue));
    }

    // 唤醒: 综合驱动 (由状态计算, 非演化变量)
    float arousal_value() const {
        return std::max(0.0f, std::min(1.0f,
            0.3f
            + hunger * 0.4f
            + (1.0f - comfort) * 0.3f
            - fatigue * 0.2f
        ));
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
        float a = arousal_value();
        out[12] = (a < 0.3f) ? 1.0f : 0.0f;
        out[13] = (a >= 0.3f && a < 0.7f) ? 1.0f : 0.0f;
        out[14] = (a >= 0.7f) ? 1.0f : 0.0f;
    }

    // 衍生计算
    float discomfort() const {
        return hunger * 0.4f + (1.0f - comfort) * 0.4f + fatigue * 0.2f;
    }

    float distress() const {
        return discomfort() * (0.5f + arousal_value() * 0.5f);
    }
};

} // namespace stage2e

#endif // SNN_EMBODIED_BODY_H
