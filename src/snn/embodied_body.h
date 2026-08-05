// src/snn/embodied_body.h
#ifndef SNN_EMBODIED_BODY_H
#define SNN_EMBODIED_BODY_H

#include <cmath>
#include <algorithm>
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1/D2: 虚拟身体内感态状态向量 — 女性学生(青少年)稳态模型
// 5维核心内感态 (hunger/temperature/comfort/fatigue/pain) + 女性生理周期
// 2026-08-04 成人化重构 (spec §3.1 生理硬编码 + 启蒙期=生理基线验证期):
//   - 移除婴儿语义: 尿布 (diaper_dirty)、婴儿哭喊求照料 (cry 语义成人化为呼救)
//   - 新增: 温度双向调节 (冷→趋暖 / 热→避热)、疲劳→睡眠压力、女性生理周期
// 2026-08-04 修正 (学生标准): 生理模型按课程阶段分层 — 课程是学生(青少年)
//   标准, 非成年女性. SNN 默认为女性 (生理期保留), 年龄=青春期少女:
//   启蒙/初中 (青春期早期): 周期较短 (24 环境步, 初潮后常见)
//   高中/成年: 周期规律 (28 环境步)
// 每环境步 (100 SNN步) 演化一次; 行为→状态改善映射在 env 层 (apply_action_effects)
// =============================================================================
struct BodyState {
    // 核心内感态 (5维, ∈[0,1])
    float hunger;       // 饥饿度: 自发累积, approach 觅食成功时下降
    float temperature;  // 体感温度: 理想0.5, 冷(<0.35)→趋暖, 热(>0.65)→避热
    float comfort;      // 舒适度: f(温度偏离, 经期不适), interact/静息时提升
    float fatigue;      // 疲劳度: 累积 (睡眠压力), 静息时恢复
    float pain;         // 疼痛度: 伤害事件/经期腹痛累积, avoid/呼救缓解

    // 女性生理周期 (2026-08-04): 相位 [0,1), 每环境步 +1/cycle_period
    //   经期 ≈ [0, 0.14) ∪ [0.93, 1)  (约 4 天: 腹痛/不适)
    //   黄体期 (PMS) ≈ [0.57, 0.93)   (情绪波动/易激惹)
    //   排卵 ≈ 0.5 (黄体期前)
    float cycle_phase;
    float cycle_period;     // 周期长度 (环境步): 启蒙/初中 24 / 高中/成年 28

    // 环境衍生状态
    float ambient_temp;     // 环境温度 [0,1] (冷/热压力源)

    // 初始化为默认状态 (从经期起点开始, 便于验证周期)
    void init_default() {
        hunger = 0.3f;
        temperature = 0.5f;
        comfort = 0.7f;
        fatigue = 0.0f;
        pain = 0.0f;
        cycle_phase = 0.0f;
        cycle_period = 24.0f;   // 默认青春期早期 (初中) 周期
        ambient_temp = 0.5f;
    }

    // 按课程阶段设定生理参数 (2026-08-04, 学生标准):
    //   启蒙/初中 (0/1): 青春期早期 — 周期较短 (24), 初潮后常见不规律
    //   高中/成年 (2/3): 周期规律 (28)
    void configure_stage(int stage) {
        cycle_period = (stage <= 1) ? 24.0f : 28.0f;
    }

    // 初始化为指定场景 (场景语义均为成人稳态)
    void init_scene(const char* scene_id) {
        init_default();
        std::string s(scene_id);
        if (s == "hunger_feeding") {          // 能量缺失
            hunger = 0.8f;
        } else if (s == "warmth_safety") {    // 寒冷环境
            temperature = 0.2f;
            ambient_temp = 0.2f;
        } else if (s == "thermal_comfort") {  // 高温环境 (热应激)
            temperature = 0.8f;
            ambient_temp = 0.8f;
        } else if (s == "startle_recover") {  // 威胁应激 (env 层注入 threat)
            // 威胁信号由 env 层提供
        } else if (s == "sleep_wake") {       // 睡眠剥夺 (疲劳累积)
            fatigue = 0.9f;
        } else if (s == "menstrual_cycle") {  // 生理期 (经期开始)
            cycle_phase = 0.0f;
            pain = 0.2f;                      // 经期初期已有基础不适
        }
    }

    // 演化方程 (每环境步调用一次; 行为→状态改善在 env 层)
    void step(float dt) {
        // 饥饿: 基础速率 + 唤醒加成 (自发累积, 能量消耗压力源)
        hunger += 0.01f * (1.0f + arousal_value() * 0.3f);
        hunger = std::max(0.0f, std::min(1.0f, hunger));

        // 温度: 向 ambient 指数收敛 (冷热双向, 无 clothing 模型)
        temperature += 0.01f * (ambient_temp - temperature);

        // 舒适度: f(温度偏离, 经期不适) — 2026-08-04 移除尿布 (婴儿语义)
        comfort = std::max(0.0f, std::min(1.0f,
            0.7f
            - std::fabs(temperature - 0.5f) * 0.6f
            - (menstrual_phase() ? 0.15f : 0.0f)   // 经期不适
        ));

        // 疲劳: 持续累积 (睡眠压力; 静息恢复由 env 行为映射处理)
        fatigue += 0.0005f;
        fatigue = std::max(0.0f, std::min(1.0f, fatigue));

        // 疼痛: 经期腹痛累积 + 自愈衰减 (伤害事件由 env 注入)
        if (menstrual_phase()) {
            pain = std::min(1.0f, pain + 0.02f);   // 经期基础腹痛
        }
        pain *= 0.97f;
        if (pain < 0.01f) pain = 0.0f;

        // 生理周期: cycle_period 环境步一个周期 (按阶段 24/28, 学生标准)
        cycle_phase += 1.0f / cycle_period;
        if (cycle_phase >= 1.0f) cycle_phase -= 1.0f;
    }

    // ===== 周期相位判断 (2026-08-04) =====
    bool menstrual_phase() const { return cycle_phase < 0.14f || cycle_phase >= 0.93f; }
    bool pms_phase() const       { return cycle_phase >= 0.57f && cycle_phase < 0.93f; }

    // 唤醒: 综合驱动 (由状态计算, 非演化变量)
    // 2026-08-04: 纳入经期腹痛 (唤醒↑) + 黄体期易激惹 (烦躁)
    float arousal_value() const {
        float pms_irrit = pms_phase() ? 0.15f : 0.0f;
        float men_stress = menstrual_phase() ? 0.10f : 0.0f;
        return std::max(0.0f, std::min(1.0f,
            0.3f
            + hunger * 0.4f
            + (1.0f - comfort) * 0.3f
            - fatigue * 0.2f
            + pms_irrit + men_stress
        ));
    }

    // 内感态信号编码 (15柱: 柱35-49, 每维3柱 low/mid/high)
    // 2026-08-04: 柱47-49 由 arousal (派生量) 改为 pain (真实内感态) —
    //   arousal 可从 hunger/comfort/fatigue 推断, 腾出感知通道给疼痛;
    //   经期/PMS 通过 comfort/pain 通道间接进入感知
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
        // pain: 柱47-49 (疼痛回避内感态; 经期腹痛也经此通道)
        out[12] = (pain < 0.3f) ? 1.0f : 0.0f;
        out[13] = (pain >= 0.3f && pain < 0.7f) ? 1.0f : 0.0f;
        out[14] = (pain >= 0.7f) ? 1.0f : 0.0f;
    }

    // 衍生计算: 不适度 (2026-08-04 纳入 PMS; 权重和 = 1.0)
    float discomfort() const {
        float pms_bonus = pms_phase() ? 0.20f : 0.0f;
        return hunger * 0.25f + (1.0f - comfort) * 0.25f
             + fatigue * 0.10f + pain * 0.20f + pms_bonus * 0.20f;
    }

    float distress() const {
        return discomfort() * (0.5f + arousal_value() * 0.5f);
    }
};

} // namespace stage2e

#endif // SNN_EMBODIED_BODY_H
