#include "affective_mapping.h"

#include <cmath>
#include <utility>
#include <vector>

namespace hb {
namespace bridge {

namespace {

inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

// PAD 区间 [-1,1]: 单向强度, signal 从 threshold 线性升至 1 满量
// 注意: 调用方需传符号正确的信号 (如负向用 -pleasure), 本函数不做绝对值,
// 否则正负情感分支会同时激活 (2026-08-05 踩坑: 愉悦高时"难过"也被正偏置)
inline float pad_strength(float signal, float threshold) {
    if (threshold >= 1.0f) return 0.0f;
    if (signal <= threshold) return 0.0f;
    return clamp01((signal - threshold) / (1.0f - threshold));
}

// 调质浓度 [0,2] 基线 1.0: 高侧 signal > 1+threshold 满量在 1+threshold+span
inline float mod_high(float signal, float threshold, float span = 0.75f) {
    if (span <= 0.0f) return 0.0f;
    return clamp01((signal - (1.0f + threshold)) / span);
}

// 调质浓度 [0,2] 基线 1.0: 低侧 signal < 1-threshold 满量在 1-threshold-span
inline float mod_low(float signal, float threshold, float span = 0.75f) {
    if (span <= 0.0f) return 0.0f;
    return clamp01(((1.0f - threshold) - signal) / span);
}

void add_bias(std::vector<std::pair<std::string, float>>& out,
              const char* word, float base, float strength) {
    if (strength <= 0.0f) {
        return;
    }
    out.emplace_back(word, base * clamp01(strength));
}

}  // namespace

SamplerParams map_to_sampler_params(const EmotionState& s, const MappingConfig& cfg) {
    SamplerParams p;
    p.temperature    = clampf(cfg.temperature_base + cfg.temperature_scale * s.temperature_delta,
                              cfg.temperature_min, cfg.temperature_max);
    p.top_p          = clampf(cfg.top_p_base + cfg.top_p_scale * s.top_p_delta,
                              cfg.top_p_min, cfg.top_p_max);
    p.repeat_penalty = clampf(cfg.repeat_penalty_base + cfg.repeat_penalty_scale * s.repetition_delta,
                              cfg.repeat_penalty_min, cfg.repeat_penalty_max);
    // top_k / min_p 当前不随情感调制 (保持常量, 未来可在 MappingConfig 扩展)
    return p;
}

std::vector<std::pair<std::string, float>> compute_logit_bias(const EmotionState& s) {
    std::vector<std::pair<std::string, float>> out;

    // ---- PAD: 愉悦度 (核心情感轴) ----
    const float pos = pad_strength(s.pleasure, 0.15f);
    const float neg = pad_strength(-s.pleasure, 0.15f);
    if (pos > 0.0f) {
        add_bias(out, "开心", 0.9f, pos);
        add_bias(out, "喜欢", 0.7f, pos);
        add_bias(out, "微笑", 0.6f, pos);
        add_bias(out, "美好", 0.5f, pos);
    }
    if (neg > 0.0f) {
        add_bias(out, "难过", 0.9f, neg);
        add_bias(out, "伤心", 0.6f, neg);
        add_bias(out, "压力", 0.6f, neg);
        add_bias(out, "低落", 0.5f, neg);
        // 负向时压低正向词汇, 强化情绪对比
        add_bias(out, "开心", -0.5f, neg);
        add_bias(out, "微笑", -0.4f, neg);
    }

    // ---- PAD: 唤醒度 ----
    const float ar = pad_strength(s.arousal, 0.2f);
    if (ar > 0.0f) {
        add_bias(out, "兴奋", 0.6f, ar);
        add_bias(out, "激动", 0.5f, ar);
    } else {
        const float calm = pad_strength(-s.arousal, 0.2f);
        if (calm > 0.0f) {
            add_bias(out, "平静", 0.5f, calm);
            add_bias(out, "放松", 0.5f, calm);
        }
    }

    // ---- PAD: 主导度 ----
    const float dom = pad_strength(s.dominance, 0.2f);
    if (dom > 0.0f) {
        add_bias(out, "自信", 0.5f, dom);
    } else {
        const float sub = pad_strength(-s.dominance, 0.2f);
        if (sub > 0.0f) {
            add_bias(out, "犹豫", 0.4f, sub);
        }
    }

    // ---- 调质浓度 (高/低侧分别激活) ----
    const float kT = 0.25f;   // 偏离基线 0.25 起激活
    add_bias(out, "快乐",  0.5f, mod_high(s.dopamine, kT));
    add_bias(out, "沮丧",  0.5f, mod_low (s.dopamine, kT));
    add_bias(out, "焦虑",  0.6f, mod_low (s.serotonin, kT));
    add_bias(out, "不安",  0.5f, mod_low (s.serotonin, kT));
    add_bias(out, "紧张",  0.5f, mod_high(s.norepinephrine, kT));
    add_bias(out, "放松",  0.4f, mod_high(s.gaba, kT));
    add_bias(out, "亲切",  0.5f, mod_high(s.oxytocin, kT));
    add_bias(out, "信任",  0.4f, mod_high(s.oxytocin, kT));
    add_bias(out, "冷漠",  0.4f, mod_low (s.oxytocin, kT));

    return out;
}

}  // namespace bridge
}  // namespace hb
