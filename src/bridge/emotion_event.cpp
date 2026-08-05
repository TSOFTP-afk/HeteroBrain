// =============================================================================
// emotion_event.cpp — 情感事件抽取实现
// =============================================================================
// 内置中文情感词典: 8 类情绪 → 调质增量模式 (与 modulatory_kernels 语义对齐)。
// 增量幅值控制在小范围 [-0.5, +0.5] 基准 × 强度倍数, 最终 clamp [-1, +1]。
// =============================================================================

#include "emotion_event.h"

#include <algorithm>
#include <cstring>

namespace vita {
namespace bridge {

namespace {

// 通道索引 (与 GENE_MAP / snn_feedback.h 一致)
enum Ch : int { DA = 0, ACH = 1, NE = 2, HT5 = 3, GABA = 4, OXY = 5 };

inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

EmotionEventExtractor::Rule make_rule(const char* kw, float da, float ach, float ne,
                                      float ht5, float gaba, float oxy, float w = 1.0f) {
    EmotionEventExtractor::Rule r;
    r.keyword = kw;
    r.delta[DA] = da; r.delta[ACH] = ach; r.delta[NE] = ne;
    r.delta[HT5] = ht5; r.delta[GABA] = gaba; r.delta[OXY] = oxy;
    r.weight = w;
    return r;
}

}  // namespace

EmotionEventExtractor::EmotionEventExtractor() {
    // ---- 快乐 / 愉悦: DA↑ ACh↑ NE↑ 5HT↓ ----
    rules_.push_back(make_rule("开心",    0.5f, 0.2f, 0.2f, -0.2f, 0.0f, 0.1f));
    rules_.push_back(make_rule("高兴",    0.5f, 0.2f, 0.2f, -0.2f, 0.0f, 0.1f));
    rules_.push_back(make_rule("快乐",    0.5f, 0.2f, 0.2f, -0.2f, 0.0f, 0.1f));
    rules_.push_back(make_rule("开心极",  0.6f, 0.3f, 0.3f, -0.3f, 0.0f, 0.1f));
    rules_.push_back(make_rule("喜悦",    0.5f, 0.2f, 0.2f, -0.1f, 0.0f, 0.1f));
    rules_.push_back(make_rule("喜欢",    0.4f, 0.2f, 0.1f, -0.1f, 0.0f, 0.1f));
    rules_.push_back(make_rule("幸福",    0.5f, 0.2f, 0.1f, -0.2f, 0.0f, 0.2f));
    rules_.push_back(make_rule("满足",    0.3f, 0.1f, -0.1f, 0.2f, 0.1f, 0.1f));
    rules_.push_back(make_rule("太棒",    0.5f, 0.3f, 0.3f, -0.2f, 0.0f, 0.1f));
    rules_.push_back(make_rule("太好了",  0.5f, 0.3f, 0.3f, -0.2f, 0.0f, 0.1f));

    // ---- 悲伤 / 低落: DA↓ 5HT↑ NE↓ ----
    rules_.push_back(make_rule("难过",   -0.5f, -0.1f, -0.3f, 0.4f, 0.1f, 0.1f));
    rules_.push_back(make_rule("伤心",   -0.5f, -0.1f, -0.3f, 0.4f, 0.1f, 0.1f));
    rules_.push_back(make_rule("悲伤",   -0.5f, -0.1f, -0.3f, 0.4f, 0.1f, 0.1f));
    rules_.push_back(make_rule("失落",   -0.4f, -0.1f, -0.2f, 0.3f, 0.1f, 0.0f));
    rules_.push_back(make_rule("沮丧",   -0.4f, -0.1f, -0.3f, 0.3f, 0.1f, 0.0f));
    rules_.push_back(make_rule("痛苦",   -0.5f, -0.2f, -0.2f, 0.4f, 0.2f, 0.0f));
    rules_.push_back(make_rule("想哭",   -0.5f, -0.1f, -0.3f, 0.5f, 0.2f, 0.2f));
    rules_.push_back(make_rule("哭",     -0.4f, -0.1f, -0.2f, 0.4f, 0.1f, 0.2f));

    // ---- 愤怒 / 烦躁: NE↑↑ DA↑ GABA↓ 5HT↓ ----
    rules_.push_back(make_rule("生气",   0.2f, 0.1f, 0.6f, -0.3f, -0.3f, -0.1f));
    rules_.push_back(make_rule("愤怒",   0.2f, 0.1f, 0.7f, -0.3f, -0.3f, -0.1f));
    rules_.push_back(make_rule("恼火",   0.2f, 0.1f, 0.6f, -0.3f, -0.3f, -0.1f));
    rules_.push_back(make_rule("气死",   0.3f, 0.1f, 0.8f, -0.3f, -0.4f, -0.1f));
    rules_.push_back(make_rule("烦",     0.1f, 0.0f, 0.4f, -0.2f, -0.2f, -0.1f));
    rules_.push_back(make_rule("烦躁",   0.1f, 0.0f, 0.5f, -0.2f, -0.3f, -0.1f));

    // ---- 恐惧 / 焦虑: NE↑↑ GABA↓ DA↓ ----
    rules_.push_back(make_rule("害怕",  -0.3f, 0.0f, 0.6f, -0.2f, -0.2f, 0.1f));
    rules_.push_back(make_rule("恐惧",  -0.3f, 0.0f, 0.7f, -0.2f, -0.3f, 0.1f));
    rules_.push_back(make_rule("担心",  -0.2f, 0.0f, 0.4f, 0.1f, 0.0f, 0.2f));
    rules_.push_back(make_rule("焦虑",  -0.3f, 0.1f, 0.6f, 0.1f, -0.2f, 0.1f));
    rules_.push_back(make_rule("紧张",  -0.2f, 0.1f, 0.5f, 0.1f, -0.1f, 0.1f));
    rules_.push_back(make_rule("不安",  -0.3f, 0.0f, 0.4f, 0.1f, 0.0f, 0.1f));

    // ---- 平静 / 放松: 5HT↑ NE↓ GABA↑ ----
    rules_.push_back(make_rule("平静",   0.0f, 0.0f, -0.3f, 0.3f, 0.3f, 0.0f));
    rules_.push_back(make_rule("放松",   0.1f, 0.0f, -0.3f, 0.3f, 0.3f, 0.0f));
    rules_.push_back(make_rule("安心",   0.1f, 0.0f, -0.2f, 0.3f, 0.2f, 0.1f));
    rules_.push_back(make_rule("舒服",   0.2f, 0.0f, -0.2f, 0.3f, 0.2f, 0.0f));

    // ---- 共情 / 关爱: Oxy↑↑ 5HT↑ NE↓ ----
    rules_.push_back(make_rule("心疼",   0.0f, 0.0f, -0.2f, 0.2f, 0.0f, 0.7f));
    rules_.push_back(make_rule("同情",   0.0f, 0.0f, -0.2f, 0.2f, 0.0f, 0.6f));
    rules_.push_back(make_rule("理解你", 0.0f, 0.0f, -0.2f, 0.2f, 0.0f, 0.6f));
    rules_.push_back(make_rule("心疼你", 0.0f, 0.0f, -0.2f, 0.2f, 0.0f, 0.8f));
    rules_.push_back(make_rule("在乎",   0.0f, 0.0f, -0.1f, 0.2f, 0.0f, 0.5f));

    // ---- 惊讶 / 惊喜: DA↑ NE↑ ACh↑ ----
    rules_.push_back(make_rule("惊讶",   0.4f, 0.3f, 0.4f, 0.0f, 0.0f, 0.0f));
    rules_.push_back(make_rule("吃惊",   0.4f, 0.3f, 0.4f, 0.0f, 0.0f, 0.0f));
    rules_.push_back(make_rule("意外",   0.2f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f));
    rules_.push_back(make_rule("居然",   0.2f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f));

    // ---- 程度副词 ----
    boosters_  = {"非常", "特别", "太", "超级", "极其", "无比", "极度", "十分", "真的"};
    dampeners_ = {"有点", "稍微", "略", "一点点", "不太"};
}

void EmotionEventExtractor::add_rule(const Rule& rule) {
    rules_.push_back(rule);
}

float EmotionEventExtractor::intensity_multiplier(const std::string& text, bool* found) const {
    *found = false;
    float mult = 1.0f;
    for (const auto& b : boosters_) {
        if (text.find(b) != std::string::npos) {
            mult *= 1.5f;
            *found = true;
        }
    }
    for (const auto& d : dampeners_) {
        if (text.find(d) != std::string::npos) {
            mult *= 0.5f;
            *found = true;
        }
    }
    // 感叹号增强: 每 3 个 ! 或 ！ 提升 1.2 倍
    int bangs = 0;
    for (char c : text) {
        if (c == '!' || c == '！') ++bangs;
    }
    if (bangs >= 3) mult *= 1.2f;
    return mult;
}

EmotionEvent EmotionEventExtractor::extract(const std::string& text) const {
    EmotionEvent ev;
    if (text.empty()) {
        return ev;  // 置信度 0, 无事件
    }

    bool has_booster = false;
    const float mult = intensity_multiplier(text, &has_booster);

    float hits = 0.0f;
    float accum[6] = {0, 0, 0, 0, 0, 0};

    for (const auto& r : rules_) {
        if (text.find(r.keyword) != std::string::npos) {
            for (int c = 0; c < 6; ++c) {
                accum[c] += r.delta[c] * r.weight * mult;
            }
            hits += r.weight;
        }
    }

    if (hits <= 0.0f) {
        return ev;  // 未命中任何情感词 → 中性事件 (置信度 0)
    }

    // clamp 到 [-1, +1] (SNN 调质增量安全范围)
    for (int c = 0; c < 6; ++c) {
        ev.modulator_delta[c] = clampf(accum[c], -1.0f, 1.0f);
    }
    // 单事件默认持续一个课程事件窗口 (与课程 offset=100 对齐)
    ev.duration_steps = 100;
    // 置信度: 命中词数饱和映射 (1 词≈0.5, 2 词≈0.75, 3 词≈0.85)
    ev.confidence = clampf(0.5f + hits * 0.25f, 0.0f, 0.95f);
    return ev;
}

}  // namespace bridge
}  // namespace vita
