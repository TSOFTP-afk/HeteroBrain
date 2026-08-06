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

// -----------------------------------------------------------------------------
// extract_user_attitude — 提取"用户对模型的态度"(社交反馈), 驱动模型自身情绪。
// -----------------------------------------------------------------------------
// 与 extract() 的关键区别:
//   * extract()        → 用户"在经历什么" (情绪)   → 只驱动共情 (Oxy), 不翻转模型 PAD
//   * extract_user_attitude() → 用户"如何对待模型" (态度) → 社交反馈, 驱动模型自身 PAD
// 语义对照 GENE_MAP 社交事件:
//   PRAISE / CRITICISM / THREAT_SOCIAL / SOCIAL_BOND
// -----------------------------------------------------------------------------
EmotionEvent extract_user_attitude(const std::string& text) {
    EmotionEvent ev;
    if (text.empty()) {
        return ev;
    }

    // 态度规则: [DA, ACh, NE, 5HT, GABA, Oxy] (与 GENE_MAP 6 维语义一致)
    struct AttRule {
        const char* kw;
        float delta[6];
        float weight;
    };
    // ① 赞赏 / 感谢 → PRAISE: DA↑ Oxy↑ (社交接纳)
    static const AttRule kPraise[] = {
        {"你真棒",  {0.25f, 0.10f, 0.15f, -0.05f, 0.00f, 0.20f}, 1.0f},
        {"太棒了",  {0.25f, 0.10f, 0.15f, -0.05f, 0.00f, 0.20f}, 1.0f},
        {"厉害",    {0.20f, 0.10f, 0.10f, -0.05f, 0.00f, 0.15f}, 0.9f},
        {"真厉害",  {0.25f, 0.10f, 0.10f, -0.05f, 0.00f, 0.15f}, 1.0f},
        {"了不起",  {0.25f, 0.10f, 0.10f, -0.05f, 0.00f, 0.15f}, 1.0f},
        {"聪明",    {0.20f, 0.10f, 0.10f, -0.05f, 0.00f, 0.10f}, 0.8f},
        {"谢谢你",  {0.20f, 0.05f, 0.05f,  0.00f, 0.00f, 0.25f}, 1.0f},
        {"感谢你",  {0.20f, 0.05f, 0.05f,  0.00f, 0.00f, 0.25f}, 1.0f},
        {"帮了大忙",{0.25f, 0.10f, 0.05f,  0.00f, 0.00f, 0.20f}, 1.0f},
        {"有你真好",{0.15f, 0.05f, 0.00f,  0.05f, 0.05f, 0.30f}, 1.0f},
    };
    // ② 批评 / 否定 → CRITICISM: 5HT↑ DA↓ Oxy↓ (社交疼痛)
    static const AttRule kCriticism[] = {
        {"你真差",  {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f}, 1.0f},
        {"没用",    {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f}, 1.0f},
        {"废物",    {-0.15f, 0.05f, 0.25f, 0.30f, 0.00f, -0.20f}, 1.0f},
        {"垃圾",    {-0.15f, 0.05f, 0.25f, 0.30f, 0.00f, -0.20f}, 1.0f},
        {"真蠢",    {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f}, 1.0f},
        {"笨蛋",    {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f}, 1.0f},
        {"失望",    {-0.10f, 0.05f, 0.15f, 0.25f, 0.00f, -0.15f}, 1.0f},
        {"差劲",    {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f}, 1.0f},
    };
    // ③ 攻击 / 厌恶 → THREAT_SOCIAL: NE↑ 5HT↑ DA↓ (社交应激)
    static const AttRule kThreat[] = {
        {"滚",      {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
        {"讨厌你",  {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
        {"恨你",    {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
        {"闭嘴",    {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
        {"烦死你了",{-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
        {"别烦我",  {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f}, 1.0f},
    };
    // ④ 需要 / 依恋 → SOCIAL_BOND: Oxy↑ (依恋)
    static const AttRule kBond[] = {
        {"需要你",  {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
        {"陪陪我",  {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
        {"只有你",  {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
        {"别离开",  {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
        {"靠你了",  {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
        {"离不开你",{0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f}, 1.0f},
    };

    float hits = 0.0f;
    float accum[6] = {0, 0, 0, 0, 0, 0};

    auto scan = [&](const AttRule* rules, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (text.find(rules[i].kw) != std::string::npos) {
                for (int c = 0; c < 6; ++c) {
                    accum[c] += rules[i].delta[c] * rules[i].weight;
                }
                hits += rules[i].weight;
            }
        }
    };
    scan(kPraise, sizeof(kPraise) / sizeof(kPraise[0]));
    scan(kCriticism, sizeof(kCriticism) / sizeof(kCriticism[0]));
    scan(kThreat, sizeof(kThreat) / sizeof(kThreat[0]));
    scan(kBond, sizeof(kBond) / sizeof(kBond[0]));

    if (hits <= 0.0f) {
        return ev;  // 未命中任何态度 → 中性 (置信度 0)
    }

    for (int c = 0; c < 6; ++c) {
        ev.modulator_delta[c] = clampf(accum[c], -1.0f, 1.0f);
    }
    ev.duration_steps = 100;
    ev.confidence = clampf(0.4f + hits * 0.2f, 0.0f, 0.95f);
    return ev;
}

// -----------------------------------------------------------------------------
// canonical 类别 → 6 维调质增量 (词典→LLM 两遍流水线的第二遍数值映射)
// -----------------------------------------------------------------------------
// 基准表与关键词词典同一语义 (与 emotion_event.cpp 顶部词典 / GENE_MAP 对齐):
//   DA↑ 愉悦 / NE↑ 唤醒 / 5HT↑ 镇静低落 / GABA↑ 平静 / Oxy↑ 共情
// 类别下标与 emotion_types.h RawEmotion 枚举一致。
// -----------------------------------------------------------------------------
namespace {

// 情绪类别基准行 [DA, ACh, NE, 5HT, GABA, Oxy]
static const float kEmotionProfile[][6] = {
    /* 0 neutral  */ {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    /* 1 happy    */ {0.5f, 0.2f, 0.2f, -0.2f, 0.0f, 0.1f},
    /* 2 sad      */ {-0.5f, -0.1f, -0.3f, 0.4f, 0.1f, 0.1f},
    /* 3 angry    */ {0.2f, 0.1f, 0.6f, -0.3f, -0.3f, -0.1f},
    /* 4 fear     */ {-0.3f, 0.0f, 0.6f, -0.2f, -0.2f, 0.1f},
    /* 5 calm     */ {0.0f, 0.0f, -0.3f, 0.3f, 0.3f, 0.0f},
    /* 6 empathy  */ {0.0f, 0.0f, -0.2f, 0.2f, 0.0f, 0.7f},
    /* 7 surprise */ {0.4f, 0.3f, 0.4f, 0.0f, 0.0f, 0.0f},
};
static constexpr int kEmotionProfileCount =
    static_cast<int>(sizeof(kEmotionProfile) / sizeof(kEmotionProfile[0]));

// 态度类别基准行 (PRAISE/CRITICISM/THREAT_SOCIAL/SOCIAL_BOND)
static const float kAttitudeProfile[][6] = {
    /* 0 neutral  */ {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    /* 1 praise   */ {0.25f, 0.10f, 0.15f, -0.05f, 0.00f, 0.20f},
    /* 2 criticism*/ {-0.10f, 0.05f, 0.20f, 0.25f, 0.00f, -0.15f},
    /* 3 threat   */ {-0.15f, 0.20f, 0.45f, 0.35f, 0.05f, -0.10f},
    /* 4 bond     */ {0.10f, 0.05f, -0.05f, 0.05f, 0.05f, 0.35f},
};
static constexpr int kAttitudeProfileCount =
    static_cast<int>(sizeof(kAttitudeProfile) / sizeof(kAttitudeProfile[0]));

}  // namespace

EmotionEvent emotion_from_category(int emotion, float intensity, float confidence) {
    EmotionEvent ev;
    if (emotion <= 0 || emotion >= kEmotionProfileCount) {
        return ev;  // neutral 或越界 → 全零事件
    }
    const float* row = kEmotionProfile[emotion];
    for (int c = 0; c < 6; ++c) {
        ev.modulator_delta[c] = clampf(row[c] * intensity, -1.0f, 1.0f);
    }
    ev.duration_steps = 100;
    ev.confidence = clampf(confidence, 0.0f, 1.0f);
    return ev;
}

EmotionEvent attitude_from_category(int attitude, float intensity, float confidence) {
    EmotionEvent ev;
    if (attitude <= 0 || attitude >= kAttitudeProfileCount) {
        return ev;  // neutral 或越界 → 全零事件
    }
    const float* row = kAttitudeProfile[attitude];
    for (int c = 0; c < 6; ++c) {
        ev.modulator_delta[c] = clampf(row[c] * intensity, -1.0f, 1.0f);
    }
    ev.duration_steps = 100;
    ev.confidence = clampf(confidence, 0.0f, 1.0f);
    return ev;
}

// -----------------------------------------------------------------------------
// 6 维增量 → canonical 类别 (最近邻, 词典先验反查)
// -----------------------------------------------------------------------------
// 在基准表上对每个类别做点积相似度, 取最大正向点积的类别; 无正向命中 → 0 (中性)。
// 强度取增量最大绝对值 (clamp [0,1]), 近似该类在 SNN 端的注入强弱。
int emotion_category_from_delta(const float delta[6], float* intensity) {
    int best = 0;
    float best_dot = 0.0f;
    float mag = 0.0f;
    for (int c = 0; c < 6; ++c) {
        const float a = delta[c] < 0.0f ? -delta[c] : delta[c];
        if (a > mag) mag = a;
    }
    for (int k = 1; k < kEmotionProfileCount; ++k) {
        float dot = 0.0f;
        for (int c = 0; c < 6; ++c) {
            dot += delta[c] * kEmotionProfile[k][c];
        }
        if (dot > best_dot) {
            best_dot = dot;
            best = k;
        }
    }
    if (intensity) {
        *intensity = clampf(mag, 0.0f, 1.0f);
    }
    return (best_dot > 0.0f) ? best : 0;
}

int attitude_category_from_delta(const float delta[6], float* intensity) {
    int best = 0;
    float best_dot = 0.0f;
    float mag = 0.0f;
    for (int c = 0; c < 6; ++c) {
        const float a = delta[c] < 0.0f ? -delta[c] : delta[c];
        if (a > mag) mag = a;
    }
    for (int k = 1; k < kAttitudeProfileCount; ++k) {
        float dot = 0.0f;
        for (int c = 0; c < 6; ++c) {
            dot += delta[c] * kAttitudeProfile[k][c];
        }
        if (dot > best_dot) {
            best_dot = dot;
            best = k;
        }
    }
    if (intensity) {
        *intensity = clampf(mag, 0.0f, 1.0f);
    }
    return (best_dot > 0.0f) ? best : 0;
}

}  // namespace bridge
}  // namespace vita
