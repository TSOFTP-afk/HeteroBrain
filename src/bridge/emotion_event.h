// =============================================================================
// emotion_event.h — 情感事件抽取 (LLM→SNN 语义锚点)
// =============================================================================
// 从对话文本中识别情感信号, 映射为 SNN 可消费的 6 维调质增量
// ([DA, ACh, NE, 5HT, GABA, Oxy], 与 snn_feedback.h / GENE_MAP 顺序一致)。
//
// 设计:
//   1. 词典驱动: 情感关键词 → 调质增量模式, 内置中文词典, 支持运行时扩展
//   2. 强度加权: 程度副词 (很/非常/太…) 与感叹号增强, 中性词衰减
//   3. 确定性纯函数: 便于单元测试与离线调参 (同 affective_mapping 原则)
//   4. 桥接层通过 SnnFeedbackSink::emit_event 注入 SNN, 不直接依赖 SNN 头文件
//
// 调质语义基准 (与 src/snn/modulatory_kernels.cu 注释一致):
//   DA↑ 愉悦 / NE↑ 唤醒 / 5HT↑ 镇静低落 / GABA↑ 平静 / Oxy↑ 共情
// =============================================================================

#ifndef VITA_BRIDGE_EMOTION_EVENT_H
#define VITA_BRIDGE_EMOTION_EVENT_H

#include <string>
#include <vector>
#include "emotion_types.h"

namespace vita {
namespace bridge {

// -----------------------------------------------------------------------------
// EmotionEvent — 一次对话情感事件 (可直接传给 SnnFeedbackSink::emit_event)
// -----------------------------------------------------------------------------
struct EmotionEvent {
    float modulator_delta[6] = {0, 0, 0, 0, 0, 0};  // [DA, ACh, NE, 5HT, GABA, Oxy]
    int   duration_steps = 0;   // 0=单次脉冲, >0=plateau 型每 100 步递减 (SNN 语义)
    float confidence = 0.0f;    // [0,1] 抽取置信度
};

// -----------------------------------------------------------------------------
// EmotionEventExtractor — 文本 → 情感事件
// 线程安全: 无内部可变状态 (词典构建后只读), 可跨线程复用
// -----------------------------------------------------------------------------
class EmotionEventExtractor {
public:
    // 词典条目: 关键词 + 调质增量 + 基础权重
    struct Rule {
        std::string keyword;         // UTF-8 关键词 (中文子串匹配)
        float delta[6];              // [DA, ACh, NE, 5HT, GABA, Oxy]
        float weight = 1.0f;         // 命中权重
    };

    EmotionEventExtractor();

    // 从文本抽取情感事件 (内部: 词典命中累加 + 强度加权 + clamp)
    EmotionEvent extract(const std::string& text) const;

    // 运行时扩展词典 (可复用性: 支持领域定制情感词)
    void add_rule(const Rule& rule);

    // 词典大小 (诊断用)
    std::size_t rule_count() const { return rules_.size(); }

private:
    // 强度加权: 返回 (倍数, 是否命中程度副词)
    float intensity_multiplier(const std::string& text, bool* found) const;

    std::vector<Rule> rules_;
    std::vector<std::string> boosters_;    // 程度增强词 (很/非常/太…)
    std::vector<std::string> dampeners_;   // 程度减弱词 (有点/稍微/略…)
};

// -----------------------------------------------------------------------------
// extract_user_attitude — 提取“用户对模型的态度”(社交反馈), 而非用户自身情绪。
// -----------------------------------------------------------------------------
// 边界区分 (2026-08-07):
//   * 用户“在经历什么”(情绪)  → 只驱动共情 (Oxy 关怀), 不翻转模型 PAD。
//   * 用户“如何对待模型”(态度) → 驱动模型自身情绪 (社交反馈)。
// 本函数识别第二类: 模型被赞赏/否定/攻击/需要 → 对应调制物增量
//   ([DA, ACh, NE, 5HT, GABA, Oxy], 与 GENE_MAP 语义一致)。
// 词典语义对照 GENE_MAP 社交事件: PRAISE/CRITICISM/THREAT_SOCIAL/SOCIAL_BOND。
EmotionEvent extract_user_attitude(const std::string& text);

// -----------------------------------------------------------------------------
// canonical 类别 → 6 维调质增量 (词典→LLM 两遍流水线的第二遍数值映射)
// -----------------------------------------------------------------------------
// LLM 只输出语义类别 (RawEmotion.emotion/attitude) + 强度, 本函数把类别映射为
// 与关键词词典同一语义的确定性调质增量 (见 emotion_event.cpp 基准表):
//   emotion_from_category: 情绪类别 → [DA,ACh,NE,5HT,GABA,Oxy] × intensity
//   attitude_from_category: 态度类别 → 同上 (PRAISE/CRITICISM/THREAT_SOCIAL/SOCIAL_BOND)
// 均返回 duration_steps=100 (与词典单事件对齐); 类别 0 (neutral) 返回全零事件。
EmotionEvent emotion_from_category(int emotion, float intensity, float confidence);
EmotionEvent attitude_from_category(int attitude, float intensity, float confidence);

// -----------------------------------------------------------------------------
// 6 维调质增量 → canonical 类别 (词典→LLM 串联流水线的词典先验反查, 2026-08-07)
// -----------------------------------------------------------------------------
// 词典抽取返回的是 6 维增量 (modulator_delta), 本身不带类别标签。串联流程把
// 词典结果作为"先验"传给 LLM 做语义裁决, 需要先把增量反查成它最接近的类别
// (在 canonical 基准表 kEmotionProfile/kAttitudeProfile 上做最近邻)。
//   delta      - 6 维调质增量 [DA,ACh,NE,5HT,GABA,Oxy]
//   intensity  - [out] 该类的强度 (≈增量最大绝对值, clamp [0,1])
// 返回 canonical 类别下标; 无正向命中 (中性/未匹配) 返回 0。
int emotion_category_from_delta(const float delta[6], float* intensity);
int attitude_category_from_delta(const float delta[6], float* intensity);

}  // namespace bridge
}  // namespace vita

#endif  // VITA_BRIDGE_EMOTION_EVENT_H
