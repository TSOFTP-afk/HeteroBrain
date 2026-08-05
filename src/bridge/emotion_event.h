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

#ifndef HETERO_BRAIN_BRIDGE_EMOTION_EVENT_H
#define HETERO_BRAIN_BRIDGE_EMOTION_EVENT_H

#include <string>
#include <vector>

namespace hb {
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

}  // namespace bridge
}  // namespace hb

#endif  // HETERO_BRAIN_BRIDGE_EMOTION_EVENT_H
