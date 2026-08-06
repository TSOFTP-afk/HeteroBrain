// =============================================================================
// emotion_bridge.h — 链接层主入口 (EmotionBridge 编排器)
// =============================================================================
// 职责 (双向通道):
//   SNN→LLM: 接收情感快照 → 映射为采样参数 + 情感 prompt →
//            经 LlmBackend SPI 应用至 LLM 生成
//   LLM→SNN: 共情/事件/奖励回流 → 经 SnnFeedbackSink SPI 注入 SNN
// 可复用性: 不依赖 SNN 头文件与 llama.cpp 头文件; 后端/反馈端运行期可替换;
//           未注入时调用安全降级 (纯计算仍可用)。
// 扩展性:   未来 spike_embedding / pca_projection / truth_filter 等 T2H 模块
//           作为独立模块挂载在 EmotionBridge 之外, 复用 emotion_types 契约,
//           不侵入本编排器。
// =============================================================================

#ifndef VITA_BRIDGE_EMOTION_BRIDGE_H
#define VITA_BRIDGE_EMOTION_BRIDGE_H

#include <string>
#include "affective_mapping.h"
#include "emotion_event.h"
#include "emotion_prompt.h"
#include "emotion_types.h"
#include "llm_backend.h"
#include "snn_feedback.h"

namespace vita {
namespace bridge {

class EmotionBridge {
public:
    // 连接外部组件 (可运行期替换; 均允许为空, 空时相关调用安全降级)
    void attach_backend(LlmBackend* backend);
    void attach_snn_feedback(SnnFeedbackSink* sink);
    // 情感事件抽取器 (默认内置中文词典; 可替换为领域定制抽取器)
    void attach_emotion_extractor(const EmotionEventExtractor* extractor);

    // SNN 每轮推送情感快照 (线程安全由调用方保证)
    void set_state(const EmotionState& state);
    const EmotionState& state() const;

    void set_mapping_config(const MappingConfig& cfg);
    const MappingConfig& mapping_config() const;

    // ---- SNN→LLM 纯计算 (不依赖后端, 便于测试/调试) ----
    SamplerParams compute_sampler_params() const;
    // 词级 logit 偏置 (logit_bias 通道): 当前情感快照 → {情感词, 偏置量} 列表
    std::vector<std::pair<std::string, float>> compute_logit_bias() const;
    std::string  build_mood_description() const;
    std::string  build_system_prompt_snippet() const;

    // ---- SNN→LLM 应用 (经已注入后端) ----
    // 依次 apply_sampler + apply_emotion_prompt
    // 返回值: 0 成功; -1 未注入后端; 其它负值为后端错误码
    int apply_to_generation();

    // ---- LLM→SNN 回流 (经已注入反馈端; 未注入时静默丢弃) ----
    void emit_empathy(float level);
    void emit_event(const float modulator_delta[6], int duration_steps);
    void emit_embodied_reward(float reward);
    // 世界事件 (事件类型, 强度) → 反馈端 emit_world_event (Phase 3a-G):
    //   事件类型直通 SNN (杏仁核 LA 注入 + 联合皮层子区域注入), 强度 [-50,50]
    void emit_world_event(int event_type, float intensity);

    // ---- LLM→SNN 语义锚点编排 ----
    // 处理一轮对话文本 (role: "user"/"assistant"):
    //   ① 通知后端回合 (backend->on_user_turn/on_assistant_turn)
    //   ② 情感事件抽取 → emit_event 注入 SNN (抽取器可注入/替换)
    // 返回值: 0 已处理; -1 无抽取器
    int process_turn(const std::string& role, const std::string& text);

private:
    EmotionState    state_;
    MappingConfig   cfg_;
    LlmBackend*     backend_ = nullptr;
    SnnFeedbackSink* snn_    = nullptr;
    const EmotionEventExtractor* extractor_ = nullptr;
};

}  // namespace bridge
}  // namespace vita

#endif  // VITA_BRIDGE_EMOTION_BRIDGE_H
