// =============================================================================
// emotion_prompt.h — 情感上下文文本生成
// =============================================================================
// 将 EmotionState 渲染为 LLM 可消费的中文情感描述:
//   build_mood_description      — 单句心情摘要 (日志/调试/轻量注入)
//   build_system_prompt_snippet — 追加到 system prompt 的情感上下文 (含共情级别)
// =============================================================================

#ifndef HETERO_BRAIN_BRIDGE_EMOTION_PROMPT_H
#define HETERO_BRAIN_BRIDGE_EMOTION_PROMPT_H

#include <string>
#include "emotion_types.h"

namespace hb {
namespace bridge {

// 单句心情描述: "当前情感: 愉悦、兴奋、主动 (愉悦 0.50, 唤醒 0.40, 主导 0.60, 共情 0.80)"
std::string build_mood_description(const EmotionState& state);

// system prompt 情感片段: 【当前情感状态】… (含共情级别, 指导 LLM 调整语气)
std::string build_system_prompt_snippet(const EmotionState& state);

}  // namespace bridge
}  // namespace hb

#endif  // HETERO_BRAIN_BRIDGE_EMOTION_PROMPT_H
