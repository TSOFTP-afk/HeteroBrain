// =============================================================================
// affective_mapping.h — 情感状态 → LLM 采样参数映射
// =============================================================================
// 确定性纯函数, 便于单元测试与离线调参。映射系数见 MappingConfig (emotion_types.h)。
// =============================================================================

#ifndef VITA_BRIDGE_AFFECTIVE_MAPPING_H
#define VITA_BRIDGE_AFFECTIVE_MAPPING_H

#include <string>
#include <utility>
#include <vector>
#include "emotion_types.h"

namespace vita {
namespace bridge {

// 情感状态 → LLM 采样参数 (线性叠加 + 硬边界 clamp)
SamplerParams map_to_sampler_params(const EmotionState& state,
                                    const MappingConfig& cfg = MappingConfig{});

// 情感状态 → 词级 logit 偏置 (logit_bias 通道, 2026-08-05)
// 每项 {情感词, 偏置量}: 偏置 > 0 提高该词 token 采样概率, < 0 压低。
// 偏置量随状态强度线性缩放 (|signal| 超过阈值才激活), 情感越强干预越大。
// 中性状态返回空数组 (无干预)。这是逐 token 层级的 SNN→LLM 调制,
// 区别于采样参数 (数值通道) 与 system prompt (文字通道)。
std::vector<std::pair<std::string, float>> compute_logit_bias(const EmotionState& state);

}  // namespace bridge
}  // namespace vita

#endif  // VITA_BRIDGE_AFFECTIVE_MAPPING_H
