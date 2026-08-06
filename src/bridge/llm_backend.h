// =============================================================================
// llm_backend.h — LLM 后端 SPI (Service Provider Interface)
// =============================================================================
// 桥接层通过本接口驱动 LLM 生成, 不依赖任何 llama.cpp 头文件:
//   1. 进程内 llama.cpp 适配器 (src/llm/llama_backend.cpp, 后续实现) 实现本接口,
//      使 AffectiveState → 采样参数真正作用于 llama_sampler;
//   2. 单元测试用 MockBackend 验证桥接编排逻辑;
//   3. 未来可替换/并存其它后端 (HTTP/子进程), 桥接层代码零改动。
// =============================================================================

#ifndef VITA_BRIDGE_LLM_BACKEND_H
#define VITA_BRIDGE_LLM_BACKEND_H

#include <string>
#include <utility>
#include <vector>
#include "emotion_types.h"

namespace vita {
namespace bridge {

class LlmBackend {
public:
    virtual ~LlmBackend() = default;

    // 应用采样参数 (生成前调用; 逐 token 调制由适配器内部基于最新 SamplerParams 实现)
    // 返回值: 0 成功, 负值错误码
    virtual int apply_sampler(const SamplerParams& params) = 0;

    // 注入情感 system prompt 片段 (追加/替换上下文中的情感槽位)
    // 返回值: 0 成功, 负值错误码
    virtual int apply_emotion_prompt(const std::string& snippet) = 0;

    // 应用逐 token 情感偏置 (logit_bias 通道, 2026-08-05):
    //   word_bias 每项 {情感词, 偏置量}; 偏置 > 0 提高该词 token 的 logits
    //   (生成期逐 token 干预采样分布), < 0 压低。后端内部负责 tokenize。
    //   空数组 = 清除偏置。默认实现空操作 (兼容旧后端), llama 后端覆盖为
    //   重建采样器链首的 llama_sampler_init_logit_bias。
    // 返回值: 0 成功, 负值错误码
    virtual int apply_logit_bias(const std::vector<std::pair<std::string, float>>& /*word_bias*/) {
        return 0;
    }

    // LLM→SNN 语义锚点: 用户/助手回合文本回调
    // (未来接事件抽取 → SnnFeedbackSink::emit_event, 实现情绪语义锚点)
    virtual void on_user_turn(const std::string& user_text) = 0;
    virtual void on_assistant_turn(const std::string& assistant_text) = 0;

    // 语义情感抽取 (词典→LLM 串联流水线的 LLM 语义裁决, 2026-08-07):
    //   对一段文本做一次短生成, 输出语义类别 (RawEmotion.emotion/attitude) + 强度。
    //   dict_hint: 词典一遍产出的先验 (类别人话摘要), 供 LLM 参照做语义裁决, 弥补
    //     关键词词典对否定句/改写/反讽的盲区; 为空则表示无词典先验。
    //   数值映射由调用方 (EmotionBridge) 经 emotion_from_category/attitude_from_category 完成。
    // 返回值: 0 成功且 out.ok=true; 负值 = 未支持/失败 (调用方降级为纯词典)。
    virtual int extract_emotion(const std::string& /*text*/, const std::string& /*dict_hint*/,
                                RawEmotion& /*out*/) {
        return -1;
    }
};

}  // namespace bridge
}  // namespace vita

#endif  // VITA_BRIDGE_LLM_BACKEND_H
