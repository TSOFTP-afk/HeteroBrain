// =============================================================================
// emotion_types.h — 桥接层数据类型契约 (纯 C++17, 无外部依赖)
// =============================================================================
// 设计原则:
//   1. 桥接层不依赖 SNN 头文件 (modulatory_kernels.cuh) 与 llama.cpp 头文件,
//      只通过本文件定义的数据契约与两个 SPI 接口 (llm_backend.h / snn_feedback.h)
//      交互。EmotionState 与 SNN 的 AffectiveState 字段一一对应, 由引擎边界
//      (heterobrain engine) 做一次字段拷贝, 保持桥接层解耦、可独立构建测试。
//   2. 扩展性: 新增情感维度 / 采样器参数时在此追加字段即可, 默认值保证
//      既有调用方与适配器不受影响 (向后兼容)。
// =============================================================================

#ifndef HETERO_BRAIN_BRIDGE_EMOTION_TYPES_H
#define HETERO_BRAIN_BRIDGE_EMOTION_TYPES_H

#include <cstdint>

namespace hb {
namespace bridge {

// -----------------------------------------------------------------------------
// EmotionState — SNN 情感读出的桥接自有快照
// 字段注释中的取值范围与 src/snn/modulatory_kernels.cuh 的 AffectiveState 一致
// -----------------------------------------------------------------------------
struct EmotionState {
    // 6 维调质浓度快照 [0, 2] (基线约 1.0)
    float dopamine       = 0.0f;
    float serotonin      = 0.0f;
    float norepinephrine = 0.0f;
    float acetylcholine  = 0.0f;
    float gaba           = 0.0f;
    float oxytocin       = 0.0f;

    // PAD 情感模型 [-1, 1]
    float pleasure  = 0.0f;   // 愉悦度
    float arousal   = 0.0f;   // 唤醒度
    float dominance = 0.0f;   // 主导度

    // LLM 生成调制信号 (delta, 叠加到 LLM 默认采样参数上)
    float temperature_delta = 0.0f;   // [-0.5, +0.5]
    float top_p_delta       = 0.0f;   // [-0.3,  0.0]
    float repetition_delta  = 0.0f;   // [ 0.0, +0.2]
    float empathy_level     = 0.0f;   // [ 0.0,  1.0]

    // 元信息
    std::int64_t step       = 0;      // SNN 训练/运行步
    float        confidence = 1.0f;   // [0, 1] 状态置信度

    // 中性基线: 调质=1.0 (平衡), PAD=0, 调制信号=0, 置信度=1
    static EmotionState neutral();
};

inline EmotionState EmotionState::neutral() {
    EmotionState s;
    s.dopamine = s.serotonin = s.norepinephrine = s.acetylcholine = s.gaba = s.oxytocin = 1.0f;
    s.confidence = 1.0f;
    return s;
}

// -----------------------------------------------------------------------------
// SamplerParams — LLM 采样参数快照 (对齐 llama.cpp 采样器可消费字段)
// -----------------------------------------------------------------------------
// 扩展采样器 (mirostat / typical_p / presence_penalty 等) 时在此追加字段,
// 默认值保持不变, 既有适配器不受影响。
// -----------------------------------------------------------------------------
struct SamplerParams {
    float temperature    = 0.8f;    // 温度 [0.1, 2.0]
    float top_p          = 0.95f;   // 核采样 [0.5, 1.0]
    int   top_k          = 40;      // top-k
    float repeat_penalty = 1.1f;    // 重复惩罚 [1.0, 2.0]
    float min_p          = 0.05f;   // min-p 采样
};

// -----------------------------------------------------------------------------
// MappingConfig — 情感 → 采样参数映射策略 (可配置系数)
// -----------------------------------------------------------------------------
// 默认值编码 SNN 文档注释的调制语义 (src/snn/modulatory_kernels.cuh §3.2):
//   temperature_delta: DA↑→+0.3, 5HT↑→-0.3, GABA↑→-0.1 (SNN 侧已算出)
// 桥接层只做 线性叠加 + 硬边界 clamp, 保护 LLM 采样器不越界。
// 未来可从 config YAML 加载覆盖 (configs/default.yaml → bridge.emotion)。
// -----------------------------------------------------------------------------
struct MappingConfig {
    float temperature_base    = 0.8f;    // temperature = base + scale * delta
    float temperature_scale   = 1.0f;
    float top_p_base          = 0.95f;
    float top_p_scale         = 1.0f;
    float repeat_penalty_base = 1.1f;
    float repeat_penalty_scale = 0.5f;

    // 硬边界 (clamp 区间, 保护采样器)
    float temperature_min = 0.1f;
    float temperature_max = 2.0f;
    float top_p_min       = 0.5f;
    float top_p_max       = 1.0f;
    float repeat_penalty_min = 1.0f;
    float repeat_penalty_max = 2.0f;
};

}  // namespace bridge
}  // namespace hb

#endif  // HETERO_BRAIN_BRIDGE_EMOTION_TYPES_H
