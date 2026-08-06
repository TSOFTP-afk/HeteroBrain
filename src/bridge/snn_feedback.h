// =============================================================================
// snn_feedback.h — SNN 反馈接收端 SPI (LLM→SNN 回流通道)
// =============================================================================
// 引擎将本接口的实现连接到 src/snn 的 host 端 setter
// (set_empathy_signal / set_event_signal / set_embodied_reward),
// 桥接层由此驱动情感回流, 不直接包含 SNN 头文件, 保持解耦。
// 通道顺序与 SNN 一致: [DA, ACh, NE, 5HT, GABA, Oxy] (GENE_MAP 顺序, 见 mod_simulator.h)。
// =============================================================================

#ifndef VITA_BRIDGE_SNN_FEEDBACK_H
#define VITA_BRIDGE_SNN_FEEDBACK_H

namespace vita {
namespace bridge {

class SnnFeedbackSink {
public:
    virtual ~SnnFeedbackSink() = default;

    // 共情信号 [0,1] → SNN set_empathy_signal
    virtual void emit_empathy(float level) = 0;

    // 6 维调质增量 [DA, ACh, NE, 5HT, GABA, Oxy] + 持续步数 → SNN set_event_signal
    // (duration_steps: 0=单次脉冲, >0=plateau 型每 100 步递减)
    virtual void emit_event(const float modulator_delta[6], int duration_steps) = 0;

    // 他人情绪弱泄入 (2026-08-07 边界重设计): 用户情绪 → 以较小的增益泄入 SNN,
    //   让模型"感知到"用户在经历某种情绪, 但自我 PAD 不被拉到用户值 (可感知但不同步)。
    // 与 emit_event 的区别: emit_event 驱动模型自身情绪 (社交反馈); 本通道是"他者"镜像,
    //   增量已按他人增益缩放 (Oxy 主 + 微 5HT/NE), 并附带文本用于工作台 OTHER 标签。
    virtual void emit_other_emotion(const float weak_delta[6], int duration_steps,
                                    const char* text, int text_len) = 0;

    // 世界事件 (事件类型, 强度) → SNN 事件通道 (Phase 3a-G, 2026-08-06):
    //   事件类型直通 SNN — 杏仁核 LA 注入 (M1) + 联合皮层子区域注入 (A),
    //   强度 [-50, 50] 线性缩放。不做语义转换 (LLM 理解转化器属后续 spec);
    //   调用方保证 event_type 落在 [0, EVT_COUNT) (事件类型见 src/snn/event_types.h)。
    virtual void emit_world_event(int event_type, float intensity) = 0;

    // 外部奖励 → SNN set_embodied_reward
    virtual void emit_embodied_reward(float reward) = 0;
};

}  // namespace bridge
}  // namespace vita

#endif  // VITA_BRIDGE_SNN_FEEDBACK_H
