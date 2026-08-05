// =============================================================================
// snn_feedback.h — SNN 反馈接收端 SPI (LLM→SNN 回流通道)
// =============================================================================
// 引擎将本接口的实现连接到 src/snn 的 host 端 setter
// (set_empathy_signal / set_event_signal / set_embodied_reward),
// 桥接层由此驱动情感回流, 不直接包含 SNN 头文件, 保持解耦。
// 通道顺序与 SNN 一致: [DA, ACh, NE, 5HT, GABA, Oxy] (GENE_MAP 顺序, 见 mod_simulator.h)。
// =============================================================================

#ifndef HETERO_BRAIN_BRIDGE_SNN_FEEDBACK_H
#define HETERO_BRAIN_BRIDGE_SNN_FEEDBACK_H

namespace hb {
namespace bridge {

class SnnFeedbackSink {
public:
    virtual ~SnnFeedbackSink() = default;

    // 共情信号 [0,1] → SNN set_empathy_signal
    virtual void emit_empathy(float level) = 0;

    // 6 维调质增量 [DA, ACh, NE, 5HT, GABA, Oxy] + 持续步数 → SNN set_event_signal
    // (duration_steps: 0=单次脉冲, >0=plateau 型每 100 步递减)
    virtual void emit_event(const float modulator_delta[6], int duration_steps) = 0;

    // 外部奖励 → SNN set_embodied_reward
    virtual void emit_embodied_reward(float reward) = 0;
};

}  // namespace bridge
}  // namespace hb

#endif  // HETERO_BRAIN_BRIDGE_SNN_FEEDBACK_H
