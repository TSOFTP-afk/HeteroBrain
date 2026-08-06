#include "emotion_bridge.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace vita {
namespace bridge {

void EmotionBridge::attach_backend(LlmBackend* backend) { backend_ = backend; }
void EmotionBridge::attach_snn_feedback(SnnFeedbackSink* sink) { snn_ = sink; }
void EmotionBridge::attach_emotion_extractor(const EmotionEventExtractor* extractor) {
    extractor_ = extractor;
}

void EmotionBridge::set_state(const EmotionState& state) { state_ = state; }
const EmotionState& EmotionBridge::state() const { return state_; }

void EmotionBridge::set_mapping_config(const MappingConfig& cfg) { cfg_ = cfg; }
const MappingConfig& EmotionBridge::mapping_config() const { return cfg_; }

SamplerParams EmotionBridge::compute_sampler_params() const {
    return map_to_sampler_params(state_, cfg_);
}

std::vector<std::pair<std::string, float>> EmotionBridge::compute_logit_bias() const {
    return ::vita::bridge::compute_logit_bias(state_);
}

std::string EmotionBridge::build_mood_description() const {
    // 限定命名空间调用, 避免成员函数同名递归
    return ::vita::bridge::build_mood_description(state_);
}

std::string EmotionBridge::build_system_prompt_snippet() const {
    return ::vita::bridge::build_system_prompt_snippet(state_);
}

int EmotionBridge::apply_to_generation() {
    if (backend_ == nullptr) {
        return -1;
    }
    const int rc_sampler = backend_->apply_sampler(compute_sampler_params());
    if (rc_sampler != 0) {
        return rc_sampler;
    }
    return backend_->apply_emotion_prompt(build_system_prompt_snippet());
}

void EmotionBridge::emit_empathy(float level) {
    if (snn_) {
        snn_->emit_empathy(level);
    }
}

void EmotionBridge::emit_event(const float modulator_delta[6], int duration_steps) {
    if (snn_) {
        snn_->emit_event(modulator_delta, duration_steps);
    }
}

void EmotionBridge::emit_other_emotion(const float weak_delta[6], int duration_steps,
                                       const char* text, int text_len) {
    if (snn_) {
        snn_->emit_other_emotion(weak_delta, duration_steps, text, text_len);
    }
}

void EmotionBridge::emit_embodied_reward(float reward) {
    if (snn_) {
        snn_->emit_embodied_reward(reward);
    }
}

void EmotionBridge::emit_world_event(int event_type, float intensity) {
    if (snn_) {
        snn_->emit_world_event(event_type, intensity);
    }
}

namespace {

// canonical 类别 → 人话标签 (词典先验摘要用, 与 emotion_types.h RawEmotion 枚举一致)
static const char* kEmotionCatName[] = {
    "neutral中性", "happy开心", "sad难过", "angry生气",
    "fear害怕", "calm平静", "empathy共情", "surprise惊讶",
};
static const char* kAttitudeCatName[] = {
    "neutral中性", "praise赞赏", "criticism批评", "threat攻击", "bond依恋",
};

inline std::string format_float(float v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return std::string(buf);
}

// 词典先验 (串联流水线的第一遍输出): 把词典 6 维增量反查成 canonical 类别的人话,
// 作为第二遍 LLM 语义裁决的参考输入。词典可能被表面词误导, 故提示 LLM 以语义为准。
inline std::string build_dict_hint(const EmotionEvent& ev, const EmotionEvent& att) {
    float ie = 0.0f, ia = 0.0f;
    const int ec = emotion_category_from_delta(ev.modulator_delta, &ie);
    const int ac = attitude_category_from_delta(att.modulator_delta, &ia);
    std::string h = "情绪=";
    h += (ec >= 0 && ec < 8) ? kEmotionCatName[ec] : kEmotionCatName[0];
    h += "(" + format_float(ie) + ")";
    h += ", 态度=";
    h += (ac >= 0 && ac < 5) ? kAttitudeCatName[ac] : kAttitudeCatName[0];
    h += "(" + format_float(ia) + ")";
    return h;
}

}  // namespace

int EmotionBridge::process_turn(const std::string& role, const std::string& text) {
    // ① 通知后端回合 (语义锚点回调)
    if (backend_) {
        if (role == "user") {
            backend_->on_user_turn(text);
        } else if (role == "assistant") {
            backend_->on_assistant_turn(text);
        }
    }
    // ② 情感处理 (2026-08-07 边界重设计 + 词典→LLM 固定两遍): 区分"用户情绪"
    //   与"用户态度"两条通道。
    //   * 用户"在经历什么" (情绪)   → 只驱动共情 (Oxy 关怀), 不翻转模型自身 PAD。
    //   * 用户"如何对待模型" (态度) → 社交反馈, 驱动模型自身情绪。
    // 抽取流水线 (固定两遍, 每轮都执行, 无置信度门控):
    //   第一遍 关键词词典 (零延迟快路径, 精确命中);
    //   第二遍 LLM 语义裁决 (弥补否定/改写/反讽的盲区), 参照词典先验做语义重判,
    //         判定结果作为最终裁决 (覆盖词典); LLM 未注入/判中性时安全降级为纯词典。
    // 无抽取器或反馈端时降级跳过。
    if (extractor_ == nullptr || snn_ == nullptr) {
        return -1;
    }

    // ---- 第一遍: 关键词词典 ----
    EmotionEvent ev = extractor_->extract(text);          // 用户情绪
    EmotionEvent att = extract_user_attitude(text);       // 用户态度

    // ---- 第二遍: LLM 语义裁决 (仅 user 回合, 避免对助手自我回复引入额外延迟) ----
    // 串联: 词典先验 → LLM 参照裁决 → 最终结果注入 SNN。
    if (backend_ && role == "user") {
        const std::string dict_hint = build_dict_hint(ev, att);
        RawEmotion raw;
        if (backend_->extract_emotion(text, dict_hint, raw) == 0 && raw.ok) {
            std::fprintf(stderr,
                         "[EmoBridge-串] dict_hint=%s | LLM裁决: emotion=%d(%.2f) "
                         "attitude=%d(%.2f) conf=%.2f\n",
                         dict_hint.c_str(), raw.emotion, raw.emotion_intensity,
                         raw.attitude, raw.attitude_intensity, raw.confidence);
            if (raw.emotion != 0) {
                ev = emotion_from_category(
                    raw.emotion, raw.emotion_intensity, raw.confidence);
            }
            if (raw.attitude != 0) {
                att = attitude_from_category(
                    raw.attitude, raw.attitude_intensity, raw.confidence);
            }
        }
    }

    // ②a 用户情绪 → 共情通道 (Phase 3a): 引擎感同身受。
    //     只经 set_empathy_signal 驱动催产素, 不再把用户情绪增量注入模型自身调制物,
    //     避免模型 PAD 镜像用户情绪 (2026-08-07 修复)。
    if (ev.confidence > 0.0f) {
        // 共情水平 = 情感强度 (confidence) 与催产素倾向 (oxy) 的加权平均, 再经 sqrt 压缩。
        //   sqrt 压缩让 0.5→0.71、0.9→0.95, 保留梯度, 不满值。
        const float oxy = ev.modulator_delta[5];   // OXY 索引 5 (与 GENE_MAP 顺序一致)
        float raw = 0.5f * ev.confidence + 0.5f * (oxy > 0.0f ? oxy : 0.0f);
        float empathy = sqrtf(raw);
        if (empathy > 1.0f) empathy = 1.0f;
        snn_->emit_empathy(empathy);

        // ②a-2 他人情绪弱泄入 (2026-08-07 边界重设计): 用户情绪 → 以较小增益泄入 SNN,
        //   让模型"感知到"用户在经历某种情绪, 但自我 PAD 不被拉到用户值 (可感知但不同步)。
        //   与他人情绪区分: 这是"他者"镜像, 主要走 Oxy (关怀) + 微 5HT/NE, 不翻转自我 PAD。
        //   附带原始文本 → 工作台 OTHER 标签 (LLM 可读出"SNN 感知到用户情绪")。
        const float kOtherGain = 0.30f;   // 他人情绪净增益 (显著低于自我事件 1.0)
        float weak[6] = {0, 0, 0, 0, 0, 0};
        weak[5] = kOtherGain * (oxy > 0.0f ? oxy : ev.confidence);  // Oxy 主 (关怀)
        weak[3] = kOtherGain * ev.modulator_delta[3] * 0.5f;        // 微 5HT
        weak[2] = kOtherGain * ev.modulator_delta[2] * 0.5f;        // 微 NE
        snn_->emit_other_emotion(weak, ev.duration_steps,
                                 text.data(), static_cast<int>(text.size()));
    }

    // ②b 用户态度 → 社交反馈, 驱动模型自身情绪 (PRAISE/CRITICISM/THREAT_SOCIAL/SOCIAL_BOND)。
    //     这是模型 PAD 的来源 — 由"用户如何对待我"决定, 而非"用户在经历什么"。
    if (att.confidence > 0.0f) {
        snn_->emit_event(att.modulator_delta, att.duration_steps);
    }
    return 0;
}

}  // namespace bridge
}  // namespace vita
