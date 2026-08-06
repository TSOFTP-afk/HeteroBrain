#include "emotion_bridge.h"

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

int EmotionBridge::process_turn(const std::string& role, const std::string& text) {
    // ① 通知后端回合 (语义锚点回调)
    if (backend_) {
        if (role == "user") {
            backend_->on_user_turn(text);
        } else if (role == "assistant") {
            backend_->on_assistant_turn(text);
        }
    }
    // ② 情感事件抽取 → 注入 SNN (无抽取器或反馈端时降级跳过)
    if (extractor_ == nullptr || snn_ == nullptr) {
        return -1;
    }
    const EmotionEvent ev = extractor_->extract(text);
    if (ev.confidence <= 0.0f) {
        return 0;  // 无情感信号, 无事件注入
    }
    snn_->emit_event(ev.modulator_delta, ev.duration_steps);
    return 0;
}

}  // namespace bridge
}  // namespace vita
