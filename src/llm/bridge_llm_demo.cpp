// =============================================================================
// bridge_llm_demo.cpp — 桥接层全链路集成验证
// =============================================================================
// 验证: EmotionBridge (SNN 情感) → LlamaBackend (进程内 llama.cpp) → MiniCPM5
// 流程: 同一问题分别用 中性 / 高唤醒 / 低唤醒 三种情感状态生成, 对比调制效果。
// 用法: bridge_llm_demo.exe [-m <model.gguf>] [-ngl <n_gpu_layers>]
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bridge/emotion_bridge.h"
#include "bridge/emotion_event.h"
#include "llm/llama_backend.h"

using namespace hb;

// -----------------------------------------------------------------------------
// DemoSink — 模拟 SNN 反馈接收端
// 真实引擎接线时, 此实现替换为 SNN setter (set_empathy_signal / set_event_signal /
// set_embodied_reward) 的包装。此处打印事件, 并按 SNN 映射模拟情感反应
// (PAD = 调质 → 映射见 modulatory_kernels.cu), 用于展示闭环。
// -----------------------------------------------------------------------------
class DemoSink final : public bridge::SnnFeedbackSink {
public:
    bridge::EmotionState state = bridge::EmotionState::neutral();  // 模拟 SNN 累积状态

    void emit_empathy(float level) override {
        std::printf("[sink] empathy=%.2f\n", level);
        state.empathy_level = level;
    }

    void emit_event(const float d[6], int duration) override {
        std::printf("[sink] event=[DA %+.2f ACh %+.2f NE %+.2f 5HT %+.2f GABA %+.2f Oxy %+.2f] dur=%d\n",
                    d[0], d[1], d[2], d[3], d[4], d[5], duration);
        // 模拟 SNN 对事件的调质反应 (增量按比例融入浓度)
        state.dopamine       += d[0] * 0.4f;
        state.acetylcholine  += d[1] * 0.4f;
        state.norepinephrine += d[2] * 0.4f;
        state.serotonin      += d[3] * 0.4f;
        state.gaba           += d[4] * 0.4f;
        state.oxytocin       += d[5] * 0.4f;
        // 重算 PAD (与 SNN 的 get_affective_state 映射一致)
        state.pleasure  = state.dopamine - 0.5f * state.serotonin - 0.3f * state.gaba;
        state.arousal   = state.norepinephrine - 0.4f * state.gaba - 0.3f * state.serotonin;
        state.dominance = state.dopamine - 0.5f * state.oxytocin;
    }

    void emit_embodied_reward(float reward) override {
        std::printf("[sink] reward=%.2f\n", reward);
    }
};

int main(int argc, char** argv) {
    std::string model_path = "F:/hb_models/MiniCPM5-1B-Q4_K_M.gguf";
    int ngl = 99;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [-m model.gguf] [-ngl N]\n", argv[0]);
            return 1;
        }
    }

    // ---- 1. 后端: 进程内 llama.cpp ----
    llm::LlamaBackend::Options opt;
    opt.model_path    = model_path;
    opt.n_gpu_layers  = ngl;
    opt.max_new_tokens = 96;  // 1B Q4 小模型易陷入复读循环, 缩短上限
    llm::LlamaBackend backend(opt);
    if (!backend.is_loaded()) {
        std::fprintf(stderr, "error: failed to load model: %s\n", model_path.c_str());
        return 1;
    }
    std::printf("[llm] model loaded: %s (ctx=%d)\n", model_path.c_str(), opt.n_ctx);

    // ---- 2. 桥接层: EmotionBridge 连接后端 ----
    bridge::EmotionBridge bridge;
    bridge.attach_backend(&backend);

    const char* kQuestion = "如果明天是晴天，你有什么想法？";

    auto run_turn = [&](const char* tag, const bridge::EmotionState& state) {
        bridge.set_state(state);
        bridge.apply_to_generation();  // 采样参数 + 情感 prompt 注入

        const auto p = bridge.compute_sampler_params();
        std::printf("\n===== %s =====\n", tag);
        std::printf("[emotion] %s\n", bridge.build_mood_description().c_str());
        std::printf("[sampler] temperature=%.2f top_p=%.2f repeat_penalty=%.2f\n",
                    p.temperature, p.top_p, p.repeat_penalty);

        backend.clear_history();  // 每轮独立, 保证对比公平
        std::string resp;
        const int rc = backend.chat(kQuestion, resp);
        if (rc != 0) {
            std::printf("[error] chat rc=%d\n", rc);
            return;
        }
        std::printf("[response] %s\n", resp.c_str());
    };

    // ---- 3. 三态对比 ----
    run_turn("中性 (neutral)", bridge::EmotionState::neutral());

    bridge::EmotionState hot;
    hot.pleasure          =  0.6f;  // 愉悦
    hot.arousal           =  0.7f;  // 兴奋
    hot.dominance         =  0.5f;  // 主动
    hot.temperature_delta =  0.5f;  // DA 高 → 温度 +0.5
    hot.empathy_level     =  0.2f;
    run_turn("高唤醒 (hot)", hot);

    bridge::EmotionState calm;
    calm.pleasure          = -0.4f;  // 低落
    calm.arousal           = -0.5f;  // 困倦
    calm.dominance         = -0.3f;  // 顺从
    calm.temperature_delta = -0.5f;  // 5HT 高 → 温度 -0.5
    calm.empathy_level     =  0.9f;
    run_turn("低唤醒共情 (calm)", calm);

    // ---- 4. LLM→SNN 语义锚点闭环 ----
    // 用户文本 → 情感事件抽取 → SNN 调质 → 情感状态更新 → 下一轮生成调制
    std::printf("\n===== 语义锚点闭环 (LLM→SNN→LLM) =====\n");
    DemoSink demoSink;
    bridge::EmotionBridge anchorBridge;
    anchorBridge.attach_backend(&backend);
    anchorBridge.attach_snn_feedback(&demoSink);
    bridge::EmotionEventExtractor extractor;
    anchorBridge.attach_emotion_extractor(&extractor);

    const char* kSad = "我今天特别难过，感觉好累";
    std::printf("[user] %s\n", kSad);
    anchorBridge.process_turn("user", kSad);           // ① 文本 → 事件 → SNN

    anchorBridge.set_state(demoSink.state);            // ② 读取 SNN 更新后的情感
    anchorBridge.apply_to_generation();                // ③ 应用到 LLM 生成参数
    const auto ap = anchorBridge.compute_sampler_params();
    std::printf("[emotion] %s\n", anchorBridge.build_mood_description().c_str());
    std::printf("[sampler] temperature=%.2f top_p=%.2f repeat_penalty=%.2f\n",
                ap.temperature, ap.top_p, ap.repeat_penalty);

    backend.clear_history();
    std::string resp2;
    if (backend.chat("我现在感觉怎么样？", resp2) == 0) {
        std::printf("[response] %s\n", resp2.c_str());
    }

    return 0;
}
