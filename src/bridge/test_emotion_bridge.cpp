// =============================================================================
// test_emotion_bridge.cpp — 桥接层单元测试 (纯 C++, 无 CUDA / llama.cpp 依赖)
// =============================================================================
// 覆盖:
//   1. EmotionState::neutral() 默认值
//   2. map_to_sampler_params: 已知 delta → 期望参数 + clamp 边界
//   3. 情感 prompt 生成: mood description / system prompt snippet 内容
//   4. EmotionBridge 编排: MockBackend/MockSink 记录调用 (SNN→LLM 应用 + LLM→SNN 回流)
//   5. 空后端降级: 未 attach 时 apply_to_generation 返回 -1, 回流静默丢弃
// 运行方式: test_emotion_bridge.exe (退出码 0 = 全部通过)
// =============================================================================

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "affective_mapping.h"
#include "emotion_bridge.h"
#include "emotion_event.h"
#include "emotion_prompt.h"
#include "emotion_types.h"

using namespace vita::bridge;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("[FAIL] line %d: %s\n", __LINE__, msg);               \
            ++g_failures;                                                     \
        } else {                                                              \
            std::printf("[PASS] %s\n", msg);                                  \
        }                                                                     \
    } while (0)

bool feq(float a, float b, float eps = 1e-4f) {
    return (a - b) < eps && (b - a) < eps;
}

// -----------------------------------------------------------------------------
// MockBackend — 记录桥接层对 LLM 后端的调用
// -----------------------------------------------------------------------------
class MockBackend final : public LlmBackend {
public:
    int apply_sampler(const SamplerParams& p) override {
        ++sampler_calls;
        last_params = p;
        return return_code;
    }
    int apply_emotion_prompt(const std::string& s) override {
        ++prompt_calls;
        last_snippet = s;
        return return_code;
    }
    void on_user_turn(const std::string& t) override { last_user = t; }
    void on_assistant_turn(const std::string& t) override { last_assistant = t; }

    int return_code = 0;
    int sampler_calls = 0;
    int prompt_calls = 0;
    SamplerParams last_params;
    std::string last_snippet;
    std::string last_user;
    std::string last_assistant;
};

// -----------------------------------------------------------------------------
// MockSink — 记录桥接层对 SNN 的反馈调用
// -----------------------------------------------------------------------------
class MockSink final : public SnnFeedbackSink {
public:
    void emit_empathy(float level) override { empathy_levels.push_back(level); }
    void emit_event(const float d[6], int duration) override {
        std::memcpy(last_event_delta, d, sizeof(last_event_delta));
        last_event_duration = duration;
        ++event_calls;
    }
    void emit_embodied_reward(float reward) override {
        last_reward = reward;
        ++reward_calls;
    }

    std::vector<float> empathy_levels;
    float last_event_delta[6] = {0, 0, 0, 0, 0, 0};
    int   last_event_duration = 0;
    int   event_calls = 0;
    float last_reward = 0.0f;
    int   reward_calls = 0;
};

void test_neutral_state() {
    EmotionState s = EmotionState::neutral();
    CHECK(feq(s.dopamine, 1.0f) && feq(s.serotonin, 1.0f) && feq(s.oxytocin, 1.0f),
          "neutral: 调质基线 1.0");
    CHECK(feq(s.pleasure, 0.0f) && feq(s.arousal, 0.0f) && feq(s.dominance, 0.0f),
          "neutral: PAD 为 0");
    CHECK(feq(s.temperature_delta, 0.0f) && feq(s.empathy_level, 0.0f),
          "neutral: 调制信号为 0");
    CHECK(feq(s.confidence, 1.0f), "neutral: 置信度 1.0");
}

void test_mapping() {
    // 中性状态 → 基线参数
    SamplerParams p = map_to_sampler_params(EmotionState::neutral());
    CHECK(feq(p.temperature, 0.8f), "mapping: 中性温度 = base");
    CHECK(feq(p.top_p, 0.95f), "mapping: 中性 top_p = base");
    CHECK(feq(p.repeat_penalty, 1.1f), "mapping: 中性 repeat_penalty = base");

    // 已知 delta → 线性叠加
    EmotionState s;
    s.temperature_delta = 0.3f;   // DA 高 → 温度升
    s.top_p_delta       = -0.2f;  // NE 高 → top_p 降 (更聚焦)
    s.repetition_delta  = 0.1f;   // NE 高 → 重复惩罚升
    SamplerParams p2 = map_to_sampler_params(s);
    CHECK(feq(p2.temperature, 1.1f), "mapping: temperature = base + delta");
    CHECK(feq(p2.top_p, 0.75f), "mapping: top_p = base + delta (负)");
    CHECK(feq(p2.repeat_penalty, 1.15f), "mapping: repeat_penalty = base + 0.5*delta");

    // clamp: 极端 delta 不越界
    EmotionState hot;
    hot.temperature_delta = 5.0f;
    hot.top_p_delta       = -5.0f;
    hot.repetition_delta  = 5.0f;
    SamplerParams p3 = map_to_sampler_params(hot);
    CHECK(feq(p3.temperature, 2.0f), "mapping: temperature clamp 上界");
    CHECK(feq(p3.top_p, 0.5f), "mapping: top_p clamp 下界");
    CHECK(feq(p3.repeat_penalty, 2.0f), "mapping: repeat_penalty clamp 上界");

    // 自定义 MappingConfig 生效
    MappingConfig cfg;
    cfg.temperature_base = 1.0f;
    cfg.temperature_scale = 0.5f;
    SamplerParams p4 = map_to_sampler_params(s, cfg);
    CHECK(feq(p4.temperature, 1.15f), "mapping: 自定义配置生效");
}

void test_prompt() {
    EmotionState s;
    s.pleasure = 0.5f;
    s.arousal = -0.4f;
    s.dominance = 0.6f;
    s.empathy_level = 0.8f;

    std::string mood = build_mood_description(s);
    CHECK(mood.find("愉悦") != std::string::npos, "prompt: 高愉悦 → 愉悦");
    CHECK(mood.find("困倦") != std::string::npos, "prompt: 低唤醒 → 困倦");

    std::string snippet = build_system_prompt_snippet(s);
    CHECK(snippet.find("当前心境") != std::string::npos, "prompt: snippet 含心境标题");
    CHECK(snippet.find("共情倾向") != std::string::npos, "prompt: snippet 含共情倾向");
}

void test_bridge_orchestration() {
    MockBackend backend;
    MockSink sink;
    EmotionBridge bridge;
    bridge.attach_backend(&backend);
    bridge.attach_snn_feedback(&sink);

    EmotionState s;
    s.temperature_delta = 0.3f;
    s.top_p_delta = -0.1f;
    s.empathy_level = 0.6f;
    bridge.set_state(s);

    CHECK(feq(bridge.state().temperature_delta, 0.3f), "bridge: 状态保存");

    // SNN→LLM 应用
    int rc = bridge.apply_to_generation();
    CHECK(rc == 0, "bridge: apply_to_generation 成功");
    CHECK(backend.sampler_calls == 1, "bridge: 采样参数应用 1 次");
    CHECK(backend.prompt_calls == 1, "bridge: 情感 prompt 注入 1 次");
    CHECK(feq(backend.last_params.temperature, 1.1f), "bridge: 后端收到调制后温度");
    CHECK(backend.last_snippet.find("共情倾向：中") != std::string::npos,
          "bridge: 后端收到情感上下文");

    // LLM→SNN 回流
    bridge.emit_empathy(0.9f);
    CHECK(sink.empathy_levels.size() == 1 && feq(sink.empathy_levels[0], 0.9f),
          "bridge: 共情回流");

    float ev[6] = {0.1f, 0, 0, 0, 0, 0.5f};
    bridge.emit_event(ev, 100);
    CHECK(sink.event_calls == 1 && sink.last_event_duration == 100, "bridge: 事件回流");
    CHECK(feq(sink.last_event_delta[0], 0.1f) && feq(sink.last_event_delta[5], 0.5f),
          "bridge: 事件增量透传");

    bridge.emit_embodied_reward(0.3f);
    CHECK(sink.reward_calls == 1 && feq(sink.last_reward, 0.3f), "bridge: 奖励回流");

    // LLM→SNN 语义锚点 (后端回调透传)
    backend.on_user_turn("我很难过");
    CHECK(backend.last_user == "我很难过", "bridge: 用户回合回调透传");
}

void test_no_backend_degrades() {
    EmotionBridge bridge;  // 未 attach 任何后端/反馈端
    EmotionState s;
    s.temperature_delta = 0.2f;
    bridge.set_state(s);

    // 纯计算仍可用
    SamplerParams p = bridge.compute_sampler_params();
    CHECK(feq(p.temperature, 1.0f), "bridge: 无后端时纯计算可用");

    // 应用返回 -1
    CHECK(bridge.apply_to_generation() == -1, "bridge: 无后端 apply 返回 -1");

    // 回流静默丢弃 (不崩溃)
    bridge.emit_empathy(0.5f);
    float ev[6] = {0, 0, 0, 0, 0, 0};
    bridge.emit_event(ev, 10);
    bridge.emit_embodied_reward(0.0f);
    CHECK(true, "bridge: 无反馈端时回流静默丢弃");
}

void test_emotion_extractor() {
    EmotionEventExtractor ex;

    // 快乐: DA+ / 5HT-
    const EmotionEvent ev1 = ex.extract("我今天很开心！");
    CHECK(ev1.confidence > 0.0f, "extractor: 命中开心");
    CHECK(ev1.modulator_delta[0] > 0.0f, "extractor: 开心 → DA+");
    CHECK(ev1.modulator_delta[3] < 0.0f, "extractor: 开心 → 5HT-");
    CHECK(ev1.duration_steps == 100, "extractor: 事件持续 100 步");

    // 悲伤: DA- / 5HT+
    const EmotionEvent ev2 = ex.extract("我非常难过");
    CHECK(ev2.modulator_delta[0] < 0.0f, "extractor: 难过 → DA-");
    CHECK(ev2.modulator_delta[3] > 0.0f, "extractor: 难过 → 5HT+");
    // 加强副词放大强度 (非常 ×1.5 → NE 增量应显著)
    CHECK(ev2.modulator_delta[2] < -0.2f, "extractor: 强悲伤 → NE- 放大");

    // 强愤怒: NE+ 显著
    const EmotionEvent ev3 = ex.extract("我非常非常生气！！！");
    CHECK(ev3.modulator_delta[2] > 0.5f, "extractor: 强愤怒 → NE+ 大");

    // 中性文本: 无事件
    const EmotionEvent ev0 = ex.extract("今天天气不错，我写了些代码。");
    CHECK(ev0.confidence <= 0.0f, "extractor: 中性文本无事件");

    // clamp: 极端强度不越界
    const EmotionEvent evx = ex.extract("我非常非常非常非常生气！！！！！！");
    CHECK(evx.modulator_delta[2] <= 1.0f, "extractor: NE clamp 上界");

    // 自定义词典扩展 (可复用性)
    EmotionEventExtractor ex2;
    const auto base = ex2.rule_count();
    ex2.add_rule({"代码写完了", {0.3f, 0, 0, 0, 0, 0}, 1.0f});
    const EmotionEvent ev4 = ex2.extract("我的代码写完了");
    CHECK(ev4.modulator_delta[0] > 0.0f, "extractor: 自定义规则生效");
    CHECK(ex2.rule_count() == base + 1, "extractor: 规则计数");
}

void test_bridge_process_turn() {
    MockBackend backend;
    MockSink sink;
    EmotionBridge bridge;
    bridge.attach_backend(&backend);
    bridge.attach_snn_feedback(&sink);
    EmotionEventExtractor ex;
    bridge.attach_emotion_extractor(&ex);

    const int rc = bridge.process_turn("user", "我今天很难过");
    CHECK(rc == 0, "bridge: process_turn 成功");
    CHECK(backend.last_user == "我今天很难过", "bridge: user 回合回调");
    CHECK(sink.event_calls == 1, "bridge: 情感事件注入 SNN");
    CHECK(sink.last_event_delta[0] < 0.0f, "bridge: 事件 DA- 透传");
    CHECK(sink.last_event_duration == 100, "bridge: 事件持续窗口");

    // 无情感词 → 仅回回合回调, 不注入事件
    const int before = sink.event_calls;
    bridge.process_turn("assistant", "好的，我知道了。");
    CHECK(backend.last_assistant == "好的，我知道了。", "bridge: assistant 回合回调");
    CHECK(sink.event_calls == before, "bridge: 中性文本不注入事件");

    // 无抽取器 → 降级返回 -1 (但回合回调仍发生)
    EmotionBridge bare;
    bare.attach_snn_feedback(&sink);
    CHECK(bare.process_turn("user", "我很开心") == -1, "bridge: 无抽取器降级");
}

}  // namespace

int main() {
    test_neutral_state();
    test_mapping();
    test_prompt();
    test_bridge_orchestration();
    test_no_backend_degrades();
    test_emotion_extractor();
    test_bridge_process_turn();

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
