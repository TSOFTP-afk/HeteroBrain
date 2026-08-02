// src/snn/test_embodied.cpp
// 沙盒 v1 单元测试 (纯host, 无CUDA依赖)
// 测试 BodyState演化 + MotorReadout读出 + 行为→奖励映射 + 教师信号
// 2026-08-01 spec §6.1/§6.2: 无 agent, 内稳态向量 + 认知动作
#define SNN_NO_CUDA  // 排除 read_motor_output (GPU版本), 避免链接cudart

#include "embodied_body.h"
#include "embodied_motor.h"
#include "embodied_env.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace stage2e;

static int g_test_pass = 0;
static int g_test_fail = 0;

#define TEST(cond, msg) do { \
    if (cond) { g_test_pass++; } \
    else { g_test_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) TEST(fabsf((a)-(b)) < (eps), msg)

// 测试 1: BodyState 默认初始化
void test_body_init_default() {
    BodyState b;
    b.init_default();
    ASSERT_NEAR(b.hunger, 0.3f, 1e-5f, "default hunger=0.3");
    ASSERT_NEAR(b.temperature, 0.5f, 1e-5f, "default temp=0.5");
    ASSERT_NEAR(b.comfort, 0.7f, 1e-5f, "default comfort=0.7");
    ASSERT_NEAR(b.fatigue, 0.0f, 1e-5f, "default fatigue=0");
    ASSERT_NEAR(b.arousal_value(), 0.51f, 1e-5f, "default arousal (formula) = 0.51");
}

// 测试 2: BodyState 场景初始化
void test_body_init_scene() {
    BodyState b;
    b.init_scene("hunger_feeding");
    ASSERT_NEAR(b.hunger, 0.8f, 1e-5f, "hunger_feeding hunger=0.8");

    b.init_scene("warmth_safety");
    ASSERT_NEAR(b.temperature, 0.2f, 1e-5f, "warmth_safety temp=0.2");

    b.init_scene("discomfort_change");
    ASSERT_NEAR(b.diaper_dirty, 0.9f, 1e-5f, "discomfort_change diaper=0.9");
}

// 测试 3: BodyState 演化 — 饥饿自发累积 (无喂食, 纯压力源)
void test_body_step_hunger() {
    BodyState b;
    b.init_default();
    float h0 = b.hunger;
    b.step(1.0f);
    TEST(b.hunger > h0, "hunger increases after step (spontaneous pressure)");
    TEST(b.hunger <= 1.0f, "hunger clamped to 1.0");
}

// 测试 4: BodyState 演化 — 温度收敛
void test_body_step_temp() {
    BodyState b;
    b.init_default();
    b.temperature = 0.2f;
    b.ambient_temp = 0.5f;
    float t0 = b.temperature;
    b.step(1.0f);
    TEST(b.temperature > t0, "temperature converges toward ambient");
}

// 测试 5: BodyState 内感态编码 (arousal 为计算值)
void test_body_encode_interoception() {
    BodyState b;
    b.init_default();
    float out[15];
    b.encode_interoception(out);
    TEST(out[1] > 0.5f, "hunger=0.3 activates mid");
    TEST(out[2] < 0.5f, "hunger=0.3 does not activate high");

    b.hunger = 0.8f;
    b.encode_interoception(out);
    TEST(out[2] > 0.5f, "hunger=0.8 activates high");
}

// 测试 6: MotorReadout host 读出 (5 认知动作)
void test_motor_readout_host() {
    bool flags[5000];
    std::memset(flags, 0, sizeof(flags));
    MotorReadout m = read_motor_output_host(flags, 5000, 50, 100);
    ASSERT_NEAR(m.action_prob[0], 0.2f, 0.01f, "no spike -> uniform prob");

    // 组0-999 (ACT_CRY 10组) 全部发放
    std::memset(flags, 0, sizeof(flags));
    for (int i = 0; i < 1000; ++i) flags[i] = true;
    m = read_motor_output_host(flags, 5000, 50, 100);
    TEST(m.action_prob[ACT_CRY] > 0.5f, "cry groups spike -> cry prob high");
    TEST(m.cry_intensity > 0.5f, "cry_intensity > 0.5");

    // 组2000-2999 (ACT_APPROACH) 全部发放
    std::memset(flags, 0, sizeof(flags));
    for (int i = 2000; i < 3000; ++i) flags[i] = true;
    m = read_motor_output_host(flags, 5000, 50, 100);
    TEST(m.approach_strength > 0.5f, "approach groups spike -> approach high");
}

// 测试 7: 沙盒 v1 — cry 求助有效性 (无 agent, 概率映射)
void test_env_help_prob() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    // hunger=0.8 → P 高
    float p = env.compute_help_prob();
    TEST(p > 0.6f, "high hunger -> high help prob");

    env.body.hunger = 0.1f;
    p = env.compute_help_prob();
    TEST(p < 0.4f, "low hunger -> lower help prob");
}

// 测试 8: 沙盒 v1 — 教师信号 (基因硬编码, spec §3.1)
void test_env_teacher_signal() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    TEST(env.get_teacher_signal() == ACT_CRY, "hunger>0.6 -> CRY");

    env.body.hunger = 0.2f;
    env.body.temperature = 0.2f;  // 偏离 0.5
    TEST(env.get_teacher_signal() == ACT_APPROACH, "cold -> APPROACH");

    env.body.temperature = 0.5f;
    env.threat_level = 0.8f;
    TEST(env.get_teacher_signal() == ACT_AVOID, "threat -> AVOID");

    env.threat_level = 0.0f;
    env.novelty_level = 0.8f;
    TEST(env.get_teacher_signal() == ACT_INTERACT, "novelty -> INTERACT");

    env.novelty_level = 0.0f;
    TEST(env.get_teacher_signal() == -1, "no teacher when comfortable");
}

// 测试 9: 沙盒 v1 — 感知信号编码 (抽象环境信号)
void test_env_sensory_signals() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    float sig[50];
    env.compute_sensory_signals(sig);
    TEST(sig[0] > 0.4f, "warm_source channel active");
    TEST(sig[37] > 0.5f, "hunger=0.8 -> interoception high");

    env.threat_level = 1.0f;
    env.compute_sensory_signals(sig);
    TEST(sig[10] > 0.5f, "threat channel active");
}

// 测试 10: 沙盒 v1 — 行为→奖励映射 (cry 降低饥饿 → 正奖励)
void test_env_reward() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    BodyState prev = env.body;

    // 模拟饥饿下降 (cry 求助成功的效果)
    env.body.hunger = 0.5f;
    float r = env.compute_reward(prev);
    TEST(r > 0, "hunger decrease -> positive reward");

    // 模拟饥饿上升
    env.body.hunger = 0.9f;
    r = env.compute_reward(prev);
    TEST(r < 0, "hunger increase -> negative reward");
}

// 测试 11: 沙盒 v1 — step_env 闭环 (认知动作, 无 agent)
void test_env_step_closed_loop() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");

    MotorReadout motor = {};
    motor.cry_intensity = 0.9f;      // 大声求助
    motor.approach_strength = 0.0f;
    motor.avoid_strength = 0.0f;
    motor.interact_intensity = 0.0f;

    // 跑 200 环境步, 验证不崩 + hunger 最终下降 (求助有效)
    bool hunger_dropped = false;
    for (int i = 0; i < 200; ++i) {
        BodyState prev = env.body;
        env.step_env(motor);
        env.last_reward = env.compute_reward(prev);
        if (env.body.hunger < 0.5f) hunger_dropped = true;
        TEST(env.body.hunger >= 0.0f && env.body.hunger <= 1.0f, "hunger in range");
        TEST(env.body.comfort >= 0.0f && env.body.comfort <= 1.0f, "comfort in range");
    }
    // 高饥饿 + 强 cry → 求助大概率有效, 200 步内应降到 <0.5
    TEST(hunger_dropped, "hunger drops with strong cry (help mapping works)");
}

int main() {
    srand(42);  // 固定种子, 可复现

    test_body_init_default();
    test_body_init_scene();
    test_body_step_hunger();
    test_body_step_temp();
    test_body_encode_interoception();
    test_motor_readout_host();
    test_env_help_prob();
    test_env_teacher_signal();
    test_env_sensory_signals();
    test_env_reward();
    test_env_step_closed_loop();

    printf("\n=== Embodied Sandbox v1 Tests ===\n");
    printf("PASS: %d\n", g_test_pass);
    printf("FAIL: %d\n", g_test_fail);
    printf("Result: %s\n", g_test_fail == 0 ? "ALL PASS" : "HAS FAILURES");
    return g_test_fail == 0 ? 0 : 1;
}
