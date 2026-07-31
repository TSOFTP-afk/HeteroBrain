// src/snn/test_embodied.cpp
// 具身环境单元测试 (纯host, 无CUDA依赖)
// 测试 BodyState演化 + MotorReadout读出 + Environment响应 + 奖励计算
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
    ASSERT_NEAR(b.arousal, 0.3f, 1e-5f, "default arousal=0.3");
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

// 测试 3: BodyState 演化 — 饥饿增加
void test_body_step_hunger() {
    BodyState b;
    b.init_default();
    b.is_fed = false;
    float h0 = b.hunger;
    b.step(1.0f);
    TEST(b.hunger > h0, "hunger increases after step");
    TEST(b.hunger <= 1.0f, "hunger clamped to 1.0");
}

// 测试 4: BodyState 演化 — 喂养降低饥饿
void test_body_step_feed() {
    BodyState b;
    b.init_default();
    b.hunger = 0.8f;
    b.is_fed = true;
    float h0 = b.hunger;
    b.step(1.0f);
    TEST(b.hunger < h0, "hunger decreases when fed");
    // hunger += 0.001*(1+0.3*0.3) - 0.3 ≈ -0.299
    ASSERT_NEAR(b.hunger, 0.8f + 0.001f*(1.0f+0.3f*0.3f) - 0.3f, 0.01f, "feed amount");
}

// 测试 5: BodyState 内感态编码
void test_body_encode_interoception() {
    BodyState b;
    b.init_default();
    float out[15];
    b.encode_interoception(out);
    // hunger=0.3 → mid 激活 (0.3 >= 0.3 && < 0.7)
    TEST(out[1] > 0.5f, "hunger=0.3 activates mid");
    TEST(out[0] < 0.5f, "hunger=0.3 does not activate low");
    TEST(out[2] < 0.5f, "hunger=0.3 does not activate high");

    b.hunger = 0.8f;
    b.encode_interoception(out);
    TEST(out[2] > 0.5f, "hunger=0.8 activates high");
}

// 测试 6: MotorReadout host 读出
void test_motor_readout_host() {
    // 模拟5000个神经元, 全部不发放
    bool flags[5000];
    std::memset(flags, 0, sizeof(flags));
    MotorReadout m = read_motor_output_host(flags, 5000, 50, 100);
    // 全不发放 → softmax均匀
    ASSERT_NEAR(m.action_prob[0], 0.2f, 0.01f, "no spike -> uniform prob");

    // 让组0-99 (ACT_CRY) 全部发放
    std::memset(flags, 0, sizeof(flags));
    for (int i = 0; i < 1000; ++i) flags[i] = true;
    m = read_motor_output_host(flags, 5000, 50, 100);
    TEST(m.action_prob[ACT_CRY] > 0.5f, "cry groups spike -> cry prob high");
    TEST(m.cry_intensity > 0.5f, "cry_intensity > 0.5");
}

// 测试 7: Environment 妈妈响应概率
void test_env_mom_response_prob() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    // hunger=0.8, arousal≈0.3+0.8*0.4=0.62, mom_fatigue=0
    float p = env.compute_mom_response_prob();
    TEST(p > 0.8f, "high hunger -> high mom response prob");

    env.body.hunger = 0.1f;
    env.body.arousal = 0.2f;
    p = env.compute_mom_response_prob();
    TEST(p < 0.3f, "low hunger -> low mom response prob");
}

// 测试 8: Environment 教师信号
void test_env_teacher_signal() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    // hunger=0.8 > 0.6 → ACT_CRY
    TEST(env.get_teacher_signal() == ACT_CRY, "hunger>0.6 -> CRY");

    env.body.hunger = 0.4f;
    env.body.is_fed = true;
    TEST(env.get_teacher_signal() == ACT_SUCK, "fed+hungry -> SUCK");

    env.body.hunger = 0.1f;
    env.body.is_fed = false;
    env.body.fatigue = 0.8f;
    TEST(env.get_teacher_signal() == ACT_GAZE, "fatigued -> GAZE");

    env.body.fatigue = 0.1f;
    TEST(env.get_teacher_signal() == -1, "no teacher when comfortable");
}

// 测试 9: Environment 感知信号编码
void test_env_sensory_signals() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    float sig[50];
    env.compute_sensory_signals(sig);

    // 妈妈不在 → 触觉柱0-4为0
    TEST(sig[0] < 0.5f, "not held -> touch=0");
    // hunger=0.8 → 内感态柱37 (hunger high) 激活
    TEST(sig[37] > 0.5f, "hunger=0.8 -> interoception high");

    // 模拟妈妈在场
    env.mom_present = true;
    env.body.is_held = true;
    env.mom_speaking = true;
    env.mom_visible = true;
    env.compute_sensory_signals(sig);
    TEST(sig[0] > 0.5f, "held -> touch=1");
    TEST(sig[10] > 0.5f, "mom speaking -> auditory");
    TEST(sig[25] > 0.5f, "mom visible -> face");
}

// 测试 10: Environment 奖励计算
void test_env_reward() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    BodyState prev = env.body;

    // 模拟饥饿下降 (喂养)
    env.body.hunger = 0.5f;  // 从0.8降到0.5
    float r = env.compute_reward(prev);
    TEST(r > 0, "hunger decrease -> positive reward");

    // 模拟饥饿上升
    env.body.hunger = 0.9f;
    r = env.compute_reward(prev);
    TEST(r < 0, "hunger increase -> negative reward");
}

// 测试 11: Environment step_env 闭环
void test_env_step_closed_loop() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");

    MotorReadout motor = {};
    motor.cry_intensity = 0.9f;  // 大声哭
    motor.suck_strength = 0.5f;
    motor.limb_movement = 0.1f;

    // 跑50个环境步, 验证不崩 + 妈妈最终会来
    bool mom_came = false;
    for (int i = 0; i < 50; ++i) {
        BodyState prev = env.body;
        env.step_env(motor);
        env.last_reward = env.compute_reward(prev);
        if (env.mom_present) mom_came = true;
        TEST(env.body.hunger >= 0.0f && env.body.hunger <= 1.0f, "hunger in range");
        TEST(env.body.comfort >= 0.0f && env.body.comfort <= 1.0f, "comfort in range");
    }
    // hunger=0.8 + cry=0.9 → 妈妈应该会在50步内来
    TEST(mom_came, "mom comes within 50 steps when hungry+crying");
}

int main() {
    srand(42);  // 固定种子, 可复现

    test_body_init_default();
    test_body_init_scene();
    test_body_step_hunger();
    test_body_step_feed();
    test_body_encode_interoception();
    test_motor_readout_host();
    test_env_mom_response_prob();
    test_env_teacher_signal();
    test_env_sensory_signals();
    test_env_reward();
    test_env_step_closed_loop();

    printf("\n=== Embodied Environment Tests ===\n");
    printf("PASS: %d\n", g_test_pass);
    printf("FAIL: %d\n", g_test_fail);
    printf("Result: %s\n", g_test_fail == 0 ? "ALL PASS" : "HAS FAILURES");
    return g_test_fail == 0 ? 0 : 1;
}
