// src/snn/embodied_motor.cpp
#include "embodied_motor.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cstdlib>

namespace stage2e {

// ===== 动作采样 RNG 播种 (2026-08-02) =====
// 修复: 动作采样用 C 全局 rand(), 原未播种 (默认 seed=1), 同 seed 多次运行
//   行为轨迹不可复现. 首次采样前统一播种:
//   - 调用方初始化时调用 motor_rng_seed(config.seed) 接入配置种子
//   - 未显式配置时用固定派生种子 0x20260801u 兜底 (固定播种保证可复现, 后续可接 config.seed)
static unsigned int g_motor_rng_seed       = 0x20260801u;
static bool         g_motor_rng_configured = false;

void motor_rng_seed(unsigned int seed) {
    g_motor_rng_seed       = seed;
    g_motor_rng_configured = true;
}

void motor_rng_ensure_seeded() {
    static bool seeded = false;
    if (!seeded) {
        srand(g_motor_rng_configured ? g_motor_rng_seed : 0x20260801u);
        seeded = true;
    }
}

// 通用读出逻辑 (host端)
MotorReadout compute_readout(const bool* spike_flags, int n_neurons,
                                     int n_groups, int group_size) {
    MotorReadout m = {};
    for (int i = 0; i < 5; ++i) m.action_raw[i] = 0.0f;
    for (int i = 0; i < 5; ++i) m.action_prob[i] = 0.0f;
    m.action_sampled = 0;
    m.cry_intensity = 0.0f;
    m.gaze_intensity = 0.0f;
    m.approach_strength = 0.0f;
    m.avoid_strength = 0.0f;
    m.interact_intensity = 0.0f;

    // 1. 按组聚合发放率
    float group_rate[50];
    for (int g = 0; g < n_groups && g < 50; ++g) {
        int count = 0;
        int start = g * group_size;
        int end = std::min(start + group_size, n_neurons);
        for (int i = start; i < end; ++i) {
            if (spike_flags[i]) ++count;
        }
        group_rate[g] = (float)count / (float)group_size;
    }

    // 2. 5个动作各对应10组
    for (int a = 0; a < 5; ++a) {
        float sum = 0;
        for (int g = a * 10; g < (a + 1) * 10; ++g) {
            sum += group_rate[g];
        }
        m.action_raw[a] = sum / 10.0f;
    }

    // 3. Softmax (τ=0.5)
    float max_val = *std::max_element(m.action_raw, m.action_raw + 5);
    float exp_sum = 0;
    for (int a = 0; a < 5; ++a) {
        m.action_prob[a] = expf((m.action_raw[a] - max_val) / 0.5f);
        exp_sum += m.action_prob[a];
    }
    for (int a = 0; a < 5; ++a) m.action_prob[a] /= exp_sum;

    // 4. 采样动作 (简单随机, 用 rand())
    // 2026-08-02: 首次采样前统一播种, 保证同 seed 多次运行行为轨迹可复现
    motor_rng_ensure_seeded();
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    m.action_sampled = 4;  // 默认最后一个
    for (int a = 0; a < 5; ++a) {
        cum += m.action_prob[a];
        if (r < cum) { m.action_sampled = a; break; }
    }

    // 5. 连续值 (spec §6.2: 认知动作, 无纯生理槽位)
    m.cry_intensity      = m.action_prob[ACT_CRY];
    m.gaze_intensity     = m.action_prob[ACT_GAZE];
    m.approach_strength  = m.action_prob[ACT_APPROACH];
    m.avoid_strength     = m.action_prob[ACT_AVOID];
    m.interact_intensity = m.action_prob[ACT_INTERACT];

    return m;
}

MotorReadout read_motor_output_host(const bool* h_spike_flags, int n_neurons,
                                     int n_groups, int group_size) {
    return compute_readout(h_spike_flags, n_neurons, n_groups, group_size);
}

// CUDA 版本 read_motor_output 已移至 embodied_motor.cu (nvcc 编译)

} // namespace stage2e
