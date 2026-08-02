// src/snn/embodied_motor.cu
// CUDA 版本: 从 GPU d_motor_spike_flags 读出动作
#include "embodied_motor.h"
#include "config.h"
#include "types.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdlib>

namespace stage2e {

MotorReadout read_motor_output(const bool* d_motor_spike_flags) {
    // 拷贝到 host
    bool h_spike_flags[5000];
    cudaMemcpy(h_spike_flags, d_motor_spike_flags, 5000 * sizeof(bool),
               cudaMemcpyDeviceToHost);
    // 50组 × 100神经元 = 5000
    return compute_readout(h_spike_flags, 5000, 50, 100);
}

// -----------------------------------------------------------------------------
// read_motor_output_weights: 从 L5→Motor 突触权重直读动作偏好 (2026-08-01)
// 沙盒 V1 行为读出直连策略参数 (REINFORCE 标准落地):
//   - 原 spike 读出: motor 神经元发放 → 组聚合 → softmax
//     问题: L5→Motor 权重更新后, 需经 AdEx 发放阈值才能影响读出,
//     发放稀疏时权重变化无法可靠传导 → 行为恒均匀 (20K 实测)
//   - 新权重读出: action_raw[a] = 动作组 a 的平均入度权重
//     权重即策略参数, 策略梯度更新直接改变行为概率分布
// -----------------------------------------------------------------------------
MotorReadout read_motor_output_weights(const BioSynapse* d_l5_to_motor_synapses,
                                       const int* d_l5_to_motor_csr_row_ptr) {
    MotorReadout m = {};
    const int n_syn = N_MOTOR_NEURONS * L5_TO_MOTOR_SYNAPSES_PER_NEURON;  // 250K
    const int n_motor = N_MOTOR_NEURONS;                                    // 5000

    // D2H 拷贝 (每 100 沙盒步一次, 250K×80B=20MB, 可接受)
    std::vector<BioSynapse> h_syn(n_syn);
    std::vector<int> h_row(n_motor + 1);
    cudaMemcpy(h_syn.data(), d_l5_to_motor_synapses, n_syn * sizeof(BioSynapse),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row.data(), d_l5_to_motor_csr_row_ptr, (n_motor + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // 每个 post 神经元 i: 入度平均权重 → 归入动作组
    // 动作组映射: 神经元 i → 组 g=i/100 (0..49) → 动作 a=g/10 (0..4)
    float raw[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int   cnt[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < n_motor; ++i) {
        int a = (i / MOTOR_GROUP_SIZE) / (N_MOTOR_GROUPS / ACT_COUNT);
        if (a < 0 || a >= ACT_COUNT) continue;
        float sum = 0.0f;
        int   row_start = h_row[i], row_end = h_row[i + 1];
        for (int k = row_start; k < row_end; ++k) sum += h_syn[k].weight;
        int in_deg = row_end - row_start;
        raw[a] += (in_deg > 0) ? (sum / (float)in_deg) : 0.0f;
        cnt[a]++;
    }
    for (int a = 0; a < ACT_COUNT; ++a) {
        m.action_raw[a] = (cnt[a] > 0) ? (raw[a] / (float)cnt[a]) : 0.0f;
    }

    // softmax (τ=0.5, 与 compute_readout 一致)
    float max_val = *std::max_element(m.action_raw, m.action_raw + ACT_COUNT);
    float exp_sum = 0.0f;
    for (int a = 0; a < ACT_COUNT; ++a) {
        m.action_prob[a] = expf((m.action_raw[a] - max_val) / 0.5f);
        exp_sum += m.action_prob[a];
    }
    for (int a = 0; a < ACT_COUNT; ++a) m.action_prob[a] /= exp_sum;

    // 采样动作
    // 2026-08-02: 本函数为 host 代码 (D2H 拷贝后 host 端采样), rand() 为 CRT 全局函数,
    //   经 motor_rng_ensure_seeded() 统一播种 (与 compute_readout 共享同一种子状态,
    //   不会重复重置序列), 保证同 seed 多次运行行为轨迹可复现.
    //   注: CUDA kernel 内 rand() 播种受限 (无低成本方案), kernel 内随机性不可复现;
    //       如需 kernel 内可复现随机需 curand (超出本任务范围).
    motor_rng_ensure_seeded();
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    m.action_sampled = ACT_COUNT - 1;
    for (int a = 0; a < ACT_COUNT; ++a) {
        cum += m.action_prob[a];
        if (r < cum) { m.action_sampled = a; break; }
    }

    // 连续值 (spec §6.2)
    m.cry_intensity      = m.action_prob[ACT_CRY];
    m.gaze_intensity     = m.action_prob[ACT_GAZE];
    m.approach_strength  = m.action_prob[ACT_APPROACH];
    m.avoid_strength     = m.action_prob[ACT_AVOID];
    m.interact_intensity = m.action_prob[ACT_INTERACT];
    return m;
}

} // namespace stage2e
