// src/snn/motor_teacher.cu
// 行为模仿教师 + 沙盒奖励驱动的 L5→Motor 突触策略学习 (REINFORCE 式)
// 修复 (2026-08-01): 原 D1 版本只记录教师信号不修改权重 (行为恒均匀根因)
//   新实现: 按 post 神经元分组, 沙盒 reward 强化/削弱对应动作组的 L5→Motor 突触权重
#include "motor_teacher.cuh"
#include "config.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace stage2e {

// 策略梯度学习率 (REINFORCE 项)
// 2026-08-01 修复: 0.01 过小, reward~0.05 时每步权重变化 ~0.0005, 3000 步内无法拉开组差
//   提高至 0.3: 单步 ΔW ~0.015, 30 沙盒步累积 ~0.45, 足以在 [0.3,0.8] 权重带上产生分化
#define MOTOR_RL_LR         0.3f
// 权重裁剪范围 (初始权重 [0.3, 0.8], 允许小幅增长/削弱)
#define MOTOR_W_MIN         0.05f
#define MOTOR_W_MAX         2.0f
// 教师引导学习率 (基因硬编码辅助, 行为克隆)
// 2026-08-01: 0.02 偏弱, 与 RL 项 (0.3) 相比引导不明显; 提高至 0.06
#define MOTOR_TEACH_LR      0.06f

// -----------------------------------------------------------------------------
// motor_policy_gradient_kernel: L5→Motor 突触权重策略梯度更新
// 生物对应: DA 奖赏系统调制运动皮层突触可塑性 (基底节-皮层通路)
//   按 post 神经元分组: 每个运动神经元属于唯一动作组 (50组 → 5动作, 每组10组×100神经元)
//   执行动作 a_sampled 概率 p_sampled:
//     ΔW[post∈a_sampled] += η_rl · reward · (1 - p_sampled)   // 强化已选动作
//     ΔW[post∉a_sampled] -= η_rl · reward · p_sampled          // 削弱未选动作
//   基因教师引导 (target_action >= 0):
//     ΔW[post∈target]     += η_teach                           // 目标动作组
//     ΔW[post∉target]     -= η_teach · 0.25                    // 其他组轻微抑制
// -----------------------------------------------------------------------------
__global__ void motor_policy_gradient_kernel(
    BioSynapse* __restrict__ synapses,        // [250K] L5→Motor 突触
    const int* __restrict__ csr_row_ptr,      // [N_MOTOR_NEURONS + 1]
    int n_motor_neurons,                      // 5000
    float reward,                             // 沙盒反馈 (内稳态改善)
    int sampled_action,                       // 采样的离散动作 [0,5)
    float p_sampled,                          // 采样动作的概率 (softmax)
    int target_action,                        // 基因教师目标动作, -1=无
    float rl_lr,                              // 策略梯度学习率
    float teach_lr)                           // 教师引导学习率
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_motor_neurons) return;

    // 动作组映射: 运动神经元 i → 组 g = i/100 (0..49) → 动作 a = g/10 (0..4)
    const int g = i / MOTOR_GROUP_SIZE;
    const int a = g / (N_MOTOR_GROUPS / 5);
    if (a < 0 || a >= 5) return;

    // --- 策略梯度项 (REINFORCE) ---
    float delta = 0.0f;
    if (sampled_action >= 0 && sampled_action < 5) {
        if (a == sampled_action) {
            delta += rl_lr * reward * (1.0f - p_sampled);
        } else {
            delta -= rl_lr * reward * p_sampled;
        }
    }

    // --- 基因教师引导项 (行为克隆, spec §3.1 硬编码允许) ---
    if (target_action >= 0 && target_action < 5) {
        if (a == target_action) {
            delta += teach_lr;
        } else {
            delta -= teach_lr * 0.25f;
        }
    }

    if (delta == 0.0f) return;

    // --- 应用权重更新 (该神经元全部入度 L5→Motor 突触) ---
    const int row_start = csr_row_ptr[i];
    const int row_end   = csr_row_ptr[i + 1];
    for (int k = row_start; k < row_end; ++k) {
        float w = synapses[k].weight + delta;
        if (w < MOTOR_W_MIN) w = MOTOR_W_MIN;
        if (w > MOTOR_W_MAX) w = MOTOR_W_MAX;
        synapses[k].weight = w;
    }
}

// -----------------------------------------------------------------------------
// host launcher
// -----------------------------------------------------------------------------
void launch_motor_teacher(const PersistentBuffers& buf,
                          int target_action,
                          float teacher_lr,
                          float reward,
                          const MotorReadout& motor)
{
    if (!buf.d_l5_to_motor_synapses || !buf.d_l5_to_motor_csr_row_ptr) return;

    // 教师学习率: 调用方传 0 时用默认基因引导强度
    const float teach = (teacher_lr > 0.0f) ? teacher_lr : MOTOR_TEACH_LR;

    motor_policy_gradient_kernel<<<(N_MOTOR_NEURONS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E,
                                   THREADS_PER_BLOCK_2E>>>(
        buf.d_l5_to_motor_synapses,
        buf.d_l5_to_motor_csr_row_ptr,
        N_MOTOR_NEURONS,
        reward,
        motor.action_sampled,
        motor.action_prob[motor.action_sampled],
        target_action,
        MOTOR_RL_LR,
        teach);
    CUDA_CHECK_LAST_2E();
}

int get_last_teacher_action() { return -1; }  // 历史接口保留, 不再有全局状态
float get_last_teacher_lr() { return 0.0f; }

} // namespace stage2e
