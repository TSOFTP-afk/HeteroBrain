// src/snn/embodied_motor.h
#ifndef SNN_EMBODIED_MOTOR_H
#define SNN_EMBODIED_MOTOR_H

#include <cstddef>

// 前置声明 (避免引入 BioSynapse 定义, 仅指针)
struct BioSynapse;

namespace stage2e {

// 5个认知动作 (50组运动皮层, 每动作10组)
// 2026-08-01 spec §6.2: 动作空间 = 认知行为最小完备集
//   移除 ACT_SUCK/ACT_HAND/ACT_FOOT (纯生理反射, 无认知学习价值)
//   新增 approach/avoid (价值判断) + interact (对象互动)
enum ActionId {
    ACT_CRY      = 0,  // 求助: 触发"援助有效"概率 → hunger 下降
    ACT_GAZE     = 1,  // 注意选择: 获取注意对象信息 (认知核心能力)
    ACT_APPROACH = 2,  // 趋近: 靠近奖励源 (温暖源/食物/对象)
    ACT_AVOID    = 3,  // 回避: 远离威胁 (安全行为)
    ACT_INTERACT = 4,  // 互动: 与对象互动 (v2 起: 同理心/依恋载体)
    ACT_COUNT    = 5
};

struct MotorReadout {
    float action_raw[5];         // 原始发放率 (softmax前)
    float action_prob[5];        // softmax概率
    int   action_sampled;        // 采样的离散动作
    float cry_intensity;         // = action_prob[ACT_CRY]      求助强度
    float gaze_intensity;        // = action_prob[ACT_GAZE]     注意强度
    float approach_strength;     // = action_prob[ACT_APPROACH] 趋近强度
    float avoid_strength;        // = action_prob[ACT_AVOID]    回避强度
    float interact_intensity;    // = action_prob[ACT_INTERACT] 互动强度
};

// 从 GPU d_motor_spike_flags 读出动作 (5K神经元, 50组×100)
// 调用后 motor 中填充读出结果
MotorReadout read_motor_output(const bool* d_motor_spike_flags);

// 从 GPU L5→Motor 突触权重读出动作 (2026-08-01 新增)
// 沙盒 V1 行为读出直连策略参数 (REINFORCE 标准落地):
//   修复: 原 spike 读出通路脆弱, 策略梯度更新的权重无法可靠传导到行为概率
//   → 直接以 L5→Motor 突触权重聚合为动作偏好 (权重即策略)
MotorReadout read_motor_output_weights(const BioSynapse* d_l5_to_motor_synapses,
                                       const int* d_l5_to_motor_csr_row_ptr);

// 纯 host 版本: 从 host spike_flags 读出 (供单元测试使用)
MotorReadout read_motor_output_host(const bool* h_spike_flags, int n_neurons,
                                     int n_groups, int group_size);

// 内部共享: 通用读出逻辑 (host端, 供 .cpp 和 .cu 调用)
MotorReadout compute_readout(const bool* spike_flags, int n_neurons,
                              int n_groups, int group_size);

// 动作采样 RNG 播种 (2026-08-02):
//   动作采样使用 C 全局 rand(), 原未按 config.seed 播种, 同 seed 多次运行
//   行为轨迹不可复现. 调用方应在初始化时调用 motor_rng_seed(config.seed);
//   未显式调用时采样入口用固定派生种子兜底 (保证可复现, 后续可接 config.seed).
void motor_rng_seed(unsigned int seed);

// 内部共享: 采样前确保 RNG 已播种 (host 端, 供 .cpp/.cu 采样函数调用)
void motor_rng_ensure_seeded();

} // namespace stage2e

#endif // SNN_EMBODIED_MOTOR_H
