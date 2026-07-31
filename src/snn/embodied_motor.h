// src/snn/embodied_motor.h
#ifndef SNN_EMBODIED_MOTOR_H
#define SNN_EMBODIED_MOTOR_H

namespace stage2e {

// 5个动作 (50组运动皮层, 每动作10组)
enum ActionId {
    ACT_CRY  = 0,  // 哭: 触发妈妈响应概率
    ACT_HAND = 1,  // 手部动作: 消耗能量
    ACT_FOOT = 2,  // 脚部动作: 消耗能量
    ACT_SUCK = 3,  // 吸吮: 喂养时 hunger-=0.3*suck
    ACT_GAZE = 4,  // 注视: 无直接效果
    ACT_COUNT = 5
};

struct MotorReadout {
    float action_raw[5];     // 原始发放率 (softmax前)
    float action_prob[5];    // softmax概率
    int   action_sampled;    // 采样的离散动作
    float cry_intensity;     // = action_prob[ACT_CRY]
    float suck_strength;     // = action_prob[ACT_SUCK]
    float limb_movement;     // = (action_prob[ACT_HAND] + action_prob[ACT_FOOT]) / 2
};

// 从 GPU d_motor_spike_flags 读出动作 (5K神经元, 50组×100)
// 调用后 motor 中填充读出结果
MotorReadout read_motor_output(const bool* d_motor_spike_flags);

// 纯 host 版本: 从 host spike_flags 读出 (供单元测试使用)
MotorReadout read_motor_output_host(const bool* h_spike_flags, int n_neurons,
                                     int n_groups, int group_size);

// 内部共享: 通用读出逻辑 (host端, 供 .cpp 和 .cu 调用)
MotorReadout compute_readout(const bool* spike_flags, int n_neurons,
                              int n_groups, int group_size);

} // namespace stage2e

#endif // SNN_EMBODIED_MOTOR_H
