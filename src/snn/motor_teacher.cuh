// src/snn/motor_teacher.cuh
#ifndef SNN_MOTOR_TEACHER_CUH
#define SNN_MOTOR_TEACHER_CUH

#include "memory_allocator.cuh"
#include "embodied_motor.h"

namespace stage2e {

// 行为模仿教师 + 沙盒奖励驱动的 L5→Motor 突触策略学习 (REINFORCE 式)
// 修复 (2026-08-01): 原 D1 版本只记录教师信号, 不修改权重 → 行为恒均匀
//   新实现: 按 post 神经元分组, 沙盒 reward 强化/削弱对应动作组的突触权重:
//     ΔW = η_rl · reward · (I[a==sampled] - p_sampled)   (策略梯度, 生物: DA 奖赏)
//        + η_teach · (I[a==target] - 0.25·I[a!=target])  (基因教师引导)
//   target_action=-1 时仅 reward 项生效
// 必须在 scheduler.step() 之前调用 (沙盒反馈后立即塑形, 无滞后)
void launch_motor_teacher(const PersistentBuffers& buf,
                          int target_action,
                          float teacher_lr,
                          float reward,
                          const MotorReadout& motor);

// 供 main.cpp 查询最近教师信号 (日志用)
int   get_last_teacher_action();
float get_last_teacher_lr();

} // namespace stage2e

#endif // SNN_MOTOR_TEACHER_CUH
