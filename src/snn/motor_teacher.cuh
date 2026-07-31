// src/snn/motor_teacher.cuh
#ifndef SNN_MOTOR_TEACHER_CUH
#define SNN_MOTOR_TEACHER_CUH

namespace stage2e {

// 行为模仿教师: 增强目标动作对应组的L5→运动皮层突触权重
// target_action=-1 时无操作
void launch_motor_teacher(
    int target_action,
    float teacher_lr);

} // namespace stage2e

#endif // SNN_MOTOR_TEACHER_CUH
