// src/snn/motor_teacher.cu
#include "motor_teacher.cuh"
#include "config.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace stage2e {

// D1阶段: 简化版, 不实际修改突触权重 (避免需要访问scheduler内部状态)
// 只记录教师信号, 让现有STDP机制自然学习
// 真正的motor_teacher_kernel留待D2阶段实现完整版

static int g_last_teacher_action = -1;
static float g_last_teacher_lr = 0.0f;

void launch_motor_teacher(
    int target_action,
    float teacher_lr)
{
    // D1: 记录教师信号 (供main.cpp日志读取)
    g_last_teacher_action = target_action;
    g_last_teacher_lr = teacher_lr;
}

// 供main.cpp查询最近教师信号
int get_last_teacher_action() { return g_last_teacher_action; }
float get_last_teacher_lr() { return g_last_teacher_lr; }

} // namespace stage2e
