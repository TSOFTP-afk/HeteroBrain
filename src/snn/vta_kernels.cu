// =============================================================================
// Phase 3a-I (M2): VTA-DA 奖赏预测误差 (RPE) 神经化 — CUDA 实现 (2026-08-06)
// =============================================================================
// 见 vta_kernels.cuh 头文件设计说明。
// 流水线 (每 SNN 步):
//   launch_vta_inject (每步, RPE 为调制窗口级持续状态) → launch_vta_forward (LIF)
// 输出: 正/负组窗口累计发放率 → scheduler 在调制更新时读并算 r_vta
//   (r_vta = pos - neg) → STDP 第三因子 DA 项叠加。
// 无学习权重 → 无 checkpoint section。
// =============================================================================

#include "vta_kernels.cuh"

#include <cuda_runtime.h>
#include <cstring>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// RPE 注入: rpe[2] 强度 → 对应组神经元注入电流
__global__ void vta_inject_kernel(float* __restrict__ input,
                                  const float* __restrict__ rpe,
                                  int group_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_VTA_NEURONS) return;
    int g = i / group_size;
    if (g >= 2) return;
    input[i] += rpe[g] * VTA_INJECT_GAIN;
}

// VTA LIF 更新: 泄漏积分 + 发放判定 + 窗口累计计数
__global__ void vta_update_kernel(float* __restrict__ v,
                                  bool* __restrict__ spike,
                                  float* __restrict__ input,
                                  unsigned int* __restrict__ accum,
                                  int group_size, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int g = i / group_size;
    float vv = v[i] * (1.0f - VTA_LEAK) + input[i];
    input[i] = 0.0f;  // 单步电流
    if (vv >= VTA_THRESHOLD) {
        spike[i] = true;
        v[i] = VTA_RESET;
        atomicAdd(&accum[g], 1u);   // 窗口累计 (读时归一化并清零)
    } else {
        spike[i] = false;
        v[i] = vv;
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

void launch_vta_inject(const float rpe[2], MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_vta_input) return;
    // rpe 拷贝到 device (2 floats, 每次调用开销可忽略)
    float* d_rpe = nullptr;
    if (cudaMalloc(&d_rpe, 2 * sizeof(float)) != cudaSuccess) return;
    if (cudaMemcpy(d_rpe, rpe, 2 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_rpe);
        return;
    }
    vta_inject_kernel<<<(N_VTA_NEURONS + 63) / 64, 64>>>(
        b.d_vta_input, d_rpe, VTA_GROUP_SIZE);
    cudaFree(d_rpe);
}

void launch_vta_forward(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_vta_v) return;
    vta_update_kernel<<<(N_VTA_NEURONS + 63) / 64, 64>>>(
        b.d_vta_v, b.d_vta_spike, b.d_vta_input, b.d_vta_accum,
        VTA_GROUP_SIZE, N_VTA_NEURONS);
}

void read_vta_output(MemoryAllocator* alloc, float out[2], int window_steps,
                     bool reset) {
    if (!out) return;
    out[0] = 0.0f;
    out[1] = 0.0f;
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_vta_accum) return;
    unsigned int h_accum[2] = {0, 0};
    cudaMemcpy(h_accum, b.d_vta_accum, 2 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    if (reset) {
        cudaMemset(b.d_vta_accum, 0, 2 * sizeof(unsigned int));
    }
    const int win = (window_steps > 0) ? window_steps : 1;
    const float norm = 1.0f / (float)(win * VTA_GROUP_SIZE);
    out[0] = (float)h_accum[0] * norm;
    out[1] = (float)h_accum[1] * norm;
}

void init_vta(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_vta_v) return;
    cudaMemset(b.d_vta_v, 0, N_VTA_NEURONS * sizeof(float));
    cudaMemset(b.d_vta_input, 0, N_VTA_NEURONS * sizeof(float));
    cudaMemset(b.d_vta_accum, 0, 2 * sizeof(unsigned int));
}

} // namespace stage2e
