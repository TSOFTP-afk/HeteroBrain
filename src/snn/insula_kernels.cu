// =============================================================================
// Phase 3a-H (M4): 脑岛内感受模块 — CUDA 实现 (2026-08-06 生物拟真 spec)
// =============================================================================
// 见 insula_kernels.cuh 头文件设计说明。
// 流水线 (每 SNN 步):
//   launch_insula_inject (每步, 内感受持续状态) → launch_insula_forward (LIF)
// 输出: 5 组发放率 → scheduler 在调制更新时读并转调质偏置 (不适→NE↑, 舒适→Oxy↑)
// 无学习权重 → 无 checkpoint section。
// =============================================================================

#include "insula_kernels.cuh"

#include <cuda_runtime.h>
#include <cstring>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// 内感受注入: 5 维强度 → 对应组神经元注入电流
__global__ void insula_inject_kernel(float* __restrict__ input,
                                     const float* __restrict__ dims,
                                     int group_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_INSULA_NEURONS) return;
    int g = i / group_size;
    if (g >= 5) return;
    input[i] += dims[g] * INSULA_INJECT_GAIN;
}

// 脑岛 LIF 更新: 泄漏积分 + 发放判定 + 窗口累计计数
__global__ void insula_update_kernel(float* __restrict__ v,
                                     bool* __restrict__ spike,
                                     float* __restrict__ input,
                                     unsigned int* __restrict__ accum,
                                     int group_size, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int g = i / group_size;
    float vv = v[i] * (1.0f - INSULA_LEAK) + input[i];
    input[i] = 0.0f;  // 单步电流
    if (vv >= INSULA_THRESHOLD) {
        spike[i] = true;
        v[i] = INSULA_RESET;
        atomicAdd(&accum[g], 1u);   // 窗口累计 (读时归一化并清零)
    } else {
        spike[i] = false;
        v[i] = vv;
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

void launch_insula_inject(const float dims[5], MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_insula_input) return;
    // dims 拷贝到 device (5 floats, 每次调用开销可忽略)
    float* d_dims = nullptr;
    if (cudaMalloc(&d_dims, 5 * sizeof(float)) != cudaSuccess) return;
    if (cudaMemcpy(d_dims, dims, 5 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_dims);
        return;
    }
    insula_inject_kernel<<<(N_INSULA_NEURONS + 63) / 64, 64>>>(
        b.d_insula_input, d_dims, INSULA_GROUP_SIZE);
    cudaFree(d_dims);
}

void launch_insula_forward(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_insula_v) return;
    insula_update_kernel<<<(N_INSULA_NEURONS + 63) / 64, 64>>>(
        b.d_insula_v, b.d_insula_spike, b.d_insula_input, b.d_insula_accum,
        INSULA_GROUP_SIZE, N_INSULA_NEURONS);
}

void read_insula_output(MemoryAllocator* alloc, float out[5], int window_steps,
                        bool reset) {
    if (!out) return;
    for (int i = 0; i < 5; ++i) out[i] = 0.0f;
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_insula_accum) return;
    unsigned int h_accum[5] = {0, 0, 0, 0, 0};
    cudaMemcpy(h_accum, b.d_insula_accum, 5 * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    if (reset) {
        cudaMemset(b.d_insula_accum, 0, 5 * sizeof(unsigned int));
    }
    const int win = (window_steps > 0) ? window_steps : 1;
    const float norm = 1.0f / (float)(win * INSULA_GROUP_SIZE);
    for (int g = 0; g < 5; ++g) {
        out[g] = (float)h_accum[g] * norm;
    }
}

void init_insula(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_insula_v) return;
    cudaMemset(b.d_insula_v, 0, N_INSULA_NEURONS * sizeof(float));
    cudaMemset(b.d_insula_input, 0, N_INSULA_NEURONS * sizeof(float));
    cudaMemset(b.d_insula_accum, 0, 5 * sizeof(unsigned int));
}

} // namespace stage2e
