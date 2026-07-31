// =============================================================================
// Stage 2e 课程训练 BPTT 调质监督 kernel (Phase 3a-D3)
// =============================================================================
// 对应 spec: docs/developmental-training-master-spec.md §5.2/§5.3
//
// 核心思想:
//   现有 BPTT 以"字节预测"为监督 (decode_error 反传)。
//   课程训练把监督信号换成"目标调质轨迹 + 目标 PAD"。
//
//   调质 readout 层 (与解码器同构, 但输出 6 维):
//     pred_mod[m] = Σ_i W_cur[i*6+m] · spike[i]      (m ∈ [0,6))
//
//   课程损失 (spec §5.3):
//     L = w_mod · MSE(pred_mod, target_mod)
//       + w_pad · MSE(pred_pad, target_pad)
//       + w_tool · CE(pred_tool, target_tool)        (高中阶段)
//
//   反传 (与 decode_error backprop 同构):
//     dL/dS_direct[i] = Σ_m W_cur[i*6+m] · error[m]
//     该梯度注入 BPTT 最终步 dL/dV[T], 复用现有反向循环。
//
//   实现:
//     - 课程模式下, BPTTTrainer::backward 的最终步初始化改用
//       curriculum 误差 (由 main/scheduler 调用本文件 kernel 计算)
//     - 权重更新: W_cur[i*6+m] -= lr_cur · error[m] · spike[i]
// =============================================================================

#include "bptt_trainer.cuh"
#include "memory_allocator.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernel 0: readout 权重初始化 (小随机值, 保证零梯度对称破缺)
//   W[i*6+m] = (rand()/RAND_MAX - 0.5) * init_scale * 2
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_init_kernel(
    float* __restrict__ readout_weights,
    int N, float init_scale, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState state;
    curand_init(seed, i, 0, &state);
    float* row = readout_weights + (size_t)i * 6;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        row[m] = (curand_uniform(&state) - 0.5f) * 2.0f * init_scale;
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 初始化 readout 权重 (N×6, 小随机值)
// -----------------------------------------------------------------------------
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed)
{
    if (!buf.d_curriculum_readout_weights) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_init_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights, N_TOTAL_NEURONS_2E, init_scale, seed);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 1: 调质 readout 前向
//   logits[m] = Σ_i W_cur[i*6+m] · spike[i]   (m ∈ [0,6))
//   用 6 个 block, 每 block 归约 60K 神经元
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_forward_kernel(
    float* __restrict__ logits,                 // [6]
    const float* __restrict__ readout_weights,  // [N × 6]
    const bool* __restrict__ spike_flags,       // [N]
    int N)
{
    const int m = blockIdx.x;                   // 0..5
    if (m >= 6) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_flags[i]) {
            sum += readout_weights[(size_t)i * 6 + m];
        }
    }
    // block 内归约
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[m] = sdata[0];
}

// -----------------------------------------------------------------------------
// Kernel 2: 误差 + 损失计算
//   error[m] = logits[m] - target[m]
//   loss = 0.5 * Σ_m error[m]²   (MSE)
// -----------------------------------------------------------------------------
__global__ void curriculum_error_kernel(
    float* __restrict__ error,                  // [6]
    const float* __restrict__ logits,           // [6]
    const float* __restrict__ target,           // [6]
    float* __restrict__ loss_out)               // [1]
{
    float sum = 0.0f;
    for (int m = threadIdx.x; m < 6; m += blockDim.x) {
        float e = logits[m] - target[m];
        error[m] = e;
        sum += 0.5f * e * e;
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) loss_out[0] = sdata[0];
}

// -----------------------------------------------------------------------------
// Kernel 3: 调质误差反传到神经元 dL/dS_direct
//   dL_dS_direct[i] = Σ_m W_cur[i*6+m] · error[m]
// -----------------------------------------------------------------------------
__global__ void curriculum_backprop_kernel(
    float* __restrict__ dL_dS_direct,           // [N]
    const float* __restrict__ readout_weights,  // [N × 6]
    const float* __restrict__ error,            // [6]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* row = readout_weights + (size_t)i * 6;
    float sum = 0.0f;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        sum += row[m] * error[m];
    }
    dL_dS_direct[i] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 4: readout 权重更新 (SGD, 无裁剪, 学习率小)
//   W[i*6+m] -= lr_cur · error[m] · spike[i]
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_update_kernel(
    float* __restrict__ readout_weights,        // [N × 6]
    const bool* __restrict__ spike_flags,       // [N]
    const float* __restrict__ error,            // [6]
    int N, float lr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float* row = readout_weights + (size_t)i * 6;
    float s = spike_flags[i] ? 1.0f : 0.0f;
    if (s == 0.0f) return;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        row[m] -= lr * error[m] * s;
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 课程前向 (spike → 6 维调质预测)
//   输出到 buf.d_curriculum_logits
// -----------------------------------------------------------------------------
void launch_curriculum_readout_forward(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_logits
        || !buf.d_spike_flags) return;
    curriculum_readout_forward_kernel<<<6, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_logits,
        buf.d_curriculum_readout_weights,
        buf.d_spike_flags,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: 课程误差 + 损失 (host 端传入目标 6 维调质)
//   输出: buf.d_curriculum_error + *out_loss
// -----------------------------------------------------------------------------
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             float* out_loss)
{
    if (!buf.d_curriculum_logits || !buf.d_curriculum_error) {
        if (out_loss) *out_loss = 0.0f;
        return;
    }
    // 目标拷贝到 device
    float* d_target = nullptr;
    cudaMalloc(&d_target, 6 * sizeof(float));
    cudaMemcpy(d_target, target_mod, 6 * sizeof(float), cudaMemcpyHostToDevice);

    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));

    curriculum_error_kernel<<<1, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_error,
        buf.d_curriculum_logits,
        d_target,
        d_loss);
    CUDA_CHECK_LAST_2E();

    if (out_loss) {
        cudaMemcpy(out_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_target);
    cudaFree(d_loss);
}

// -----------------------------------------------------------------------------
// host wrapper: 调质误差反传到神经元 dL/dS_direct
//   输出到 direct_grad (由调用方复用, 如 BPTT 的 d_v_grad_ 临时缓冲)
// -----------------------------------------------------------------------------
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_error) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_backprop_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        dL_dS_direct,
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_error,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: readout 权重更新
// -----------------------------------------------------------------------------
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_error
        || !buf.d_spike_flags) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights,
        buf.d_spike_flags,
        buf.d_curriculum_error,
        N_TOTAL_NEURONS_2E, lr);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
