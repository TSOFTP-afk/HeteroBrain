// =============================================================================
// Stage 2e 课程训练 BPTT 调质 + 工具调用监督 kernel (Phase 3a-D3)
// =============================================================================
// 对应 spec: docs/developmental-training-master-spec.md §5.2/§5.3
//
// 核心思想 (知识框架设计):
//   SNN 不存储具体知识, 只学习"情境 → 情感反应 + 工具调用决策"的框架,
//   具体知识由 TF (MiniCPM5-1B + RAG + 黑板) 承担。
//
//   两个 readout 头 (与解码器同构):
//     调质头: pred_mod[m] = Σ_i W_mod[i*6+m] · spike[i]   (m ∈ [0,6))
//     工具头: pred_tool[t] = Σ_i W_tool[i*7+t] · spike[i]  (t ∈ [0,7))
//             t ∈ [0,6) = 6 类工具, t = 6 = 不调用
//
//   课程损失 (spec §5.3 扩展, 初中起训工具):
//     L = w_mod · MSE(pred_mod, target_mod)
//       + w_tool · CE(pred_tool, target_tool)     (softmax over 7)
//
//   反传 (两路误差合并注入 BPTT 最终步梯度):
//     dL/dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                     + Σ_t w_tool·W_tool[i,t]·err_tool[t]
//
//   权重更新: W_mod[i*6+m] -= lr·w_mod·err_mod[m]·spike[i]
//             W_tool[i*7+t] -= lr·w_tool·err_tool[t]·spike[i]
// =============================================================================

#include "bptt_curriculum.cuh"
#include "bptt_trainer.cuh"
#include "memory_allocator.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernel 0: readout 权重初始化 (小随机值, 保证零梯度对称破缺)
//   W_mod[i*6+m]  = (rand-0.5)*2*init_scale
//   W_tool[i*7+t] = (rand-0.5)*2*init_scale
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_init_kernel(
    float* __restrict__ readout_weights,   // [N × 6]
    float* __restrict__ tool_weights,      // [N × 7]
    int N, float init_scale, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState state;
    curand_init(seed, i, 0, &state);

    float* mod_row = readout_weights + (size_t)i * 6;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        mod_row[m] = (curand_uniform(&state) - 0.5f) * 2.0f * init_scale;
    }
    float* tool_row = tool_weights + (size_t)i * CURRICULUM_N_TOOL;
    #pragma unroll
    for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
        tool_row[t] = (curand_uniform(&state) - 0.5f) * 2.0f * init_scale;
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 初始化 readout 权重 (调质 N×6 + 工具 N×7)
// -----------------------------------------------------------------------------
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_init_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights, buf.d_curriculum_tool_weights,
        N_TOTAL_NEURONS_2E, init_scale, seed);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 1a: 调质 readout 前向
//   logits[m] = Σ_i W_mod[i*6+m] · spike[i]   (m ∈ [0,6))
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
// Kernel 1b: 工具 readout 前向
//   tool_logits[t] = Σ_i W_tool[i*7+t] · spike[i]   (t ∈ [0,7))
//   用 7 个 block
// -----------------------------------------------------------------------------
__global__ void curriculum_tool_forward_kernel(
    float* __restrict__ tool_logits,            // [7]
    const float* __restrict__ tool_weights,     // [N × 7]
    const bool* __restrict__ spike_flags,       // [N]
    int N)
{
    const int t = blockIdx.x;                   // 0..6
    if (t >= CURRICULUM_N_TOOL) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_flags[i]) {
            sum += tool_weights[(size_t)i * CURRICULUM_N_TOOL + t];
        }
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) tool_logits[t] = sdata[0];
}

// -----------------------------------------------------------------------------
// Kernel 2: 误差 + 损失计算
//   调质: error_mod[m] = logits[m] - target[m];  L_mod = 0.5·Σ error²
//   工具: softmax 后 CE, error_tool[t] = p[t] - y[t] (dL/dz)
//   总损失: L = w_mod·L_mod + w_tool·L_tool
// -----------------------------------------------------------------------------
__global__ void curriculum_error_kernel(
    float* __restrict__ mod_error,              // [6]
    const float* __restrict__ mod_logits,       // [6]
    const float* __restrict__ target_mod,       // [6]
    float* __restrict__ tool_error,             // [7]
    const float* __restrict__ tool_logits,      // [7]
    int target_tool,                            // 0-6
    float w_mod, float w_tool,
    float* __restrict__ loss_out)               // [1]
{
    float sum = 0.0f;
    // 调质部分: 跨线程
    for (int m = threadIdx.x; m < 6; m += blockDim.x) {
        float e = mod_logits[m] - target_mod[m];
        mod_error[m] = e;
        sum += 0.5f * w_mod * e * e;
    }
    // 工具部分: thread 0 单独算 (softmax 需全 7 个 logits)
    if (threadIdx.x == 0) {
        float maxv = -1e30f;
        for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
            maxv = fmaxf(maxv, tool_logits[t]);
        }
        float denom = 0.0f;
        for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
            denom += expf(tool_logits[t] - maxv);
        }
        float ce = 0.0f;
        for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
            float p = expf(tool_logits[t] - maxv) / denom;
            // dL_tool/dz_t = p_t - y_t (softmax 梯度)
            tool_error[t] = p - ((t == target_tool) ? 1.0f : 0.0f);
            if (t == target_tool) {
                ce = -logf(fmaxf(p, 1e-30f));
            }
        }
        sum += w_tool * ce;
    }
    // block 内归约 (调质部分跨线程 + thread 0 的工具部分)
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
// Kernel 3: 调质 + 工具误差合并反传到神经元 dL/dS_direct
//   dL_dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                   + Σ_t w_tool·W_tool[i,t]·err_tool[t]
// -----------------------------------------------------------------------------
__global__ void curriculum_backprop_kernel(
    float* __restrict__ dL_dS_direct,           // [N]
    const float* __restrict__ readout_weights,  // [N × 6]
    const float* __restrict__ tool_weights,     // [N × 7]
    const float* __restrict__ mod_error,        // [6]
    const float* __restrict__ tool_error,       // [7]
    int N, float w_mod, float w_tool)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    const float* mod_row = readout_weights + (size_t)i * 6;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        sum += w_mod * mod_row[m] * mod_error[m];
    }
    const float* tool_row = tool_weights + (size_t)i * CURRICULUM_N_TOOL;
    #pragma unroll
    for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
        sum += w_tool * tool_row[t] * tool_error[t];
    }
    dL_dS_direct[i] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 4: readout 权重更新 (SGD, 无裁剪, 学习率小)
//   W_mod[i*6+m]  -= lr·w_mod·err_mod[m]·spike[i]
//   W_tool[i*7+t] -= lr·w_tool·err_tool[t]·spike[i]
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_update_kernel(
    float* __restrict__ readout_weights,        // [N × 6]
    float* __restrict__ tool_weights,           // [N × 7]
    const bool* __restrict__ spike_flags,       // [N]
    const float* __restrict__ mod_error,        // [6]
    const float* __restrict__ tool_error,       // [7]
    int N, float lr, float w_mod, float w_tool)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (!spike_flags[i]) return;

    float* mod_row = readout_weights + (size_t)i * 6;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        mod_row[m] -= lr * w_mod * mod_error[m];
    }
    float* tool_row = tool_weights + (size_t)i * CURRICULUM_N_TOOL;
    #pragma unroll
    for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
        tool_row[t] -= lr * w_tool * tool_error[t];
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 课程前向 (spike → 调质 6 维预测 + 工具 7 类注意力)
//   输出到 buf.d_curriculum_logits + d_curriculum_tool_logits
// -----------------------------------------------------------------------------
void launch_curriculum_readout_forward(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_logits || !buf.d_curriculum_tool_logits
        || !buf.d_spike_flags) return;
    curriculum_readout_forward_kernel<<<6, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_logits,
        buf.d_curriculum_readout_weights,
        buf.d_spike_flags,
        N_TOTAL_NEURONS_2E);
    curriculum_tool_forward_kernel<<<CURRICULUM_N_TOOL, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_tool_logits,
        buf.d_curriculum_tool_weights,
        buf.d_spike_flags,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: 课程误差 + 损失 (host 端传入目标调质 + 目标工具)
//   输出: buf.d_curriculum_error + d_curriculum_tool_error + *out_loss
// -----------------------------------------------------------------------------
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             int target_tool,
                             float w_mod, float w_tool,
                             float* out_loss)
{
    if (!buf.d_curriculum_logits || !buf.d_curriculum_tool_logits
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error) {
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
        buf.d_curriculum_tool_error,
        buf.d_curriculum_tool_logits,
        target_tool,
        w_mod, w_tool,
        d_loss);
    CUDA_CHECK_LAST_2E();

    if (out_loss) {
        cudaMemcpy(out_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_target);
    cudaFree(d_loss);
}

// -----------------------------------------------------------------------------
// host wrapper: 调质 + 工具误差合并反传到神经元 dL/dS_direct
//   输出到 direct_grad (由调用方复用, 如 BPTT 的 d_v_grad_ 临时缓冲)
// -----------------------------------------------------------------------------
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct,
                                float w_mod, float w_tool)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_backprop_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        dL_dS_direct,
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_tool_weights,
        buf.d_curriculum_error,
        buf.d_curriculum_tool_error,
        N_TOTAL_NEURONS_2E, w_mod, w_tool);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: readout 权重更新 (调质 + 工具)
// -----------------------------------------------------------------------------
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr,
                                      float w_mod, float w_tool)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error
        || !buf.d_spike_flags) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_tool_weights,
        buf.d_spike_flags,
        buf.d_curriculum_error,
        buf.d_curriculum_tool_error,
        N_TOTAL_NEURONS_2E, lr, w_mod, w_tool);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
