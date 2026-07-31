// =============================================================================
// Stage 2e BPTT (Backpropagation Through Time) 代理梯度训练器实现
// =============================================================================
// 设计要点:
//   1. 前向 (forward): T 步简化 LIF 重放, 保存 V/S 历史用于反向
//      - 不调用 AdEx (主循环已调用), 仅用 LIF 重算保存 BPTT 视角下的历史
//      - CSR 稀疏突触前向: 每线程一神经元, 遍历 row_ptr[i]..row_ptr[i+1]-1
//
//   2. 反向 (backward): 从 t=T-1 到 t=0, 累积 dL/dW
//      - 最终步: dL/dS[T] 由 decode_error 经 W_decode^T 反传得到
//      - 每步: v_grad (V 通道) + s_grad_reset (reset 通道) + dL_dS_via_W (W 通道)
//      - 代理梯度 sigma'(alpha*(V-theta)) 合并 V/S 通道
//      - atomicAdd 用于 dL/dW 累积和 dL/dS_via_W 反向 matvec
//
//   3. 更新 (update): SGD + 全局梯度裁剪
//      - 多 block reduction 计算 ||dL/dW||_2 (atomicAdd 汇总)
//      - scale = min(1.0, grad_clip / norm)
//      - W -= lr * scale * dL/dW (同步 synapses.weight + weights_cache)
//
// 性能考虑:
//   - 10.7M 突触, 每线程一突触, ~41K blocks × 256 threads
//   - atomicAdd 用于跨突触累积 (dL_dW, dL_dS_via_W)
//   - 前向 CSR: 每神经元遍历 ~200 突触 (SYNAPSES_PER_NEURON_2E)
//   - 反向 dL_dS_via_W: 10.7M atomicAdd (可接受, 无需反向 CSR)
// =============================================================================

#include "bptt_trainer.cuh"
#include "bptt_curriculum.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// 文件作用域 device 暂存缓冲: loss 标量 + 梯度范数平方标量
// =============================================================================
// 仿照 decode_kernels.cu 的 d_loss_scratch 模式: 懒分配, 单次分配后复用
// d_loss_scratch: float (loss 标量)
// d_norm_sq_scratch: double (Task F5: 改用 double 累积, 避免 10M 突触平方和溢出 float 上限)
// =============================================================================
namespace {
float*  d_loss_scratch   = nullptr;
double* d_norm_sq_scratch = nullptr;  // Task F5: float -> double

inline void ensure_scratch() {
    if (d_loss_scratch == nullptr) {
        CUDA_CHECK_2E(cudaMalloc(&d_loss_scratch, sizeof(float)));
        CUDA_CHECK_2E(cudaMemset(d_loss_scratch, 0, sizeof(float)));
    }
    if (d_norm_sq_scratch == nullptr) {
        CUDA_CHECK_2E(cudaMalloc(&d_norm_sq_scratch, sizeof(double)));
        CUDA_CHECK_2E(cudaMemset(d_norm_sq_scratch, 0, sizeof(double)));
    }
}
} // anonymous namespace

// =============================================================================
// Kernel 1: LIF 单步前向 (BPTT 视角, 真实 spike)
// =============================================================================
//   V[t+1] = beta * V[t] * (1 - S[t]) + I[t+1]
//   S[t+1] = (V[t+1] >= threshold) ? 1.0f : 0.0f
//
// 说明:
//   - S_prev 是上一步的真实 spike (0 或 1), 作为 reset 项
//   - 发放后 V 被重置: V[t+1] = beta * V[t] * 0 + I = I (S[t]=1 时)
//   - 数值稳定: 检测 NaN/Inf 并归零, 防止梯度爆炸传播
// =============================================================================
__global__ void bptt_forward_step_kernel(
    float* __restrict__ V_next,
    float* __restrict__ S_next,
    const float* __restrict__ V_prev,
    const float* __restrict__ S_prev,
    const float* __restrict__ I_input,
    int N, float beta, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // LIF 单步前向: reset 项 (1 - S[t]) 在发放后清零膜电压
    float v = beta * V_prev[i] * (1.0f - S_prev[i]) + I_input[i];

    // 真实 spike (非平滑), 用于 BPTT 前向重放
    float s = (v >= threshold) ? 1.0f : 0.0f;

    // 数值稳定: 防止 V 漂移到 NaN/Inf 污染后续梯度计算
    if (!isfinite(v)) {
        v = 0.0f;
        s = 0.0f;
    }

    V_next[i] = v;
    S_next[i] = s;
}

// =============================================================================
// Kernel 2: CSR 稀疏突触前向: I[i] = Σ_j W[i,j] * S_prev[j]
// =============================================================================
// CSR 格式:
//   row_ptr[i]..row_ptr[i+1]-1 是神经元 i (突触后) 的所有输入突触索引
//   synapses[s].pre_idx 是突触前神经元, synapses[s].weight 是权重
//
// 计算:
//   I[i] = Σ_{s=row_ptr[i]}^{row_ptr[i+1]-1} synapses[s].weight * S_prev[synapses[s].pre_idx]
//
// 启动: <<<ceil(N/256), 256>>>, 每线程一神经元
// =============================================================================
__global__ void bptt_synapse_forward_csr_kernel(
    float* __restrict__ I_out,
    const BioSynapse* __restrict__ synapses,
    const int* __restrict__ row_ptr,
    const float* __restrict__ S_prev,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 遍历神经元 i 的所有输入突触
    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    float sum = 0.0f;
    for (int s = start; s < end; ++s) {
        int pre = synapses[s].pre_idx;
        sum += synapses[s].weight * S_prev[pre];
    }

    // 数值稳定
    if (!isfinite(sum)) sum = 0.0f;

    I_out[i] = sum;
}

// =============================================================================
// Kernel 3a: 反向 V 通道梯度 + S reset 通道梯度
// =============================================================================
// 输入: dL_dV[i] = dL/dV[t+1, i] (当前步完整梯度, 来自未来或最终步初始化)
//       V_prev[i] = V[t, i], S_prev[i] = S[t, i]
//
// 输出:
//   v_grad[i]       = dL/dV[t+1,i] * beta * (1 - S[t,i])
//     (V[t] 通过 beta*V[t]*(1-S[t]) 项直接影响 V[t+1], 不经过 sigma')
//   s_grad_reset[i] = dL/dV[t+1,i] * (-beta * V[t,i])
//     (S[t] 通过 -beta*V[t]*S[t] reset 项影响 V[t+1], 需经过 sigma')
// =============================================================================
__global__ void bptt_backward_v_grad_kernel(
    float* __restrict__ v_grad,
    float* __restrict__ s_grad_reset,
    const float* __restrict__ dL_dV,
    const float* __restrict__ V_prev,
    const float* __restrict__ S_prev,
    int N, float beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float grad = dL_dV[i];
    // Task F5: 梯度数值稳定 - 对 dL_dV 也加 isfinite 检查
    if (!isfinite(grad)) grad = 0.0f;

    // V 直接通道: dL/dV[t]_via_V = dL/dV[t+1] * ∂V[t+1]/∂V[t]
    //   ∂V[t+1]/∂V[t] = beta * (1 - S[t])  (S[t]=1 时 V 被重置, 梯度为 0)
    v_grad[i] = grad * beta * (1.0f - S_prev[i]);

    // S 经 reset 通道: dL/dS[t]_via_reset = dL/dV[t+1] * ∂V[t+1]/∂S[t]
    //   ∂V[t+1]/∂S[t] = -beta * V[t]  (S[t] 通过 reset 项 -beta*V[t]*S[t] 影响 V[t+1])
    // Task F5: V_prev 钳位到 [-5, 5] 防止 s_grad_reset 爆炸
    //   诊断: BPTT 前向 I = W*S 累积可达 90 (180 突触 × 0.5 权重), V=90 时
    //         s_grad_reset = dL_dV * (-0.9 * 90) = -81 * dL_dV, 50 步反向指数爆炸
    //   钳位到 5.0: s_grad_reset 上限 = 4.5 * dL_dV, 反向累积可控
    //   注: 这是代理梯度训练的标准技巧 (如 Zenke 2021, Neftci 2019)
    float v_clamped = V_prev[i];
    if (v_clamped > 5.0f) v_clamped = 5.0f;
    else if (v_clamped < -5.0f) v_clamped = -5.0f;
    s_grad_reset[i] = grad * (-beta * v_clamped);
}

// =============================================================================
// Kernel 3b: 累积 dL/dW (CSR 稀疏, 每突触一线程)
// =============================================================================
// 数学:
//   I[post] = Σ_s W[s] * S_prev[pre_s]
//   dL/dW[s] = Σ_t dL/dI[t+1, post] * ∂I[t+1,post]/∂W[s]
//            = Σ_t dL/dV[t+1, post] * S_prev[pre_s]   (dI/dV = 1, 因 V[t+1] = ... + I[t+1])
//
// 用 atomicAdd 累积到 dL_dW[s], 因为同一突触在多个时间步累积
// =============================================================================
__global__ void bptt_backward_dW_kernel(
    float* __restrict__ dL_dW,
    const float* __restrict__ dL_dV,
    const BioSynapse* __restrict__ synapses,
    const float* __restrict__ S_prev,
    int n_synapses)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_synapses) return;

    int post = synapses[s].post_idx;
    int pre = synapses[s].pre_idx;

    // dL/dW[s] += dL/dV[t+1, post] * S[t, pre]
    float grad = dL_dV[post] * S_prev[pre];

    // 数值稳定: 跳过 NaN/Inf 梯度, 避免污染累积
    if (isfinite(grad)) {
        atomicAdd(&dL_dW[s], grad);
    }
}

// =============================================================================
// Kernel 4: 计算 dL/dS_via_W (CSR 反向稀疏 matvec, atomicAdd)
// =============================================================================
// 数学:
//   dL/dS[t, j]_via_W = Σ_i dL/dV[t+1, i] * W[i, j]
//                     = Σ_i dL/dV[t+1, i] * (synapses[s].weight for s where post=i, pre=j)
//
// 简化方案: 遍历每个突触, atomicAdd 到 dL_dS_via_W[pre]
//   dL_dS_via_W[pre] += dL_dV[post] * weight
//
// 注意: 调用前必须清零 dL_dS_via_W (用 bptt_zero_array_kernel)
// =============================================================================
__global__ void bptt_compute_grad_S_prev_csr_kernel(
    float* __restrict__ dL_dS_via_W,
    const float* __restrict__ dL_dV,
    const BioSynapse* __restrict__ synapses,
    int n_synapses)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_synapses) return;

    int post = synapses[s].post_idx;
    int pre = synapses[s].pre_idx;

    // dL/dS[t, pre]_via_W += dL/dV[t+1, post] * W[s]
    float grad = dL_dV[post] * synapses[s].weight;

    // 数值稳定
    if (isfinite(grad)) {
        atomicAdd(&dL_dS_via_W[pre], grad);
    }
}

// =============================================================================
// Kernel 5: 合并 V 通道和 S 通道梯度 (代理梯度)
// =============================================================================
// 代理梯度:
//   x = alpha * (V_prev - theta)
//   sigma(x) = 1 / (1 + exp(-x))
//   dS/dV = alpha * sigma(x) * (1 - sigma(x))
//
// 合并:
//   dL_dV_out = v_grad + (dL_dS_via_W + s_grad_reset) * dS/dV
//
// 说明:
//   - v_grad 是 V 直接通道 (不经过 sigma')
//   - (dL_dS_via_W + s_grad_reset) 是 S 通道总梯度 (经过 sigma')
//   - dL_dS_direct 在中间步为 0, 最终步通过 bptt_init_final_grad_kernel 单独处理
// =============================================================================
__global__ void bptt_combine_grad_kernel(
    float* __restrict__ dL_dV_out,
    const float* __restrict__ v_grad,
    const float* __restrict__ dL_dS_via_W,
    const float* __restrict__ s_grad_reset,
    const float* __restrict__ V_prev,
    int N, float threshold, float surrogate_alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 代理梯度: dS/dV = alpha * sigma(x) * (1 - sigma(x))
    float x = surrogate_alpha * (V_prev[i] - threshold);
    float sigma = 1.0f / (1.0f + expf(-x));
    float dS_dV = surrogate_alpha * sigma * (1.0f - sigma);

    // S 通道总梯度 = via_W + via_reset (via_direct 在中间步为 0)
    float dL_dS_total = dL_dS_via_W[i] + s_grad_reset[i];

    // dL/dV[t] = V 直接通道 + S 通道 * 代理梯度
    float result = v_grad[i] + dL_dS_total * dS_dV;

    // 数值稳定: 防止 NaN/Inf 传播
    if (!isfinite(result)) result = 0.0f;

    dL_dV_out[i] = result;
}

// =============================================================================
// Kernel 5b: dL_dV 范数裁剪 (Task F5: 防止 50 步反向累积指数爆炸)
// =============================================================================
// 数学:
//   1. 计算 ||dL_dV||_2 (单 block reduction, N=60K 适合单 block 256 threads)
//   2. 若 norm > clip_thresh, scale = clip_thresh / norm; 否则 scale = 1.0
//   3. dL_dV[i] *= scale
//
// 注意:
//   - 这是"反向传播梯度裁剪", 与 update() 中的"权重梯度裁剪"互补
//   - 反向时每步裁剪 dL_dV, 防止 50 步累积指数爆炸
//   - 前向 V_history 不变, 仅缩放反向梯度
//   - 阈值 dL_dV_CLIP_THRESH = 10.0 (经验值, Neftci 2019 建议 1-100)
//
// 启动: <<<1, 256, 256*sizeof(float)>>>
// =============================================================================
__global__ void bptt_clip_dL_dV_kernel(
    float* __restrict__ dL_dV,
    int N,
    float clip_thresh,
    float* __restrict__ norm_sq_scratch)
{
    extern __shared__ float s_partial[];

    int tid = threadIdx.x;

    // Phase 1: 计算 ||dL_dV||^2 (单 block, 256 threads, 60K 神经元)
    // 每线程累加多个元素 (60K / 256 ≈ 234 元素/线程)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float g = dL_dV[i];
        if (isfinite(g)) {
            local_sum += g * g;
        }
    }
    s_partial[tid] = local_sum;
    __syncthreads();

    // Block 级 tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    float norm_sq = s_partial[0];
    if (tid == 0) {
        atomicExch(norm_sq_scratch, norm_sq);  // 写给 host
    }
    __syncthreads();

    // Phase 2: 广播 scale, 应用裁剪
    // 用 shared memory 广播 scale (避免再次读 device memory)
    __shared__ float s_scale;
    if (tid == 0) {
        float norm = sqrtf(norm_sq);
        if (!isfinite(norm) || norm > clip_thresh) {
            s_scale = (isfinite(norm) && norm > 1e-12f) ? (clip_thresh / norm) : 0.0f;
        } else {
            s_scale = 1.0f;
        }
    }
    __syncthreads();

    float scale = s_scale;
    for (int i = tid; i < N; i += blockDim.x) {
        float g = dL_dV[i];
        if (isfinite(g)) {
            dL_dV[i] = g * scale;
        } else {
            dL_dV[i] = 0.0f;
        }
    }
}

// =============================================================================
// Kernel 6: SGD 更新 (同步 synapses.weight 和 weights_cache)
// =============================================================================
//   W -= lr * scale * dL_dW
//
// scale 由 host 端计算 (梯度裁剪): scale = min(1.0, grad_clip / ||dL_dW||)
// lr 由 host 端计算 (warmup 调度): lr = get_current_lr(current_step)
//
// 同步: 同时更新 d_synapses[s].weight 和 d_weights_cache[s]
//   (d_weights_cache 是 d_synapses.weight 的镜像, 用于快速统计)
// =============================================================================
__global__ void bptt_sgd_update_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ weights_cache,
    const float* __restrict__ dL_dW,
    int n_synapses, float lr, float scale)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_synapses) return;

    float grad = dL_dW[s];
    if (!isfinite(grad)) return;  // 跳过 NaN/Inf 梯度

    // SGD 更新: W -= lr * scale * dL_dW
    float delta = lr * scale * grad;
    float new_weight = synapses[s].weight - delta;

    // 数值稳定: 防止权重漂移到 NaN/Inf
    if (!isfinite(new_weight)) {
        new_weight = synapses[s].weight;  // 保持原值
    }

    synapses[s].weight = new_weight;
    weights_cache[s] = new_weight;
}

// =============================================================================
// Kernel 7: 梯度范数平方计算 (多 block reduction + atomicAdd)
// =============================================================================
// 算法:
//   1. 每个 block 用 shared memory 归约部分平方和 (256 threads tree reduction)
//   2. block 的部分和通过 atomicAdd 累加到 out_norm_sq
//   3. host 端 sqrt 得到 ||dL_dW||_2
//
// 性能:
//   - 10.7M 突触 / 256 = ~41K blocks, 每 block 做一次 atomicAdd
//   - 41K atomicAdd 序列化约 0.4ms, 可接受
//
// 注意: 调用前必须清零 out_norm_sq (host 端 cudaMemset)
// =============================================================================
// Task F5: 改用 double 累积, 避免 10M 突触平方和溢出 float 上限 (3.4e38)
//   单梯度 1e9 平方 = 1e18, 10M 累积 = 1e25 (float 可表示)
//   但单梯度 1e15 平方 = 1e30, 10M 累积 = 1e37 (接近上限)
//   单梯度 1e18 平方 = 1e36, 10M 累积溢出 inf
//   double 上限 1.8e308, 完全足够
__global__ void bptt_grad_norm_sq_kernel(
    const float* __restrict__ dL_dW, int n, double* __restrict__ out_norm_sq)
{
    extern __shared__ double s_partial_d[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 加载并平方 (跳过 NaN/Inf), 用 double 累积
    double val = 0.0;
    if (i < n) {
        float g = dL_dW[i];
        if (isfinite(g)) {
            val = (double)g * (double)g;
        }
    }
    s_partial_d[tid] = val;
    __syncthreads();

    // Block 级 tree reduction (double)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial_d[tid] += s_partial_d[tid + stride];
        }
        __syncthreads();
    }

    // block 的部分和 atomicAdd 到全局输出 (double atomicAdd, sm_60+)
    if (tid == 0) {
        atomicAdd(out_norm_sq, s_partial_d[0]);
    }
}

// =============================================================================
// 辅助 Kernel: 初始化 V[0]/S[0] 从 buf.d_neurons 和 buf.d_spike_flags 复制
// =============================================================================
__global__ void bptt_init_history_kernel(
    float* __restrict__ V_history_0,
    float* __restrict__ S_history_0,
    const NeuronStateAdEx* __restrict__ neurons,
    const bool* __restrict__ spike_flags,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 从主循环状态初始化 BPTT 历史
    V_history_0[i] = neurons[i].membrane_potential;
    S_history_0[i] = spike_flags[i] ? 1.0f : 0.0f;
}

// =============================================================================
// 辅助 Kernel: 解码误差反传到神经元级 dL/dS[T]
// =============================================================================
// 数学:
//   logits[b] = Σ_i W_decode[i*256+b] * S[i]
//   error[b] = p[b] - one_hot(target) = dL/d_logits[b]  (cross-entropy + softmax)
//   dL/dS[i] = Σ_b dL/d_logits[b] * ∂logits[b]/∂S[i]
//            = Σ_b error[b] * W_decode[i*256+b]
//
// 注意符号: error = p - one_hot 正是 dL/d_logits (无需额外取负)
// (与 decode_eligibility_update_kernel 的 credit = -blame 不同, 这里是纯梯度)
// =============================================================================
__global__ void bptt_decode_error_backprop_kernel(
    float* __restrict__ dL_dS_direct,
    const float* __restrict__ decode_weights,
    const float* __restrict__ decode_error,
    int n_neurons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // dL/dS[i] = Σ_b W_decode[i*256+b] * error[b]
    // W_decode 行主序 (N×256), 相邻 b 连续 → warp 内合并访问
    const float* row = decode_weights + (size_t)i * 256;
    float sum = 0.0f;
    #pragma unroll 4
    for (int b = 0; b < 256; ++b) {
        sum += row[b] * decode_error[b];
    }

    // 数值稳定
    if (!isfinite(sum)) sum = 0.0f;

    dL_dS_direct[i] = sum;
}

// =============================================================================
// 辅助 Kernel: 初始化最终步 dL/dV[T] = dL/dS[T] * sigma'(alpha*(V[T]-theta))
// =============================================================================
__global__ void bptt_init_final_grad_kernel(
    float* __restrict__ dL_dV,
    const float* __restrict__ dL_dS_direct,
    const float* __restrict__ V_final,
    int N, float threshold, float surrogate_alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 代理梯度: dS/dV = alpha * sigma(x) * (1 - sigma(x))
    float x = surrogate_alpha * (V_final[i] - threshold);
    float sigma = 1.0f / (1.0f + expf(-x));
    float dS_dV = surrogate_alpha * sigma * (1.0f - sigma);

    // dL/dV[T] = dL/dS[T] * dS/dV[T]
    float result = dL_dS_direct[i] * dS_dV;

    // 数值稳定
    if (!isfinite(result)) result = 0.0f;

    dL_dV[i] = result;
}

// =============================================================================
// 辅助 Kernel: 计算 loss = 0.5 * Σ_b decode_error[b]^2
// =============================================================================
// 单 block 256 threads, tree reduction
// (decode_error 为 256 维, 单 block 足够)
__global__ void bptt_compute_loss_kernel(
    const float* __restrict__ decode_error,
    float* __restrict__ loss_out,
    int n)
{
    __shared__ float s_data[256];
    int tid = threadIdx.x;

    // 加载平方值
    float val = (tid < n) ? decode_error[tid] : 0.0f;
    s_data[tid] = val * val;
    __syncthreads();

    // Tree reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < n) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    // 输出 loss = 0.5 * Σ error^2
    if (tid == 0) {
        *loss_out = 0.5f * s_data[0];
    }
}

// =============================================================================
// 辅助 Kernel: 清零数组
// =============================================================================
__global__ void bptt_zero_array_kernel(float* __restrict__ arr, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    arr[i] = 0.0f;
}

// =============================================================================
// BPTTTrainer 方法实现
// =============================================================================

// -----------------------------------------------------------------------------
// 构造函数: 分配所有 device 缓冲并清零
// -----------------------------------------------------------------------------
BPTTTrainer::BPTTTrainer(const BPTTConfig& config, int n_neurons, int n_synapses)
    : config_(config), n_neurons_(n_neurons), n_synapses_(n_synapses),
      d_V_history_(nullptr), d_S_history_(nullptr),
      d_dL_dV_(nullptr), d_dL_dW_(nullptr),
      d_I_buffer_(nullptr), d_dL_dS_via_W_(nullptr),
      d_v_grad_(nullptr), d_s_grad_reset_(nullptr),
      last_loss_(0.0f), last_grad_norm_(0.0f)
{
    const int T = config_.window_size;
    const int N = n_neurons_;
    const int NS = n_synapses_;

    // V/S history: [(T+1) × N]
    size_t bytes_V = (size_t)(T + 1) * N * sizeof(float);
    CUDA_CHECK_2E(cudaMalloc(&d_V_history_, bytes_V));
    CUDA_CHECK_2E(cudaMemset(d_V_history_, 0, bytes_V));

    size_t bytes_S = (size_t)(T + 1) * N * sizeof(float);
    CUDA_CHECK_2E(cudaMalloc(&d_S_history_, bytes_S));
    CUDA_CHECK_2E(cudaMemset(d_S_history_, 0, bytes_S));

    // 梯度缓冲 (N)
    size_t bytes_N = (size_t)N * sizeof(float);
    CUDA_CHECK_2E(cudaMalloc(&d_dL_dV_,       bytes_N));
    CUDA_CHECK_2E(cudaMalloc(&d_I_buffer_,    bytes_N));
    CUDA_CHECK_2E(cudaMalloc(&d_dL_dS_via_W_, bytes_N));
    CUDA_CHECK_2E(cudaMalloc(&d_v_grad_,      bytes_N));
    CUDA_CHECK_2E(cudaMalloc(&d_s_grad_reset_,bytes_N));
    CUDA_CHECK_2E(cudaMemset(d_dL_dV_,        0, bytes_N));
    CUDA_CHECK_2E(cudaMemset(d_I_buffer_,     0, bytes_N));
    CUDA_CHECK_2E(cudaMemset(d_dL_dS_via_W_,  0, bytes_N));
    CUDA_CHECK_2E(cudaMemset(d_v_grad_,       0, bytes_N));
    CUDA_CHECK_2E(cudaMemset(d_s_grad_reset_, 0, bytes_N));

    // dL/dW: [N_SYNAPSES] (清零, 供 backward 累积)
    size_t bytes_NS = (size_t)NS * sizeof(float);
    CUDA_CHECK_2E(cudaMalloc(&d_dL_dW_, bytes_NS));
    CUDA_CHECK_2E(cudaMemset(d_dL_dW_, 0, bytes_NS));

    // 确保标量暂存缓冲已分配
    ensure_scratch();
}

// -----------------------------------------------------------------------------
// 析构函数: 释放所有 device 缓冲
// -----------------------------------------------------------------------------
BPTTTrainer::~BPTTTrainer()
{
    if (d_V_history_)    cudaFree(d_V_history_);
    if (d_S_history_)    cudaFree(d_S_history_);
    if (d_dL_dV_)        cudaFree(d_dL_dV_);
    if (d_dL_dW_)        cudaFree(d_dL_dW_);
    if (d_I_buffer_)     cudaFree(d_I_buffer_);
    if (d_dL_dS_via_W_)  cudaFree(d_dL_dS_via_W_);
    if (d_v_grad_)       cudaFree(d_v_grad_);
    if (d_s_grad_reset_) cudaFree(d_s_grad_reset_);
    // 注: d_loss_scratch / d_norm_sq_scratch 为文件作用域静态, 生命周期 = 程序, 不在此释放
}

// -----------------------------------------------------------------------------
// forward: T 步 LIF 重放, 保存 V/S 历史
// -----------------------------------------------------------------------------
void BPTTTrainer::forward(PersistentBuffers& buf, int window_start_step)
{
    (void)window_start_step;  // 仅用于日志/调试, 不影响前向计算

    const int T = config_.window_size;
    const int N = n_neurons_;
    const float beta = config_.beta;
    const float threshold = config_.threshold;

    int blocks_N = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    // 1. 初始化 V[0], S[0] 从 buf.d_neurons 和 buf.d_spike_flags 复制
    bptt_init_history_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
        d_V_history_,            // V_history[0 * N]
        d_S_history_,            // S_history[0 * N]
        buf.d_neurons,
        buf.d_spike_flags,
        N);
    CUDA_CHECK_LAST_2E();

    // 2. 循环 t = 0..T-1: I = W * S[t]; V[t+1] = LIF_step(V[t], S[t], I)
    for (int t = 0; t < T; ++t) {
        // 2a. CSR 稀疏突触前向: I = W * S[t]
        bptt_synapse_forward_csr_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_I_buffer_,
            buf.d_synapses,
            buf.d_csr_row_ptr,
            d_S_history_ + (size_t)t * N,   // S[t]
            N);
        CUDA_CHECK_LAST_2E();

        // 2b. LIF 单步前向: V[t+1], S[t+1]
        bptt_forward_step_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_V_history_ + (size_t)(t + 1) * N,   // V[t+1]
            d_S_history_ + (size_t)(t + 1) * N,   // S[t+1]
            d_V_history_ + (size_t)t * N,         // V[t]
            d_S_history_ + (size_t)t * N,         // S[t]
            d_I_buffer_,
            N, beta, threshold);
        CUDA_CHECK_LAST_2E();
    }
    // 注: forward 不修改 buf.d_neurons (主循环状态), 仅写入 d_V_history_/d_S_history_
}

// -----------------------------------------------------------------------------
// backward: 从 t=T-1 到 t=0, 累积 dL/dW
// -----------------------------------------------------------------------------
void BPTTTrainer::backward(PersistentBuffers& buf, uint8_t target_byte)
{
    (void)target_byte;  // target_byte 已在主循环 decode_error 中使用, 这里仅用 d_decode_error

    const int T = config_.window_size;
    const int N = n_neurons_;
    const int NS = n_synapses_;
    const float threshold = config_.threshold;
    const float alpha = config_.surrogate_alpha;
    const float beta = config_.beta;

    int blocks_N  = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    int blocks_NS = (NS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    // 1. 最终步初始化: dL/dV[T] = dL/dS[T] * sigma'(V[T])
    //    dL/dS[T] 由 decode_error 经 W_decode^T 反传得到

    // 1a. 解码误差反传: dL_dS_direct[i] = Σ_b W_decode[i*256+b] * error[b]
    //     (复用 d_v_grad_ 作为临时 dL_dS_direct 缓冲, 避免额外分配)
    float* d_dL_dS_direct = d_v_grad_;
    if (buf.d_decode_weights && buf.d_decode_error) {
        bptt_decode_error_backprop_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_direct,
            buf.d_decode_weights,
            buf.d_decode_error,
            N);
        CUDA_CHECK_LAST_2E();
    } else {
        // 防御: 解码器未初始化, 用零梯度 (loss 无法反传, 但不崩溃)
        bptt_zero_array_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_direct, N);
        CUDA_CHECK_LAST_2E();
    }

    // 1b. 初始化 dL/dV[T] = dL/dS[T] * sigma'(alpha*(V[T]-theta))
    //     V[T] = V_history[T * N]
    bptt_init_final_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
        d_dL_dV_,
        d_dL_dS_direct,
        d_V_history_ + (size_t)T * N,   // V[T]
        N, threshold, alpha);
    CUDA_CHECK_LAST_2E();

    // 2. 反向循环 t = T-1 .. 0
    for (int t = T - 1; t >= 0; --t) {
        const float* V_prev = d_V_history_ + (size_t)t * N;   // V[t]
        const float* S_prev = d_S_history_ + (size_t)t * N;   // S[t]
        // 此时 d_dL_dV_ 存储 dL/dV[t+1]

        // 2a. V 通道梯度 + S reset 通道梯度
        bptt_backward_v_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_v_grad_,           // V 直接通道
            d_s_grad_reset_,     // S 经 reset 通道
            d_dL_dV_,            // dL/dV[t+1]
            V_prev,              // V[t]
            S_prev,              // S[t]
            N, beta);
        CUDA_CHECK_LAST_2E();

        // 2b. 累积 dL/dW: dL_dW[s] += dL/dV[t+1, post] * S[t, pre]
        bptt_backward_dW_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
            d_dL_dW_,
            d_dL_dV_,            // dL/dV[t+1]
            buf.d_synapses,
            S_prev,              // S[t]
            NS);
        CUDA_CHECK_LAST_2E();

        // 2c. 计算 dL/dS_via_W: dL_dS_via_W[pre] = Σ_post dL/dV[t+1, post] * W[s]
        //     先清零 (atomicAdd 累积)
        bptt_zero_array_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_via_W_, N);
        CUDA_CHECK_LAST_2E();

        bptt_compute_grad_S_prev_csr_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_via_W_,
            d_dL_dV_,            // dL/dV[t+1]
            buf.d_synapses,
            NS);
        CUDA_CHECK_LAST_2E();

        // 2d. 合并梯度: dL/dV[t] = v_grad + (dL_dS_via_W + s_grad_reset) * sigma'(V[t])
        //     覆盖 d_dL_dV_ (从 dL/dV[t+1] 变为 dL/dV[t], 供下一轮迭代)
        bptt_combine_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dV_,            // 输出: dL/dV[t] (覆盖 dL/dV[t+1])
            d_v_grad_,
            d_dL_dS_via_W_,
            d_s_grad_reset_,
            V_prev,              // V[t] (用于代理梯度)
            N, threshold, alpha);
        CUDA_CHECK_LAST_2E();

        // Task F5: 每步反向后对 dL_dV 做范数裁剪, 防止 50 步累积指数爆炸
        //   阈值 10.0 (经验值), 单 block 256 threads
        //   复用 d_loss_scratch (float*) 作为 norm_sq 暂存 (此时 loss 已计算完毕, 可覆盖)
        bptt_clip_dL_dV_kernel<<<1, THREADS_PER_BLOCK_2E, THREADS_PER_BLOCK_2E * sizeof(float)>>>(
            d_dL_dV_, N, 10.0f, d_loss_scratch);
        CUDA_CHECK_LAST_2E();
    }
    // 循环结束后 d_dL_dV_ 存储 dL/dV[0] (不再使用), d_dL_dW_ 累积了 T 步的权重梯度

    // 3. 计算 loss: 0.5 * Σ_b decode_error[b]^2
    ensure_scratch();
    if (buf.d_decode_error) {
        bptt_compute_loss_kernel<<<1, 256>>>(
            buf.d_decode_error,
            d_loss_scratch,
            256);
        CUDA_CHECK_LAST_2E();

        // 同步拷贝 loss 到 host
        CUDA_CHECK_2E(cudaMemcpy(&last_loss_, d_loss_scratch,
                                  sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        last_loss_ = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// backward_curriculum: 课程模式反向 (Phase 3a-D3)
//   与 backward() 的唯一区别: 最终步 dL/dS[T] 由调质 readout 误差驱动
//   (替代 decode_error)。反向循环和权重更新完全复用。
// -----------------------------------------------------------------------------
void BPTTTrainer::backward_curriculum(PersistentBuffers& buf)
{
    const int T = config_.window_size;
    const int N = n_neurons_;
    const int NS = n_synapses_;
    const float threshold = config_.threshold;
    const float alpha = config_.surrogate_alpha;
    const float beta = config_.beta;

    int blocks_N  = (N + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    int blocks_NS = (NS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    // 1. 最终步初始化: dL/dV[T] = dL/dS[T] * sigma'(V[T])
    //    dL/dS[T] = Σ_m W_cur[i*6+m] · error[m]  (调质 readout 误差反传)

    // 1a. 调质误差反传: dL_dS_direct[i] = Σ_m W_cur[i*6+m] · error[m]
    //     (复用 d_v_grad_ 作为临时 dL_dS_direct 缓冲, 与 backward() 一致)
    float* d_dL_dS_direct = d_v_grad_;
    if (buf.d_curriculum_readout_weights && buf.d_curriculum_error) {
        launch_curriculum_backprop(buf, d_dL_dS_direct);
    } else {
        // 防御: readout 未初始化, 用零梯度
        bptt_zero_array_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_direct, N);
        CUDA_CHECK_LAST_2E();
    }

    // 1b. 初始化 dL/dV[T] = dL/dS[T] * sigma'(alpha*(V[T]-theta))
    bptt_init_final_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
        d_dL_dV_,
        d_dL_dS_direct,
        d_V_history_ + (size_t)T * N,   // V[T]
        N, threshold, alpha);
    CUDA_CHECK_LAST_2E();

    // 2. 反向循环 t = T-1 .. 0 (与 backward() 完全相同)
    for (int t = T - 1; t >= 0; --t) {
        const float* V_prev = d_V_history_ + (size_t)t * N;   // V[t]
        const float* S_prev = d_S_history_ + (size_t)t * N;   // S[t]

        bptt_backward_v_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_v_grad_, d_s_grad_reset_, d_dL_dV_, V_prev, S_prev, N, beta);
        CUDA_CHECK_LAST_2E();

        bptt_backward_dW_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
            d_dL_dW_, d_dL_dV_, buf.d_synapses, S_prev, NS);
        CUDA_CHECK_LAST_2E();

        bptt_zero_array_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_via_W_, N);
        CUDA_CHECK_LAST_2E();

        bptt_compute_grad_S_prev_csr_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
            d_dL_dS_via_W_, d_dL_dV_, buf.d_synapses, NS);
        CUDA_CHECK_LAST_2E();

        bptt_combine_grad_kernel<<<blocks_N, THREADS_PER_BLOCK_2E>>>(
            d_dL_dV_, d_v_grad_, d_dL_dS_via_W_, d_s_grad_reset_,
            V_prev, N, threshold, alpha);
        CUDA_CHECK_LAST_2E();

        bptt_clip_dL_dV_kernel<<<1, THREADS_PER_BLOCK_2E, THREADS_PER_BLOCK_2E * sizeof(float)>>>(
            d_dL_dV_, N, 10.0f, d_loss_scratch);
        CUDA_CHECK_LAST_2E();
    }

    // 3. 计算 loss: 0.5 * Σ_m curriculum_error[m]^2
    ensure_scratch();
    if (buf.d_curriculum_error) {
        bptt_compute_loss_kernel<<<1, 256>>>(
            buf.d_curriculum_error,
            d_loss_scratch,
            6);
        CUDA_CHECK_LAST_2E();
        CUDA_CHECK_2E(cudaMemcpy(&last_loss_, d_loss_scratch,
                                  sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        last_loss_ = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// update: SGD + 全局梯度裁剪
// -----------------------------------------------------------------------------
void BPTTTrainer::update(PersistentBuffers& buf, int current_step)
{
    const int NS = n_synapses_;
    const float grad_clip = config_.grad_clip;
    const float lr = get_current_lr(current_step);

    int blocks_NS = (NS + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    // Task F5: shared memory 改用 double (与 bptt_grad_norm_sq_kernel 一致)
    size_t smem = THREADS_PER_BLOCK_2E * sizeof(double);

    // 1. 计算 ||dL_dW||^2 (多 block reduction + atomicAdd, double 累积)
    ensure_scratch();
    CUDA_CHECK_2E(cudaMemset(d_norm_sq_scratch, 0, sizeof(double)));

    bptt_grad_norm_sq_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E, smem>>>(
        d_dL_dW_, NS, d_norm_sq_scratch);
    CUDA_CHECK_LAST_2E();

    // 2. host 端: norm = sqrt(norm_sq); scale = min(1.0, grad_clip / norm)
    //    Task F5: 改用 double 读取, 并检测 inf/nan (梯度爆炸时跳过该窗口更新)
    double norm_sq_d = 0.0;
    CUDA_CHECK_2E(cudaMemcpy(&norm_sq_d, d_norm_sq_scratch,
                              sizeof(double), cudaMemcpyDeviceToHost));

    float norm = 0.0f;
    float scale = 0.0f;  // Task F5: 默认 scale=0 (爆炸时跳过更新)

    if (!isfinite(norm_sq_d) || norm_sq_d < 0.0) {
        // Task F5: norm_sq 本身是 inf/nan (不应发生, 因 kernel 已过滤, 但防御性处理)
        // scale=0 → 跳过整个窗口的权重更新, 防止 NaN 污染
        scale = 0.0f;
        norm = INFINITY;
    } else if (norm_sq_d < 1e-24) {
        // 零梯度: 无需更新, scale=0 (delta=0, 实际无变化)
        scale = 0.0f;
        norm = 0.0f;
    } else {
        norm = sqrtf((float)norm_sq_d);
        if (!isfinite(norm) || norm > 1e15f) {
            // Task F5: norm 仍爆炸 (V_prev 钳位未完全生效的兜底)
            // scale=0 → 跳过更新, 避免用极端梯度污染权重
            scale = 0.0f;
        } else {
            scale = (norm > grad_clip) ? (grad_clip / norm) : 1.0f;
        }
    }

    last_grad_norm_ = norm;

    // 3. SGD 更新: W -= lr * scale * dL_dW
    //    同时更新 d_synapses[s].weight 和 d_weights_cache[s]
    //    Task F5: scale=0 时 kernel 仍会启动, 但 delta=0, 实际无变化 (节省条件分支)
    bptt_sgd_update_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
        buf.d_synapses,
        buf.d_weights_cache,
        d_dL_dW_,
        NS, lr, scale);
    CUDA_CHECK_LAST_2E();

    // 4. 清零 d_dL_dW_ 供下一个窗口使用
    bptt_zero_array_kernel<<<blocks_NS, THREADS_PER_BLOCK_2E>>>(
        d_dL_dW_, NS);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// get_current_lr: warmup 学习率调度
// -----------------------------------------------------------------------------
float BPTTTrainer::get_current_lr(int current_step) const
{
    // warmup: lr 线性增长 lr * (step / warmup_steps)
    if (config_.warmup_steps > 0 && current_step < config_.warmup_steps) {
        return config_.lr * (float)current_step / (float)config_.warmup_steps;
    }
    // warmup 后: 固定 lr
    return config_.lr;
}

} // namespace stage2e
