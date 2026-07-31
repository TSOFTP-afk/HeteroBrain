#ifndef SNN_STAGE2E_BPTT_TRAINER_CUH
#define SNN_STAGE2E_BPTT_TRAINER_CUH

// =============================================================================
// Stage 2e BPTT (Backpropagation Through Time) 代理梯度训练器
// =============================================================================
// 对应设计: Stage 1 BPTT kernel 移植到 Stage 2e CSR 稀疏格式
//
// 数学:
//   LIF 前向 (BPTT 视角, 简化 LIF, 非主循环 AdEx):
//     I[t+1] = W * S[t]                            (CSR 稀疏突触前向)
//     V[t+1] = beta * V[t] * (1 - S[t]) + I[t+1]   (reset 后衰减)
//     S[t+1] = (V[t+1] >= theta) ? 1 : 0           (真实 spike, 非平滑)
//
//   代理梯度 (反向用):
//     dS/dV = alpha * sigma(x) * (1 - sigma(x)),  x = alpha * (V - theta)
//     sigma(x) = 1 / (1 + exp(-x))
//
//   反向 (t = T-1 -> 0):
//     dL/dW[i,j] += dL/dV[t+1,i] * S[t,j]          (权重梯度累积)
//     dL/dV[t]_via_V = dL/dV[t+1] * beta * (1 - S[t])         (V 直接通道)
//     dL/dS[t,j]_via_W = sum_i dL/dV[t+1,i] * W[i,j]          (S 经 W 通道)
//     dL/dS[t,j]_via_reset = dL/dV[t+1,j] * (-beta * V[t,j])  (S 经 reset 通道)
//     dL/dV[t] = dL/dV[t]_via_V
//              + (dL/dS[t]_via_W + dL/dS[t]_via_reset) * dS/dV
//
//   最终步初始化 (t = T):
//     dL/dS[T,i] = sum_b decode_error[b] * W_decode[i*256+b]   (解码误差反传)
//     dL/dV[T,i] = dL/dS[T,i] * sigma'(alpha*(V[T,i]-theta))
//
//   权重更新 (SGD + 全局梯度裁剪):
//     1. norm = ||dL/dW||_2
//     2. scale = min(1.0, grad_clip / norm)
//     3. W -= lr * scale * dL/dW
//
// 注意:
//   - forward() 是"重放", 用简化 LIF 重算 V/S 历史, 不调用 AdEx
//   - 主循环的真实前向 (AdEx) 已发生, 这里仅保存 BPTT 视角下的历史
//   - CSR 稀疏格式: d_csr_row_ptr[i]..d_csr_row_ptr[i+1]-1 为神经元 i 的输入突触
//   - 权重更新同步到 d_synapses[i].weight 和 d_weights_cache[i]
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include <cuda_runtime.h>

namespace stage2e {

// -----------------------------------------------------------------------------
// BPTT 配置参数
// -----------------------------------------------------------------------------
struct BPTTConfig {
    int   window_size;          // 截断窗口长度 (默认 50)
    float lr;                   // 基础学习率 (默认 0.01)
    float grad_clip;            // 梯度裁剪全局范数 (默认 5.0)
    int   warmup_steps;         // 学习率 warmup 步数 (默认 1000)
    float surrogate_alpha;      // 代理梯度 sigmoid 斜率 (默认 4.0)
    float beta;                 // LIF 膜电压衰减系数 (与 AdEx tau_m 一致, 默认 0.9)
    float threshold;            // 发放阈值 V_norm (默认 1.0, 对应 ADEX_V_THRESH_NORM)

    BPTTConfig()
        : window_size(50), lr(0.01f), grad_clip(5.0f),
          warmup_steps(1000), surrogate_alpha(4.0f),
          beta(0.9f), threshold(1.0f) {}
};

// -----------------------------------------------------------------------------
// BPTT 训练器
// -----------------------------------------------------------------------------
// 缓冲布局 (device):
//   d_V_history_     [(T+1) × N float]   前向保存的膜电压历史, V_history[t*N + i]
//   d_S_history_     [(T+1) × N float]   前向保存的真实 spike 历史, S_history[t*N + i]
//   d_dL_dV_         [N float]           当前步 dL/dV (递推用, 反向时逐步覆盖)
//   d_dL_dW_         [N_SYNAPSES float]  累积的 dL/dW (与 d_synapses 一一对应)
//   d_I_buffer_      [N float]           突触前向电流缓冲
//   d_dL_dS_via_W_   [N float]           S 通道梯度 (via W, 反向 matvec)
//   d_v_grad_        [N float]           V 通道梯度 (直接通道)
//   d_s_grad_reset_  [N float]           S 通道梯度 (via V reset 项)
// -----------------------------------------------------------------------------
class BPTTTrainer {
public:
    // 构造: 分配 V/S history 缓冲和梯度缓冲
    //   T = config.window_size
    //   N = n_neurons (通常 = N_TOTAL_NEURONS_2E)
    //   分配所有上述 device 缓冲并清零
    BPTTTrainer(const BPTTConfig& config, int n_neurons, int n_synapses);
    ~BPTTTrainer();

    // 前向一个窗口: T 步 LIF 重放
    //   - 使用真实 spike (非平滑), 保存 V/S 历史
    //   - 不修改主循环的 d_neurons 状态 (用 history 副本计算)
    //   - 从 buf.d_neurons.membrane_potential 和 buf.d_spike_flags 初始化 V[0]/S[0]
    //   - 循环 t = 0..T-1: I = W * S[t]; V[t+1] = beta*V[t]*(1-S[t]) + I; S[t+1] = (V[t+1]>=theta)
    //   window_start_step 仅用于日志/调试, 不影响前向计算
    void forward(PersistentBuffers& buf, int window_start_step);

    // 反向一个窗口: 从 t=T-1 到 t=0
    //   - 用 sigmoid 代理梯度 sigma'(alpha*(V-theta))
    //   - 累积 dL/dW 到 d_dL_dW_
    //   - target_byte 用于最终步 loss 计算 (cross-entropy via 解码器)
    //   - 最终步初始化: dL/dS[T] 由 buf.d_decode_error 经 W_decode^T 反传得到
    void backward(PersistentBuffers& buf, uint8_t target_byte);

    // 课程模式反向 (Phase 3a-D3): 最终步梯度由调质 readout 误差驱动
    //   - 前置: 调用方已通过 launch_curriculum_readout_forward + launch_curriculum_error
    //           计算出 buf.d_curriculum_error
    //   - 最终步初始化: dL/dS[T] = Σ_m W_cur[i*6+m]·error[m] (替代 decode error)
    //   - 反向循环与 backward() 相同
    //   - loss 记录到 last_loss_ (调质 MSE)
    void backward_curriculum(PersistentBuffers& buf, float w_mod, float w_tool);

    // 应用权重更新: SGD + 全局梯度裁剪
    //   1. 计算 ||dL_dW|| 全局范数 (多 block reduction + atomicAdd)
    //   2. host 端: scale = min(1.0, grad_clip / norm); norm=0 时 scale=1.0 (无更新)
    //   3. W -= lr * scale * dL_dW (lr 含 warmup 调度)
    //   4. 同步到 d_synapses[i].weight 和 d_weights_cache[i]
    //   5. 清零 d_dL_dW_ 供下一个窗口使用
    //   current_step 用于 warmup 学习率调度
    void update(PersistentBuffers& buf, int current_step);

    // 获取最近一次 backward 的 loss
    float get_last_loss() const { return last_loss_; }

    // 获取最近一次 update 的梯度范数
    float get_last_grad_norm() const { return last_grad_norm_; }

    // 获取当前有效学习率 (含 warmup)
    //   warmup 期: lr * (current_step / warmup_steps)
    //   warmup 后: lr
    float get_current_lr(int current_step) const;

    const BPTTConfig& config() const { return config_; }

private:
    BPTTConfig config_;
    int n_neurons_;
    int n_synapses_;

    // device 缓冲
    float* d_V_history_;        // [(T+1) × N]
    float* d_S_history_;        // [(T+1) × N]
    float* d_dL_dV_;            // [N]
    float* d_dL_dW_;            // [N_SYNAPSES]
    float* d_I_buffer_;         // [N]
    float* d_dL_dS_via_W_;      // [N]
    float* d_v_grad_;           // [N]
    float* d_s_grad_reset_;     // [N]

    float last_loss_;
    float last_grad_norm_;
};

// =============================================================================
// CUDA kernel 声明
// =============================================================================

// Kernel 1: LIF 单步前向 (BPTT 视角, 真实 spike)
//   V[t+1] = beta * V[t] * (1 - S[t]) + I[t+1]
//   S[t+1] = (V[t+1] >= threshold) ? 1.0f : 0.0f
// 启动配置: <<<ceil(N/256), 256>>>
__global__ void bptt_forward_step_kernel(
    float* __restrict__ V_next,
    float* __restrict__ S_next,
    const float* __restrict__ V_prev,
    const float* __restrict__ S_prev,
    const float* __restrict__ I_input,
    int N, float beta, float threshold);

// Kernel 2: CSR 稀疏突触前向: I[i] = Σ_j W[i,j] * S_prev[j]
//   遍历神经元 i 的所有输入突触 (row_ptr[i]..row_ptr[i+1]-1)
//   I[i] = Σ_s synapses[s].weight * S_prev[synapses[s].pre_idx]
// 启动配置: <<<ceil(N/256), 256>>>
__global__ void bptt_synapse_forward_csr_kernel(
    float* __restrict__ I_out,
    const BioSynapse* __restrict__ synapses,
    const int* __restrict__ row_ptr,
    const float* __restrict__ S_prev,
    int N);

// Kernel 3a: 反向 V 通道梯度 + S reset 通道梯度
//   v_grad[i]       = dL_dV[i] * beta * (1 - S_prev[i])     (V 直接通道)
//   s_grad_reset[i] = dL_dV[i] * (-beta * V_prev[i])        (S 经 V reset 项)
// 启动配置: <<<ceil(N/256), 256>>>
__global__ void bptt_backward_v_grad_kernel(
    float* __restrict__ v_grad,
    float* __restrict__ s_grad_reset,
    const float* __restrict__ dL_dV,
    const float* __restrict__ V_prev,
    const float* __restrict__ S_prev,
    int N, float beta);

// Kernel 3b: 累积 dL/dW (CSR 稀疏, 每突触一线程)
//   dL_dW[s] += dL_dV[post_idx] * S_prev[pre_idx]
//   (因为 I[post] = Σ W[s] * S_prev[pre], 所以 dI[post]/dW[s] = S_prev[pre])
// 启动配置: <<<ceil(N_SYN/256), 256>>>
__global__ void bptt_backward_dW_kernel(
    float* __restrict__ dL_dW,
    const float* __restrict__ dL_dV,
    const BioSynapse* __restrict__ synapses,
    const float* __restrict__ S_prev,
    int n_synapses);

// Kernel 4: 计算 dL/dS_via_W (CSR 反向稀疏 matvec, atomicAdd)
//   dL_dS_via_W[pre] += dL_dV[post] * W[s]
//   (dL/dS[t,j]_via_W = Σ_i dL/dV[t+1,i] * W[i,j])
//   用 atomicAdd 避免构建反向 CSR, 每突触一线程
// 启动配置: <<<ceil(N_SYN/256), 256>>>
__global__ void bptt_compute_grad_S_prev_csr_kernel(
    float* __restrict__ dL_dS_via_W,
    const float* __restrict__ dL_dV,
    const BioSynapse* __restrict__ synapses,
    int n_synapses);

// Kernel 5: 合并 V 通道和 S 通道梯度 (代理梯度)
//   dS/dV = alpha * sigma(x) * (1 - sigma(x)),  x = alpha * (V_prev - theta)
//   dL_dV_out = v_grad + (dL_dS_via_W + s_grad_reset) * dS/dV
//   (dL_dS_direct 在中间步为 0, 最终步通过 init_final_grad 单独处理)
// 启动配置: <<<ceil(N/256), 256>>>
__global__ void bptt_combine_grad_kernel(
    float* __restrict__ dL_dV_out,
    const float* __restrict__ v_grad,
    const float* __restrict__ dL_dS_via_W,
    const float* __restrict__ s_grad_reset,
    const float* __restrict__ V_prev,
    int N, float threshold, float surrogate_alpha);

// Kernel 5b: dL_dV 范数裁剪 (Task F5: 防止 50 步反向累积指数爆炸)
//   每步反向后裁剪 dL_dV 到 ||dL_dV|| <= clip_thresh, 防止梯度指数增长
// 启动配置: <<<1, 256, 256*sizeof(float)>>> (单 block, 60K 神经元)
__global__ void bptt_clip_dL_dV_kernel(
    float* __restrict__ dL_dV,
    int N,
    float clip_thresh,
    float* __restrict__ norm_sq_scratch);

// Kernel 6: SGD 更新 (同步 synapses.weight 和 weights_cache)
//   W -= lr * scale * dL_dW
// 启动配置: <<<ceil(N_SYN/256), 256>>>
__global__ void bptt_sgd_update_kernel(
    BioSynapse* __restrict__ synapses,
    float* __restrict__ weights_cache,
    const float* __restrict__ dL_dW,
    int n_synapses, float lr, float scale);

// Kernel 7: 梯度范数平方计算 (多 block reduction + atomicAdd)
//   每个 block 用 shared memory 归约部分平方和, 再 atomicAdd 到 out_norm_sq
//   host 端 sqrt 得到 ||dL_dW||_2
// Task F5: 改用 double 累积 (out_norm_sq + shared memory), 避免 10M 突触平方和溢出
// 启动配置: <<<ceil(N/256), 256, 256*sizeof(double)>>>
__global__ void bptt_grad_norm_sq_kernel(
    const float* __restrict__ dL_dW, int n, double* __restrict__ out_norm_sq);

// ---- 辅助 kernel ----

// 初始化 V[0]/S[0] 从 buf.d_neurons 和 buf.d_spike_flags 复制
__global__ void bptt_init_history_kernel(
    float* __restrict__ V_history_0,
    float* __restrict__ S_history_0,
    const NeuronStateAdEx* __restrict__ neurons,
    const bool* __restrict__ spike_flags,
    int N);

// 解码误差反传到神经元级 dL/dS[T]
//   dL_dS_direct[i] = Σ_b decode_error[b] * W_decode[i*256+b]
__global__ void bptt_decode_error_backprop_kernel(
    float* __restrict__ dL_dS_direct,
    const float* __restrict__ decode_weights,
    const float* __restrict__ decode_error,
    int n_neurons);

// 初始化最终步 dL/dV[T] = dL/dS[T] * sigma'(alpha*(V[T]-theta))
__global__ void bptt_init_final_grad_kernel(
    float* __restrict__ dL_dV,
    const float* __restrict__ dL_dS_direct,
    const float* __restrict__ V_final,
    int N, float threshold, float surrogate_alpha);

// 计算 loss = 0.5 * Σ_b decode_error[b]^2 (单 block 256 threads)
__global__ void bptt_compute_loss_kernel(
    const float* __restrict__ decode_error,
    float* __restrict__ loss_out,
    int n);

// 清零数组
__global__ void bptt_zero_array_kernel(float* __restrict__ arr, int N);

} // namespace stage2e

#endif // SNN_STAGE2E_BPTT_TRAINER_CUH
