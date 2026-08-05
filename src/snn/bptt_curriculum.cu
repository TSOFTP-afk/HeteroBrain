// =============================================================================
// Stage 2e 课程训练 BPTT 调质 + 工具调用 + PAD 情感监督 kernel (Phase 3a-D3)
// =============================================================================
// 对应 spec: docs/developmental-training-master-spec.md §5.2/§5.3
//
// 核心思想 (知识框架设计):
//   SNN 不存储具体知识, 只学习"情境 → 情感反应 + 工具调用决策"的框架,
//   具体知识由 TF (MiniCPM5-1B + RAG + 黑板) 承担。
//
//   三个 readout 头 (与解码器同构):
//     调质头: pred_mod[m] = Σ_i W_mod[i*6+m] · spike[i]   (m ∈ [0,6))
//             ⚠️ m 为 GENE_MAP 顺序 [DA, ACh, NE, 5HT, GABA, Oxy]
//             (通道契约见 bptt_curriculum.cuh, 索引常量 MOD_CH_* 见 mod_simulator.h;
//              勿按 personality 顺序误读)
//     工具头: pred_tool[t] = Σ_i W_tool[i*7+t] · spike[i]  (t ∈ [0,7))
//             t ∈ [0,6) = 6 类工具, t = 6 = 不调用
//     PAD 头:  pred_pad[p] = Σ_i W_pad[i*3+p] · spike[i]   (p ∈ [0,3))
//             [Pleasure, Arousal, Dominance]  (2026-08-02 Task 5)
//
//   课程损失 (spec §5.3 扩展, 初中起训工具 + PAD):
//     L = w_mod · MSE(pred_mod, target_mod)
//       + w_pad · MSE(pred_pad, target_pad)
//       + w_tool · CE(pred_tool, target_tool)     (softmax over 7, 2026-08-03 起类别加权)
//
//   反传 (三路误差合并注入 BPTT 最终步梯度 / N3F eligibility):
//     dL/dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                     + Σ_p w_pad·W_pad[i,p]·err_pad[p]
//                     + Σ_t w_tool·W_tool[i,t]·err_tool[t]
//
//   权重更新: W_mod[i*6+m]  -= lr·w_mod·err_mod[m]·spike[i]
//             W_tool[i*7+t] -= lr·w_tool·err_tool[t]·spike[i]
//             W_pad[i*3+p]  -= lr·w_pad·err_pad[p]·spike[i]
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
//   W_pad[i*3+p]  = (rand-0.5)*2*init_scale   (2026-08-02 Task 5)
//   W_conc[m*6+j] = (rand-0.5)*2*init_scale   (2026-08-04 方案2: 浓度头, 6×6)
//     浓度头由 i==0 的线程初始化 (仅 36 个权重, 独立于神经元)
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_init_kernel(
    float* __restrict__ readout_weights,   // [N × 6]
    float* __restrict__ tool_weights,      // [N × 7]
    float* __restrict__ pad_weights,       // [N × 3]
    float* __restrict__ conc_weights,      // [6 × 6] 2026-08-04 方案2 (可空)
    int N, float init_scale, unsigned long long seed)
{
    // 浓度头 (与神经元无关, 由 block0 thread0 初始化)
    if (conc_weights && blockIdx.x == 0 && threadIdx.x == 0) {
        curandState cstate;
        curand_init(seed ^ 0xC0A11Cu, 1, 0, &cstate);
        #pragma unroll
        for (int k = 0; k < 6 * 6; ++k) {
            conc_weights[k] = (curand_uniform(&cstate) - 0.5f) * 2.0f * init_scale;
        }
    }
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
    // PAD 头 (缓冲缺失时跳过, 保持与 mod/tool 同 seed 序列)
    if (pad_weights) {
        float* pad_row = pad_weights + (size_t)i * 3;
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            pad_row[p] = (curand_uniform(&state) - 0.5f) * 2.0f * init_scale;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 0b: PAD readout 权重专用初始化 (仅旧 checkpoint 缺失 PAD 节时调用)
//   只初始化 W_pad, 不触碰 mod/tool 权重 — 避免重随机化已加载的 readout 头
// -----------------------------------------------------------------------------
__global__ void curriculum_pad_readout_init_kernel(
    float* __restrict__ pad_weights,       // [N × 3]
    int N, float init_scale, unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curandState state;
    curand_init(seed, i, 0, &state);

    float* pad_row = pad_weights + (size_t)i * 3;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        pad_row[p] = (curand_uniform(&state) - 0.5f) * 2.0f * init_scale;
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 初始化 readout 权重 (调质 N×6 + 工具 N×7 + PAD N×3)
// -----------------------------------------------------------------------------
void launch_curriculum_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_init_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights, buf.d_curriculum_tool_weights,
        buf.d_curriculum_pad_weights,
        buf.d_curriculum_conc_weights,          // 2026-08-04 方案2: 浓度头 (可空)
        N_TOTAL_NEURONS_2E, init_scale, seed);
    CUDA_CHECK_LAST_2E();
}

void launch_curriculum_pad_readout_init(PersistentBuffers& buf, float init_scale, unsigned long long seed)
{
    if (!buf.d_curriculum_pad_weights) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_pad_readout_init_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_pad_weights,
        N_TOTAL_NEURONS_2E, init_scale, seed);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 0c: 浓度 readout 头专用初始化 (2026-08-04 方案2)
//   只初始化 6×6 浓度头权重, 不触碰 mod/tool/pad 头 (避免重随机化已加载权重)
// -----------------------------------------------------------------------------
__global__ void curriculum_conc_readout_init_kernel(
    float* __restrict__ conc_weights,           // [6 × 6]
    float init_scale, unsigned long long seed)
{
    if (threadIdx.x != 0) return;
    curandState cstate;
    curand_init(seed ^ 0xC0A11Cu, 1, 0, &cstate);
    #pragma unroll
    for (int k = 0; k < 6 * 6; ++k) {
        conc_weights[k] = (curand_uniform(&cstate) - 0.5f) * 2.0f * init_scale;
    }
}

void launch_curriculum_conc_readout_init(PersistentBuffers& buf, float init_scale,
                                         unsigned long long seed)
{
    if (!buf.d_curriculum_conc_weights) return;
    curriculum_conc_readout_init_kernel<<<1, 1>>>(
        buf.d_curriculum_conc_weights, init_scale, seed);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 1a: 调质 readout 前向
//   logits[m] = Σ_i W_mod[i*6+m] · rate[i]   (m ∈ [0,6))
//             + Σ_j W_conc[m*6+j] · conc[j]  (2026-08-04 方案2: 浓度头并联)
//   用 6 个 block, 每 block 归约 60K 神经元
//   输入为课程窗口内累计 spike 平均发放率 (rate ∈ [0,1]),
//   而非最后一帧 spike — 事件调质在窗口内持续调制发放,
//   累计率比单帧更能编码"知识链 vs 情感链"的事件类型信号
//   浓度头: conc[j] = 网络调质浓度 (各神经元相同, 读 [0]),
//     顺序与 readout 通道一致 [DA, ACh, NE, 5HT, GABA, Oxy]
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_forward_kernel(
    float* __restrict__ logits,                 // [6]
    const float* __restrict__ readout_weights,  // [N × 6]
    const float* __restrict__ spike_rates,      // [N] 累计平均发放率
    const float* __restrict__ conc_weights,     // [6×6] 2026-08-04 方案2 (可空)
    const float* __restrict__ da_conc,          // [N] 网络浓度 (读 [0])
    const float* __restrict__ ach_conc,
    const float* __restrict__ ne_conc,
    const float* __restrict__ ht5_conc,
    const float* __restrict__ gaba_conc,
    const float* __restrict__ oxy_conc,
    int N)
{
    const int m = blockIdx.x;                   // 0..5
    if (m >= 6) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_rates[i] > 0.0f) {
            sum += readout_weights[(size_t)i * 6 + m] * spike_rates[i];
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
    if (threadIdx.x == 0) {
        float out = sdata[0];
        // 2026-08-04 方案2: 浓度头 (浓度缓冲缺失时跳过)
        if (conc_weights && da_conc) {
            const float conc[6] = { da_conc[0], ach_conc[0], ne_conc[0],
                                    ht5_conc[0], gaba_conc[0], oxy_conc[0] };
            const float* crow = conc_weights + m * 6;
            #pragma unroll
            for (int j = 0; j < 6; ++j) out += crow[j] * conc[j];
        }
        logits[m] = out;
    }
}

// -----------------------------------------------------------------------------
// Kernel 1b: 工具 readout 前向
//   tool_logits[t] = Σ_i W_tool[i*7+t] · rate[i]   (t ∈ [0,7))
//   用 7 个 block; 输入同为窗口累计平均发放率
// -----------------------------------------------------------------------------
__global__ void curriculum_tool_forward_kernel(
    float* __restrict__ tool_logits,            // [7]
    const float* __restrict__ tool_weights,     // [N × 7]
    const float* __restrict__ spike_rates,      // [N] 累计平均发放率
    int N)
{
    const int t = blockIdx.x;                   // 0..6
    if (t >= CURRICULUM_N_TOOL) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_rates[i] > 0.0f) {
            sum += tool_weights[(size_t)i * CURRICULUM_N_TOOL + t] * spike_rates[i];
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
// Kernel 1c: PAD 情感 readout 前向 (2026-08-02 Task 5)
//   logits_pad[p] = Σ_i W_pad[i*3+p] · rate[i]   (p ∈ [0,3))
//   [Pleasure, Arousal, Dominance]; 用 3 个 block, 每 block 归约一个 P 通道
//   与 mod/tool 前向同构, 输入同为窗口累计平均发放率
// -----------------------------------------------------------------------------
__global__ void curriculum_pad_forward_kernel(
    float* __restrict__ pad_logits,             // [3]
    const float* __restrict__ pad_weights,      // [N × 3]
    const float* __restrict__ spike_rates,      // [N] 累计平均发放率
    int N)
{
    const int p = blockIdx.x;                   // 0..2
    if (p >= 3) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_rates[i] > 0.0f) {
            sum += pad_weights[(size_t)i * 3 + p] * spike_rates[i];
        }
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) pad_logits[p] = sdata[0];
}

// -----------------------------------------------------------------------------
// host wrapper: PAD readout 前向 (累计帧: 窗口平均发放率 → logits_pad[3])
// -----------------------------------------------------------------------------
void launch_curriculum_pad_forward(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_pad_weights || !buf.d_curriculum_pad_logits
        || !buf.d_curriculum_accum_spikes) return;
    curriculum_pad_forward_kernel<<<3, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_pad_logits,
        buf.d_curriculum_pad_weights,
        buf.d_curriculum_accum_spikes,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 2: 误差 + 损失计算
//   调质: error_mod[m] = logits[m] - target[m];  L_mod = 0.5·Σ error²
//   PAD:  error_pad[p] = logits_pad[p] - target_pad[p]; L_pad = 0.5·Σ error²
//  工具: softmax 后 CE, error_tool[t] = p[t] - y[t] (dL/dz)
//  总损失: L = w_mod·L_mod + w_pad·L_pad + w_tool·L_tool
//  注: w_pad=0 (或 pad 缓冲缺失, wrapper 置 0) 时 PAD 项数学上无贡献 (安全默认)
// -----------------------------------------------------------------------------
__global__ void curriculum_error_kernel(
    float* __restrict__ mod_error,              // [6]
    const float* __restrict__ mod_logits,       // [6]
    const float* __restrict__ target_mod,       // [6]
    float* __restrict__ tool_error,             // [7]
    const float* __restrict__ tool_logits,      // [7]
    int target_tool,                            // 0-6
    float* __restrict__ pad_error,              // [3]
    const float* __restrict__ pad_logits,       // [3]
    const float* __restrict__ target_pad,       // [3]
    float w_mod, float w_tool, float w_pad,
    float* __restrict__ loss_out)               // [1]
{
    float sum = 0.0f;
    // 调质部分: 跨线程
    for (int m = threadIdx.x; m < 6; m += blockDim.x) {
        float e = mod_logits[m] - target_mod[m];
        mod_error[m] = e;
        sum += 0.5f * w_mod * e * e;
    }
    // PAD 部分: 跨线程 (w_pad=0 时跳过, 避免触碰空指针且贡献恒为 0)
    if (w_pad != 0.0f) {
        for (int p = threadIdx.x; p < 3; p += blockDim.x) {
            float e = pad_logits[p] - target_pad[p];
            pad_error[p] = e;
            sum += 0.5f * w_pad * e * e;
        }
    }
    // 工具部分: thread 0 单独算 (softmax 需全 7 个 logits)
    if (threadIdx.x == 0) {
        // 2026-08-03 工具类分化: 类别加权 CE (inverse-frequency, 归一化使平均权重=1)
        //   依据 curriculum_middle_school.jsonl 2000 样本工具分布统计:
        //   类 0-4 各 ~190-210 样本, 类 5 无样本, 类 6(不调用) 1018 样本 (50.9%).
        //   原均匀 CE 下多数类(6)梯度主导 → readout 全预测 6 (55% 准确率假象).
        //   加权后少数类梯度放大 ~5x, 迫使 readout 分化工具决策.
        //   L = -w_y·log(p_y);  dL/dz_t = w_y·(p_t - y_t)  (softmax 加权梯度)
        const float cls_w[CURRICULUM_N_TOOL] = {
            1.128f, 1.187f, 1.063f, 1.151f, 1.193f, 1.058f, 0.220f
        };
        const float w_y = cls_w[target_tool];
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
            // 加权 softmax 梯度: dL_tool/dz_t = w_y·(p_t - y_t)
            tool_error[t] = w_y * (p - ((t == target_tool) ? 1.0f : 0.0f));
            if (t == target_tool) {
                ce = -w_y * logf(fmaxf(p, 1e-30f));
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
// Kernel 3: 调质 + PAD + 工具误差合并反传到神经元 dL/dS_direct
//   dL_dS_direct[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]
//                   + Σ_p w_pad·W_pad[i,p]·err_pad[p]
//                   + Σ_t w_tool·W_tool[i,t]·err_tool[t]
//   注: w_pad=0 时 PAD 项跳过 (安全默认, 不触碰空指针)
// -----------------------------------------------------------------------------
__global__ void curriculum_backprop_kernel(
    float* __restrict__ dL_dS_direct,           // [N]
    const float* __restrict__ readout_weights,  // [N × 6]
    const float* __restrict__ tool_weights,     // [N × 7]
    const float* __restrict__ pad_weights,      // [N × 3]
    const float* __restrict__ mod_error,        // [6]
    const float* __restrict__ tool_error,       // [7]
    const float* __restrict__ pad_error,        // [3]
    int N, float w_mod, float w_tool, float w_pad)
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
    if (w_pad != 0.0f) {
        const float* pad_row = pad_weights + (size_t)i * 3;
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            sum += w_pad * pad_row[p] * pad_error[p];
        }
    }
    dL_dS_direct[i] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 4: readout 权重更新 (SGD, 无裁剪, 学习率小)
//   W_mod[i*6+m]  -= lr·w_mod·err_mod[m]·rate[i]
//   W_tool[i*7+t] -= lr·w_tool·err_tool[t]·rate[i]
//   W_pad[i*3+p]  -= lr·w_pad·err_pad[p]·rate[i]   (2026-08-02 Task 5)
//   W_conc[m*6+j] -= lr·w_mod·err_mod[m]·conc[j]   (2026-08-04 方案2, i==0 线程)
//   更新量按窗口累计平均发放率 rate[i] 缩放 (与 readout 前向输入一致)
//   浓度头输入 conc[j] 为网络浓度 (各神经元相同, 读 [0])
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_update_kernel(
    float* __restrict__ readout_weights,        // [N × 6]
    float* __restrict__ tool_weights,           // [N × 7]
    float* __restrict__ pad_weights,            // [N × 3]
    float* __restrict__ conc_weights,           // [6×6] 2026-08-04 方案2 (可空)
    const float* __restrict__ spike_rates,      // [N] 累计平均发放率
    const float* __restrict__ mod_error,        // [6]
    const float* __restrict__ tool_error,       // [7]
    const float* __restrict__ pad_error,        // [3]
    const float* __restrict__ da_conc,          // [N] 网络浓度 (读 [0])
    const float* __restrict__ ach_conc,
    const float* __restrict__ ne_conc,
    const float* __restrict__ ht5_conc,
    const float* __restrict__ gaba_conc,
    const float* __restrict__ oxy_conc,
    int N, float lr, float w_mod, float w_tool, float w_pad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 2026-08-04 方案2: 浓度头更新 (仅 i==0 线程, 36 个权重与神经元无关)
    if (i == 0 && conc_weights && da_conc) {
        const float conc[6] = { da_conc[0], ach_conc[0], ne_conc[0],
                                ht5_conc[0], gaba_conc[0], oxy_conc[0] };
        #pragma unroll
        for (int m = 0; m < 6; ++m) {
            float* crow = conc_weights + m * 6;
            #pragma unroll
            for (int j = 0; j < 6; ++j) {
                float w = crow[j] - lr * w_mod * mod_error[m] * conc[j];
                crow[j] = fminf(fmaxf(w, -CURRICULUM_READOUT_WEIGHT_CLIP),
                                CURRICULUM_READOUT_WEIGHT_CLIP);
            }
        }
    }

    if (spike_rates[i] <= 0.0f) return;

    // 2026-08-01 spec §7.8 修复: SGD 更新后裁剪权重
    //   防 spike_rates 系统性偏高的神经元上 readout 权重发散
    float* mod_row = readout_weights + (size_t)i * 6;
    #pragma unroll
    for (int m = 0; m < 6; ++m) {
        float w = mod_row[m] - lr * w_mod * mod_error[m] * spike_rates[i];
        mod_row[m] = fminf(fmaxf(w, -CURRICULUM_READOUT_WEIGHT_CLIP),
                           CURRICULUM_READOUT_WEIGHT_CLIP);
    }
    float* tool_row = tool_weights + (size_t)i * CURRICULUM_N_TOOL;
    #pragma unroll
    for (int t = 0; t < CURRICULUM_N_TOOL; ++t) {
        float w = tool_row[t] - lr * w_tool * tool_error[t] * spike_rates[i];
        tool_row[t] = fminf(fmaxf(w, -CURRICULUM_READOUT_WEIGHT_CLIP),
                            CURRICULUM_READOUT_WEIGHT_CLIP);
    }
    // PAD 头: w_pad=0 (或缓冲缺失, wrapper 置 0) 时跳过 (安全默认)
    if (w_pad != 0.0f) {
        float* pad_row = pad_weights + (size_t)i * 3;
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            float w = pad_row[p] - lr * w_pad * pad_error[p] * spike_rates[i];
            pad_row[p] = fminf(fmaxf(w, -CURRICULUM_READOUT_WEIGHT_CLIP),
                               CURRICULUM_READOUT_WEIGHT_CLIP);
        }
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 课程前向 (窗口累计发放率 → 调质 6 维预测 + 工具 7 类注意力)
//   输出到 buf.d_curriculum_logits + d_curriculum_tool_logits
// -----------------------------------------------------------------------------
void launch_curriculum_readout_forward(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_logits || !buf.d_curriculum_tool_logits
        || !buf.d_curriculum_accum_spikes) return;
    curriculum_readout_forward_kernel<<<6, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_logits,
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_accum_spikes,
        buf.d_curriculum_conc_weights,            // 2026-08-04 方案2 (可空)
        buf.d_da_concentration,                   // 网络浓度 (读 [0])
        buf.d_ach_concentration,
        buf.d_ne_concentration,
        buf.d_ht5_concentration,
        buf.d_gaba_concentration,
        buf.d_oxytocin_concentration,
        N_TOTAL_NEURONS_2E);
    curriculum_tool_forward_kernel<<<CURRICULUM_N_TOOL, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_tool_logits,
        buf.d_curriculum_tool_weights,
        buf.d_curriculum_accum_spikes,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: 课程误差 + 损失 (host 端传入目标调质 + 目标工具 + 目标 PAD)
//   输出: buf.d_curriculum_error + d_curriculum_tool_error
//         + d_curriculum_pad_error + *out_loss
// -----------------------------------------------------------------------------
void launch_curriculum_error(PersistentBuffers& buf,
                             const float target_mod[6],
                             int target_tool,
                             const float target_pad[3],
                             float w_mod, float w_tool, float w_pad,
                             float* out_loss)
{
    if (!buf.d_curriculum_logits || !buf.d_curriculum_tool_logits
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error) {
        if (out_loss) *out_loss = 0.0f;
        return;
    }
    // 2026-08-01 spec §7.9 修复: 消除静态懒分配 (不可重入 + 生命周期游离)
    // 目标/loss 缓冲移入 PersistentBuffers (d_curriculum_target/d_curriculum_loss),
    // 与 allocator 生命周期一致, 多流/多 GPU 场景安全
    if (!buf.d_curriculum_target || !buf.d_curriculum_loss) {
        if (out_loss) *out_loss = 0.0f;
        return;
    }
    cudaMemcpy(buf.d_curriculum_target, target_mod, 6 * sizeof(float), cudaMemcpyHostToDevice);
    // PAD 缓冲缺失时降级: w_pad=0 (Kernel 内 PAD 块跳过, 不触碰空指针)
    const bool has_pad = buf.d_curriculum_pad_logits && buf.d_curriculum_pad_error;
    if (has_pad) {
        // d_curriculum_target 布局 [9]: [0..6) 调质目标 + [6..9) PAD 目标
        cudaMemcpy(buf.d_curriculum_target + 6, target_pad, 3 * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    curriculum_error_kernel<<<1, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_error,
        buf.d_curriculum_logits,
        buf.d_curriculum_target,
        buf.d_curriculum_tool_error,
        buf.d_curriculum_tool_logits,
        target_tool,
        has_pad ? buf.d_curriculum_pad_error : nullptr,
        has_pad ? buf.d_curriculum_pad_logits : nullptr,
        has_pad ? buf.d_curriculum_target + 6 : nullptr,
        w_mod, w_tool, has_pad ? w_pad : 0.0f,
        buf.d_curriculum_loss);
    CUDA_CHECK_LAST_2E();

    if (out_loss) {
        cudaMemcpy(out_loss, buf.d_curriculum_loss, sizeof(float), cudaMemcpyDeviceToHost);
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 调质 + PAD + 工具误差合并反传到神经元 dL/dS_direct
//   输出到 direct_grad (由调用方复用, 如 BPTT 的 d_v_grad_ 临时缓冲)
//   w_pad 默认 0.0f: bptt_trainer.backward_curriculum (不可修改文件) 以 4 参调用,
//   PAD 项数学上无贡献 (安全默认); N3F eligibility 路径不走本函数
// -----------------------------------------------------------------------------
void launch_curriculum_backprop(PersistentBuffers& buf, float* dL_dS_direct,
                                float w_mod, float w_tool, float w_pad)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error) return;
    // PAD 缓冲缺失时降级: w_pad=0 (Kernel 内 PAD 块跳过, 不触碰空指针)
    const bool has_pad = buf.d_curriculum_pad_weights && buf.d_curriculum_pad_error;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_backprop_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        dL_dS_direct,
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_tool_weights,
        has_pad ? buf.d_curriculum_pad_weights : nullptr,
        buf.d_curriculum_error,
        buf.d_curriculum_tool_error,
        has_pad ? buf.d_curriculum_pad_error : nullptr,
        N_TOTAL_NEURONS_2E, w_mod, w_tool, has_pad ? w_pad : 0.0f);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// host wrapper: readout 权重更新 (调质 + 工具 + PAD, 按窗口累计发放率缩放)
// -----------------------------------------------------------------------------
void launch_curriculum_readout_update(PersistentBuffers& buf, float lr,
                                      float w_mod, float w_tool, float w_pad)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_error || !buf.d_curriculum_tool_error
        || !buf.d_curriculum_accum_spikes) return;
    // PAD 缓冲缺失时降级: w_pad=0 (Kernel 内 PAD 块跳过, 不触碰空指针)
    const bool has_pad = buf.d_curriculum_pad_weights && buf.d_curriculum_pad_error;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_readout_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_tool_weights,
        has_pad ? buf.d_curriculum_pad_weights : nullptr,
        buf.d_curriculum_conc_weights,            // 2026-08-04 方案2 (可空)
        buf.d_curriculum_accum_spikes,
        buf.d_curriculum_error,
        buf.d_curriculum_tool_error,
        has_pad ? buf.d_curriculum_pad_error : nullptr,
        buf.d_da_concentration,                   // 网络浓度 (读 [0])
        buf.d_ach_concentration,
        buf.d_ne_concentration,
        buf.d_ht5_concentration,
        buf.d_gaba_concentration,
        buf.d_oxytocin_concentration,
        N_TOTAL_NEURONS_2E, lr, w_mod, w_tool, has_pad ? w_pad : 0.0f);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 5a: 课程窗口累计 spike 清零
// -----------------------------------------------------------------------------
__global__ void curriculum_accum_clear_kernel(float* __restrict__ accum, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    accum[i] = 0.0f;
}

// -----------------------------------------------------------------------------
// Kernel 5b: 逐帧累计 spike 平均发放率
//   accum[i] += (spike_flags[i] ? 1.0f : 0.0f) / window_size
//   窗口末 accum[i] ∈ [0,1] = 该窗口内神经元 i 的平均发放率
// -----------------------------------------------------------------------------
__global__ void curriculum_accumulate_kernel(
    float* __restrict__ accum,
    const bool* __restrict__ spike_flags,
    int N, float inv_window)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (spike_flags[i]) {
        accum[i] += inv_window;
    }
}

// -----------------------------------------------------------------------------
// host wrapper: 课程窗口累计 spike 清零 / 逐帧累计
// -----------------------------------------------------------------------------
void launch_curriculum_accum_clear(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_accum_spikes) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_accum_clear_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_accum_spikes, N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

void launch_curriculum_accumulate(PersistentBuffers& buf, int window_size)
{
    if (!buf.d_curriculum_accum_spikes || !buf.d_spike_flags) return;
    if (window_size <= 0) window_size = 1;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    curriculum_accumulate_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_accum_spikes, buf.d_spike_flags,
        N_TOTAL_NEURONS_2E, 1.0f / (float)window_size);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 6 (N3F): readout 前向 — 当前帧 bool spike (在线教学信号用)
//   logits[m] = Σ_i W_mod[i*6+m] · spike[i]   (spike ∈ {0,1})
//   tool_logits[t] = Σ_i W_tool[i*7+t] · spike[i]
// -----------------------------------------------------------------------------
__global__ void curriculum_readout_forward_frame_kernel(
    float* __restrict__ mod_logits,             // [6]
    float* __restrict__ tool_logits,            // [7]
    const float* __restrict__ readout_weights,  // [N × 6]
    const float* __restrict__ tool_weights,     // [N × 7]
    const bool* __restrict__ spike_flags,       // [N] 当前帧
    int N)
{
    // 调质头: blockIdx.x = 0..5
    if (blockIdx.x < 6) {
        const int m = blockIdx.x;
        float sum = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (spike_flags[i]) {
                sum += readout_weights[(size_t)i * 6 + m];
            }
        }
        __shared__ float sdata[256];
        sdata[threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) mod_logits[m] = sdata[0];
    } else if (blockIdx.x < 6 + CURRICULUM_N_TOOL) {
        // 工具头: blockIdx.x = 6..12 → t = blockIdx.x - 6
        const int t = blockIdx.x - 6;
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
}

void launch_curriculum_readout_forward_frame(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_readout_weights || !buf.d_curriculum_tool_weights
        || !buf.d_curriculum_logits || !buf.d_curriculum_tool_logits
        || !buf.d_spike_flags) return;
    curriculum_readout_forward_frame_kernel<<<6 + CURRICULUM_N_TOOL, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_logits,
        buf.d_curriculum_tool_logits,
        buf.d_curriculum_readout_weights,
        buf.d_curriculum_tool_weights,
        buf.d_spike_flags,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 6c (N3F): PAD readout 前向 — 当前帧 bool spike (2026-08-02 Task 5)
//   logits_pad[p] = Σ_i W_pad[i*3+p] · spike[i]   (spike ∈ {0,1})
//   用 3 个 block, 每 block 归约一个 P 通道
// -----------------------------------------------------------------------------
__global__ void curriculum_pad_forward_frame_kernel(
    float* __restrict__ pad_logits,             // [3]
    const float* __restrict__ pad_weights,      // [N × 3]
    const bool* __restrict__ spike_flags,       // [N] 当前帧
    int N)
{
    const int p = blockIdx.x;                   // 0..2
    if (p >= 3) return;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (spike_flags[i]) {
            sum += pad_weights[(size_t)i * 3 + p];
        }
    }
    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) pad_logits[p] = sdata[0];
}

void launch_curriculum_pad_forward_frame(PersistentBuffers& buf)
{
    if (!buf.d_curriculum_pad_weights || !buf.d_curriculum_pad_logits
        || !buf.d_spike_flags) return;
    curriculum_pad_forward_frame_kernel<<<3, THREADS_PER_BLOCK_2E>>>(
        buf.d_curriculum_pad_logits,
        buf.d_curriculum_pad_weights,
        buf.d_spike_flags,
        N_TOTAL_NEURONS_2E);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// Kernel 8 (N3F, 2026-08-01 spec §7.1): 具身奖励 → 神经元级 eligibility
//   neuron_elig[i] = λ·elig[i] + g·reward   (reward ∈ [-1,1], uniform 广播)
//   生物对应: DA 奖赏系统广播到所有表达受体的神经元
//   reward > 0 → 强化 (eligibility 上升, STDP 证据增强 LTP)
//   reward < 0 → 削弱 (eligibility 下降, STDP 证据增强 LTD)
// -----------------------------------------------------------------------------
__global__ void embodied_eligibility_update_kernel(
    float* __restrict__ neuron_eligibility,  // [N] in/out
    float reward, int N, float decay_factor, float gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float old_val = neuron_eligibility[i];
    float new_val = decay_factor * old_val + gain * reward;
    if (new_val > 1.0f)  new_val = 1.0f;
    if (new_val < -1.0f) new_val = -1.0f;
    neuron_eligibility[i] = new_val;
}

void launch_embodied_eligibility_update(PersistentBuffers& buf,
                                        float reward, float decay_factor, float gain)
{
    if (!buf.d_neuron_eligibility) return;
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    embodied_eligibility_update_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        buf.d_neuron_eligibility, reward, N_TOTAL_NEURONS_2E, decay_factor, gain);
    CUDA_CHECK_LAST_2E();
}

} // namespace stage2e
