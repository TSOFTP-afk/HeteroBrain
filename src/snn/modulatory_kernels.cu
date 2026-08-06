// =============================================================================
// Stage 2e 调质系统 + DA价值函数 + 字节选择性统计 实现 (P2)
// =============================================================================
// 设计要点:
//   1. modulatory_kernel: 4种调质浓度衰减 + 信号驱动
//   2. da_value_function: 亚柱级 V(s) = w_value · φ(s), TD学习
//   3. byte_histogram: 每注入步统计 spike count per byte
// =============================================================================

#include "modulatory_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

namespace stage2e {

// ==================== GPU kernel ====================

// 调质浓度衰减 + 信号驱动 (每100步调用, 但浓度本身每步衰减)
// 每 thread 处理一个神经元
// Phase 3a: 扩充到 6 维 (加 GABA/催产素)
__global__ void modulatory_kernel(
    float* __restrict__ da_conc,
    float* __restrict__ ach_conc,
    float* __restrict__ ne_conc,
    float* __restrict__ ht5_conc,
    float* __restrict__ gaba_conc,      // Phase 3a
    float* __restrict__ oxytocin_conc,  // Phase 3a
    int n_neurons,
    float da_signal,       // DA 信号 (δ + 基线)
    float ach_signal,      // ACh 信号 (惊奇 + 注意力)
    float ne_signal,       // NE 信号 (KL散度触发)
    float ht5_signal,      // 5HT 信号 (预测误差持续负)
    float gaba_signal,     // GABA 信号 (NE 反馈抗焦虑, Phase 3a)
    float oxytocin_signal, // 催产素信号 (共情驱动, Phase 3a)
    float da_decay,        // DA 衰减率 (exp(-100/tau))
    float ach_decay,
    float ne_decay,
    float ht5_decay,
    float gaba_decay,      // Phase 3a
    float oxytocin_decay)  // Phase 3a
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;

    // 衰减
    da_conc[i]       *= da_decay;
    ach_conc[i]      *= ach_decay;
    ne_conc[i]       *= ne_decay;
    ht5_conc[i]      *= ht5_decay;
    gaba_conc[i]     *= gaba_decay;       // Phase 3a
    oxytocin_conc[i] *= oxytocin_decay;   // Phase 3a

    // 信号驱动 (加到所有神经元, 后续突触级受体差异化响应)
    da_conc[i]       += da_signal;
    ach_conc[i]      += ach_signal;
    ne_conc[i]       += ne_signal;
    ht5_conc[i]      += ht5_signal;
    gaba_conc[i]     += gaba_signal;      // Phase 3a
    oxytocin_conc[i] += oxytocin_signal;  // Phase 3a

    // clamp
    if (da_conc[i]       < 0.0f) da_conc[i]       = 0.0f;
    if (da_conc[i]       > 2.0f) da_conc[i]       = 2.0f;
    if (ach_conc[i]      < 0.0f) ach_conc[i]      = 0.0f;
    if (ach_conc[i]      > 2.0f) ach_conc[i]      = 2.0f;
    if (ne_conc[i]       < 0.0f) ne_conc[i]       = 0.0f;
    if (ne_conc[i]       > 2.0f) ne_conc[i]       = 2.0f;
    if (ht5_conc[i]      < 0.0f) ht5_conc[i]      = 0.0f;
    if (ht5_conc[i]      > 2.0f) ht5_conc[i]      = 2.0f;
    if (gaba_conc[i]     < 0.0f) gaba_conc[i]     = 0.0f;  // Phase 3a
    if (gaba_conc[i]     > 2.0f) gaba_conc[i]     = 2.0f;
    if (oxytocin_conc[i] < 0.0f) oxytocin_conc[i] = 0.0f;
    if (oxytocin_conc[i] > 2.0f) oxytocin_conc[i] = 2.0f;
}

// 亚柱发放直方图计算 (从 spike_flags 聚合到 200 维亚柱级)
// 每 thread 处理一个亚柱 (50柱 × 4亚柱 = 200)
__global__ void subcolumn_fr_kernel(
    const bool* __restrict__ spike_flags,
    float* __restrict__ subcolumn_fr,
    int n_neurons,
    int neurons_per_subcolumn)
{
    int sc = blockIdx.x * blockDim.x + threadIdx.x;
    if (sc >= W_VALUE_DIM) return;

    int start = sc * neurons_per_subcolumn;
    int end = start + neurons_per_subcolumn;
    if (end > n_neurons) end = n_neurons;

    int count = 0;
    for (int i = start; i < end; ++i) {
        if (spike_flags[i]) count++;
    }
    // 发放率 = spike count / 神经元数
    subcolumn_fr[sc] = static_cast<float>(count) / neurons_per_subcolumn;
}

// V(s) = w_value · φ(s)  (线性价值函数)
// φ(s) = subcolumn_fr (200维)
__global__ void value_function_kernel(
    const float* __restrict__ subcolumn_fr,
    const float* __restrict__ w_value,
    float* __restrict__ v_out,
    int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        float v = 0.0f;
        for (int k = 0; k < dim; ++k) {
            v += w_value[k] * subcolumn_fr[k];
        }
        *v_out = v;
    }
}

// TD学习: w_value += η · δ · φ(s)
// 注意: W_pred 完整矩阵更新已移至 w_pred_update_kernel (Task 10)
__global__ void td_update_kernel(
    float* __restrict__ w_value,
    const float* __restrict__ subcolumn_fr,
    float delta,
    int dim)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;

    // w_value 更新 (TD error 驱动)
    w_value[k] += ETA_VALUE * delta * subcolumn_fr[k];
    // clamp 防发散
    if (w_value[k] > 1.0f) w_value[k] = 1.0f;
    if (w_value[k] < -1.0f) w_value[k] = -1.0f;
}

// EMA 基线更新 (W_pred 预测已移至 w_pred_predict_kernel, Task 10)
__global__ void baseline_update_kernel(
    float* __restrict__ baseline_fr,
    const float* __restrict__ subcolumn_fr,
    int dim)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= dim) return;

    // EMA 基线
    baseline_fr[k] = NOVELTY_EMA_BETA * baseline_fr[k]
                   + (1.0f - NOVELTY_EMA_BETA) * subcolumn_fr[k];
}

// ==================== Task 10: W_pred 完整矩阵 ====================

// W_pred 完整矩阵预测: pred_fr[j] = Σ_k w_pred[j*dim+k] * fr_prev[k]
// 每 thread 处理一个 j (200 个线程), 内部循环 k (200 次)
// 矩阵布局: row-major, w_pred[j*dim+k] 是第 j 行第 k 列
__global__ void w_pred_predict_kernel(
    float* __restrict__ pred_fr,
    const float* __restrict__ w_pred,
    const float* __restrict__ fr_prev,    // 上一步亚柱发放率
    int dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;

    // 完整矩阵-向量乘法: pred_fr = W_pred · fr_prev
    float sum = 0.0f;
    for (int k = 0; k < dim; ++k) {
        sum += w_pred[j * dim + k] * fr_prev[k];
    }
    pred_fr[j] = sum;
}

// W_pred 完整矩阵更新: w_pred[j*dim+k] += η_pred · (fr_j - pred_j) · fr_k_prev
// 每 thread 处理一个 j (200 个线程), 内部循环 k (200 次)
// 学习规则: 预测误差 (fr_j - pred_j) × 输入 fr_k_prev 的外积
__global__ void w_pred_update_kernel(
    float* __restrict__ w_pred,
    const float* __restrict__ subcolumn_fr,   // 当前亚柱发放率 fr_j
    const float* __restrict__ pred_fr,         // 当前预测 pred_j
    const float* __restrict__ fr_prev,         // 上一步亚柱发放率 fr_k_prev
    int dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dim) return;

    // 第 j 行的预测误差
    float pred_error = subcolumn_fr[j] - pred_fr[j];
    // 更新第 j 行所有列 k: w_pred[j*dim+k] += η_pred · pred_error · fr_prev[k]
    for (int k = 0; k < dim; ++k) {
        w_pred[j * dim + k] += ETA_PRED * pred_error * fr_prev[k];
        // clamp 防发散 (与原对角项更新一致)
        if (w_pred[j * dim + k] > 1.0f) w_pred[j * dim + k] = 1.0f;
        if (w_pred[j * dim + k] < -1.0f) w_pred[j * dim + k] = -1.0f;
    }
}

// 预测成功率 (余弦相似度): pred_succ = (cos(pred_fr, fr_subcol) + 1) / 2
// 单线程计算 (200 维, 计算量小, 启动开销不值得并行化)
// 余弦相似度 ∈ [-1, 1], 映射到 [0, 1] 用于 ACh 注意力调制
__global__ void cosine_similarity_kernel(
    const float* __restrict__ pred_fr,
    const float* __restrict__ subcolumn_fr,
    float* __restrict__ out_pred_succ,
    int dim)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float dot = 0.0f;
        float norm_pred = 0.0f;
        float norm_fr = 0.0f;
        for (int k = 0; k < dim; ++k) {
            dot += pred_fr[k] * subcolumn_fr[k];
            norm_pred += pred_fr[k] * pred_fr[k];
            norm_fr += subcolumn_fr[k] * subcolumn_fr[k];
        }
        norm_pred = sqrtf(norm_pred);
        norm_fr = sqrtf(norm_fr);
        float denom = norm_pred * norm_fr;
        // 防 0 除 (冷启动时 fr_prev=0 → pred_fr=0, 范数为 0)
        float cos_sim = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        // 映射到 [0, 1]: (cos + 1) / 2
        *out_pred_succ = (cos_sim + 1.0f) * 0.5f;
    }
}

// 字节选择性直方图 (每注入步)
// 统计当前字节对应的总 spike 数
__global__ void byte_histogram_kernel(
    const bool* __restrict__ spike_flags,
    int* __restrict__ byte_histogram,
    int n_neurons,
    int current_byte)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_neurons) return;
    if (spike_flags[i]) {
        atomicAdd(&byte_histogram[current_byte], 1);
    }
}

// ==================== Host 端 launch ====================

static float h_v_s = 0.0f;
static float h_v_sp = 0.0f;
static float* d_v_scratch = nullptr;

// Task 10: 预测成功率 (余弦相似度版本, 由 launch_da_value_function 计算)
// launch_modulatory 使用此值替代外部传入的二元 pred_succ
static float h_pred_succ_cos = 0.5f;   // 冷启动默认 0.5 (中性)
static float* d_pred_succ_scratch = nullptr;

// Phase 3a: 共情驱动信号缓存 (由 set_empathy_signal 设置, launch_modulatory 读取后清零)
static float h_empathy_signal = 0.0f;

// Phase 3a-C1: 6 维事件驱动调质信号缓存 (与 h_empathy_signal 同构, 但扩展到 6 维)
// Phase 3a-C2 修复: 改为累加模式, 支持同一 step 多事件叠加
//   旧实现: h_event_signal[i] = v (覆盖, 后调用的事件冲掉前面的)
//   新实现: h_event_signal[i] += v (累加, 多事件叠加后一次性注入)
//   生命周期:
//     - 每步 dispatch_pending 开始前由 reset_event_signal() 清零
//     - 同一 step 多个事件依次累加
//     - launch_modulatory 读取后清零 (单次触发模型)
static float h_event_signal[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

// Phase 3a-F (M3): HPA 应激慢轴 — 皮质醇 host 标量 (零显存)
//   事件应激累积 → 慢衰减 (CORTISOL_TAU, 分钟级) → 调制 5HT/GABA 基线
static float g_cortisol = 0.0f;
static int   h_event_duration_steps = 0;  // 剩余持续步数 (0=单次脉冲)
static int   h_event_pending_count = 0;    // 当前 step 累加的事件数

// Phase 3a-D1: 具身训练 reward + curiosity 缓存
//   h_embodied_reward: 由 set_embodied_reward 设置, launch_da_value_function 读取覆盖外部 reward
//   h_curiosity_ach: 由 set_curiosity_ach 设置, launch_modulatory 读取叠加到 ACh 通道
//   注意: h_curiosity_ach 是持续信号 (不像 h_event_signal 单次触发), 每次调用不清零
static float h_embodied_reward = 0.0f;
static float h_curiosity_ach = 0.0f;

// Phase 3a: 上轮 NE 均值缓存 (用于 GABA 抗焦虑反馈计算)
static float h_last_ne_mean = 0.05f;

// Phase 3a-B: 6 维受体灵敏度 (稳态补偿, 防止病理滑移)
//   索引顺序与 CHANNEL_ORDER 一致: [DA, ACh, NE, 5HT, GABA, Oxy]
//   初始 1.0 (满灵敏), 持续超阈时下调, 低于阈时缓慢回升
//   有效注入信号 = 原始信号 × receptor_sensitivity[ch]
static float h_receptor_sensitivity[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
// 各 channel 对应的稳态基线阈值 (与 config.h HOMEOSTATIC_BASELINE_* 对齐)
static const float HOMEOSTATIC_BASELINE[6] = {
    HOMEOSTATIC_BASELINE_DA,
    HOMEOSTATIC_BASELINE_ACH,
    HOMEOSTATIC_BASELINE_NE,
    HOMEOSTATIC_BASELINE_HT5,
    HOMEOSTATIC_BASELINE_GABA,
    HOMEOSTATIC_BASELINE_OXY,
};

void set_empathy_signal(float empathy_signal) {
    if (empathy_signal < 0.0f) empathy_signal = 0.0f;
    if (empathy_signal > 1.0f) empathy_signal = 1.0f;
    h_empathy_signal = empathy_signal;
}

// Phase 3a-C2 修复: 每步开始前清零事件信号缓存, 允许同 step 多事件叠加
// 调用时机: EventScheduler::dispatch_pending 在处理 step 的事件之前先调用此函数
void reset_event_signal() {
    for (int i = 0; i < 6; ++i) h_event_signal[i] = 0.0f;
    h_event_pending_count = 0;
    // 注意: duration 不在这里清, 因为它由事件本身指定
}

// Phase 3a-C2 修复: 累加模式, 支持同一 step 多事件叠加
// 旧实现是 h_event_signal[i] = v (覆盖), 会导致后调用的事件冲掉前面的
// 新实现是 h_event_signal[i] += v (累加), 多事件叠加后一次性注入
// 叠加后 clamp 到 [-1.5, 1.5] (允许超调, 但有上限防爆炸)
void set_event_signal(const float modulator_delta[6], int duration_steps) {
    for (int i = 0; i < 6; ++i) {
        float v = modulator_delta[i];
        // 单事件 delta 先 clamp 到 [-1, 1]
        if (v < -1.0f) v = -1.0f;
        if (v > 1.0f) v = 1.0f;
        // 累加 (非覆盖)
        h_event_signal[i] += v;
        // 叠加后 clamp 到 [-1.5, 1.5] (允许 2-3 事件叠加超调, 但防止爆炸)
        if (h_event_signal[i] < -1.5f) h_event_signal[i] = -1.5f;
        if (h_event_signal[i] > 1.5f) h_event_signal[i] = 1.5f;
    }
    // duration 取最大值 (长持续时间事件不应被短事件截断)
    if (duration_steps > h_event_duration_steps) {
        h_event_duration_steps = duration_steps;
    }
    h_event_pending_count++;
}

// Phase 3a-C2: 查询当前 step 累加的事件数 (诊断用)
int get_event_pending_count() {
    return h_event_pending_count;
}

// Phase 3a-F (M3): HPA 应激慢轴 (皮质醇) host 接口
//   stress ∈ [0,1]: 事件应激量 (threat/criticism/social_loss → 高; praise/奖赏 → 0)
//   应激累积: cortisol += stress * CORTISOL_STRESS_GAIN, 衰减在 launch_modulatory
void set_cortisol_stress(float stress) {
    if (stress < 0.0f) stress = 0.0f;
    if (stress > 1.0f) stress = 1.0f;
    g_cortisol += stress * CORTISOL_STRESS_GAIN;
    if (g_cortisol > 1.0f) g_cortisol = 1.0f;
}

float get_cortisol_level() {
    return g_cortisol;
}

void set_cortisol_level(float v) {
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    g_cortisol = v;
}

// Phase 3a-D1: 具身训练 reward + curiosity 接口
void set_embodied_reward(float reward) {
    if (reward < -1.0f) reward = -1.0f;
    if (reward > 1.0f) reward = 1.0f;
    h_embodied_reward = reward;
}

void set_curiosity_ach(float pred_error) {
    if (pred_error < 0.0f) pred_error = 0.0f;
    if (pred_error > 1.0f) pred_error = 1.0f;
    h_curiosity_ach = pred_error * 0.3f;  // clamp ACh增量到[0, 0.3]
}

float get_last_embodied_reward() { return h_embodied_reward; }
float get_last_curiosity_ach() { return h_curiosity_ach; }

ModulatoryRuntimeState export_modulatory_runtime_state() {
    return {h_v_s, h_v_sp};
}

void import_modulatory_runtime_state(const ModulatoryRuntimeState& state) {
    h_v_s = state.v_s;
    h_v_sp = state.v_sp;
    // h_pred_succ_cos 不持久化: 重启后由下一步 launch_da_value_function 重算
}

static void ensure_v_scratch() {
    if (d_v_scratch == nullptr) {
        cudaMalloc(&d_v_scratch, sizeof(float));
        cudaMemset(d_v_scratch, 0, sizeof(float));
    }
}

static void ensure_pred_succ_scratch() {
    if (d_pred_succ_scratch == nullptr) {
        cudaMalloc(&d_pred_succ_scratch, sizeof(float));
        cudaMemset(d_pred_succ_scratch, 0, sizeof(float));
    }
}

void launch_modulatory(MemoryAllocator* alloc, int step,
                       float reward_signal, float novelty,
                       float pred_succ, float kl_divergence,
                       float da_delta,
                       float prediction_error_norm,
                       float empathy_signal,
                       const float* stage_baseline,
                       bool deterministic,
                       int mod_interval)
{
    PersistentBuffers& b = alloc->buffers();
    // DA 释放覆盖 [0, N_TOTAL_NEURONS_2E) = [0, 60000)
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;

    // ===== Phase 3a-B: 稳态补偿 — 采样上轮 6 维浓度均值, 更新受体灵敏度 =====
    // 时序: 此处采样的是上一轮 kernel 调用后的浓度 (等价于原末尾采样)
    //       提前到开头是为了让本轮 GABA 反馈和受体灵敏度共用一次 D2H 拷贝
    float current_means[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    {
        const int n_sample = 256;
        float h_da_sample[n_sample], h_ach_sample[n_sample];
        float h_ne_sample[n_sample],  h_ht5_sample[n_sample];
        float h_gaba_sample[n_sample], h_oxy_sample[n_sample];
        CUDA_CHECK_2E(cudaMemcpy(h_da_sample,   b.d_da_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_2E(cudaMemcpy(h_ach_sample,  b.d_ach_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_2E(cudaMemcpy(h_ne_sample,   b.d_ne_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_2E(cudaMemcpy(h_ht5_sample,  b.d_ht5_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_2E(cudaMemcpy(h_gaba_sample, b.d_gaba_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_2E(cudaMemcpy(h_oxy_sample,  b.d_oxytocin_concentration,
                                  n_sample * sizeof(float), cudaMemcpyDeviceToHost));
        double s_da = 0, s_ach = 0, s_ne = 0, s_ht5 = 0, s_gaba = 0, s_oxy = 0;
        for (int i = 0; i < n_sample; ++i) {
            s_da   += h_da_sample[i];
            s_ach  += h_ach_sample[i];
            s_ne   += h_ne_sample[i];
            s_ht5  += h_ht5_sample[i];
            s_gaba += h_gaba_sample[i];
            s_oxy  += h_oxy_sample[i];
        }
        current_means[0] = static_cast<float>(s_da   / n_sample);
        current_means[1] = static_cast<float>(s_ach  / n_sample);
        current_means[2] = static_cast<float>(s_ne   / n_sample);
        current_means[3] = static_cast<float>(s_ht5  / n_sample);
        current_means[4] = static_cast<float>(s_gaba / n_sample);
        current_means[5] = static_cast<float>(s_oxy  / n_sample);

        // 更新 h_last_ne_mean (供本轮 GABA 反馈, 与原末尾采样等价)
        h_last_ne_mean = current_means[2];

        // 受体灵敏度更新: 超阈下调, 低阈回升 (双向稳态)
        //   down: sensitivity *= (1 - rate * max(0, mean - baseline))
        //   up:   sensitivity *= (1 + upreg_rate * max(0, baseline - mean))
        for (int ch = 0; ch < 6; ++ch) {
            float excess = current_means[ch] - HOMEOSTATIC_BASELINE[ch];
            if (excess > 0.0f) {
                // 下调: 持续超阈 → 受体脱敏
                h_receptor_sensitivity[ch] *= (1.0f - HOMEOSTATIC_RATE * excess);
            } else {
                // 上调: 低于基线 → 受体缓慢恢复
                h_receptor_sensitivity[ch] *= (1.0f + HOMEOSTATIC_UPREG_RATE * (-excess));
            }
            // clamp 到 [MIN, MAX]
            if (h_receptor_sensitivity[ch] < RECEPTOR_SENSITIVITY_MIN)
                h_receptor_sensitivity[ch] = RECEPTOR_SENSITIVITY_MIN;
            if (h_receptor_sensitivity[ch] > RECEPTOR_SENSITIVITY_MAX)
                h_receptor_sensitivity[ch] = RECEPTOR_SENSITIVITY_MAX;
        }
    }

    // 调质信号计算
    // Phase 3a-D3: 课程模式阶段基线 (顺序映射 profile.baseline_mod = [DA, 5HT, NE, ACh, GABA, Oxy])
    //   注意: 内部变量顺序是 da/ach/ne/ht5/gaba/oxy, 与数组索引非直接对应
    //   非课程模式 (stage_baseline == nullptr): 使用 config.h 默认常量, 行为与修复前一致
    float base_da   = stage_baseline ? stage_baseline[0] : DA_BASE;
    float base_ht5  = stage_baseline ? stage_baseline[1] : 0.1f;
    float base_ne   = stage_baseline ? stage_baseline[2] : 0.05f;
    float base_ach  = stage_baseline ? stage_baseline[3] : 0.2f;
    float base_gaba = stage_baseline ? stage_baseline[4] : GABA_BASE;
    float base_oxy  = stage_baseline ? stage_baseline[5] : OXYTOCIN_BASE;

    // Phase 3a-F (M3): HPA 应激慢轴 — 皮质醇衰减 + 基线调制 (补分钟级时间尺度)
    //   衰减: 每 mod_interval 步按 CORTISOL_TAU 指数衰减 (慢通道, 事件后恢复惯性)
    //   调制: 慢性应激 → 5HT 基线下降 (抑郁模型) + GABA 基线上升 (代偿性抑制)
    g_cortisol *= expf(-(float)mod_interval / CORTISOL_TAU);
    if (g_cortisol > 0.0f) {
        base_ht5  = fmaxf(0.0f, base_ht5  - g_cortisol * CORTISOL_HT5_MOD);
        base_gaba = fminf(1.0f, base_gaba + g_cortisol * CORTISOL_GABA_MOD);
    }

    // DA 信号 = 基线 + 预测误差耦合 + TD error 驱动
    // 课程模式 (deterministic): 关闭网络依赖动力学项, 浓度 = 阶段基线 + 事件
    float da_signal;
    if (deterministic) {
        da_signal = base_da;   // 关闭 DA_GAIN·(1-pred_err) 与 TD delta 项
    } else {
        da_signal = base_da + DA_GAIN * (1.0f - prediction_error_norm);
        da_signal += (da_delta > 0 ? da_delta : 0.3f * da_delta);
        if (da_signal < 0.0f) da_signal = 0.0f;
    }

    // ACh: 基线 + 惊奇 (novelty) + 注意力 (余弦相似度预测成功率)
    // 课程模式: 关闭 novelty/pred_succ/curiosity 项
    float ach_signal;
    if (deterministic) {
        ach_signal = base_ach;   // 关闭 novelty/pred_succ/curiosity
    } else {
        ach_signal = base_ach + 0.3f * novelty + 0.1f * h_pred_succ_cos;
        ach_signal += h_curiosity_ach;
    }

    // NE: 基线 + KL 散度触发
    float ne_signal = base_ne;
    if (!deterministic && kl_divergence > 0.5f) {
        ne_signal += 0.5f * kl_divergence;
    }

    // 5HT: 基线 + 预测误差持续负时上升
    float ht5_signal = base_ht5;
    if (!deterministic && da_delta < -0.5f) {
        ht5_signal += 0.3f * fabsf(da_delta);
    }

    // Phase 3a: GABA 信号 = 基线 + NE 反馈 (抗焦虑负反馈)
    //   NE 越高 → GABA 越高 (抑制过度唤醒, 防止焦虑循环)
    //   使用上一轮 NE 均值 (h_last_ne_mean) 避免本轮 NE 信号自身依赖
    float gaba_signal = base_gaba;
    if (!deterministic && h_last_ne_mean > 0.3f) {
        gaba_signal += GABA_GAIN * (h_last_ne_mean - 0.3f);
    }

    // Phase 3a: 催产素信号 = 基线 + 共情驱动
    //   empathy_signal 来自 set_empathy_signal() (由 LLM/用户反馈触发)
    //   若调用方未传 empathy_signal, 使用内部缓存的 h_empathy_signal
    float eff_empathy = (empathy_signal > 0.0f) ? empathy_signal : h_empathy_signal;
    float oxytocin_signal = base_oxy;
    if (!deterministic) {
        oxytocin_signal += OXYTOCIN_GAIN * eff_empathy;
    }
    // 无论是否 deterministic, 读取后清零内部缓存 (单次触发模型)
    h_empathy_signal = 0.0f;

    // Phase 3a-C1: 读取事件驱动 6 维调质增量 (优先级高于 empathy)
    //   事件信号叠加到各通道, 然后统一经过受体灵敏度衰减 (稳态补偿仍生效)
    // Phase 3a-C2 修复: h_event_signal 已是累加结果 (同一 step 多事件叠加)
    bool has_event = false;
    float eff_event[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
        if (fabsf(h_event_signal[i]) > 1e-6f) {
            eff_event[i] = h_event_signal[i];
            has_event = true;
        }
    }
    if (has_event) {
        // Phase 3a-C2: 调质非线性交互 (生物脑中调质之间存在拮抗/协同)
        //   索引: [0]=DA, [1]=ACh, [2]=NE, [3]=5HT, [4]=GABA, [5]=Oxy
        //   规则 1: DA ↔ 5HT 拮抗 (DA↑ 时 5HT 效应减弱, 反之亦然)
        //     生物学依据: 中脑边缘 DA 通路与中缝核 5HT 通路相互抑制
        //     实现: 同号部分互相削减 (DA+ 与 5HT+ 同时存在时互相抵消一部分)
        //   规则 2: NE → GABA 抑制 (NE 释放会减少 GABA 能抑制)
        //     生物学依据: 蓝斑 NE 投射到 GABA 能中间神经元产生抑制
        //   规则 3: Oxy 放大 DA 奖赏 (催产素增强 DA 奖赏效应)
        //     生物学依据: Oxy 与 DA 在伏隔核协同, 促进社交奖赏
        float da = eff_event[0];
        float ach = eff_event[1];
        float ne = eff_event[2];
        float ht5 = eff_event[3];
        float gaba = eff_event[4];
        float oxy = eff_event[5];

        // 规则 1: DA-5HT 拮抗 (仅在两者同号时生效, 避免反向增强)
        if (da > 0.0f && ht5 > 0.0f) {
            float antagonism = 0.2f * fminf(da, ht5);
            da -= antagonism;
            ht5 -= antagonism;
        }
        // 规则 2: NE 抑制 GABA (NE↑ 时 GABA 释放减弱)
        if (ne > 0.0f && gaba > 0.0f) {
            float inhibition = 0.3f * ne * gaba;
            gaba -= inhibition;
            if (gaba < 0.0f) gaba = 0.0f;
        }
        // 规则 3: Oxy 放大 DA 奖赏 (仅在 DA 正向时生效)
        if (oxy > 0.0f && da > 0.0f) {
            da *= (1.0f + 0.5f * oxy);
        }

        // 写回 eff_event
        eff_event[0] = da;
        eff_event[1] = ach;
        eff_event[2] = ne;
        eff_event[3] = ht5;
        eff_event[4] = gaba;
        eff_event[5] = oxy;

        da_signal       = fmaxf(0.0f, da_signal       + eff_event[0]);
        ach_signal      = fmaxf(0.0f, ach_signal      + eff_event[1]);
        ne_signal       = fmaxf(0.0f, ne_signal       + eff_event[2]);
        ht5_signal      = fmaxf(0.0f, ht5_signal      + eff_event[3]);
        gaba_signal     = fmaxf(0.0f, gaba_signal     + eff_event[4]);
        oxytocin_signal = fmaxf(0.0f, oxytocin_signal + eff_event[5]);
        // 单次触发: 清零缓存 (plateau 型保留直到 duration 归零)
        if (h_event_duration_steps <= 0) {
            for (int i = 0; i < 6; ++i) h_event_signal[i] = 0.0f;
            h_event_pending_count = 0;
        } else {
            h_event_duration_steps -= mod_interval;  // 每次调用消耗一个调制间隔 (2026-08-05 参数化)
        }
    }

    // Phase 3a-B: 应用受体灵敏度 (有效注入信号 = 原始信号 × sensitivity)
    //   顺序: [DA, ACh, NE, 5HT, GABA, Oxy] 与 h_receptor_sensitivity 索引一致
    da_signal       *= h_receptor_sensitivity[0];
    ach_signal      *= h_receptor_sensitivity[1];
    ne_signal       *= h_receptor_sensitivity[2];
    ht5_signal      *= h_receptor_sensitivity[3];
    gaba_signal     *= h_receptor_sensitivity[4];
    oxytocin_signal *= h_receptor_sensitivity[5];

    // 衰减率: mod_interval 步 / tau (2026-08-05 参数化: 原硬编码 100 步)
    float da_decay       = expf(-(float)mod_interval / DA_TAU);
    float ach_decay      = expf(-(float)mod_interval / ACH_TAU);
    float ne_decay       = expf(-(float)mod_interval / NE_TAU);
    float ht5_decay      = expf(-(float)mod_interval / HT5_TAU);
    float gaba_decay     = expf(-(float)mod_interval / GABA_TAU);          // Phase 3a
    float oxytocin_decay = expf(-(float)mod_interval / OXYTOCIN_TAU);      // Phase 3a

    modulatory_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_da_concentration, b.d_ach_concentration,
        b.d_ne_concentration, b.d_ht5_concentration,
        b.d_gaba_concentration, b.d_oxytocin_concentration,    // Phase 3a
        N_TOTAL_NEURONS_2E,
        da_signal, ach_signal, ne_signal, ht5_signal,
        gaba_signal, oxytocin_signal,                          // Phase 3a
        da_decay, ach_decay, ne_decay, ht5_decay,
        gaba_decay, oxytocin_decay);                           // Phase 3a
}

void launch_da_value_function(MemoryAllocator* alloc, int step,
                              float reward, float* out_v_s, float* out_v_sp)
{
    PersistentBuffers& b = alloc->buffers();
    ensure_v_scratch();
    ensure_pred_succ_scratch();

    // 1. 计算亚柱发放直方图 (200维, 当前步 subcolumn_fr)
    int neurons_per_sc = N_TOTAL_NEURONS_2E / W_VALUE_DIM;
    int sc_blocks = (W_VALUE_DIM + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    subcolumn_fr_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_spike_flags, b.d_subcolumn_fr, N_TOTAL_NEURONS_2E, neurons_per_sc);

    // 2. 计算 V(s) = w_value · subcolumn_fr (当前状态价值)
    value_function_kernel<<<1, 1>>>(
        b.d_subcolumn_fr, b.d_w_value, d_v_scratch, W_VALUE_DIM);
    CUDA_CHECK_2E(cudaMemcpy(&h_v_s, d_v_scratch, sizeof(float), cudaMemcpyDeviceToHost));

    // 3. W_pred 完整矩阵预测: pred_fr[j] = Σ_k w_pred[j*dim+k] * fr_prev[k]
    //    使用上一步的 subcolumn_fr (d_subcol_fr_prev) 作为输入
    //    冷启动首步 fr_prev=0 → pred_fr=0, 由 cosine_similarity_kernel 防 0 除处理
    w_pred_predict_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_pred_fr, b.d_w_pred, b.d_subcol_fr_prev, W_VALUE_DIM);

    // 4. 计算预测成功率 (余弦相似度, Task 10)
    //    pred_succ = (cos(pred_fr, subcolumn_fr) + 1) / 2 ∈ [0, 1]
    //    pred_fr 基于上一步 fr 预测当前 fr, 与当前实际 fr 对比
    cosine_similarity_kernel<<<1, 1>>>(
        b.d_pred_fr, b.d_subcolumn_fr, d_pred_succ_scratch, W_VALUE_DIM);
    CUDA_CHECK_2E(cudaMemcpy(&h_pred_succ_cos, d_pred_succ_scratch,
                              sizeof(float), cudaMemcpyDeviceToHost));

    // 5. EMA 基线更新 (仅 EMA, pred_fr 已由步骤 3 计算)
    baseline_update_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_baseline_fr, b.d_subcolumn_fr, W_VALUE_DIM);

    // 6. 计算 V(s') = w_value · pred_fr (用预测的 fr 估计下一状态价值)
    value_function_kernel<<<1, 1>>>(
        b.d_pred_fr, b.d_w_value, d_v_scratch, W_VALUE_DIM);
    CUDA_CHECK_2E(cudaMemcpy(&h_v_sp, d_v_scratch, sizeof(float), cudaMemcpyDeviceToHost));

    // 7. TD error: δ = R + γ·V(s') - V(s)
    // Phase 3a-D1: 具身模式覆盖 reward (h_embodied_reward 非零时使用)
    float effective_reward = reward;
    if (fabsf(h_embodied_reward) > 1e-6f) {
        effective_reward = h_embodied_reward;
    }
    float delta = effective_reward + TD_GAMMA * h_v_sp - h_v_s;

    // 8. w_value TD 学习更新 (w_pred 更新已拆分至步骤 9)
    td_update_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_w_value, b.d_subcolumn_fr, delta, W_VALUE_DIM);

    // 9. W_pred 完整矩阵更新: w_pred[j*dim+k] += η_pred · (fr_j - pred_j) · fr_k_prev
    //    外积更新: 预测误差 (fr - pred) × 上一步输入 fr_prev
    w_pred_update_kernel<<<sc_blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_w_pred, b.d_subcolumn_fr, b.d_pred_fr, b.d_subcol_fr_prev, W_VALUE_DIM);

    // 10. 保存当前 subcolumn_fr 到 d_subcol_fr_prev (供下一步预测使用)
    CUDA_CHECK_2E(cudaMemcpy(b.d_subcol_fr_prev, b.d_subcolumn_fr,
                              W_VALUE_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

    if (out_v_s)  *out_v_s  = h_v_s;
    if (out_v_sp) *out_v_sp = h_v_sp;
}

void launch_byte_histogram(MemoryAllocator* alloc, uint8_t current_byte)
{
    PersistentBuffers& b = alloc->buffers();
    int blocks = (N_TOTAL_NEURONS_2E + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    byte_histogram_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        b.d_spike_flags, b.d_byte_histogram, N_TOTAL_NEURONS_2E, current_byte);
}

void get_byte_histogram(MemoryAllocator* alloc, int* out_hist)
{
    PersistentBuffers& b = alloc->buffers();
    cudaMemcpy(out_hist, b.d_byte_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
}

ModulatoryStats get_modulatory_stats(MemoryAllocator* alloc)
{
    PersistentBuffers& b = alloc->buffers();
    ModulatoryStats stats = {};

    // 采样前 1000 个神经元的调质浓度均值 (Phase 3a: 扩充到 6 维)
    const int sample = 1000;
    float h_da[sample], h_ach[sample], h_ne[sample], h_ht5[sample];
    float h_gaba[sample], h_oxytocin[sample];  // Phase 3a
    cudaMemcpy(h_da,  b.d_da_concentration,       sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ach, b.d_ach_concentration,      sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ne,  b.d_ne_concentration,       sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ht5, b.d_ht5_concentration,      sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gaba,     b.d_gaba_concentration,     sample * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oxytocin, b.d_oxytocin_concentration, sample * sizeof(float), cudaMemcpyDeviceToHost);

    double s_da = 0, s_ach = 0, s_ne = 0, s_ht5 = 0;
    double s_gaba = 0, s_oxy = 0;  // Phase 3a
    for (int i = 0; i < sample; ++i) {
        s_da  += h_da[i];
        s_ach += h_ach[i];
        s_ne  += h_ne[i];
        s_ht5 += h_ht5[i];
        s_gaba += h_gaba[i];       // Phase 3a
        s_oxy  += h_oxytocin[i];   // Phase 3a
    }
    stats.da_mean       = static_cast<float>(s_da  / sample);
    stats.ach_mean      = static_cast<float>(s_ach / sample);
    stats.ne_mean       = static_cast<float>(s_ne  / sample);
    stats.ht5_mean      = static_cast<float>(s_ht5 / sample);
    stats.gaba_mean     = static_cast<float>(s_gaba / sample);          // Phase 3a
    stats.oxytocin_mean = static_cast<float>(s_oxy  / sample);          // Phase 3a
    stats.v_s  = h_v_s;
    stats.v_sp = h_v_sp;
    // Task 10: 预测成功率 = 余弦相似度版本 (替代原二元判断)
    stats.pred_succ = h_pred_succ_cos;
    return stats;
}

// =============================================================================
// Phase 3a: AffectiveState readout — 6 维调质 → PAD 情感模型 + LLM 调制信号
// =============================================================================
// 详见 docs/snn-emotion-and-workspace-direction.md §3.2, §6.7
//
// PAD 情感模型映射 (Mehrabian 1996):
//   P (愉悦) = +DA - 0.5·5HT - 0.3·GABA        (DA 驱动愉悦, 5HT/GABA 抑制)
//   A (唤醒) = +NE - 0.4·GABA - 0.3·5HT        (NE 驱动唤醒, GABA/5HT 镇静)
//   D (主导) = +DA - 0.5·催产素                  (催产素高→更顺从/共情, 降低主导)
//
// LLM 生成调制信号:
//   temperature_delta: DA↑→+0.3 (更兴奋), 5HT↑→-0.3 (冷静), GABA↑→-0.1 (平静)
//   top_p_delta:       NE↑→-0.2 (更聚焦, 减少尾部词采样)
//   repetition_delta:  NE↑→+0.1 (高唤醒时避免重复跑题)
//   empathy_level:     催产素 / 2 → [0,1] (注入 system prompt)
//
// 置信度: 高 GABA/NE 表示系统处于不确定/警觉状态, 置信度降低
// =============================================================================
AffectiveState get_affective_state(MemoryAllocator* alloc, int step)
{
    ModulatoryStats stats = get_modulatory_stats(alloc);
    AffectiveState state;

    // 6 维调质快照
    state.dopamine      = stats.da_mean;
    state.serotonin     = stats.ht5_mean;
    state.norepinephrine = stats.ne_mean;
    state.acetylcholine = stats.ach_mean;
    state.gaba          = stats.gaba_mean;
    state.oxytocin      = stats.oxytocin_mean;

    // PAD 情感模型映射 (clamped to [-1, 1])
    state.pleasure  = state.dopamine - 0.5f * state.serotonin - 0.3f * state.gaba;
    state.arousal   = state.norepinephrine - 0.4f * state.gaba - 0.3f * state.serotonin;
    state.dominance = state.dopamine - 0.5f * state.oxytocin;
    if (state.pleasure  >  1.0f) state.pleasure  =  1.0f;
    if (state.pleasure  < -1.0f) state.pleasure  = -1.0f;
    if (state.arousal   >  1.0f) state.arousal   =  1.0f;
    if (state.arousal   < -1.0f) state.arousal   = -1.0f;
    if (state.dominance >  1.0f) state.dominance =  1.0f;
    if (state.dominance < -1.0f) state.dominance = -1.0f;

    // LLM 生成调制信号 (delta, 叠加到 LLM 默认参数上)
    state.temperature_delta =  0.3f * state.dopamine
                             - 0.3f * state.serotonin
                             - 0.1f * state.gaba;
    state.top_p_delta       = -0.2f * state.norepinephrine;
    state.repetition_delta  =  0.1f * state.norepinephrine;
    state.empathy_level     = state.oxytocin * 0.5f;  // [0, 1] (oxytocin ∈ [0, 2])
    if (state.empathy_level > 1.0f) state.empathy_level = 1.0f;
    if (state.empathy_level < 0.0f) state.empathy_level = 0.0f;

    // 元信息
    state.step      = step;
    // 置信度: GABA/NE 越高 → 系统越警觉/不确定 → 置信度越低
    state.confidence = 1.0f - (state.gaba + state.norepinephrine) * 0.25f;
    if (state.confidence > 1.0f) state.confidence = 1.0f;
    if (state.confidence < 0.0f) state.confidence = 0.0f;

    return state;
}

} // namespace stage2e
