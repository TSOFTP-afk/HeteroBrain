// =============================================================================
// Phase 3a-F (M1): 杏仁核情感学习核心 — CUDA 实现 (2026-08-06 生物拟真 spec)
// =============================================================================
// 见 amygdala_kernels.cuh 头文件设计说明。
// 流水线 (每 SNN 步):
//   amygdala_event_inject (事件 dispatch 时, 非每步)
//     → amygdala_la_update (LA 积分+发放)
//     → amygdala_ba_accum (LA spike → BA 输入)
//     → amygdala_ba_update (BA 积分+发放)
//     → amygdala_trace + amygdala_stdp_ltp/ltd (LA→BA 权重学习)
// =============================================================================

#include "amygdala_kernels.cuh"

#include <cuda_runtime.h>
#include <vector>

namespace stage2e {

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// 事件注入: 事件类型 → LA 对应组注入电流 (gain 随强度缩放)
__global__ void amygdala_event_inject_kernel(float* __restrict__ input_la,
                                             int group_start, int group_size,
                                             float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= group_size) return;
    input_la[group_start + idx] += gain;
}

// LA 更新: 泄漏积分 + 发放判定 (简单 LIF)
__global__ void amygdala_la_update_kernel(float* __restrict__ v_la,
                                          bool* __restrict__ spike_la,
                                          float* __restrict__ input_la,
                                          int n_la) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_la) return;
    float v = v_la[i] * (1.0f - AMYGDALA_LEAK) + input_la[i];
    input_la[i] = 0.0f;  // 单次注入
    if (v >= AMYGDALA_THRESHOLD) {
        spike_la[i] = true;
        v_la[i] = AMYGDALA_RESET;
    } else {
        spike_la[i] = false;
        v_la[i] = v;
    }
}

// BA 输入累积: 每个 BA 神经元遍历所有 LA, 累加 spike_la[i] * W[i][j]
__global__ void amygdala_ba_accum_kernel(float* __restrict__ input_ba,
                                         const bool* __restrict__ spike_la,
                                         const float* __restrict__ W,
                                         int n_la, int n_ba) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_ba) return;
    float acc = 0.0f;
    for (int i = 0; i < n_la; ++i) {
        if (spike_la[i]) acc += W[(size_t)i * n_ba + j];
    }
    input_ba[j] += acc;
}

// BA 更新: 泄漏积分 + 发放判定
__global__ void amygdala_ba_update_kernel(float* __restrict__ v_ba,
                                          bool* __restrict__ spike_ba,
                                          float* __restrict__ input_ba,
                                          int n_ba) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_ba) return;
    float v = v_ba[j] * (1.0f - AMYGDALA_LEAK) + input_ba[j];
    input_ba[j] = 0.0f;
    if (v >= AMYGDALA_THRESHOLD) {
        spike_ba[j] = true;
        v_ba[j] = AMYGDALA_RESET;
    } else {
        spike_ba[j] = false;
        v_ba[j] = v;
    }
}

// STDP traces: 发放后置 1, 否则指数衰减 (近因窗口)
__global__ void amygdala_trace_la_kernel(float* __restrict__ trace_la,
                                         const bool* __restrict__ spike_la,
                                         int n_la) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_la) return;
    trace_la[i] = trace_la[i] * 0.95f + (spike_la[i] ? 1.0f : 0.0f);
}

__global__ void amygdala_trace_ba_kernel(float* __restrict__ trace_ba,
                                         const bool* __restrict__ spike_ba,
                                         int n_ba) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_ba) return;
    trace_ba[j] = trace_ba[j] * 0.95f + (spike_ba[j] ? 1.0f : 0.0f);
}

// LTP: BA 发放 → W[i][j] += A_plus * trace_la[i] (pre 近期活动增强)
__global__ void amygdala_stdp_ltp_kernel(float* __restrict__ W,
                                         const bool* __restrict__ spike_ba,
                                         const float* __restrict__ trace_la,
                                         int n_la, int n_ba) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_la * n_ba;
    if (idx >= total) return;
    int i = idx / n_ba;
    int j = idx % n_ba;
    if (spike_ba[j]) {
        float w = W[idx] + AMYGDALA_STDP_A_PLUS * trace_la[i];
        if (w > AMYGDALA_W_MAX) w = AMYGDALA_W_MAX;
        W[idx] = w;
    }
}

// LTD: LA 发放 → W[i][j] -= A_minus * trace_ba[j] (post 近期活动削弱)
__global__ void amygdala_stdp_ltd_kernel(float* __restrict__ W,
                                         const bool* __restrict__ spike_la,
                                         const float* __restrict__ trace_ba,
                                         int n_la, int n_ba) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_la * n_ba;
    if (idx >= total) return;
    int i = idx / n_ba;
    int j = idx % n_ba;
    if (spike_la[i]) {
        float w = W[idx] - AMYGDALA_STDP_A_MINUS * trace_ba[j];
        if (w < 0.0f) w = 0.0f;
        W[idx] = w;
    }
}

// 初始化权重: W_INIT 随机小值 + 正负事件组→正负 BA 组偏置
__global__ void amygdala_init_kernel(float* __restrict__ W,
                                     int n_la, int n_ba, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_la * n_ba;
    if (idx >= total) return;
    int i = idx / n_ba;
    int j = idx % n_ba;
    // 确定性 xorshift (与 network_init 风格一致)
    uint32_t x = seed ^ (uint32_t)(idx * 2654435761u + 0x9E3779B9u);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    float r = (float)(x & 0xFFFFFFu) / 16777216.0f;
    float w = AMYGDALA_W_INIT * (0.5f + r);
    int group = i / amygdala_event_group_size();
    if (group < EVT_COUNT) {
        if (amygdala_event_is_negative(group)) {
            if (j < AMYGDALA_BA_NEG_GROUP) w += AMYGDALA_NEG_BIAS;
        } else {
            if (j >= AMYGDALA_BA_NEG_GROUP) w += AMYGDALA_POS_BIAS;
        }
    }
    if (w > AMYGDALA_W_MAX) w = AMYGDALA_W_MAX;
    W[idx] = w;
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

// 事件注入挂起状态 (host 端, 单事件持续 AMYGDALA_INJECT_DURATION 步的电流注入)
//   生物学: 感觉事件持续数百 ms 而非单步脉冲; LA 在注入窗口内累积积分,
//   发放率随强度升高 (强事件更快过阈值、窗口内发放更多次), 而非"是否发放"二值
static float h_amyg_pending_gain[EVT_COUNT] = {0.0f};
static int   h_amyg_pending_steps[EVT_COUNT] = {0};

void launch_amygdala_event_inject(MemoryAllocator* alloc, int event_type, float intensity) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_amyg_input_la || event_type < 0 || event_type >= EVT_COUNT) return;
    int gs = amygdala_event_group_size();
    int start = event_type * gs;
    // 强度归一化 [0,50] → [0.5, 1.5] 电流增益
    float mag = intensity < 0 ? -intensity : intensity;
    if (mag > 50.0f) mag = 50.0f;
    float gain = AMYGDALA_INJECT_GAIN * (0.5f + 0.5f * mag / 50.0f);
    // 挂起注入: 接下来 AMYGDALA_INJECT_DURATION 步由 launch_amygdala_forward 注入
    //   (同类型事件在窗口内叠加时后事件覆盖前事件, 强度语义取最后事件)
    h_amyg_pending_gain[event_type] = gain;
    h_amyg_pending_steps[event_type] = AMYGDALA_INJECT_DURATION;
}

void launch_amygdala_forward(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_amyg_v_la) return;
    const int n_la = N_AMYGDALA_LA;
    const int n_ba = N_AMYGDALA_BA;
    const int gs = amygdala_event_group_size();

    // 挂起事件电流注入 (每步一次, 步数递减)
    for (int e = 0; e < EVT_COUNT; ++e) {
        if (h_amyg_pending_steps[e] > 0) {
            amygdala_event_inject_kernel<<<1, 64>>>(
                b.d_amyg_input_la, e * gs, gs, h_amyg_pending_gain[e]);
            h_amyg_pending_steps[e]--;
        }
    }

    amygdala_la_update_kernel<<<(n_la + 63) / 64, 64>>>(
        b.d_amyg_v_la, b.d_amyg_spike_la, b.d_amyg_input_la, n_la);
    amygdala_ba_accum_kernel<<<(n_ba + 63) / 64, 64>>>(
        b.d_amyg_input_ba, b.d_amyg_spike_la, b.d_amyg_w_la_ba, n_la, n_ba);
    amygdala_ba_update_kernel<<<(n_ba + 63) / 64, 64>>>(
        b.d_amyg_v_ba, b.d_amyg_spike_ba, b.d_amyg_input_ba, n_ba);
}

void launch_amygdala_stdp(MemoryAllocator* alloc) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_amyg_w_la_ba) return;
    const int n_la = N_AMYGDALA_LA;
    const int n_ba = N_AMYGDALA_BA;
    const int total = n_la * n_ba;

    amygdala_trace_la_kernel<<<(n_la + 63) / 64, 64>>>(
        b.d_amyg_trace_la, b.d_amyg_spike_la, n_la);
    amygdala_trace_ba_kernel<<<(n_ba + 63) / 64, 64>>>(
        b.d_amyg_trace_ba, b.d_amyg_spike_ba, n_ba);
    amygdala_stdp_ltp_kernel<<<(total + 255) / 256, 256>>>(
        b.d_amyg_w_la_ba, b.d_amyg_spike_ba, b.d_amyg_trace_la, n_la, n_ba);
    amygdala_stdp_ltd_kernel<<<(total + 255) / 256, 256>>>(
        b.d_amyg_w_la_ba, b.d_amyg_spike_la, b.d_amyg_trace_ba, n_la, n_ba);
}

void read_amygdala_ba_output(MemoryAllocator* alloc, float* out_neg, float* out_pos) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_amyg_spike_ba) { if (out_neg) *out_neg = 0.0f; if (out_pos) *out_pos = 0.0f; return; }
    const int n_ba = N_AMYGDALA_BA;
    // 注意: 不能用 std::vector<bool> (bit 压缩特化, 无 .data(), 且语义为 bit 数组)
    // 设备端 bool 是 1 字节, 用 unsigned char 中转等宽拷贝
    std::vector<unsigned char> h_spike(n_ba);
    cudaMemcpy(h_spike.data(), b.d_amyg_spike_ba, n_ba * sizeof(bool),
               cudaMemcpyDeviceToHost);
    int neg_cnt = 0, pos_cnt = 0;
    for (int j = 0; j < AMYGDALA_BA_NEG_GROUP; ++j) if (h_spike[j]) ++neg_cnt;
    for (int j = AMYGDALA_BA_NEG_GROUP; j < n_ba; ++j) if (h_spike[j]) ++pos_cnt;
    if (out_neg) *out_neg = (float)neg_cnt / (float)AMYGDALA_BA_NEG_GROUP;
    if (out_pos) *out_pos = (float)pos_cnt / (float)(n_ba - AMYGDALA_BA_NEG_GROUP);
}

void init_amygdala(MemoryAllocator* alloc, uint32_t seed) {
    if (!alloc) return;
    PersistentBuffers& b = alloc->buffers();
    if (!b.d_amyg_w_la_ba) return;
    const int n_la = N_AMYGDALA_LA;
    const int n_ba = N_AMYGDALA_BA;
    const int total = n_la * n_ba;

    CUDA_CHECK_2E(cudaMemset(b.d_amyg_v_la, 0, n_la * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_amyg_v_ba, 0, n_ba * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_amyg_input_la, 0, n_la * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_amyg_input_ba, 0, n_ba * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_amyg_trace_la, 0, n_la * sizeof(float)));
    CUDA_CHECK_2E(cudaMemset(b.d_amyg_trace_ba, 0, n_ba * sizeof(float)));
    amygdala_init_kernel<<<(total + 255) / 256, 256>>>(
        b.d_amyg_w_la_ba, n_la, n_ba, seed);
}

} // namespace stage2e
