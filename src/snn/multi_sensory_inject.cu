// src/snn/multi_sensory_inject.cu
#include "multi_sensory_inject.cuh"
#include <cuda_runtime.h>

namespace stage2e {

// 每 thread 处理一个柱 (50柱)
__global__ void multi_sensory_inject_kernel(
    const float* __restrict__ sensory_signals,  // [50]
    float* __restrict__ input_current,           // d_input_current
    const ThalamicGateState* __restrict__ gate_states,
    int n_columns,
    int neurons_per_column,
    int l4_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;

    float s = sensory_signals[col];
    if (s < 0.01f) return;  // 信号太弱, 跳过

    float gate = gate_states[col].gate_signal;
    float effective = s * gate;
    if (effective < 0.01f) return;

    // 在柱内 L4 层激活 K 个神经元
    int K = (int)(100.0f * effective);  // 最多100个
    if (K < 1) K = 1;

    int sensory_base = col * neurons_per_column;  // L4在柱首

    // xorshift32 哈希选择K个神经元 (与 input_encoding.cu 一致)
    unsigned int state = (unsigned int)(col * 2654435761u + 12345u);
    for (int k = 0; k < K; ++k) {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        int neuron_offset = (int)(state % (unsigned int)l4_size);
        int neuron_idx = sensory_base + neuron_offset;
        // 2026-08-01 修复: 感知注入增益 POP_CODING_GAIN → SENSORY_INJECT_GAIN
        //   根因: 感知注入修复生效后, 每环境步用 POP_CODING_GAIN=80 驱动 50 柱×100
        //   = 5000 个 L4 神经元, 对网络过强 → L4 活动剧增 (spikes 700→1234) →
        //   PCA Oja 输入分布突变 → W_norm 15→5.5e8 发散 → WM 签名异常 → 前额叶仍不激活.
        //   修复: 感知信号本质是"弱引导" (沙盒 v1 极简: 内感态向量), 用独立温和增益,
        //   维持感知可分辨又不破坏 PCA 统计稳定性.
        atomicAdd(&input_current[neuron_idx], SENSORY_INJECT_GAIN * effective);
    }
}

void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states)
{
    // 50柱, 1 block × 50 threads
    float* d_sensory = nullptr;
    cudaMalloc(&d_sensory, 50 * sizeof(float));
    cudaMemcpy(d_sensory, sensory, 50 * sizeof(float), cudaMemcpyHostToDevice);

    multi_sensory_inject_kernel<<<1, 50>>>(
        d_sensory,
        d_input_current,
        d_gate_states,
        N_COLUMNS_2E,          // 50
        NEURONS_PER_COLUMN_2E, // 1000
        COL_L4_SIZE_2E         // 200
    );

    cudaFree(d_sensory);
}

// =============================================================================
// Phase 3a-G (A): 事件→联合皮层直通注入
// =============================================================================

// 事件类型 k → 联合皮层子区域 [start, start+region_size) 均匀注入电流
//   简单均匀注入 (无哈希/无门控): 事件是"情感语义锚点", 整个子区域被激活,
//   与文本流的柱级群体编码形成可区分的发放模式
__global__ void event_cortex_inject_kernel(
    float* __restrict__ input_current,
    int start, int region_size, float gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= region_size) return;
    input_current[start + i] += gain;
}

void launch_event_cortex_inject(int event_type, float gain, float* d_input_current) {
    if (!d_input_current || event_type < 0 || event_type >= EVT_COUNT) return;
    const int region = EVENT_CORTEX_REGION_SIZE;
    const int start = event_type * region;
    event_cortex_inject_kernel<<<(region + 255) / 256, 256>>>(
        d_input_current, start, region, gain);
}

} // namespace stage2e
