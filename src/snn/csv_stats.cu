// =============================================================================
// CSV 采样 GPU 归约: 把每步诊断统计从"全量 D2H 拷贝 + CPU 遍历"
// 改为 GPU 单次遍历归约, 消除每步 ~137MB 的 DeviceToHost 流量
// (曾导致 N3F 训练 0.044s/步 → 0.118s/步 的额外开销)
// =============================================================================
#include "csv_stats.cuh"
#include "config.h"
#include <cuda_runtime.h>

#define CSV_STATS_THREADS 256
#define CSV_STATS_BLOCKS  256

namespace stage2e {

__device__ inline void atomicMinFloat(float* addr, float v) {
    unsigned int* a = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *a;
    while (true) {
        float old_f = __uint_as_float(old);
        float new_f = fminf(old_f, v);
        unsigned int new_v = __float_as_uint(new_f);
        unsigned int res = atomicCAS(a, old, new_v);
        if (res == old) break;
        old = res;
    }
}

__device__ inline void atomicMaxFloat(float* addr, float v) {
    unsigned int* a = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *a;
    while (true) {
        float old_f = __uint_as_float(old);
        float new_f = fmaxf(old_f, v);
        unsigned int new_v = __float_as_uint(new_f);
        unsigned int res = atomicCAS(a, old, new_v);
        if (res == old) break;
        old = res;
    }
}

// 单次遍历: sum + 非零计数, block 内树归约后 atomicAdd 到全局
__global__ void reduce_sum_count_kernel(const float* __restrict__ d, int n,
                                        float* out_sum, int* out_nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float s = 0.0f;
    int c = 0;
    for (int j = i; j < n; j += stride) {
        float v = d[j];
        s += v;
        if (v != 0.0f) ++c;
    }
    __shared__ float ss[CSV_STATS_THREADS];
    __shared__ int   sc[CSV_STATS_THREADS];
    ss[threadIdx.x] = s;
    sc[threadIdx.x] = c;
    __syncthreads();
    for (int k = CSV_STATS_THREADS / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            ss[threadIdx.x] += ss[threadIdx.x + k];
            sc[threadIdx.x] += sc[threadIdx.x + k];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, ss[0]);
        atomicAdd(out_nz, sc[0]);
    }
}

void device_sum_count(const float* d, int n, float* out_sum, int* out_nz)
{
    if (!d || n <= 0 || !out_sum || !out_nz) return;
    static float* s_d_sum = nullptr;
    static int*   s_d_nz  = nullptr;
    if (!s_d_sum) cudaMalloc(&s_d_sum, sizeof(float));
    if (!s_d_nz)  cudaMalloc(&s_d_nz, sizeof(int));
    cudaMemsetAsync(s_d_sum, 0, sizeof(float));
    cudaMemsetAsync(s_d_nz, 0, sizeof(int));
    reduce_sum_count_kernel<<<CSV_STATS_BLOCKS, CSV_STATS_THREADS>>>(d, n, s_d_sum, s_d_nz);
    cudaMemcpy(out_sum, s_d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_nz, s_d_nz, sizeof(int), cudaMemcpyDeviceToHost);
}

// 单次遍历: 权重 sum / abs_sum / min / max
// 采样窗口 [offset, offset+n) 环形滚动: 每步 offset += n, 长期覆盖全数组
__global__ void synapse_weight_stats_kernel(const BioSynapse* __restrict__ s, int n, int offset,
                                            float* out_sum, float* out_abs,
                                            float* out_min, float* out_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float ssum = 0.0f, absum = 0.0f, mn = 1e30f, mx = -1e30f;
    for (int j = i; j < n; j += stride) {
        int idx = j + offset;
        if (idx >= N_TOTAL_SYNAPSES_2E) idx -= N_TOTAL_SYNAPSES_2E;  // 环形回绕
        float w = s[idx].weight;
        ssum += w;
        absum += fabsf(w);
        mn = fminf(mn, w);
        mx = fmaxf(mx, w);
    }
    __shared__ float ss[CSV_STATS_THREADS];
    __shared__ float sa[CSV_STATS_THREADS];
    __shared__ float smn[CSV_STATS_THREADS];
    __shared__ float smx[CSV_STATS_THREADS];
    ss[threadIdx.x]  = ssum;
    sa[threadIdx.x]  = absum;
    smn[threadIdx.x] = mn;
    smx[threadIdx.x] = mx;
    __syncthreads();
    for (int k = CSV_STATS_THREADS / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) {
            ss[threadIdx.x]  += ss[threadIdx.x + k];
            sa[threadIdx.x]  += sa[threadIdx.x + k];
            smn[threadIdx.x]  = fminf(smn[threadIdx.x], smn[threadIdx.x + k]);
            smx[threadIdx.x]  = fmaxf(smx[threadIdx.x], smx[threadIdx.x + k]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, ss[0]);
        atomicAdd(out_abs, sa[0]);
        atomicMinFloat(out_min, smn[0]);
        atomicMaxFloat(out_max, smx[0]);
    }
}

void device_synapse_weight_stats(const BioSynapse* synapses, int n, int offset,
                                 float* mean_w, float* mean_abs_w,
                                 float* min_w, float* max_w)
{
    if (!synapses || n <= 0) return;
    if (n > N_TOTAL_SYNAPSES_2E) n = N_TOTAL_SYNAPSES_2E;
    // offset 归一化到 [0, N_TOTAL), 保证 [offset, offset+n) 环形窗口合法
    offset %= N_TOTAL_SYNAPSES_2E;
    if (offset < 0) offset += N_TOTAL_SYNAPSES_2E;
    static float* s_d_out = nullptr;
    if (!s_d_out) cudaMalloc(&s_d_out, 4 * sizeof(float));
    // out[0]=sum, out[1]=abs_sum, out[2]=min(哨兵), out[3]=max(哨兵)
    cudaMemsetAsync(s_d_out, 0, 2 * sizeof(float));
    float init[2] = {1e30f, -1e30f};
    cudaMemcpyAsync(s_d_out + 2, init, 2 * sizeof(float), cudaMemcpyHostToDevice);
    synapse_weight_stats_kernel<<<CSV_STATS_BLOCKS, CSV_STATS_THREADS>>>(
        synapses, n, offset, s_d_out, s_d_out + 1, s_d_out + 2, s_d_out + 3);
    float h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpy(h, s_d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    *mean_w      = static_cast<float>(h[0] / n);
    *mean_abs_w  = static_cast<float>(h[1] / n);
    *min_w       = (h[2] >= 1e29f) ? 0.0f : h[2];
    *max_w       = (h[3] <= -1e29f) ? 0.0f : h[3];
}

} // namespace stage2e
