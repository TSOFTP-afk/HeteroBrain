// =============================================================================
// Stage 2e 认知工作台 (Cognitive Workbench) kernel 实现 (Phase 3b)
// =============================================================================
// 设计要点:
//   - wb_write_kernel: 单 block, 256 线程 (每线程一槽)
//       阶段1: 每线程算槽位 cosine 相似度 (空槽 → 0, 可被覆盖)
//       阶段2: thread 0 串行归约找 max_sim + argmax
//       阶段3: thread 0 决策 (新颖 → 写入未保护槽 / 刷新已有)
//   - wb_maintain_kernel: 单 block, 256 线程, activation *= WB_DECAY
//   - wb_read_head_kernel: 单 block, 256 线程
//       阶段1: 每线程算 score = activation (空槽为 0)
//       阶段2: thread 0 串行 softmax → d_read_attn
//       阶段3: thread 0 注意力加权重建签名 → d_read_signal (LLM 导出)
//       阶段4: 每线程对注意力 > 阈值 的槽位, 绑定前额叶组 (slot % 50),
//             内联 PCA 反投影注入 d_prefrontal_input
//
// 前额叶布局 (与 WM 一致):
//   全局神经元索引 [0, 50000) = 联合皮层, [50000, 55000) = 前额叶 (50 组 × 100)
//   d_prefrontal_input 索引 [0, 5000) 对应前额叶 [50000, 55000)
//   slot i 绑定组 (i % PREFRONTAL_GROUPS)
// =============================================================================

#include "cognitive_workbench.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

namespace stage2e {

// =============================================================================
// 工作台写入 kernel (新颖检测 + LRU 替换 + 情感印记 + 类型标签 + 写保护)
// =============================================================================
//   cosine(sig, pat) = (sig·pat) / (|sig|·|pat|)
//   空槽 (tag == UNUSED) → cosine = 0 (可被新颖模式覆盖)
//   决策:
//     - signature 为零 (网络静默期) → 跳过
//     - max_sim < novelty_threshold: 新颖 → 从写头游标起找首个 unprotection 槽,
//       写入签名 + 情感印记, tag/confidence/timestamp 更新, activation=1.0
//     - 否则: 刷新匹配槽 (activation=1.0, timestamp=current_step)
// =============================================================================
__global__ void wb_write_kernel(
    WorkbenchSlot* __restrict__ d_slots,
    const float* __restrict__ d_signature,
    const float* __restrict__ d_emotion,
    int* __restrict__ d_write_cursor,
    int current_step,
    int n_slots,
    float novelty_threshold,
    SlotTag tag)
{
    const int tid = threadIdx.x;

    __shared__ float s_sim[WB_CAPACITY];    // 每槽 cosine 相似度
    __shared__ float s_sig_norm_sq;         // signature 范数平方
    __shared__ float s_max_sim;             // 最大相似度
    __shared__ int   s_max_idx;             // 最大相似度槽位

    // ---------- 阶段1a: signature 范数平方 (thread 0, 广播) ----------
    if (tid == 0) {
        float norm_sq = 0.0f;
        for (int k = 0; k < WB_SIGNATURE_DIM; ++k) {
            norm_sq += d_signature[k] * d_signature[k];
        }
        s_sig_norm_sq = norm_sq;
        s_max_sim = -2.0f;
        s_max_idx = 0;
    }
    __syncthreads();
    const float sig_norm_sq = s_sig_norm_sq;

    // ---------- 阶段1b: 每线程算一个槽位的 cosine 相似度 ----------
    float my_sim = -2.0f;
    if (tid < n_slots) {
        const WorkbenchSlot& slot = d_slots[tid];
        if (slot.tag == (uint8_t)SlotTag::UNUSED) {
            // 空槽: cosine = 0, 可被新颖模式覆盖
            my_sim = 0.0f;
        } else if (sig_norm_sq > 1e-20f) {
            float dot = 0.0f;
            float pat_norm_sq = 0.0f;
            for (int k = 0; k < WB_SIGNATURE_DIM; ++k) {
                float p = slot.signature[k];
                dot += d_signature[k] * p;
                pat_norm_sq += p * p;
            }
            if (pat_norm_sq > 1e-20f) {
                my_sim = dot / (sqrtf(sig_norm_sq) * sqrtf(pat_norm_sq));
            } else {
                my_sim = 0.0f;
            }
        }
        s_sim[tid] = my_sim;
    }
    __syncthreads();

    // ---------- 阶段2: thread 0 串行归约找最大相似度 + 索引 ----------
    if (tid == 0) {
        float max_sim = -2.0f;
        int max_idx = 0;
        for (int i = 0; i < n_slots; ++i) {
            if (s_sim[i] > max_sim) {
                max_sim = s_sim[i];
                max_idx = i;
            }
        }
        s_max_sim = max_sim;
        s_max_idx = max_idx;
    }
    __syncthreads();

    // ---------- 阶段3: 决策 (thread 0 执行写入) ----------
    if (tid == 0) {
        if (sig_norm_sq <= 1e-20f) {
            return;  // 网络静默期不写入工作台
        }

        if (s_max_sim < novelty_threshold) {
            // 新颖: 从写头游标起找首个未保护槽 (写保护防 LRU 覆盖关键槽)
            int cursor = atomicAdd(d_write_cursor, 1);
            int target = -1;
            for (int probe = 0; probe < n_slots; ++probe) {
                int idx = (cursor + probe) % n_slots;
                if (d_slots[idx].protection == 0) {
                    target = idx;
                    break;
                }
            }
            if (target < 0) {
                return;  // 全部槽受保护, 放弃本次写入
            }
            WorkbenchSlot& slot = d_slots[target];
            for (int k = 0; k < WB_SIGNATURE_DIM; ++k) {
                slot.signature[k] = d_signature[k];
            }
            for (int m = 0; m < WB_EMOTION_DIM; ++m) {
                slot.emotion[m] = d_emotion[m];   // 情感印记快照
            }
            slot.confidence = 1.0f;
            slot.activation = 1.0f;
            slot.timestamp  = current_step;
            slot.tag        = (uint8_t)tag;
            slot.protection = 0;
            // 双模态: SNN 写槽时清空文本 (LRU 覆盖可能残留旧 LLM 文本)
            for (int t = 0; t < WB_TEXT_CAPACITY; ++t) {
                slot.text[t] = 0;
            }
        } else {
            // 已有模式: 刷新匹配槽 (重置 activation, 更新时间戳)
            WorkbenchSlot& slot = d_slots[s_max_idx];
            slot.activation = 1.0f;
            slot.timestamp  = current_step;
        }
    }
}

// =============================================================================
// 工作台维持 kernel (每步指数衰减)
// =============================================================================
__global__ void wb_maintain_kernel(
    WorkbenchSlot* __restrict__ d_slots,
    int n_slots,
    float decay_factor)
{
    const int tid = threadIdx.x;
    if (tid >= n_slots) return;
    d_slots[tid].activation *= decay_factor;
}

// =============================================================================
// 工作台读头 kernel (softmax 注意力 + 重建签名 + PCA 反投影注入)
// =============================================================================
//   score_i = activation_i (空槽 tag==UNUSED 或 activation<=0 → 0)
//   attn_i = score_i / Σ_j score_j        (softmax, 读头注意力)
//   read_signal[k] = Σ_i attn_i · sig_i[k] (注意力加权重建, LLM 导出)
//   注入: 对 attn > 1e-3 的槽 i, 绑定前额叶组 (i % 50),
//         recon[j] = mean_fr[base+j] + Σ_k sig_i[k]·W[base+j][k]
//         inject += recon[j] × attn_i × inject_gain
// =============================================================================
__global__ void wb_read_head_kernel(
    const WorkbenchSlot* __restrict__ d_slots,
    const float* __restrict__ d_pca_W,
    const float* __restrict__ d_mean_fr,
    float* __restrict__ d_read_attn,
    float* __restrict__ d_read_signal,
    float* __restrict__ d_prefrontal_input,
    int n_slots,
    int n_prefrontal,
    int group_size,
    float inject_gain,
    int n_pca_components,
    int n_neurons)
{
    const int tid = threadIdx.x;

    __shared__ float s_attn[WB_CAPACITY];   // softmax 注意力

    // ---------- 阶段1: 每线程算槽位 score ----------
    float score = 0.0f;
    if (tid < n_slots) {
        const WorkbenchSlot& slot = d_slots[tid];
        if (slot.tag != (uint8_t)SlotTag::UNUSED && slot.activation > 0.0f) {
            score = slot.activation;
        }
    }
    s_attn[tid] = (tid < n_slots) ? score : 0.0f;
    __syncthreads();

    // ---------- 阶段2: softmax (thread 0 串行) → d_read_attn ----------
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n_slots; ++i) sum += s_attn[i];
        if (sum > 1e-20f) {
            for (int i = 0; i < n_slots; ++i) {
                s_attn[i] /= sum;
                d_read_attn[i] = s_attn[i];
            }
        } else {
            for (int i = 0; i < n_slots; ++i) {
                s_attn[i] = 0.0f;
                d_read_attn[i] = 0.0f;
            }
        }
    }
    __syncthreads();

    // ---------- 阶段3: 注意力加权重建签名 → d_read_signal (thread 0) ----------
    if (tid == 0) {
        for (int k = 0; k < WB_SIGNATURE_DIM; ++k) {
            float acc = 0.0f;
            for (int i = 0; i < n_slots; ++i) {
                acc += s_attn[i] * d_slots[i].signature[k];
            }
            d_read_signal[k] = acc;
        }
    }
    __syncthreads();

    // ---------- 阶段4: 注意力 > 阈值的槽位 PCA 反投影注入前额叶 ----------
    if (tid < n_slots) {
        const float a = s_attn[tid];
        if (a > 1e-3f) {
            const WorkbenchSlot& slot = d_slots[tid];
            const int group = tid % PREFRONTAL_GROUPS;
            const int pf_start = n_neurons - n_prefrontal;
            const int group_offset = group * group_size;

            // 预加载签名到寄存器 (避免内层 j 循环重复读 global memory)
            float pat[WB_SIGNATURE_DIM];
            for (int k = 0; k < WB_SIGNATURE_DIM; ++k) {
                pat[k] = slot.signature[k];
            }

            for (int j = 0; j < group_size; ++j) {
                const int neuron_idx = pf_start + group_offset + j;
                if (neuron_idx >= n_neurons) break;

                // PCA 反投影: recon = mean_fr + Σ_k pat[k]·W[neuron_idx][k]
                float recon = d_mean_fr[neuron_idx];
                const size_t w_base = (size_t)neuron_idx * n_pca_components;
                #pragma unroll 8
                for (int k = 0; k < n_pca_components; ++k) {
                    recon += pat[k] * d_pca_W[w_base + k];
                }

                // 注入电流 = recon × attn × gain
                float current = recon * a * inject_gain;
                atomicAdd(&d_prefrontal_input[group_offset + j], current);
            }
        }
    }
}

// =============================================================================
// Host 端 wrapper 函数
// =============================================================================

void launch_wb_write(WorkbenchSlot* d_slots, const float* d_sig,
                     const float* d_emotion, int* d_cursor, int step,
                     int n, float thr, SlotTag tag)
{
    if (!d_slots || !d_sig) return;
    wb_write_kernel<<<1, n>>>(d_slots, d_sig, d_emotion, d_cursor, step,
                              n, thr, tag);
    CUDA_CHECK_LAST_2E();
}

void launch_wb_maintain(WorkbenchSlot* d_slots, int n, float decay)
{
    if (!d_slots) return;
    wb_maintain_kernel<<<1, n>>>(d_slots, n, decay);
    CUDA_CHECK_LAST_2E();
}

void launch_wb_read_head(const WorkbenchSlot* d_slots,
                         const float* d_pca_W, const float* d_mean_fr,
                         float* d_read_attn, float* d_read_signal,
                         float* d_prefrontal_input, int n_slots, int n_pf,
                         int group_size, float inject_gain, int n_comp, int n_neurons)
{
    if (!d_slots || !d_pca_W || !d_mean_fr || !d_prefrontal_input) return;
    wb_read_head_kernel<<<1, n_slots>>>(
        d_slots, d_pca_W, d_mean_fr, d_read_attn, d_read_signal,
        d_prefrontal_input, n_slots, n_pf, group_size, inject_gain,
        n_comp, n_neurons);
    CUDA_CHECK_LAST_2E();
}

// -----------------------------------------------------------------------------
// 工作台前额叶注入合并 (2026-08-07)
// 与 wm merge 同构: 把 d_wb_input (工作台重建的前额叶发放率) 累加到
// d_input_current 的前额叶神经元区间 (全局索引 pf_start..pf_start+n_pf).
// 必须在 delay_inject (清零 input_current) 之后、lif_adex 之前调用.
// 缩放: 读头注入 recon×attn×WB_INJECT_GAIN ~ 0.003-0.17, 需乘增益驱动前额叶,
//   折中取与 WM 一致的 20 倍 (避免过强导致前额叶持续爆发).
// -----------------------------------------------------------------------------
#define WB_PREFRONTAL_MERGE_GAIN 20.0f

__global__ void merge_wb_prefrontal_input_kernel(
    const float* __restrict__ d_wb_input,     // [n_pf] 工作台重建的前额叶发放率
    float* __restrict__ d_input_current,      // [N] 主网络输入电流 (延迟注入后)
    int n_pf,                                 // N_PREFRONTAL_NEURONS = 5000
    int pf_start)                             // N_ASSOCIATION_NEURONS_2E = 50000
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pf) return;
    float v = d_wb_input[i];
    // NaN 防护: 重建若产生 NaN, 跳过注入 (NaN>0 恒 false, 显式防一下)
    if (v > 0.0f && v == v) {
        atomicAdd(&d_input_current[pf_start + i], v * WB_PREFRONTAL_MERGE_GAIN);
    }
}

void launch_merge_wb_prefrontal_input(const float* d_wb_input, float* d_input_current,
                                      int n_pf, int pf_start)
{
    if (!d_wb_input || !d_input_current) return;
    int blocks = (n_pf + THREADS_PER_BLOCK_2E - 1) / THREADS_PER_BLOCK_2E;
    merge_wb_prefrontal_input_kernel<<<blocks, THREADS_PER_BLOCK_2E>>>(
        d_wb_input, d_input_current, n_pf, pf_start);
    CUDA_CHECK_LAST_2E();
}

std::vector<WorkbenchSlot> read_wb_slots(const WorkbenchSlot* d_slots,
                                          int n_slots, int max_count) {
    std::vector<WorkbenchSlot> out;
    if (!d_slots || n_slots <= 0) {
        return out;
    }
    std::vector<WorkbenchSlot> h((size_t)n_slots);
    if (cudaMemcpy(h.data(), d_slots, (size_t)n_slots * sizeof(WorkbenchSlot),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return out;
    }
    // 按 activation 降序, 取前 max_count 条 (活跃记忆优先)
    std::sort(h.begin(), h.end(),
              [](const WorkbenchSlot& a, const WorkbenchSlot& b) {
                  return a.activation > b.activation;
              });
    if (max_count > 0 && (int)h.size() > max_count) {
        h.resize((size_t)max_count);
    }
    return out = std::move(h);
}

// =============================================================================
// LLM 工具协议实现 (Phase 3b 双模态)
// =============================================================================

// 单线程 kernel: 把 host 文本写入工作台指定槽位 (text + tag + protection + 时间戳)
__global__ void wb_text_write_kernel(
    WorkbenchSlot* __restrict__ d_slots,
    int idx,
    const char* __restrict__ d_text,
    int text_len,
    uint8_t tag,
    uint8_t protect,
    int step)
{
    if (threadIdx.x != 0) return;
    if (idx < 0) return;
    WorkbenchSlot& slot = d_slots[idx];
    for (int t = 0; t < WB_TEXT_CAPACITY; ++t) {
        slot.text[t] = (t < text_len) ? d_text[t] : 0;
    }
    slot.tag        = tag;
    slot.protection = protect;
    slot.timestamp  = step;
    slot.confidence = 1.0f;
    slot.activation = 1.0f;
}

int launch_wb_text_write(WorkbenchSlot* d_slots, int idx,
                         const char* host_text, int text_len,
                         uint8_t tag, uint8_t protect, int step) {
    if (!d_slots || idx < 0) {
        return -1;
    }
    if (text_len < 0) {
        text_len = 0;
    }
    if (text_len > WB_TEXT_CAPACITY) {
        text_len = WB_TEXT_CAPACITY;   // 截断超长文本
    }
    // 拷文本到 device (可空文本 → 长度 0)
    char* d_text = nullptr;
    if (text_len > 0) {
        if (cudaMalloc(&d_text, (size_t)text_len) != cudaSuccess) {
            return -1;
        }
        if (cudaMemcpy(d_text, host_text, (size_t)text_len, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_text);
            return -1;
        }
    }
    wb_text_write_kernel<<<1, 1>>>(d_slots, idx, d_text, text_len, tag, protect, step);
    if (d_text) {
        cudaFree(d_text);
    }
    CUDA_CHECK_LAST_2E();
    return idx;
}

std::vector<WorkbenchSlot> search_wb_text(const WorkbenchSlot* d_slots,
                                           int n_slots, const std::string& query,
                                           int max_hits) {
    std::vector<WorkbenchSlot> out;
    if (!d_slots || n_slots <= 0) {
        return out;
    }
    std::vector<WorkbenchSlot> h((size_t)n_slots);
    if (cudaMemcpy(h.data(), d_slots, (size_t)n_slots * sizeof(WorkbenchSlot),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return out;
    }
    // 查询转小写 (大小写不敏感子串匹配)
    std::string q = query;
    for (auto& c : q) c = (char)std::tolower((unsigned char)c);
    std::vector<WorkbenchSlot> hits;
    for (auto& s : h) {
        if (s.tag == (uint8_t)SlotTag::UNUSED) {
            continue;
        }
        if (!q.empty()) {
            std::string t(s.text, s.text + WB_TEXT_CAPACITY);
            // 截至首 0 字节
            const size_t z = t.find('\0');
            if (z != std::string::npos) t.resize(z);
            std::string tl = t;
            for (auto& c : tl) c = (char)std::tolower((unsigned char)c);
            if (tl.find(q) == std::string::npos) {
                continue;
            }
        }
        hits.push_back(s);
    }
    // 激活降序
    std::sort(hits.begin(), hits.end(),
              [](const WorkbenchSlot& a, const WorkbenchSlot& b) {
                  return a.activation > b.activation;
              });
    if (max_hits > 0 && (int)hits.size() > max_hits) {
        hits.resize((size_t)max_hits);
    }
    return hits;
}

int find_wb_write_slot(const WorkbenchSlot* d_slots, int n_slots) {
    if (!d_slots || n_slots <= 0) {
        return -1;
    }
    std::vector<WorkbenchSlot> h((size_t)n_slots);
    if (cudaMemcpy(h.data(), d_slots, (size_t)n_slots * sizeof(WorkbenchSlot),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1;
    }
    // 优先空槽 (tag==UNUSED): 直接复用
    for (int i = 0; i < n_slots; ++i) {
        if (h[(size_t)i].tag == (uint8_t)SlotTag::UNUSED) {
            return i;
        }
    }
    // 无空槽 → 未保护 (protection==0) 中 activation 最低者 (LRU 替换)
    int best = -1;
    float min_act = 1e30f;
    for (int i = 0; i < n_slots; ++i) {
        const WorkbenchSlot& s = h[(size_t)i];
        if (s.protection != 0) {
            continue;
        }
        if (s.activation < min_act) {
            min_act = s.activation;
            best = i;
        }
    }
    return best;   // 全受保护 → -1 (不可写)
}

} // namespace stage2e