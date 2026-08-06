#ifndef SNN_STAGE2E_COGNITIVE_WORKBENCH_CUH
#define SNN_STAGE2E_COGNITIVE_WORKBENCH_CUH

// =============================================================================
// Stage 2e 认知工作台 (Cognitive Workbench) kernel 声明 (Phase 3b)
// =============================================================================
// 生物学: 前额叶工作记忆外的"外部草稿纸" (Clark & Chalmers 扩展心智) —
//   SNN 受控的可读写工作区, 替代原 50 槽 WM 成为"认知工作空间" Layer 2。
// 与 WM 的关键区别:
//   - 容量 256 (WB_CAPACITY) vs 50
//   - 逐 slot 读写擦 (读写头) vs 整体 LRU 刷新
//   - 带类型标签 (SlotTag) + 情感印记 (emotion[6]) + 时间戳 + 写保护
// 闭环流程:
//   1. wb_write_kernel (每 WB_WRITE_INTERVAL 步):
//      - 余弦相似度匹配当前 PCA 签名与 256 槽
//      - max_sim < WB_NOVELTY_THRESHOLD → 新颖, 写入写头游标
//        (跳过保护槽), 记录情感印记 + 时间戳 + 类型标签
//      - 否则 → 刷新匹配槽 (activation=1.0, timestamp=step)
//   2. wb_maintain_kernel (每步):
//      - activation *= WB_DECAY (指数遗忘, 模拟短期记忆消退)
//   3. wb_read_head_kernel (每步):
//      - 对槽位 activation 做 softmax 得到读头注意力 (d_read_attn)
//      - 注意力加权重建签名 → d_read_signal (供 LLM 导出)
//      - 活跃槽位 PCA 反投影注入 d_wb_prefrontal_input (WB_INJECT_GAIN)
// 缓冲区 (memory_allocator PersistentBuffers 分配):
//   - d_wb_slots [256 × WorkbenchSlot]        工作台槽位 (63.5KB)
//   - d_wb_write_cursor [1]                    写头 LRU 游标
//   - d_wb_read_attn [256]                     读头注意力权重
//   - d_wb_read_signal [50]                    读头重建签名 (LLM 导出)
//   - d_wb_prefrontal_input [5000]             工作台注入前额叶缓冲 (20KB)
// =============================================================================

#include "config.h"
#include "types.h"
#include <cuda_runtime.h>
#include <vector>

namespace stage2e {

// -----------------------------------------------------------------------------
// CUDA kernel 声明
// -----------------------------------------------------------------------------

// 工作台写入 (新颖检测 + LRU 替换, 含情感印记/时间戳/类型标签/写保护)
//   启动: <<<1, n_slots>>>, n_slots = WB_CAPACITY = 256 (单 block, 每线程一槽)
//   决策:
//     - signature 为零 (网络静默期) → 跳过
//     - max_sim < novelty_threshold → 新颖, 从游标起找首个未保护槽写入
//     - 否则 → 刷新匹配槽 (activation=1.0, timestamp=current_step)
__global__ void wb_write_kernel(
    WorkbenchSlot* __restrict__ d_slots,        // [n_slots]
    const float* __restrict__ d_signature,       // [WB_SIGNATURE_DIM] PCA 签名 (L2 归一化)
    const float* __restrict__ d_emotion,         // [WB_EMOTION_DIM] 当前 6 维调质快照
    int* __restrict__ d_write_cursor,            // [1] 写头 LRU 游标
    int current_step,                            // 当前步 (写入 timestamp)
    int n_slots,                                 // 槽位数 (= WB_CAPACITY = 256)
    float novelty_threshold,                     // (= WB_NOVELTY_THRESHOLD = 0.7)
    SlotTag tag);                                // 本次写入的类型标签

// 工作台维持 (每步指数衰减, 无注入)
//   启动: <<<1, n_slots>>>
//   activation *= decay_factor (= WB_DECAY = 0.995)
__global__ void wb_maintain_kernel(
    WorkbenchSlot* __restrict__ d_slots,        // [n_slots]
    int n_slots,                                 // 槽位数
    float decay_factor);                         // 衰减因子

// 工作台读头 (softmax 注意力 + PCA 反投影注入前额叶)
//   启动: <<<1, n_slots>>>, 单 block
//   阶段1: 每线程算槽位 score = activation (空槽/无效槽为 0)
//   阶段2: thread 0 串行 softmax → d_read_attn
//   阶段3: thread 0 注意力加权重建签名 → d_read_signal
//   阶段4: 每线程对注意力 > 阈值 的槽位, 绑定前额叶组 (slot % 50) 做 PCA 反投影注入
__global__ void wb_read_head_kernel(
    const WorkbenchSlot* __restrict__ d_slots,  // [n_slots]
    const float* __restrict__ d_pca_W,           // [n_neurons × n_pca_components] PCA 基 (row-major)
    const float* __restrict__ d_mean_fr,         // [n_neurons] 滑动平均发放率
    float* __restrict__ d_read_attn,             // [n_slots] 读头注意力权重
    float* __restrict__ d_read_signal,           // [WB_SIGNATURE_DIM] 重建签名
    float* __restrict__ d_prefrontal_input,      // [n_prefrontal] 注入缓冲 (需事先清零)
    int n_slots,                                 // 256
    int n_prefrontal,                            // 5000
    int group_size,                              // 100
    float inject_gain,                           // (= WB_INJECT_GAIN = 0.5)
    int n_pca_components,                        // 50
    int n_neurons);                              // 联合皮层+前额叶 = 55000

// -----------------------------------------------------------------------------
// Host 端 wrapper 函数 (接受裸指针, 由 scheduler 集成调用)
// -----------------------------------------------------------------------------

// 工作台写入 (每 WB_WRITE_INTERVAL 步)
//   n = WB_CAPACITY, thr = WB_NOVELTY_THRESHOLD
void launch_wb_write(WorkbenchSlot* d_slots, const float* d_sig,
                     const float* d_emotion, int* d_cursor, int step,
                     int n, float thr, SlotTag tag);

// 工作台维持 (每步衰减)
void launch_wb_maintain(WorkbenchSlot* d_slots, int n, float decay);

// 工作台读头 (每步: 注意力 + 重建 + 注入)
//   d_prefrontal_input 需在调用前清零 (scheduler 统一负责)
void launch_wb_read_head(const WorkbenchSlot* d_slots,
                         const float* d_pca_W, const float* d_mean_fr,
                         float* d_read_attn, float* d_read_signal,
                         float* d_prefrontal_input, int n_slots, int n_pf,
                         int group_size, float inject_gain, int n_comp, int n_neurons);

// 工作台前额叶注入合并: 把 d_wb_input 并入 d_input_current 前额叶区间
//   必须在 delay_inject (清零 input_current) 之后、lif_adex 之前调用
void launch_merge_wb_prefrontal_input(const float* d_wb_input, float* d_input_current,
                                      int n_pf, int pf_start);

// 读工作台槽位 (host 端, 供引擎把工作台内容拼进 LLM system prompt)
// 拷回 n_slots 个槽, 按 activation 降序, 返回最多 max_count 条
std::vector<WorkbenchSlot> read_wb_slots(const WorkbenchSlot* d_slots, int n_slots, int max_count);

// -----------------------------------------------------------------------------
// LLM 工具协议 (Phase 3b 双模态, 2026-08-07)
//   引擎层把工作台作为 LLM 可读写的认知工具: LLM 经 write/read/search 干预工作台。
//   双模态: LLM 写文本+类型标签; SNN 写 PCA 签名+情感印记, 两者共存于同槽。
// -----------------------------------------------------------------------------

// LLM 写入工作台单个槽位: 覆盖 text + tag + protection + 时间戳, 激活置满。
//   d_slots: 工作台槽位; idx: 目标槽 (0..n_slots-1); host_text: UTF-8 文本 (host 内存);
//   text_len: 文本字节数; 返回实际写入槽号 (idx 越界则 -1)。
int launch_wb_text_write(WorkbenchSlot* d_slots, int idx,
                         const char* host_text, int text_len,
                         uint8_t tag, uint8_t protect, int step);

// LLM 搜索工作台: host 端在拷贝的槽位文本上做大小写不敏感子串匹配,
//   返回激活降序且命中的槽位 (最多 max_hits 条)。query 为空则等同于读全部活跃槽。
std::vector<WorkbenchSlot> search_wb_text(const WorkbenchSlot* d_slots, int n_slots,
                                           const std::string& query, int max_hits);

// 找一个可写槽的物理索引 (LLM write 工具用): 优先空槽 (tag==UNUSED),
//   否则未保护 (protection==0) 中 activation 最低者 (LRU 替换); 全受保护则 -1。
int find_wb_write_slot(const WorkbenchSlot* d_slots, int n_slots);

} // namespace stage2e

#endif // SNN_STAGE2E_COGNITIVE_WORKBENCH_CUH