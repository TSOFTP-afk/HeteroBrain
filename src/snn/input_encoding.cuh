#ifndef SNN_STAGE2E_INPUT_ENCODING_CUH
#define SNN_STAGE2E_INPUT_ENCODING_CUH

// =============================================================================
// Stage 2e 输入编码 (P1, §2.3 群体编码)
// =============================================================================
// 对应设计文档 §2.3: 每个输入字节激活 50 柱 × 50 神经元/柱 = 2500 个感觉神经元
//   - 哈希映射: 群体索引 = hash(byte, column_id) % N_per_column
//   - 柱间差异化: 同一字节在不同柱激活不同神经元子集
//   - 信息密度: 2500 个活跃神经元 / 8 (旧 one-hot) = 312× 提升
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "thalamic_gate.cuh"
#include <cstdint>

namespace stage2e {

// 把一个字节编码为群体编码输入, 累积到 d_input_current (在 delay_inject 之后调用)
// 每 INPUT_INJECT_INTERVAL 步注入一个新字节
// P1 阶段: 用伪字节流 (0..255 循环, 或 step % 256)
//
// 输入: byte (0..255)
// 输出: d_input_current[sensory_idx] += POP_CODING_GAIN * gain * gate
//       (50 柱 × 50 神经元/柱 = 2500 个感觉神经元被激活)
// d_gate_states: 丘脑门控状态数组 (设备指针), nullptr 时全开 (向后兼容)
//   注意: 不能简单强转为 float* 因为 ThalamicGateState 是 16B 结构体,
//   gate_signal 字段在 offset 0 但相邻柱的 gate_signal 间隔 16 字节而非 4 字节
void launch_input_inject(MemoryAllocator* alloc, uint8_t byte,
                         const ThalamicGateState* d_gate_states = nullptr);

// 计算当前步应注入的字节 (P1: 简单的 step % 256 循环)
// 后续阶段: 替换为真实文本语料 (DailyDialog / LCCC)
uint8_t get_byte_for_step(int step);

// 加载 UTF-8 文本语料 (LCCC 子集) 到全局缓冲
// 调用后 get_byte_for_step 将从文本流读取而非 step%256 循环
// 返回: 加载的字节数 (0 表示失败)
size_t load_text_corpus(const char* filepath);

// 检查文本语料是否已加载
bool is_text_loaded();

// 获取已加载文本的字节数
size_t text_corpus_size();

// 获取文本缓冲指定位置的字节 (越界返回 0)
uint8_t get_text_byte_at(size_t idx);

// 2026-08-05: 设置文本流注入间隔 (原编译宏 INPUT_INJECT_INTERVAL=3)
//   main.cpp 启动时调用 (默认 3; 长线剧本模式建议 1 使文本流与窗口事件同步)
void set_input_inject_interval(int interval);

// Checkpoint/resume uses the logical corpus cursor, not a raw host pointer.
size_t text_stream_position();
bool set_text_stream_position(size_t position);
uint64_t text_corpus_fingerprint();

// 把外部文本 (对话内容) 追加到文本流尾部 (2026-08-05, 引擎对话模式用)。
// 追加后 get_byte_for_step 循环读取时可能读到新内容 → WM/Hippo 记忆系统
// 会编码对话内容的神经签名 (SNN 真正"看见"对话)。
// 过滤语义与 load_text_corpus 一致: \r\n\t → 空格; 不影响已算好的指纹 (resume 后调用)。
// 返回追加后的文本流总字节数。
size_t append_text_stream(const char* bytes, size_t n);

// =============================================================================
// BPE Token Stream Mode (Task C1)
// =============================================================================
// 与字节模式并存: int32 BPE token (distilgpt2 vocab 0..50256) 注入 L4 层
// token_id % N_COLUMNS_2E 选择偏好柱, 偏好柱 gain_in, 其余 gain_out
// 所有 BPE 函数为新增, 不修改上方字节模式函数 (向后兼容)
// =============================================================================

// 加载 BPE token 流 (.bin, int32 数组) 到全局缓冲
// 调用后 get_token_for_step 将从 BPE 流读取而非 step%50257 循环
// 返回: 加载的 token 数 (0 表示失败)
size_t load_bpe_stream(const char* filepath);

// 检查 BPE 流是否已加载
bool is_bpe_loaded();

// 获取已加载 BPE 流的 token 数
size_t bpe_stream_size();

// 获取 BPE 流指定位置的 token (越界返回 0)
int32_t get_bpe_token_at(size_t idx);

// BPE 流游标 (checkpoint/resume, 独立于字节模式游标)
size_t bpe_stream_position();
bool set_bpe_stream_position(size_t position);
uint64_t bpe_stream_fingerprint();

// 计算当前步应注入的 BPE token (镜像 get_byte_for_step)
// 真实模式: 从 BPE 流循环读取 (每调用一次推进游标)
// 回退模式: 未加载时使用 step % 50257 (distilgpt2 vocab size)
int32_t get_token_for_step(int step);

// 把一个 BPE token 编码为群体编码输入, 累积到 d_input_current
// (在 delay_inject 之后调用, 镜像 launch_input_inject)
// 输入: token (0..50256)
// 输出: d_input_current[sensory_idx] += POP_CODING_GAIN * gain * gate
//       (N_COLUMNS_2E 柱 × POP_CODING_K_PER_COLUMN 神经元/柱 被激活)
// token_id % N_COLUMNS_2E 选择偏好柱 (gain_in), 其余柱 gain_out
// d_gate_states: 丘脑门控状态数组 (设备指针), nullptr 时全开 (向后兼容)
void launch_bpe_inject(MemoryAllocator* alloc, int32_t token,
                       const ThalamicGateState* d_gate_states = nullptr);

} // namespace stage2e

#endif // SNN_STAGE2E_INPUT_ENCODING_CUH
