// =============================================================================
// snn_llm_bridge.h — SNN 子系统与 LLM (llama.cpp / MiniCPM5-1B) 之间的桥接桩
// =============================================================================
// 说明:
//   本头文件为 header-only 桩实现, 提供 SNN 训练子系统与 LLM 子系统之间的
//   接口占位。当前所有函数返回错误码 (-1), 不调用任何 llama.cpp / ggml 函数,
//   不依赖 ggml.dll 或 llama.dll, 仅用于让依赖此接口的调用方能够编译通过。
//
//   Phase 3 T2H distillation will replace these stubs with llama.cpp calls.
//   (Phase 3 T2H 蒸馏阶段将把以下桩实现替换为真正的 llama.cpp 调用)
// =============================================================================

#ifndef SNN_LLM_BRIDGE_H
#define SNN_LLM_BRIDGE_H

#include <vector>
#include <string>
#include <cstdint>

// -----------------------------------------------------------------------------
// 桥接接口函数 (C 风格签名, 便于跨编译单元 / 跨语言边界调用)
// -----------------------------------------------------------------------------

// 文本 → token id 序列。
//   text        : 输入文本 (UTF-8, NUL 结尾)
//   token_ids   : 输出缓冲区, 由调用方分配, 容量 >= max_tokens
//   max_tokens  : token_ids 缓冲区最大容量 (int 个元素)
// 返回值: 实际写入的 token 数量; -1 表示错误或未实现。
//
// 注意: 当前为桩实现, 永远返回 -1。
// Phase 3 T2H distillation will replace these stubs with llama.cpp calls.
inline int snn_llm_tokenize(const char* text, int* token_ids, int max_tokens) {
    (void)text;
    (void)token_ids;
    (void)max_tokens;
    return -1;
}

// token id 序列 → 文本。
//   token_ids   : 输入 token id 数组
//   n_tokens    : token_ids 中的元素个数
//   text        : 输出缓冲区, 由调用方分配, 容量 >= max_text_len 字节
//   max_text_len: text 缓冲区最大容量 (字节)
// 返回值: 实际写入的文本长度 (字节, 不含 NUL); -1 表示错误或未实现。
//
// 注意: 当前为桩实现, 永远返回 -1。
// Phase 3 T2H distillation will replace these stubs with llama.cpp calls.
inline int snn_llm_detokenize(const int* token_ids, int n_tokens, char* text, int max_text_len) {
    (void)token_ids;
    (void)n_tokens;
    (void)text;
    (void)max_text_len;
    return -1;
}

// token id 序列 → embedding 向量。
//   token_ids : 输入 token id 数组
//   n_tokens  : token_ids 中的元素个数
//   embeddings: 输出缓冲区, 由调用方分配, 容量 >= n_tokens * embed_dim 个 float
//   embed_dim : 单个 token 的 embedding 维度
// 返回值: 0 表示成功; -1 表示错误或未实现。
//
// 注意: 当前为桩实现, 永远返回 -1。
// Phase 3 T2H distillation will replace these stubs with llama.cpp calls.
inline int snn_llm_embed(const int* token_ids, int n_tokens, float* embeddings, int embed_dim) {
    (void)token_ids;
    (void)n_tokens;
    (void)embeddings;
    (void)embed_dim;
    return -1;
}

#endif  // SNN_LLM_BRIDGE_H
