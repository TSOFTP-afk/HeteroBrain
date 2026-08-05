// =============================================================================
// llama_backend.h — 进程内 llama.cpp 后端适配器 (实现 bridge::LlmBackend)
// =============================================================================
// 职责:
//   1. 加载 MiniCPM5 GGUF 模型 (llama_model_load_from_file)
//   2. 构建采样器链 (logit_bias + temp/top_k/top_p/min_p/penalties/dist)
//   3. 按 EmotionBridge 的调制信号动态重建采样器链 (llama_sampler 参数不可变,
//      故采用 重建整条链 的方式, 天然支持"逐轮情感调制")
//   4. 聊天式生成: chat template + 情感 system prompt + 多轮历史
// 链接: F:\hb_build 预编译静态库 (llama.lib + ggml*.lib + CUDA libs),
//       链接清单与 llama-simple-chat.exe 一致。
// logit_bias 通道 (2026-08-05 已实现): 情感词表 → llama_sampler_init_logit_bias
//   置于采样器链首, 生成期逐 token 对候选 logits 加性改写 (见 apply_logit_bias)。
// =============================================================================

#ifndef HETERO_BRAIN_LLM_LLAMA_BACKEND_H
#define HETERO_BRAIN_LLM_LLAMA_BACKEND_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bridge/emotion_types.h"
#include "bridge/llm_backend.h"

#include "llama.h"

struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;

namespace hb {
namespace llm {

// -----------------------------------------------------------------------------
// LlamaBackend — 进程内 llama.cpp 后端
// 线程安全: 调用方需保证单线程顺序访问 (生成期间不并发调制)
// -----------------------------------------------------------------------------
class LlamaBackend final : public bridge::LlmBackend {
public:
    struct Options {
        std::string model_path;         // GGUF 模型路径 (必填)
        int         n_ctx = 4096;       // 上下文窗口
        int         n_batch = 2048;     // 批大小 (多轮历史 + 长回复后 prompt 可能上千 token, 512 会触发 GGML_ASSERT)
        int         n_gpu_layers = 99;  // 卸载到 GPU 的层数
        std::uint32_t seed = 42;        // 采样随机种子
        int         max_new_tokens = 256;  // 单轮最大生成 token
    };

    explicit LlamaBackend(Options opt);
    ~LlamaBackend() override;

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    // 模型加载是否成功
    bool is_loaded() const { return model_ != nullptr; }

    // ---- bridge::LlmBackend 实现 ----
    int apply_sampler(const bridge::SamplerParams& params) override;
    int apply_emotion_prompt(const std::string& snippet) override;
    // 逐 token 情感偏置 (logit_bias 通道): word_bias {情感词, 偏置量} 在 backend
    // 内 tokenize (懒缓存词表), 重建采样器链并在链首插入 logit_bias 采样器。
    // 空数组 = 清除偏置。偏置在生成期作用于每个候选 token 的 logits (加性)。
    int apply_logit_bias(const std::vector<std::pair<std::string, float>>& word_bias) override;
    void on_user_turn(const std::string& user_text) override;
    void on_assistant_turn(const std::string& assistant_text) override;

    // ---- llama 特定扩展 ----
    // 执行一轮完整对话: user_text 作为用户输入, 生成助手回复写入 response。
    // 生成前自动注入: 情感 system prompt + 历史消息 + 当前用户消息 (chat template)。
    // 返回值: 0 成功; 负值错误码 (-1 未加载, -2 模板错误, -3 tokenize 失败,
    //   -4 上下文超限, -5 decode 失败, -6 token 解码失败)
    int chat(const std::string& user_text, std::string& response);

    // OpenAI 兼容 serve 模式: 直接用调用方提供的完整 messages 构建 prompt 生成,
    // 不读写内部 history_ (对话历史由客户端管理, 与第三方软件界面完全一致)。
    // messages 仅接受 "user"/"assistant" role, 其余 (如 "system") 忽略;
    // 情感 system prompt 仍由引擎注入在首条。返回值同 chat()。
    int chat_messages(const std::vector<std::pair<std::string, std::string>>& messages,
                      std::string& response);

    // 清空对话历史 (保留情感 prompt)
    void clear_history();

    // 预热: 在进入交互循环前强制完成一次 forward 解码, 触发 ggml-cuda 惰性初始化
    // (cuBLAS 句柄/工作区/图分配)。SNN 与 llama 共驻同一 GPU 时, 若首个 llama
    // decode 发生在 SNN 重活 (重放/评估) 之后, 首个 logits 会垃圾化 → 采样即 EOG
    // (空回复)。预热后首个"真实" decode 变为第二次调用, 规避该问题。
    // 返回值: 0 成功; 负值错误码 (同 chat)
    int warmup();

    const bridge::SamplerParams& sampler_params() const { return params_; }

private:
    void rebuild_sampler_chain();

    // 给定完整 ChatML prompt (已含 system + 历史 + assistant 头), 执行
    // tokenize + 分段 decode + 采样循环。返回 0 或负错误码 (同 chat)
    int generate(const std::string& prompt, std::string& response);

    Options               opt_;
    llama_model*          model_ = nullptr;
    llama_context*        ctx_   = nullptr;
    llama_sampler*        smpl_  = nullptr;
    const llama_vocab*    vocab_ = nullptr;
    bridge::SamplerParams params_;
    std::string           emotion_snippet_;   // 情感 system prompt 片段
    std::vector<std::pair<std::string, std::string>> history_;  // {role, content}

    // ---- logit_bias 通道 (2026-08-05) ----
    std::vector<llama_logit_bias> bias_;                        // 当前生效的 token 偏置
    std::unordered_map<std::string, std::vector<llama_token>> lexicon_cache_;  // 情感词→token 懒缓存
};

}  // namespace llm
}  // namespace hb

#endif  // HETERO_BRAIN_LLM_LLAMA_BACKEND_H
