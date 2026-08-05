// =============================================================================
// llama_backend.cpp — 进程内 llama.cpp 后端适配器实现
// =============================================================================
// 核心逻辑参考 F:\hb_llama\examples\simple-chat\simple-chat.cpp (已验证可跑通
// MiniCPM5-1B-Q4_K_M.gguf)。关键差异:
//   1. 采样器链可动态重建 (apply_sampler 按 EmotionBridge 调制信号重建)
//   2. 情感 system prompt 注入 (每轮生成前作为首条 system 消息)
//   3. 多轮历史由本类维护, 调用方只需调 chat()
// =============================================================================

#include "llama_backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

namespace hb {
namespace llm {

namespace {

// 只输出 llama 错误日志 (避免刷屏)
void quiet_log(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fprintf(stderr, "%s", text);
    }
}

// 角色规则: 1B 模型对隐式角色分不清"你/我", 必须在 system 明示
// (实测无规则时会把情感片段"当前心境"当成用户的心境来描述, 或复读 system)。
// 2026-08-05: 追加项目背景 (thetrueai 异构引擎身份);
// 思考模式保留: 不含"不要输出思考过程"类句子 (解码层已做 </think> EOG 豁免)。
const char* kRoleRule =
    "你是温柔体贴的中文 AI 助手。请牢记: 对话中“我”指用户本人, "
    "“你”指你自己(助手)。始终以助手的视角回答用户的问题, "
    "不要把自己当成用户。\n"
    "项目背景: 你由 thetrueai 异构引擎驱动——脉冲神经网络(SNN)情感核心"
    "持续演算你的情绪状态(愉悦/唤醒/主导/共情), Qwen 大语言模型负责文字表达, "
    "你的语气与措辞会实时反映当前情绪基调。请直接给出回答。";

}  // namespace

LlamaBackend::LlamaBackend(Options opt) : opt_(std::move(opt)) {
    if (opt_.model_path.empty()) {
        std::fprintf(stderr, "LlamaBackend: empty model path\n");
        return;
    }

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();

    // ---- 加载模型 ----
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = opt_.n_gpu_layers;
    model_ = llama_model_load_from_file(opt_.model_path.c_str(), mp);
    if (!model_) {
        std::fprintf(stderr, "LlamaBackend: failed to load model: %s\n", opt_.model_path.c_str());
        return;
    }
    vocab_ = llama_model_get_vocab(model_);

    // ---- 初始化上下文 ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = opt_.n_ctx;
    cp.n_batch = opt_.n_batch;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        std::fprintf(stderr, "LlamaBackend: failed to init context\n");
        llama_model_free(model_);
        model_ = nullptr;
        return;
    }

    // ---- 初始化采样器链 (默认参数) ----
    params_ = bridge::SamplerParams{};
    rebuild_sampler_chain();
}

LlamaBackend::~LlamaBackend() {
    if (smpl_)  llama_sampler_free(smpl_);
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

void LlamaBackend::rebuild_sampler_chain() {
    // llama_sampler 创建后参数不可变 → 情感调制时整链重建 (成本可忽略)
    llama_sampler* next = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // logit_bias 置于链首: 直接改写原始 logits (与官方 sampling.cpp 一致,
    // llama_sampler_logit_bias_apply 对候选 token 做 logit += bias),
    // 后续 temp/top_k/top_p/penalties 再加工偏置后的分布
    if (!bias_.empty()) {
        llama_sampler_chain_add(next, llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(vocab_), static_cast<int>(bias_.size()), bias_.data()));
    }
    llama_sampler_chain_add(next, llama_sampler_init_min_p(params_.min_p, 1));
    llama_sampler_chain_add(next, llama_sampler_init_temp(params_.temperature));
    llama_sampler_chain_add(next, llama_sampler_init_top_k(params_.top_k));
    llama_sampler_chain_add(next, llama_sampler_init_top_p(params_.top_p, 1));
    // penalties 置于 top-k/top-p 之后 (llama.h 注释: 避免全词表重复扫描)
    llama_sampler_chain_add(next, llama_sampler_init_penalties(-1, params_.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(next, llama_sampler_init_dist(opt_.seed));
    if (smpl_) {
        llama_sampler_free(smpl_);
    }
    smpl_ = next;
}

int LlamaBackend::apply_sampler(const bridge::SamplerParams& params) {
    if (!is_loaded()) {
        return -1;
    }
    params_ = params;
    rebuild_sampler_chain();
    return 0;
}

int LlamaBackend::apply_emotion_prompt(const std::string& snippet) {
    if (!is_loaded()) {
        return -1;
    }
    emotion_snippet_ = snippet;
    return 0;
}

int LlamaBackend::apply_logit_bias(
    const std::vector<std::pair<std::string, float>>& word_bias) {
    if (!is_loaded()) {
        return -1;
    }
    bias_.clear();
    if (!word_bias.empty()) {
        for (const auto& wb : word_bias) {
            // 懒缓存: 同一情感词只 tokenize 一次 (Qwen3 BPE 词表, 中文情感词
            // 通常拆成 1 个 token; 多 token 时逐 token 施加相同偏置)
            std::vector<llama_token>* toks = nullptr;
            const auto it = lexicon_cache_.find(wb.first);
            if (it != lexicon_cache_.end()) {
                toks = &it->second;
            } else {
                const int n = -llama_tokenize(vocab_, wb.first.c_str(),
                                              static_cast<int>(wb.first.size()),
                                              nullptr, 0, false, false);
                if (n <= 0) {
                    continue;  // 词表查不到该词 (非常见), 跳过
                }
                std::vector<llama_token> t(static_cast<size_t>(n));
                if (llama_tokenize(vocab_, wb.first.c_str(),
                                   static_cast<int>(wb.first.size()),
                                   t.data(), t.size(), false, false) < 0) {
                    continue;
                }
                toks = &lexicon_cache_.emplace(wb.first, std::move(t)).first->second;
            }
            for (const llama_token tok : *toks) {
                bias_.push_back(llama_logit_bias{tok, wb.second});
            }
        }
    }
    // 偏置数组变化 → 重建采样器链 (logit_bias 采样器参数在创建时固化)
    rebuild_sampler_chain();
    return 0;
}

void LlamaBackend::on_user_turn(const std::string& text) {
    if (!text.empty()) {
        history_.emplace_back("user", text);
    }
}

void LlamaBackend::on_assistant_turn(const std::string& text) {
    if (!text.empty()) {
        history_.emplace_back("assistant", text);
    }
}

void LlamaBackend::clear_history() {
    history_.clear();
}

int LlamaBackend::warmup() {
    if (!is_loaded()) {
        return -1;
    }
    llama_memory_clear(llama_get_memory(ctx_), true);
    // 极短 prompt: 仅 assistant 头, 触发完整图构建 + 采样数步
    const char* kPrompt = "<|im_start|>assistant\n";
    const int n = -llama_tokenize(vocab_, kPrompt, std::strlen(kPrompt), nullptr, 0, true, true);
    if (n <= 0) {
        return -3;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    if (llama_tokenize(vocab_, kPrompt, std::strlen(kPrompt),
                       tokens.data(), tokens.size(), true, true) < 0) {
        return -3;
    }
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx_, batch) != 0) {
        return -5;
    }
    // 采样并续解码数步: 触发 ggml-cuda 惰性初始化 + sampler 链首次使用。
    // SNN 与 llama 共驻同一 GPU 时, 首个 llama sample/decode 若发生在 SNN
    // 重活之后, 首个 logits 会垃圾化 → 采样即 EOG (空回复, 60K 首轮复现)。
    // 预热先于任何 SNN 步进执行, 把"首个使用"消耗在干净状态。
    for (int i = 0; i < 6; ++i) {
        llama_token id = llama_sampler_sample(smpl_, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, id)) {
            break;
        }
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx_, batch) != 0) {
            return -5;
        }
    }
    return 0;
}

int LlamaBackend::chat(const std::string& user_text, std::string& response) {
    if (!is_loaded()) {
        return -1;
    }

    // 清空 KV cache: 每次全量重建 prompt (情感 system + 历史 + 当前 user)。
    // 若不清空, 上一轮残留 token 会与新一轮 prompt 混叠, 模型上下文错乱
    // (实测表现为固定短语重复输出)。代价: 多轮对话每轮重算完整历史,
    // 对 1B 模型可接受; 后续可优化为 simple-chat 的增量 decode 机制。
    llama_memory_clear(llama_get_memory(ctx_), true);

    // ---- 1. 拼接 ChatML prompt ----
    // 不能使用 llama_chat_apply_template: 该 llama.cpp 版本的
    // LLM_CHAT_TEMPLATE_MINICPM 是 MiniCPM-3B-OpenHermes 的 <用户>/<AI> 格式,
    // 与 ChatML 不符 (实测模型收到陌生格式后输出重复文本),
    // 且该版本无 jinja 解析器。故按 ChatML 语义手写拼接:
    //   <|im_start|>system\n{角色规则 + snippet}<|im_end|>\n (可选)
    //   <|im_start|>user\n{u}<|im_end|>\n  × N
    //   <|im_start|>assistant\n (generation prompt)
    // 2026-08-05 修复: 引擎经 process_turn → on_user_turn 已把当前 user 追加进
    // history_, 若 chat() 再拼一次/追加一次会造成 user 消息重复 (prompt 畸形,
    // 模型可能立即输出 EOG 导致空回复)。此处检测"末尾已是同文本 user"去重。
    const bool user_in_history = !history_.empty() &&
                                 history_.back().first == "user" &&
                                 history_.back().second == user_text;

    std::string prompt;
    if (!emotion_snippet_.empty()) {
        prompt += "<|im_start|>system\n" + std::string(kRoleRule) + "\n" +
                  emotion_snippet_ + "<|im_end|>\n";
    } else {
        prompt += "<|im_start|>system\n" + std::string(kRoleRule) + "<|im_end|>\n";
    }
    for (const auto& h : history_) {
        prompt += "<|im_start|>" + h.first + "\n" + h.second + "<|im_end|>\n";
    }
    if (!user_in_history) {
        prompt += "<|im_start|>user\n" + user_text + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";

    if (!user_in_history) {
        history_.emplace_back("user", user_text);   // 未由 process_turn 预加时补记
    }
    const int rc = generate(prompt, response);
    if (rc != 0) {
        return rc;
    }
    // assistant 回合: 若已由 on_assistant_turn 追加 (process_turn 链路) 则不重复
    if (history_.empty() || history_.back().first != "assistant" ||
        history_.back().second != response) {
        history_.emplace_back("assistant", response);
    }
    return 0;
}

int LlamaBackend::chat_messages(
    const std::vector<std::pair<std::string, std::string>>& messages,
    std::string& response) {
    if (!is_loaded()) {
        return -1;
    }
    // serve 模式: 对话历史由客户端全量提供, 内部 history_ 不参与。
    llama_memory_clear(llama_get_memory(ctx_), true);

    std::string prompt;
    if (!emotion_snippet_.empty()) {
        prompt += "<|im_start|>system\n" + std::string(kRoleRule) + "\n" +
                  emotion_snippet_ + "<|im_end|>\n";
    } else {
        prompt += "<|im_start|>system\n" + std::string(kRoleRule) + "<|im_end|>\n";
    }
    for (const auto& m : messages) {
        if (m.first != "user" && m.first != "assistant") {
            continue;
        }
        if (m.second.empty()) {
            continue;
        }
        prompt += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return generate(prompt, response);
}

int LlamaBackend::generate(const std::string& prompt, std::string& response) {
    // ---- tokenize ----
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;
    const int n_prompt = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                                         nullptr, 0, is_first, true);
    if (n_prompt <= 0) {
        return -3;
    }
    std::vector<llama_token> tokens(static_cast<size_t>(n_prompt));
    if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                       tokens.data(), tokens.size(), is_first, true) < 0) {
        return -3;
    }

    // ---- 分段 decode prompt (每段 ≤ n_batch): 多轮历史 + 长回复后 prompt token
    // 数可能超过 n_batch, 一次整批 decode 会触发 GGML_ASSERT(n_tokens_all <=
    // n_batch) 崩溃 (2026-08-05 实测第二轮历史回放时崩溃)。
    {
        int pos = 0;
        while (pos < n_prompt) {
            const int n_tok = std::min(opt_.n_batch, n_prompt - pos);
            llama_batch b = llama_batch_get_one(tokens.data() + pos, n_tok);
            if (llama_decode(ctx_, b) != 0) {
                return -5;
            }
            pos += n_tok;
        }
    }

    // ---- decode + sample 循环 ----
    response.clear();
    int n_gen = 0;
    int first_sampled = -1;
    int bias_hits = 0;       // 采样命中偏置 token 的次数 (logit_bias 生效证据)
    llama_token feed = 0;          // 每步采样结果, 作为下一步 decode 输入
    bool need_decode = false;      // prompt 已 decode, 首次 sample 前无需再 decode
    while (n_gen < opt_.max_new_tokens) {
        const int n_ctx = llama_n_ctx(ctx_);
        const int n_used = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;
        if (n_used + 1 > n_ctx) {
            return -4;  // 上下文超限
        }
        if (need_decode) {
            llama_batch b = llama_batch_get_one(&feed, 1);
            if (llama_decode(ctx_, b) != 0) {
                return -5;
            }
        }
        llama_token id = llama_sampler_sample(smpl_, ctx_, -1);
        if (n_gen == 0) {
            first_sampled = id;
        }
        if (!bias_.empty()) {
            for (const auto& b : bias_) {
                if (b.token == id) {
                    ++bias_hits;
                    break;
                }
            }
        }
        if (llama_vocab_is_eog(vocab_, id)) {
            // Qwen3 思考模型: `</think>` 也是 EOG, 但 think 块之后还有正式回答。
            // 豁免 `</think>` (消费该 token 后继续生成), 直到真正的 <|im_end|>,
            // 否则回复只剩思考过程、没有最终答案 (2026-08-05 实测)。
            char tbuf[64];
            const int tn = llama_token_to_piece(vocab_, id, tbuf, sizeof(tbuf), 0, true);
            if (tn > 0 && std::string(tbuf, tn).find("think") != std::string::npos) {
                feed = id;
                need_decode = true;
                ++n_gen;
                continue;
            }
            if (n_gen == 0) {
                std::fprintf(stderr, "[llm] first sample is EOG: id=%d n_prompt=%d\n",
                             id, n_prompt);
            }
            break;
        }
        char buf[256];
        const int n = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            return -6;
        }
        response.append(buf, static_cast<size_t>(n));
        feed = id;
        need_decode = true;
        ++n_gen;
    }
    if (response.empty() && first_sampled >= 0) {
        std::fprintf(stderr, "[llm] empty response, first token id=%d (n_prompt=%d)\n",
                     first_sampled, n_prompt);
    }
    if (!bias_.empty()) {
        std::fprintf(stderr, "[llm] logit_bias: %zu token 偏置, 采样命中 %d/%d\n",
                     bias_.size(), bias_hits, n_gen);
    }
    return 0;
}

}  // namespace llm
}  // namespace hb
