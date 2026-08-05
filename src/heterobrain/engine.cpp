// =============================================================================
// engine.cpp — 异构引擎实现
// =============================================================================
// SNN 初始化流程复用 snn_train (main.cpp) 的关键路径:
//   cudaSetDevice → MemoryAllocator → (resume 跳过 init_network) → scheduler
//   → load_checkpoint → 主循环 (scheduler.step + 情感读取 + LLM 对话)
// =============================================================================

#include "engine.h"

#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "scheduler.cuh"            // stage2e::BioMechanismScheduler
#include "memory_allocator.cuh"     // stage2e::MemoryAllocator
#include "modulatory_kernels.cuh"   // get_affective_state / set_event_signal 等
#include "input_encoding.cuh"       // load_text_corpus (resume 指纹校验) / append_text_stream
#include "wm_kernels.cuh"           // read_wm_slots (工作记忆读出)
#include "hippocampal_kernels.cuh"  // read_hippo_memories (海马记忆读出)

#include "http_server.h"            // OpenAI 兼容 serve 模式
#include "mini_json.h"

namespace hb {
namespace engine {

// -----------------------------------------------------------------------------
// SNN 内部状态 (不透明, 头文件不暴露 CUDA/调度器依赖)
// -----------------------------------------------------------------------------
struct EmotionEngine::SnnState {
    stage2e::MemoryAllocator* allocator = nullptr;
    stage2e::BioMechanismScheduler* scheduler = nullptr;
};

// -----------------------------------------------------------------------------
// SNN 反馈端: 把 SNN host setter 包装为 bridge::SnnFeedbackSink
// (引擎的 LLM→SNN 回流落点; 引擎之外也可替换为其它实现)
// -----------------------------------------------------------------------------
namespace {

class SnnSink final : public bridge::SnnFeedbackSink {
public:
    void emit_empathy(float level) override {
        stage2e::set_empathy_signal(level);
    }
    void emit_event(const float modulator_delta[6], int duration_steps) override {
        stage2e::set_event_signal(modulator_delta, duration_steps);
    }
    void emit_embodied_reward(float reward) override {
        stage2e::set_embodied_reward(reward);
    }
};

// AffectiveState (SNN 读出) → EmotionState (桥接层契约), 字段一一对应
bridge::EmotionState to_emotion_state(const stage2e::AffectiveState& a) {
    bridge::EmotionState e;
    e.dopamine       = a.dopamine;
    e.serotonin      = a.serotonin;
    e.norepinephrine = a.norepinephrine;
    e.acetylcholine  = a.acetylcholine;
    e.gaba           = a.gaba;
    e.oxytocin       = a.oxytocin;
    e.pleasure       = a.pleasure;
    e.arousal        = a.arousal;
    e.dominance      = a.dominance;
    e.temperature_delta = a.temperature_delta;
    e.top_p_delta       = a.top_p_delta;
    e.repetition_delta  = a.repetition_delta;
    e.empathy_level     = a.empathy_level;
    e.step              = a.step;
    e.confidence        = a.confidence;
    return e;
}

// OpenAI 风格错误响应体
std::string error_json(const std::string& code, const std::string& msg) {
    json::Value e = json::Value::make_obj();
    e.obj.emplace_back("message", json::Value(msg));
    e.obj.emplace_back("type", json::Value("invalid_request_error"));
    e.obj.emplace_back("code", json::Value(code));
    json::Value root = json::Value::make_obj();
    root.obj.emplace_back("error", std::move(e));
    return json::dump(root);
}

// 消融实验: 冻结情感文字时使用的中性基调 (对照基线, 不随 SNN 状态变化)
const char* kNeutralEmotionPrompt =
    "你的情绪基调：平静、平稳、中立。你的共情倾向：中。";

}  // namespace

// -----------------------------------------------------------------------------
// 构造: 初始化 SNN + LLM + 桥接层
// -----------------------------------------------------------------------------
EmotionEngine::EmotionEngine(Options opt) : opt_(std::move(opt)) {
    // ---- 1. CUDA 设备 ----
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0 || opt_.device >= dev_count) {
        std::fprintf(stderr, "[engine] no CUDA device (count=%d, requested=%d)\n",
                     dev_count, opt_.device);
        return;
    }
    cudaSetDevice(opt_.device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, opt_.device);
    std::printf("[engine] GPU: %s (%.0f MB)\n", prop.name,
                prop.totalGlobalMem / (1024.0 * 1024.0));

    // ---- 2. SNN 初始化 (resume: 拓扑随 checkpoint 恢复, 跳过 init_network) ----
    snn_ = new SnnState();
    snn_->allocator = new stage2e::MemoryAllocator(
        static_cast<std::size_t>(opt_.memory_budget_mb) * 1024ULL * 1024ULL);
    if (snn_->allocator->allocate_all() == 0) {
        std::fprintf(stderr, "[engine] SNN memory allocation failed\n");
        return;
    }

    snn_->scheduler = new stage2e::BioMechanismScheduler(snn_->allocator);
    snn_->scheduler->mod_update_interval = opt_.mod_update_interval;
    snn_->scheduler->bptt_enabled = false;               // 纯 STDP + 调制 (无 BPTT 窗口重放)
    snn_->scheduler->set_weights_freeze(opt_.freeze_weights);
    snn_->scheduler->skip_replay = true;                 // 对话模式跳过睡眠重放 (防 llama decode 污染)

    // ---- 2.5 SNN 文本语料 (resume 指纹校验 + 步进输入流) ----
    // 加载成功 → 严格校验语料指纹 (与 snn_train 一致); 失败 → 跳过校验 (引擎仅做情感演化)
    bool corpus_loaded = false;
    if (!opt_.text_path.empty()) {
        const size_t n = stage2e::load_text_corpus(opt_.text_path.c_str());
        corpus_loaded = n > 0;
        if (corpus_loaded) {
            std::printf("[engine] SNN corpus loaded: %s (%zu bytes)\n",
                        opt_.text_path.c_str(), n);
        } else {
            std::fprintf(stderr, "[engine] WARN corpus load failed (skip fingerprint check): %s\n",
                         opt_.text_path.c_str());
        }
    }
    snn_->scheduler->skip_corpus_check = !corpus_loaded;

    std::uint32_t checkpoint_seed = 0;
    const int rc = snn_->scheduler->load_checkpoint(
        opt_.resume_path.c_str(), &step_, &checkpoint_seed);
    if (rc != 0) {
        std::fprintf(stderr, "[engine] checkpoint resume failed (code=%d): %s\n",
                     rc, opt_.resume_path.c_str());
        return;
    }
    std::printf("[engine] SNN resumed: %s (step=%d, seed=%u, mod_interval=%d)\n",
                opt_.resume_path.c_str(), step_, checkpoint_seed,
                opt_.mod_update_interval);

    // ---- 3. LLM (进程内 llama.cpp) ----
    llm::LlamaBackend::Options lopt;
    lopt.model_path     = opt_.llm_model_path;
    lopt.n_gpu_layers   = opt_.llm_n_gpu_layers;
    lopt.n_ctx          = opt_.llm_n_ctx;
    lopt.max_new_tokens = 1024;  // Qwen3 思考模式: 思考块 + 正式回答需更长预算
    llm_ = std::make_unique<llm::LlamaBackend>(lopt);
    if (!llm_->is_loaded()) {
        std::fprintf(stderr, "[engine] LLM load failed: %s\n", opt_.llm_model_path.c_str());
        return;
    }
    std::printf("[engine] LLM loaded: %s\n", opt_.llm_model_path.c_str());

    // ---- 3.5 LLM 预热 (规避首个 llama decode 在 SNN 重活后 logits 垃圾化) ----
    if (llm_->warmup() != 0) {
        std::fprintf(stderr, "[engine] WARN LLM warmup failed (首轮生成可能为空)\n");
    }

    // ---- 4. 桥接层接线 ----
    bridge_.attach_backend(llm_.get());
    static SnnSink snn_sink;   // SNN 侧无生命周期问题 (仅转发到全局 setter)
    bridge_.attach_snn_feedback(&snn_sink);
    bridge_.attach_emotion_extractor(&extractor_);

    ready_ = true;
    std::printf("[engine] ready. 调制更新间隔=%d 步 (~%.1fs), 每轮训练推进=%d 步\n",
                opt_.mod_update_interval, opt_.mod_update_interval * 0.14,
                opt_.steps_per_turn);
}

EmotionEngine::~EmotionEngine() {
    if (snn_) {
        if (snn_->scheduler) {
            delete snn_->scheduler;
        }
        if (snn_->allocator) {
            snn_->allocator->free_all();
            delete snn_->allocator;
        }
        delete snn_;
    }
}

// -----------------------------------------------------------------------------
// 交互主循环
// -----------------------------------------------------------------------------
int EmotionEngine::run() {
    if (!ready_) {
        return 1;
    }

    std::string line;
    while (true) {
        // 多行输入: 逐行累积, 空行提交 (连续两下 Enter = 发送一条消息)
        // 提示符: 首行 "> ", 续行 "... "; 首行回车 / exit / quit / EOF = 退出
        std::string user;
        std::printf("\n> ");
        std::fflush(stdout);
        while (std::getline(std::cin, line)) {
            if (user.empty() && (line == "exit" || line == "quit")) {
                return 0;
            }
            if (line.empty()) {
                break;  // 空行 → 提交累积的消息
            }
            if (!user.empty()) {
                user.push_back('\n');
            }
            user += line;
            std::printf("... ");
            std::fflush(stdout);
        }
        if (user.empty()) {
            break;  // EOF 或首行直接回车 → 退出
        }

        snn_and_mood(user);

        // ④ LLM 生成 (CMD 模式: 引擎内部历史)
        std::string resp;
        if (llm_->chat(user, resp) != 0) {
            std::fprintf(stderr, "[engine] LLM generation failed\n");
            continue;
        }
        std::printf("[AI] %s\n", resp.c_str());
        bridge_.process_turn("assistant", resp);
    }
    return 0;
}

// CMD 与 serve 共用的 SNN 回合: 事件注入 → 训练推进 → 情感读出 → 采样调制
void EmotionEngine::snn_and_mood(const std::string& user_msg) {
    SnnState* s = snn_;

    // 0. 对话文本进入 SNN 文本流 (2026-08-05):
    //    追加后 get_byte_for_step 循环读取可能读到对话内容 → WM/Hippo 记忆
    //    系统编码对话的神经签名 (SNN 真正"看见"对话, 而非只喂语料)。
    if (!user_msg.empty()) {
        stage2e::append_text_stream(user_msg.c_str(), user_msg.size());
        stage2e::append_text_stream(" ", 1);   // 消息间分隔
    }
    recent_dialog_.push_back(user_msg);
    if (recent_dialog_.size() > 5) {
        recent_dialog_.erase(recent_dialog_.begin());
    }

    // ① LLM→SNN: 事件抽取 → 调质注入
    bridge_.process_turn("user", user_msg);

    // ② 训练推进 (在线学习): STDP + 调制链路演化
    for (int i = 0; i < opt_.steps_per_turn; ++i) {
        s->scheduler->step(step_++);
    }
    // SNN kernel 与 ggml-cuda 使用不同 CUDA 流: 不显式同步时, 首次 llama
    // decode 会与仍在排队的 SNN 异步 kernel 并发执行 → 首个 logits 垃圾化
    // → 采样器立即返回 EOG → 空回复 (2026-08-05 实测: steps=0 正常,
    // steps=10 首轮空、次轮正常, 因 get_affective_state 的同步拷贝兜底)。
    // 此处显式同步 + 清错, 保证 LLM 前 GPU 状态干净。
    cudaDeviceSynchronize();
    {
        const cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "[engine] CUDA error after SNN steps: %s\n",
                         cudaGetErrorString(cerr));
            cudaGetLastError();  // 清空错误状态
        }
    }

    // ③ SNN→LLM: 读取情感 → 桥接层调制 (两条独立通道, 消融开关可分别冻结)
    const stage2e::AffectiveState aff =
        stage2e::get_affective_state(s->allocator, step_);
    bridge_.set_state(to_emotion_state(aff));

    //   数值通道: 采样参数 (模型不可见, 直接改 llama 采样分布)
    //   文字通道: 情感 system prompt (模型可见, 语义引导)
    //   logit_bias 通道 (2026-08-05): 逐 token 改写采样分布, 属数值通道,
    //     ablate_sampler 时一并冻结
    std::vector<std::pair<std::string, float>> word_bias;
    if (opt_.ablate_sampler) {
        llm_->apply_sampler(bridge::SamplerParams{});   // 冻结默认采样参数
    } else {
        llm_->apply_sampler(bridge_.compute_sampler_params());
        word_bias = bridge_.compute_logit_bias();
    }
    llm_->apply_logit_bias(word_bias);
    if (opt_.ablate_prompt) {
        llm_->apply_emotion_prompt(kNeutralEmotionPrompt);  // 冻结中性情绪文字
    } else {
        // 情绪基调 + SNN 记忆片段 (工作记忆/海马/最近对话) 一并注入
        llm_->apply_emotion_prompt(bridge_.build_system_prompt_snippet() +
                                   build_memory_snippet());
    }

    const auto& p = llm_->sampler_params();   // 打印实际生效的参数 (消融时可见默认值)
    std::printf("[情感] %s\n", bridge_.build_mood_description().c_str());
    std::printf("[采样] temperature=%.2f top_p=%.2f repeat_penalty=%.2f\n",
                p.temperature, p.top_p, p.repeat_penalty);
    // logit_bias 日志: 打印本轮 SNN 状态实际应用的词级偏置 (逐 token 干预证据)
    std::printf("[bias] ");
    if (word_bias.empty()) {
        std::printf("无 (SNN 中性或 ablate_sampler 冻结)\n");
    } else {
        for (const auto& wb : word_bias) {
            std::printf("%s%+.2f ", wb.first.c_str(), wb.second);
        }
        std::printf("\n");
    }
    std::fflush(stdout);   // serve 模式 stdout 重定向到管道时行缓冲, 需显式 flush 才能在日志实时看到
}

// -----------------------------------------------------------------------------
// SNN 记忆片段 → system prompt (2026-08-05 一期: 记忆数值 + 最近对话回显)
// -----------------------------------------------------------------------------
std::string EmotionEngine::build_memory_snippet() {
    if (!snn_ || !snn_->allocator) {
        return std::string();
    }
    auto& buf = snn_->allocator->buffers();

    // 工作记忆: activation 降序取前 5
    std::vector<WMSlot> wm;
    if (buf.d_wm_slots) {
        wm = stage2e::read_wm_slots(buf.d_wm_slots, WM_SLOTS, 5);
    }
    // 海马记忆: importance top-5
    std::vector<HippoIndex> hippo;
    if (buf.d_hippo_indices && buf.d_hippo_top_k && buf.d_hippo_filled_count) {
        hippo = stage2e::read_hippo_memories(buf.d_hippo_indices, buf.d_hippo_top_k,
                                             buf.d_hippo_filled_count, 5, HIPP_INDEX_SIZE);
    }

    std::string out = "\n你的记忆（SNN 神经检索）：";
    int active = 0;
    float max_act = 0.0f;
    for (const auto& w : wm) {
        if (w.activation > 0.1f) {
            ++active;
        }
        if (w.activation > max_act) {
            max_act = w.activation;
        }
    }
    out += "工作记忆 " + std::to_string(active) + "/" + std::to_string(WM_SLOTS) + " 槽活跃";
    if (max_act > 0.0f) {
        char mb[16];
        std::snprintf(mb, sizeof(mb), "%.2f", max_act);
        out += std::string("（最高激活 ") + mb + "）";
    }
    if (!hippo.empty()) {
        const HippoIndex& top = hippo.front();
        char ib[16];
        std::snprintf(ib, sizeof(ib), "%.2f", top.importance);
        out += "；海马记忆 top-1 importance " + std::string(ib) +
               "（约步 " + std::to_string(top.pattern_start_step) + " 写入）";
    }
    // 海马记忆内容解码 (二期: SNN 签名 → 字节概率分布, 单步预测器)
    if (!hippo.empty() && snn_ && snn_->scheduler) {
        const int n_mem = (int)hippo.size() < 3 ? (int)hippo.size() : 3;
        out += "\nSNN 解码的字节指纹：";
        for (int mi = 0; mi < n_mem; ++mi) {
            uint8_t bytes[4];
            float probs[4];
            const int got = snn_->scheduler->decode_signature_top_bytes(
                hippo[mi].pattern_signature, bytes, probs, 4);
            if (got <= 0) {
                break;
            }
            out += "\n- 记忆" + std::to_string(mi + 1) + "（步 " +
                   std::to_string(hippo[mi].pattern_start_step) + "）：";
            for (int j = 0; j < got; ++j) {
                char tmp[32];
                if (bytes[j] >= 0x20 && bytes[j] < 0x7F) {
                    std::snprintf(tmp, sizeof(tmp), "'%c'=%.0f%%",
                                  (char)bytes[j], probs[j] * 100.0f);
                } else {
                    std::snprintf(tmp, sizeof(tmp), "0x%02X=%.0f%%",
                                  bytes[j], probs[j] * 100.0f);
                }
                out += tmp;
                if (j + 1 < got) {
                    out += " ";
                }
            }
        }
    }
    if (!recent_dialog_.empty()) {
        out += "。你最近记住的用户话语：";
        for (const auto& msg : recent_dialog_) {
            std::string t = msg;
            if (t.size() > 60) {
                t = t.substr(0, 60) + "…";
            }
            out += "\n- 用户说：“" + t + "”";
        }
    }
    out += "。";

    // 调试日志: 记忆系统状态 (验证 SNN 记忆接入生效)
    {
        int active = 0;
        for (const auto& w : wm) if (w.activation > 0.1f) ++active;
        std::printf("[记忆] WM活跃=%d/%d 最高激活=%.2f", active, WM_SLOTS, max_act);
        if (!hippo.empty()) {
            std::printf(" 海马top=%zu 解码[", hippo.size());
            if (snn_ && snn_->scheduler) {
                const int n_dbg = (int)hippo.size() < 3 ? (int)hippo.size() : 3;
                for (int mi = 0; mi < n_dbg; ++mi) {
                    uint8_t db[2];
                    float dp[2];
                    const int got = snn_->scheduler->decode_signature_top_bytes(
                        hippo[mi].pattern_signature, db, dp, 2);
                    if (mi) std::printf(" | ");
                    std::printf("M%d(步%d):", mi + 1, hippo[mi].pattern_start_step);
                    for (int j = 0; j < got; ++j) {
                        if (db[j] >= 0x20 && db[j] < 0x7F) {
                            std::printf("'%c'=%.0f%%", (char)db[j], dp[j] * 100.0f);
                        } else {
                            std::printf("0x%02X=%.0f%%", db[j], dp[j] * 100.0f);
                        }
                        if (j + 1 < got) std::printf(" ");
                    }
                }
            }
            std::printf("]");
        }
        std::printf(" 回显=%zu条\n", recent_dialog_.size());
        std::fflush(stdout);
    }
    return out;
}

// -----------------------------------------------------------------------------
// OpenAI 兼容 HTTP 服务 (serve 模式)
// -----------------------------------------------------------------------------
int EmotionEngine::run_serve() {
    if (!ready_) {
        return 1;
    }
    if (opt_.port <= 0 || opt_.port > 65535) {
        std::fprintf(stderr, "[engine] invalid port: %d\n", opt_.port);
        return -2;
    }

    const std::string expected_auth = "Bearer " + opt_.api_key;
    const std::string model_name = opt_.model_name;

    // POST /v1/chat/completions 处理
    auto handle_chat = [this, &model_name](const net::HttpRequest& req,
                                           net::HttpResponse& resp) {
        json::Value body;
        std::string jerr;
        if (!json::parse(req.body, body, &jerr)) {
            resp.status = 400;
            resp.body = error_json("invalid_request", "bad JSON: " + jerr);
            return;
        }
        const json::Value* msgs = body.find("messages");
        if (!msgs || msgs->type != json::Value::kArr || msgs->arr.empty()) {
            resp.status = 400;
            resp.body = error_json("invalid_request", "messages array required");
            return;
        }

        // 提取: 最新 user 消息 (SNN 事件注入) + 全量 user/assistant (LLM prompt,
        // 与客户端界面历史一致; system 消息忽略 — 引擎已有角色规则+情感片段)
        std::string user_msg;
        std::vector<std::pair<std::string, std::string>> llm_msgs;
        for (const auto& m : msgs->arr) {
            if (m.type != json::Value::kObj) {
                continue;
            }
            const json::Value* r = m.find("role");
            const json::Value* c = m.find("content");
            if (!r || !c) {
                continue;
            }
            const std::string role = r->as_str();
            const std::string content = c->as_str();
            if (role == "user") {
                user_msg = content;
            }
            if ((role == "user" || role == "assistant") && !content.empty()) {
                llm_msgs.emplace_back(role, content);
            }
        }
        if (user_msg.empty()) {
            resp.status = 400;
            resp.body = error_json("invalid_request", "no user message found");
            return;
        }

        // 同一链路: 事件注入 → 训练推进 → 情感调制 (日志打到服务控制台)
        snn_and_mood(user_msg);

        // LLM 生成 (serve 模式: 客户端全量历史)
        std::string resp_text;
        const int rc = llm_->chat_messages(llm_msgs, resp_text);
        if (rc != 0) {
            std::fprintf(stderr, "[engine] LLM generation failed (rc=%d)\n", rc);
            resp.status = 500;
            resp.body = error_json("generation_failed", "LLM generation failed");
            return;
        }
        bridge_.process_turn("assistant", resp_text);

        // OpenAI chat.completion 响应
        json::Value choice = json::Value::make_obj();
        choice.obj.emplace_back("index", json::Value(0.0));
        json::Value msg = json::Value::make_obj();
        msg.obj.emplace_back("role", json::Value("assistant"));
        msg.obj.emplace_back("content", json::Value(resp_text));
        choice.obj.emplace_back("message", std::move(msg));
        choice.obj.emplace_back("finish_reason", json::Value("stop"));
        json::Value choices = json::Value::make_arr();
        choices.arr.push_back(std::move(choice));

        json::Value usage = json::Value::make_obj();
        usage.obj.emplace_back("prompt_tokens", json::Value(0.0));
        usage.obj.emplace_back("completion_tokens", json::Value(0.0));
        usage.obj.emplace_back("total_tokens", json::Value(0.0));

        json::Value root = json::Value::make_obj();
        root.obj.emplace_back("id", json::Value("chatcmpl-" + std::to_string(step_)));
        root.obj.emplace_back("object", json::Value("chat.completion"));
        root.obj.emplace_back("created", json::Value((double)std::time(nullptr)));
        root.obj.emplace_back("model", json::Value(model_name));
        root.obj.emplace_back("choices", std::move(choices));
        root.obj.emplace_back("usage", std::move(usage));
        resp.body = json::dump(root);
    };

    net::HttpServer srv;
    return srv.run(opt_.port,
        [this, &expected_auth, &model_name, &handle_chat](const net::HttpRequest& req,
                                                          net::HttpResponse& resp) -> bool {
            // 鉴权 (CORS 预检 OPTIONS 除外)
            if (req.method != "OPTIONS") {
                const std::string auth = req.header("Authorization");
                if (auth != expected_auth) {
                    resp.status = 401;
                    resp.body = error_json("invalid_api_key", "Invalid API key");
                    return true;
                }
            }
            if (req.method == "OPTIONS") {
                resp.status = 204;
                return true;
            }
            if (req.method == "GET" && req.path == "/v1/models") {
                json::Value m = json::Value::make_obj();
                m.obj.emplace_back("id", json::Value(model_name));
                m.obj.emplace_back("object", json::Value("model"));
                m.obj.emplace_back("created", json::Value(0.0));
                m.obj.emplace_back("owned_by", json::Value("thetrueai"));
                json::Value data = json::Value::make_arr();
                data.arr.push_back(std::move(m));
                json::Value root = json::Value::make_obj();
                root.obj.emplace_back("object", json::Value("list"));
                root.obj.emplace_back("data", std::move(data));
                resp.body = json::dump(root);
                return true;
            }
            if (req.method == "POST" && req.path == "/v1/chat/completions") {
                handle_chat(req, resp);
                return true;
            }
            resp.status = 404;
            resp.body = error_json("not_found", "Not found: " + req.method + " " + req.path);
            return true;
        });
}

}  // namespace engine
}  // namespace hb
