// =============================================================================
// engine.h — 异构引擎 (SNN 情感核心 + EmotionBridge + LlamaBackend 进程内接线)
// =============================================================================
// 主循环 (在线学习形态):
//   用户输入 → 事件抽取 → SNN 调质 (set_event_signal) → 训练推进 (STDP 持续演化)
//   → get_affective_state → EmotionBridge → LLM 生成
// 设计:
//   - SNN 内部状态经不透明指针 (SnnState) 持有, 本头文件不暴露 CUDA/调度器依赖
//   - 桥接层复用 (EmotionBridge/LlmBackend/EmotionEventExtractor 均在 src/bridge、src/llm)
//   - 调制更新间隔可配置 (mod_update_interval): 引擎对话模式默认 10 步 ≈ 1.4s/更新,
//     远快于训练模式 100 步 ≈ 14s (scheduler 慢时间尺度已参数化)
// =============================================================================

#ifndef VITA_ENGINE_H
#define VITA_ENGINE_H

#include <memory>
#include <string>
#include <vector>

#include "bridge/emotion_bridge.h"
#include "llm/llama_backend.h"

namespace vita {
namespace engine {

class EmotionEngine {
public:
    struct Options {
        std::string resume_path;           // SNN checkpoint (.snn2e, 必填)
        int  memory_budget_mb = 4096;      // 显存预算 (MiB, 本地 6GB 卡 SNN 全量 ~1.6GB)
        int  device = 0;                   // CUDA 设备
        std::string llm_model_path;        // GGUF 模型路径 (必填)
        int  llm_n_gpu_layers = 99;        // LLM GPU 层数
        int  llm_n_ctx = 4096;             // LLM 上下文窗口
        std::string text_path = "data/smoke_test.txt";  // SNN 文本语料 (resume 指纹校验用)
        int  mod_update_interval = 10;     // 调制更新间隔 (步): 情感响应时延 = 间隔 × 140ms
        int  steps_per_turn = 10;          // 每轮对话训练推进步数
        bool freeze_weights = false;       // true=冻结突触 (纯推理); false=在线学习 (STDP 持续演化)
        std::uint32_t seed = 42;
        // ---- OpenAI 兼容 serve 模式 (--serve) ----
        bool serve = false;                // true=HTTP 服务 (第三方 OpenAI 客户端); false=CMD 交互
        int  port = 8899;                  // 监听端口 (127.0.0.1 仅本机)
        std::string api_key = "thetrueai"; // Authorization: Bearer <api_key>
        std::string model_name = "thetrueai";  // GET /v1/models 返回的模型 id
        // ---- 消融实验开关 (验证 SNN 真实作用的对照, 2026-08-05) ----
        // 两条调制通道可分别冻结:
        //   ablate_prompt  = 情感 prompt 文字固定为中性 (保留数值通道: 采样参数)
        //   ablate_sampler = 采样参数固定为默认值 (保留文字通道: 情绪基调)
        //   --ablate-all  = 两者同时冻结 (无 SNN 影响的对照基线)
        bool ablate_prompt = false;
        bool ablate_sampler = false;
    };

    explicit EmotionEngine(Options opt);
    ~EmotionEngine();

    EmotionEngine(const EmotionEngine&) = delete;
    EmotionEngine& operator=(const EmotionEngine&) = delete;

    bool is_ready() const { return ready_; }

    // 交互主循环: getline 用户输入 → 事件注入 → 训练推进 → 情感调制 → LLM 生成
    // 返回值: 0 正常退出; 1 错误
    int run();

    // OpenAI 兼容 HTTP 服务 (serve 模式): 监听 127.0.0.1:port,
    //   GET /v1/models | POST /v1/chat/completions, Bearer 鉴权。
    // 每请求: 事件注入 → 训练推进 → 情感调制 → LLM 生成 (与 CMD 模式同链路)
    // 返回值: 0 正常退出; 负值启动失败
    int run_serve();

private:
    struct SnnState;

    // CMD/serve 共用的 SNN 回合: 事件抽取注入 → 训练推进 → 情感读出 → 采样调制。
    // 调用方随后执行 LLM 生成 (CMD: chat 内部历史; serve: chat_messages 客户端历史)
    void snn_and_mood(const std::string& user_msg);

    // 构建 SNN 记忆片段 (工作记忆 + 海马 + 最近对话), 追加到 system prompt
    std::string build_memory_snippet();

    Options   opt_;
    bool      ready_ = false;
    SnnState* snn_ = nullptr;              // SNN 内部状态 (engine.cpp 定义)
    int       step_ = 0;                   // SNN 当前步
    std::unique_ptr<llm::LlamaBackend> llm_;
    bridge::EmotionBridge bridge_;
    bridge::EmotionEventExtractor extractor_;
    std::vector<std::string> recent_dialog_;   // 最近 N 条用户消息 (host 侧对话记忆回显)
};

}  // namespace engine
}  // namespace vita

#endif  // VITA_ENGINE_H
