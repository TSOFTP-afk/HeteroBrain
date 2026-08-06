// =============================================================================
// main.cpp — 异构引擎 CLI 入口
// =============================================================================
// 用法:
//   vita_engine.exe --resume <snn.ckpt> --llm <model.gguf>
//                          [--mod-interval N] [--steps-per-turn N]
//                          [--freeze-weights] [--device N]
//                          [--memory-budget-mb N]
//                          [--serve [--port N] [--api-key K] [--model-name M]]
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>   // SetConsoleCP/SetConsoleOutputCP: 控制台 UTF-8 编码
#endif

#include "engine.h"

namespace {

// 强制控制台使用 UTF-8 输入/输出 (2026-08-05):
//   CMD 默认代码页 GBK(936), 而引擎输出/接收均为 UTF-8 → 中文乱码。
//   此处进程内设置 CP_UTF8, 无需用户手动 chcp 65001。
void setup_console_utf8() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
}

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s --resume <snn.ckpt> --llm <model.gguf> [options]\n"
        "\n"
        "required:\n"
        "  --resume <path>          SNN checkpoint (.snn2e)\n"
        "  --llm <path>             GGUF 模型路径 (MiniCPM5-1B-Q4_K_M.gguf)\n"
        "\n"
        "options:\n"
        "  --mod-interval <N>       调制更新间隔 (默认 10 步 ≈ 1.4s/更新; 训练语义 100)\n"
        "  --steps-per-turn <N>     每轮对话训练推进步数 (默认 10)\n"
        "  --text <path>            SNN 文本语料 (resume 指纹校验, 默认 data/smoke_test.txt;\n"
        "                           传空串 \"\" 跳过语料加载与指纹校验)\n"
        "  --freeze-weights         冻结突触权重 (纯推理; 默认 false = 在线学习)\n"
        "  --device <N>             CUDA 设备 (默认 0)\n"
        "  --memory-budget-mb <N>   SNN 显存预算 MiB (默认 1543)\n"
        "  --llm-ngl <N>            LLM GPU 层数 (默认 99)\n"
        "\n"
        "serve mode (OpenAI 兼容 API, 供第三方客户端/壳子连接):\n"
        "  --serve                  以 HTTP 服务运行而非 CMD 交互\n"
        "  --port <N>               监听端口 (默认 8899, 仅 127.0.0.1 本机)\n"
        "  --api-key <K>            Bearer API key (默认 thetrueai)\n"
        "  --model-name <M>         模型名, 客户端需填此值 (默认 thetrueai)\n"
        "                           端点: GET /v1/models | POST /v1/chat/completions\n"
        "\n"
        "ablation (消融实验, 验证 SNN 真实作用):\n"
        "  --ablate-prompt          冻结情感 prompt 文字为中性 (只留采样参数通道)\n"
        "  --ablate-sampler         冻结采样参数为默认值 (只留情绪文字通道)\n"
        "  --ablate-all             两者同时冻结 (无 SNN 影响的对照基线)\n",
        argv0);
}

}  // namespace

// 交互选择运行模式 (在启动动画之后): 1=对话模式, 2=HTTP 服务
// 仅当未用 --serve 强制指定时调用; 非法输入回退到对话模式
static int select_run_mode() {
    std::printf("\n选择运行模式 / Select mode:\n");
    std::printf("  1) 对话模式 (Interactive chat)\n");
    std::printf("  2) HTTP 服务 (OpenAI-compatible serve)\n");
    std::printf("> ");
    std::fflush(stdout);
    std::string line;
    std::getline(std::cin, line);
    return (line == "2") ? 2 : 1;
}

int main(int argc, char** argv) {
    setup_console_utf8();
    vita::engine::EmotionEngine::Options opt;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            opt.resume_path = argv[++i];
        } else if (std::strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            opt.llm_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mod-interval") == 0 && i + 1 < argc) {
            opt.mod_update_interval = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps-per-turn") == 0 && i + 1 < argc) {
            opt.steps_per_turn = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            opt.text_path = argv[++i];
        } else if (std::strcmp(argv[i], "--freeze-weights") == 0) {
            opt.freeze_weights = true;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            opt.device = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--memory-budget-mb") == 0 && i + 1 < argc) {
            opt.memory_budget_mb = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--llm-ngl") == 0 && i + 1 < argc) {
            opt.llm_n_gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--serve") == 0) {
            opt.serve = true;
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            opt.port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
            opt.api_key = argv[++i];
        } else if (std::strcmp(argv[i], "--model-name") == 0 && i + 1 < argc) {
            opt.model_name = argv[++i];
        } else if (std::strcmp(argv[i], "--ablate-prompt") == 0) {
            opt.ablate_prompt = true;
        } else if (std::strcmp(argv[i], "--ablate-sampler") == 0) {
            opt.ablate_sampler = true;
        } else if (std::strcmp(argv[i], "--ablate-all") == 0) {
            opt.ablate_prompt = true;
            opt.ablate_sampler = true;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (opt.resume_path.empty() || opt.llm_model_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    vita::engine::EmotionEngine engine(opt);
    if (!engine.is_ready()) {
        std::fprintf(stderr, "[main] engine initialization failed\n");
        return 1;
    }
    // 动画已在引擎构造中播放 (初始化完成后)。
    // 若未用 --serve 强制指定，则在动画之后交互选择运行模式。
    if (opt.serve) {
        return engine.run_serve();
    }
    const int mode = select_run_mode();
    if (mode == 2) {
        return engine.run_serve();
    }
    return engine.run();
}
