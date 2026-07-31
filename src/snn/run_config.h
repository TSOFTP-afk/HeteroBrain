#ifndef SNN_STAGE2E_RUN_CONFIG_H
#define SNN_STAGE2E_RUN_CONFIG_H

#include "config.h"
#include <cstdint>
#include <string>

namespace stage2e {

struct RunConfig {
    int total_steps = 10000;
    int device = 0;
    uint32_t seed = 42;
    uint64_t memory_budget_mb = DEFAULT_VRAM_BUDGET_MB;
    int checkpoint_interval = 50000;
    int keep_checkpoints = 3;
    bool e0_mode = false;
    bool synthetic_input = false;
    bool strict_criteria = false;
    bool show_help = false;
    std::string text_path = "data/lccc_sample_1mb.txt";
    std::string csv_path;
    std::string checkpoint_dir = "checkpoints";
    std::string resume_path;
    // Task 10: 在线解码评估参数
    float decode_lr = 0.001f;           // 解码学习率 (传递给 scheduler, kernel 暂用编译常量)
    bool eval_mode = false;             // 仅推理模式 (不更新 W_decode)
    std::string eval_text_path;         // held-out 评估文本路径 (非空时用于评估)
    // ==================== BPTT 代理梯度训练参数 (Task D1) ====================
    bool bptt_mode = true;              // 启用 BPTT 训练 (默认 true, 主训练算法)
    bool no_structural_rebuild = false;  // 跳过 P3-D 结构重建 (纯 STDP 模式防 GPU hang)
    int  bptt_window_size = 50;         // 截断窗口长度
    float bptt_lr = 0.001f;             // Task F5: 基础学习率 (从 0.01 降至 0.001, 防止梯度爆炸)
    float bptt_clip = 5.0f;             // 梯度裁剪全局范数
    int  bptt_warmup_steps = 1000;      // 学习率 warmup 步数
    float bptt_surrogate_alpha = 4.0f;  // 代理梯度 sigmoid 斜率
    std::string input_mode = "bpe";     // 输入模式: "bpe" (默认) 或 "byte"
    std::string bpe_data_path;          // BPE 数据文件路径 (.bin)
    // ==================== Phase 3a-C1: 事件驱动调质注入 ====================
    bool event_stream_enabled = false;
    std::string event_stream_path;      // --event-stream PATH
    // ==================== Phase 3a-D3: 课程训练 ====================
    std::string curriculum_path;        // --curriculum PATH (课程 JSONL)
    int curriculum_stage = 1;           // --stage 1=初中(默认) 2=高中 0=启蒙 3=成年
    float curriculum_lr = 0.001f;       // --curriculum-lr readout 学习率
    bool curriculum_eval = false;       // --curriculum-eval 评估模式 (冻结权重, 统计工具/调质准确率)
    int  curriculum_eval_samples = 20;  // --curriculum-eval-samples N 评估样本数
    // ==================== Phase 3a-D1: 具身发育训练 ====================
    bool embodied_mode = false;
    std::string embodied_scene = "hunger_feeding";  // 默认场景
};

bool parse_run_config(int argc, char** argv, RunConfig* config, std::string* error);
const char* run_config_usage();

} // namespace stage2e

#endif
