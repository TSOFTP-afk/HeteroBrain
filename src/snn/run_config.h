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
    // 2026-08-05: 文本流注入间隔 (原编译宏 INPUT_INJECT_INTERVAL=3)
    //   默认 3 (与历史实验一致); 长线剧本模式建议 1 (每段 400 步注入 400 字节
    //   ≈ 每段 110 字叙事 330 字节, 事件与文本流时间同步)
    int  input_inject_interval = 3;     // --input-inject-interval N (1-10)
    // ==================== Phase 3a-C1: 事件驱动调质注入 ====================
    bool event_stream_enabled = false;
    std::string event_stream_path;      // --event-stream PATH
    // ==================== Phase 3a-D3: 课程训练 ====================
    std::string curriculum_path;        // --curriculum PATH (课程 JSONL)
    int curriculum_stage = 1;           // --stage 1=初中(默认) 2=高中 0=启蒙 3=成年
    float curriculum_lr = 0.001f;       // --curriculum-lr readout 学习率
    int curriculum_readout_warmup_steps = 0;  // --curriculum-readout-warmup N
                                        //   N3F: 前 N 步冻结 readout 更新 (纯前向),
                                        //   避免随机初始化 readout 的早期误差 (spec §7.7)
    bool curriculum_eval = false;       // --curriculum-eval 评估模式 (冻结权重, 统计工具/调质准确率)
    int  curriculum_eval_samples = 100; // --curriculum-eval-samples N 评估样本数
    bool eval_emergent = false;         // --eval-emergent 情绪涌现诊断 (L1 事件扩散 / L2 readout 依赖 / L3 模式区分度)
                                        //   2026-08-01 spec §7.3: 默认 20 → 100
                                        //   (20 样本统计意义不足: 12 个 target=6 全对即可达 60% 准确率)
    bool curriculum_continuous = false; // --curriculum-continuous 连续课程模式
                                        //   窗口边界不复位浓度模拟器 → 前序窗口慢通道残留
                                        //   (Oxy tau=500) 传导到后续窗口, target 含远程因果影响;
                                        //   配套 generate_serial_curriculum.py 连续叙事数据
                                        //   (2026-08-05 新增, 与 2026-08-04 P0 修复不冲突:
                                        //   冷启动仍为默认, 连续模式为显式受控延续)
    std::string learning_rule = "bptt"; // --learning-rule bptt|n3f (默认 bptt)
                                        //   bptt: 窗口重放反传 (Phase 3a-D3 现行)
                                        //   n3f : 调质门控三因子在线学习 (Neuromodulator-Gated
                                        //         3-Factor): 每步课程误差 → neuron_eligibility →
                                        //         STDP 证据调制, 无窗口无重放无历史缓冲
                                        //   ⚠ N3F 参数复用语义 (spec §7.10):
                                        //     - bptt_window_size: N3F 中作为 readout 更新间隔 (窗口概念一致)
                                        //     - bptt_lr: N3F 不使用 (N3F 不初始化 BPTT trainer, 该参数被忽略)
                                        //     - curriculum_lr: N3F 中作为 readout 学习率 (语义一致)
    // ==================== Phase 3a-D1: 具身发育训练 ====================
    bool embodied_mode = false;
    std::string embodied_scene = "hunger_feeding";  // 默认场景
};

bool parse_run_config(int argc, char** argv, RunConfig* config, std::string* error);
const char* run_config_usage();

} // namespace stage2e

#endif
