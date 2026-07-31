#ifndef SNN_STAGE2E_SCHEDULER_CUH
#define SNN_STAGE2E_SCHEDULER_CUH

// =============================================================================
// Stage 2e 统一调度器 (P1)
// =============================================================================
// 对应设计文档 §5.4: v4 调度器
//
// P1 阶段:
//   - 快时间尺度 kernel 全部替换为真实实现 (AdEx + NMDA + STDP + STP)
//   - 中/慢时间尺度仍为占位 (留给 Phase 2-4)
//   - 群体编码输入注入
//   - spike count 统计 (P1 判据: 极差 > 100, 簇状发放出现)
//
// 流水线:
//   delay_inject → input_inject → lif_adex → synapse_nmda
//       → stdp_dual_trace → stdp_stp → delay_dispatch
//   (每10步) camkii, stdp_eligibility, inhibitory_network
//   (每100步) modulatory, scaling, wm_update
//   (每1000步) structural_plasticity, developmental
//   (每10000步) replay
// =============================================================================

#include "config.h"
#include "types.h"
#include "memory_allocator.cuh"
#include "thalamic_gate.cuh"
#include "decode_kernels.cuh"
#include "bptt_trainer.cuh"
#include <climits>
#include <cmath>
#include <vector>

namespace stage2e {

// -----------------------------------------------------------------------------
// 发育阶段参数表 (v2 修复 5)
// -----------------------------------------------------------------------------
struct DevPhaseTable {
    DevPhaseParams phases[5];

    DevPhaseTable() {
        // 胚胎期 (0 - 5K) — 缩短以加速涌现验证 (原 30K)
        phases[0] = {0.0f, 0.0f, 0.0f,  0.0f, 0.1f, 1.0f, DEV_PHASE_EMBRYO_END};
        // 突触发生期 (5K - 200K) — plast_gain 5.0→1.0 防止权重快速饱和
        phases[1] = {1.0f, 0.0f, 0.01f, 0.8f, 0.5f, 1.0f, DEV_PHASE_SYNAPTO_END};
        // 关键期 (200K - 800K)
        phases[2] = {3.0f, 0.0f, 0.0f,  1.0f, 0.8f, 1.2f, DEV_PHASE_CRITICAL_END};
        // 修剪期 (800K - 1.5M)
        phases[3] = {1.0f, 0.05f, 0.0f, 0.6f, 0.3f, 1.5f, DEV_PHASE_PRUNE_END};
        // 成熟期 (1.5M - 3M)
        phases[4] = {0.3f, 0.02f, 0.0f, 0.4f, 0.2f, 2.0f, DEV_PHASE_MATURE_END};
    }

    DevPhase get_phase(int step) const {
        for (int i = 0; i < 5; ++i) {
            if (step < phases[i].end_step) return static_cast<DevPhase>(i);
        }
        return DevPhase::MATURE;
    }

    const DevPhaseParams& get_params(int step) const {
        return phases[static_cast<int>(get_phase(step))];
    }
};

// -----------------------------------------------------------------------------
// 调度器
// -----------------------------------------------------------------------------
class BioMechanismScheduler {
public:
    BioMechanismScheduler(MemoryAllocator* alloc);
    ~BioMechanismScheduler();

    // E0 消融模式: 关闭三因素调制 + CaMKII + 调质系统 (纯 STDP 基线)
    bool e0_ablation = false;

    // Task 10: 评估模式控制 (参照 e0_ablation 公共成员模式)
    // decode_update_weights: false 时 decode_step 不更新 W_decode (仅前向预测, 用于评估)
    //   默认 true = 训练模式 (保持向后兼容)
    //   main.cpp 根据 config.eval_mode 设置: eval_mode=true → decode_update_weights=false
    bool decode_update_weights = true;
    // decode_lr: 解码学习率 (由 RunConfig.decode_lr 传入, kernel 暂用编译常量 DECODE_LEARNING_RATE)
    //   存储此处供后续 kernel 改造使用, 当前仅作元数据记录
    float decode_lr = DECODE_LEARNING_RATE;

    // ==================== Task D2: BPTT 集成配置 ====================
    // bptt_enabled: true 时启用 BPTT 代理梯度训练 (主训练算法)
    //   默认 false, 由 main.cpp 根据 RunConfig.bptt_mode 设置
    //   false 时 scheduler.step() 走纯 STDP 路径 (向后兼容 e0_mode)
    bool bptt_enabled = false;
    // skip_structural_rebuild: 跳过 P3-D 结构重建 (纯 STDP 模式防 GPU hang,
    //   与 BPTT 模式拓扑保持稳定行为一致; 由 RunConfig.no_structural_rebuild 设置)
    bool skip_structural_rebuild = false;
    // bptt_input_mode: 0=byte (旧), 1=bpe (新)
    //   由 main.cpp 根据 RunConfig.input_mode 设置
    int  bptt_input_mode = 0;
    // bptt_window_boundary: BPTT 窗口边界步数 (window_size 的倍数)
    //   每到此步触发一次 forward+backward+update
    int  bptt_window_size = 50;

    // 主步进函数
    void step(int current_step);

    // 获取统计
    const NetworkStats2e& stats() const { return stats_; }
    int delay_ring_idx() const { return delay_ring_idx_; }

    // ==================== Task 4-5: 在线解码接口 ====================
    // 解码一步: 前向 (每步) + 误差驱动权重更新 (注入步 + K 步延迟后)
    //   current_input_byte: 当前注入的字节 (非注入步可传 0)
    //   is_inject_step:     当前步是否为注入步
    //   update_weights:     是否执行权重更新 (false=仅前向预测, 用于评估)
    void decode_step(uint8_t current_input_byte, bool is_inject_step, bool update_weights);

    // 最近一次解码 cross-entropy loss (host 缓存)
    float get_last_decode_loss() const { return last_decode_loss_; }

    // 最近一次解码预测字节 (host 缓存, 0..255)
    int get_last_predicted_byte() const { return last_predicted_byte_; }

    // 解码 perplexity 统计 (供 main.cpp 读取)
    float get_decode_avg_loss() const {
        // Task F3.1: 使用全局累积器 (不被周期性日志重置)
        return loss_global_count_ > 0 ? cross_entropy_loss_global_ / loss_global_count_ : 0.0f;
    }
    float get_decode_perplexity() const {
        float avg = get_decode_avg_loss();
        return avg > 0.0f ? expf(avg) : 0.0f;
    }
    float get_decode_accuracy() const {
        return predict_total_count_ > 0
            ? (float)correct_predict_count_ / (float)predict_total_count_
            : 0.0f;
    }
    int get_decode_correct_count() const { return correct_predict_count_; }
    int get_decode_total_count() const { return predict_total_count_; }

    // P1 统计
    int total_steps_executed() const { return total_steps_; }
    int total_spikes_accum() const { return total_spikes_accum_; }
    int min_spikes_per_step() const { return min_spikes_per_step_; }
    int max_spikes_per_step() const { return max_spikes_per_step_; }
    int spike_range() const {
        return max_spikes_per_step_ - min_spikes_per_step_;
    }
    int total_burst_steps() const { return total_burst_steps_; }
    int total_single_neuron_burst_spikes() const { return total_single_neuron_burst_spikes_; }
    long long arrived_events_accum() const { return arrived_events_accum_; }
    long long dispatched_events_accum() const { return dispatched_events_accum_; }
    long long dropped_events_accum() const { return dropped_events_accum_; }
    int max_delay_slot_depth() const { return max_delay_slot_depth_; }
    int p3_inhibitory_updates() const { return p3_inhibitory_updates_; }
    int p3_wm_updates() const { return p3_wm_updates_; }
    float p3_last_activity_drive() const { return p3_last_activity_drive_; }
    int p3_kwta_updates() const { return p3_kwta_updates_; }
    int p3_kwta_active_columns() const { return p3_kwta_active_columns_; }
    int p3_kwta_winner_estimate() const { return p3_kwta_winner_estimate_; }
    int p3_kwta_suppressed_estimate() const { return p3_kwta_suppressed_estimate_; }
    int p3_kwta_target_per_column() const { return p3_kwta_target_per_column_; }

    // P3-C 语义聚类评估 (silhouette + JS divergence + 柱间差异)
    int    p3_semantic_eval_updates() const { return p3_semantic_eval_updates_; }
    double p3_silhouette_score() const { return p3_silhouette_score_; }
    double p3_js_divergence_mean() const { return p3_js_divergence_mean_; }
    double p3_js_divergence_max() const { return p3_js_divergence_max_; }
    double p3_column_ratio() const { return p3_column_ratio_; }
    int    p3_semantic_eval_step() const { return p3_semantic_eval_last_step_; }

    // 丘脑-皮层门控指标 (§1.1 注意力门控)
    float gate_mean() const { return gate_mean_; }
    float gate_open_ratio() const { return gate_open_ratio_; }

    // Phase R2 模块 C (Task 6.5): L6 反馈闭环指标
    int   l6_total_spikes_last() const { return l6_total_spikes_last_; }       // 上一步 L6 spike 总数
    float l6_activity_ema_mean() const { return l6_activity_ema_mean_; }       // L6 活动 EMA 跨柱均值
    // 层间指标 (最近一次 semantic_eval)
    int    layer_eval_step() const { return p3_semantic_eval_last_step_; }
    const double* layer_activation_delay() const { return layer_activation_delay_; }  // [5]
    const float*   layer_chi2_sig_ratio() const { return layer_chi2_sig_ratio_; }     // [5]
    const double* layer_chi2_mean() const { return layer_chi2_mean_; }                // [5]

    // P3-C: 显式触发一次语义聚类评估 (短测末尾由 main.cpp 调用)
    void run_semantic_eval(int step) { launch_semantic_eval(step); }

    // 完整 checkpoint: 所有持久 GPU 状态、调度器状态和文本游标。
    // next_step 表示恢复后第一个尚未执行的绝对 step。
    int save_checkpoint(int next_step, const char* dir, uint32_t topology_seed);
    int load_checkpoint(const char* path, int* next_step, uint32_t* topology_seed);
    int prune_checkpoints(const char* dir, int keep_latest);

    float burst_ratio() const {
        return total_steps_ > 0 ? (100.0f * total_burst_steps_ / total_steps_) : 0.0f;
    }

    // ==================== Task 2: PCA 集成接口 ====================
    // PCA 签名提取: 从当前联合皮层发放率计算 K 维 PCA 签名 (L2 归一化)
    //   供海马编码和 WM 写入调用
    //   d_signature_out: [PCA_N_COMPONENTS] 输出签名 (device, 调用方分配)
    void compute_pca_signature(float* d_signature_out);

    // PCA 反投影: 从 K 维签名重建 N 维联合皮层发放率
    //   供睡眠重放和 WM 注入调用
    //   d_signature:         [PCA_N_COMPONENTS] 输入签名 (device)
    //   d_reconstructed_out: [N_ASSOCIATION_NEURONS_2E] 输出重建 (device, 调用方分配)
    void pca_back_project(const float* d_signature, float* d_reconstructed_out);

    // PCA 更新次数计数 (供 FINAL_METRICS 输出)
    int pca_update_count() const { return pca_update_count_; }

    // ==================== Task 18: 睡眠重放状态隔离接口 ====================
    // enter_sleep_state: 进入睡眠态, 保存当前 thalamic_gain (gate_mean_) 和 ach_level,
    //   设置 ach_level *= SLEEP_ACH_FACTOR (巩固模式), is_sleeping_=true
    // exit_sleep_state: 退出睡眠态, 恢复保存的 ach_level, is_sleeping_=false
    // is_sleeping: 供 step() 内 input_inject / modulatory 步查询, true 时跳过外部输入
    void enter_sleep_state(int step);
    void exit_sleep_state(int step);
    bool is_sleeping() const { return is_sleeping_; }

    // ==================== Task D2: BPTT 接口 ====================
    // 初始化 BPTT 训练器 (在 main.cpp 中调用, 一次)
    //   config: BPTT 配置 (window_size, lr, clip, warmup, alpha, beta, threshold)
    //   成功返回 true, 失败 (内存不足等) 返回 false
    bool init_bptt(const BPTTConfig& config);

    // 释放 BPTT 训练器资源 (析构前调用)
    void shutdown_bptt();

    // BPTT 状态查询 (供 main.cpp 打印日志)
    bool bptt_active() const { return bptt_trainer_ != nullptr && bptt_enabled; }
    float bptt_last_loss() const { return bptt_last_loss_; }
    float bptt_last_grad_norm() const { return bptt_last_grad_norm_; }
    float bptt_current_lr() const { return bptt_current_lr_; }
    int   bptt_window_size_cfg() const { return bptt_window_size; }

    // main.cpp 在 BPE 注入后调用, 设置当前 BPTT target token
    void set_bptt_target_token(int32_t token) { bptt_last_target_token_ = token; }

    // ==================== Phase 3a-D3: 课程训练接口 ====================
    // 启用课程模式: 设置目标调质 + 目标工具 + readout 学习率 + 损失权重
    //   BPTT 反向用 调质误差 + 工具误差 替代解码误差 (backward_curriculum)
    //   target_tool: 0-5 = 6 类工具, 6 = 不调用
    void set_curriculum_mode(const float target_mod[6], int target_tool,
                             float readout_lr, float w_mod, float w_tool);
    void disable_curriculum_mode();
    bool curriculum_active() const { return curriculum_mode_; }
    // 最近一次课程 BPTT loss (w_mod·MSE + w_tool·CE)
    float curriculum_last_loss() const { return curriculum_last_loss_; }

    // 冻结 BPTT 更新 (评估模式): bptt_step 只做前向 + loss 记录, 不反传不更新
    void set_bptt_freeze(bool freeze) { bptt_freeze_ = freeze; }
    bool bptt_freeze() const { return bptt_freeze_; }

    // Task D3: 暴露 d_gate_states_ 供 main.cpp 在 BPE 注入时使用
    // (BPE 模式下 main.cpp 在 step() 之前调用 launch_bpe_inject, 需要传入门控状态)
    const ThalamicGateState* d_gate_states_for_inject() const { return d_gate_states_; }

private:
    friend struct SchedulerCheckpointAccess;
    MemoryAllocator* alloc_;
    NetworkStats2e stats_;
    DevPhaseTable phase_table_;
    int delay_ring_idx_;
    int last_phase_;

    // P1 统计
    int total_steps_;
    int total_spikes_accum_;
    int inject_spikes_accum_;     // P1 修正: 注入步累计脉冲 (用于排除 burst 误判)
    int min_spikes_per_step_;
    int max_spikes_per_step_;
    int total_burst_steps_;  // 簇状发放步数
    int total_single_neuron_burst_spikes_;
    long long arrived_events_accum_;
    long long dispatched_events_accum_;
    long long dropped_events_accum_;
    int max_delay_slot_depth_;
    int p3_inhibitory_updates_;
    int p3_wm_updates_;
    float p3_last_activity_drive_;
    int p3_kwta_updates_;
    int p3_kwta_active_columns_;
    int p3_kwta_winner_estimate_;
    int p3_kwta_suppressed_estimate_;
    int p3_kwta_target_per_column_;

    // P3-C 语义聚类评估状态
    int    p3_semantic_eval_updates_;
    int    p3_semantic_eval_last_step_;
    double p3_silhouette_score_;
    double p3_js_divergence_mean_;
    double p3_js_divergence_max_;
    double p3_column_ratio_;

    // Phase R2 模块 C (Task 6): L6 反馈闭环 + 层间指标
    // L6 spike count 设备缓冲 (50 柱 × 4B = 200B)
    int* d_l6_column_spikes_;
    // 层间字节响应 (5 层 × 256 字节 = 5KB, 用于 chi2 计算)
    int* d_layer_byte_responses_;
    // 层间 chi2 统计设备缓冲 (5×int sig + 5×int act + 5×float chi2_sum = 40B)
    int*   d_layer_sig_count_;
    int*   d_layer_act_count_;
    float* d_layer_chi2_sum_;
    // 期望比例 (256 × 4B = 1KB, host 上传到 device, 用于 chi2 kernel)
    float* d_injections_per_byte_;
    // host 端 L6 统计 (用于 print_step_log)
    int   l6_total_spikes_last_;
    float l6_activity_ema_mean_;
    // host 端层间指标缓存 (最近一次 semantic_eval 结果)
    double layer_activation_delay_[5];  // 每层平均激活延迟 (相对注入步)
    float  layer_chi2_sig_ratio_[5];     // 每层卡方显著神经元比例
    double layer_chi2_mean_[5];          // 每层卡方均值

    // device 端 spike 计数器 (用于 atomicAdd 统计)
    int* d_spike_counter_;
    int* d_single_neuron_burst_counter_;
    int* d_p3_column_spikes_;
    int* d_p3_kwta_stats_;
    int* d_p3_column_byte_responses_;  // P3-C: 50 柱 × 256 字节 = 50KB

    // 丘脑-皮层门控 (§1.1 注意力门控)
    ThalamicGateState* d_gate_states_;        // 门控状态 (N_COLUMNS_2E 个, 每柱 16B)
    unsigned int* d_byte_history_;             // 字节历史计数 (256 个 uint32, 用于 novelty)
    float* d_gate_stats_;                      // 门控统计 [gate_mean, gate_open_ratio, gate_min, gate_max]
    float gate_mean_;                          // 最近一次门控均值 (host 缓存)
    float gate_open_ratio_;                    // 最近一次门控开启比例 (host 缓存)

    // ==================== Task 4-5: 在线解码状态 ====================
    float last_decode_loss_ = 0.0f;            // 最近一次解码 cross-entropy loss
    int   last_predicted_byte_ = 0;            // 最近一次解码预测字节 (0..255)
    int   decode_step_counter_ = 0;            // 解码步计数 (用于每 100 步行归一化调度)
    float cross_entropy_loss_accum_ = 0.0f;    // perplexity 窗口累积器 (日志后重置)
    int   loss_accum_count_ = 0;               // 窗口累积次数 (日志后重置)
    int   correct_predict_count_ = 0;          // 准确率统计: 正确预测次数 (全局, 不重置)
    int   predict_total_count_ = 0;            // 准确率统计: 总预测次数 (全局, 不重置)
    // Task F3.1: 全局累积器 (供 final_perplexity 使用, 不被周期性日志重置)
    float cross_entropy_loss_global_ = 0.0f;   // 全局 cross-entropy 累积 (整个训练周期)
    int   loss_global_count_ = 0;              // 全局累积次数
    // 输入字节历史缓冲 (环形, 大小 = PREDICTION_DELAY_STEPS)
    // 用途: 延迟 K 步匹配 — 当缓冲区填满后, 最旧字节作为前向解码的 target
    //       (网络当前活动反映 K 步前注入字节的延迟效应, 解码器学习读出该字节)
    uint8_t input_byte_history_[PREDICTION_DELAY_STEPS] = {0};
    int   input_byte_history_idx_ = 0;         // 环形缓冲写指针
    int   input_byte_history_count_ = 0;       // 已记录的字节数 (warmup 完成后 = PREDICTION_DELAY_STEPS)

    // ==================== Task 2: PCA 集成状态 ====================
    // PCA W 矩阵 CPU 镜像 [N_ASSOCIATION_NEURONS_2E × PCA_N_COMPONENTS]
    // CPU 端 Oja 在线学习, 每 PCA_SYNC_INTERVAL 步同步到 GPU d_pca_W
    std::vector<float> h_pca_W_;
    // 联合皮层发放率快照 [N_ASSOCIATION_NEURONS_2E] (从 d_spike_flags 转换)
    std::vector<float> h_fr_snapshot_;
    // 联合皮层滑动平均发放率 [N_ASSOCIATION_NEURONS_2E] (PCA 中心化用)
    std::vector<float> h_mean_fr_;
    // spike_flags (bool) → float 转换用 host 中转缓冲 [N_ASSOCIATION_NEURONS_2E]
    std::vector<uint8_t> h_spike_buf_;
    // 滑动平均 EMA 系数 (越大越平滑, 0.99 = 时间常数 ~100 步)
    float h_mean_fr_ema_ = 0.99f;
    // PCA 更新次数计数 (每次 Oja 更新 +1, 供 FINAL_METRICS 输出)
    int pca_update_count_ = 0;
    // GPU 端 PCA 辅助缓冲 (compute_pca_signature / pca_back_project 用)
    float* d_pca_fr_ = nullptr;      // [N_ASSOCIATION_NEURONS_2E] 当前发放率 (float)
    float* d_pca_mean_ = nullptr;    // [N_ASSOCIATION_NEURONS_2E] 滑动平均发放率 GPU 镜像

    // ==================== Task 4-5: 睡眠重放临时缓冲 ====================
    float* d_replay_sig_ = nullptr;          // [PCA_N_COMPONENTS] 重放用签名临时缓冲
    float* d_replay_recon_ = nullptr;        // [N_ASSOCIATION_NEURONS_2E] PCA 反投影重建临时缓冲
    int replay_cycle_count_ = 0;             // 重放周期计数 (供 FINAL_METRICS 输出)

    // ==================== Task 8: 结构可塑性临时缓冲 ====================
    int* d_new_synapse_pairs_ = nullptr;     // [COACT_MAX_NEW_SYNAPSES × 2] 新突触对
    int* d_new_synapse_count_ = nullptr;     // [1] 新突触计数
    float* d_new_modulator_scores_ = nullptr;// [COACT_MAX_NEW_SYNAPSES] 调质分数
    int* d_prune_marks_ = nullptr;           // [N_TOTAL_SYNAPSES_2E] 修剪标记
    int* d_prune_count_ = nullptr;           // [1] 修剪计数
    int structural_rebuild_count_ = 0;       // 结构重建次数计数
    int coact_sample_count_ = 0;             // 共激活采样次数计数

    // ==================== Task 18: 睡眠重放状态隔离 ====================
    // 保存进入睡眠态前的 thalamic_gain (gate_mean_) 和 ach_level (stats_.ach_level)
    // 重放期间 is_sleeping_=true, input_inject 步跳过外部字节注入,
    // modulatory 步跳过 ACh 信号刷新 (让 ACh 自然保持低水平, 符合慢波睡眠生理特征)
    float saved_thalamic_gain_ = 0.5f;       // 进入睡眠前保存的 gate_mean_ (默认 GATE_INITIAL_SIGNAL)
    float saved_ach_level_     = 0.0f;       // 进入睡眠前保存的 stats_.ach_level
    bool  is_sleeping_         = false;      // 睡眠态标志 (true=重放进行中, 外部输入已抑制)
    int   sleep_cycle_count_   = 0;          // 睡眠周期计数 (供 FINAL_METRICS 输出)

    // ==================== Task D2: BPTT 训练器 ====================
    BPTTTrainer* bptt_trainer_ = nullptr;  // 懒分配, init_bptt() 后非空
    int bptt_step_counter_ = 0;             // 当前 BPTT 窗口内步数 (0..window_size-1)
    float bptt_last_loss_ = 0.0f;           // 最近一次 BPTT backward 的 loss
    float bptt_last_grad_norm_ = 0.0f;      // 最近一次 BPTT update 的梯度范数
    float bptt_current_lr_ = 0.0f;          // 当前有效学习率 (含 warmup)
    int32_t bptt_last_target_token_ = 0;    // 当前 BPTT target token (由 main.cpp 设置)

    // ==================== Phase 3a-D3: 课程训练状态 ====================
    bool  curriculum_mode_ = false;         // 课程模式激活
    float curriculum_target_mod_[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // 目标调质
    int   curriculum_target_tool_ = 6;      // 目标工具 (0-5 = 6 工具, 6 = 不调用)
    float curriculum_readout_lr_ = 0.001f;  // readout 权重学习率
    float curriculum_w_mod_ = 1.0f;         // 调质损失权重
    float curriculum_w_tool_ = 0.3f;        // 工具损失权重 (初中 0.3)
    float curriculum_last_loss_ = 0.0f;     // 最近一次课程 BPTT loss
    bool  bptt_freeze_ = false;             // 冻结 BPTT 更新 (评估模式)

    // PCA 增量更新 (每 PCA_UPDATE_INTERVAL 步, CPU 端 Oja's rule)
    void launch_pca_update_cpu(int step);

    // --- P1 占位 kernel 启动器 (中/慢时间尺度, Phase 2-4 实现) ---
    void launch_camkii_kernel(int step);
    void launch_stdp_eligibility(int step);
    void launch_inhibitory_network(int step);

    void launch_modulatory(int step);
    void launch_scaling(int step);
    void launch_wm_update(int step);

    void launch_structural_plasticity(int step);
    void launch_developmental(int step);

    void launch_replay(int step);

    // P3-C: 语义聚类评估 (silhouette + JS divergence + 柱间差异)
    void launch_semantic_eval(int step);

    // ==================== Task D2: BPTT 单步处理 ====================
    // BPTT 单步处理: 在主 step() 末尾调用
    //   1. bptt_step_counter_ 累加, 当达到 window_size 时触发完整 forward+backward+update
    //   2. forward 用简化 LIF 重放 T 步 (不调用 AdEx, 不修改主循环状态)
    //   3. backward 用代理梯度累积 dL/dW
    //   4. update SGD + 梯度裁剪, 同步到 d_synapses 和 d_weights_cache
    //   5. 重置 bptt_step_counter_ = 0, 缓存 last_loss/grad_norm/lr
    //   current_byte: 当前注入字节 (用于 backward 的 target)
    //   current_token: 当前 BPE token (input_mode=1 时用作 target, 取低字节)
    void bptt_step(int current_step, uint8_t current_byte, int32_t current_token);

    // 日志
    void print_step_log(int step);
    void print_phase_change(int step, DevPhase new_phase);
};

} // namespace stage2e

#endif // SNN_STAGE2E_SCHEDULER_CUH
