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
#include "personality_profiles.h"
#include "curriculum_loader.h"
#include "mod_simulator.h"
#include <climits>
#include <cmath>
#include <string>
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

    // 跳过 checkpoint corpus 指纹校验 (2026-08-05 引擎接线):
    //   引擎对话模式不关心文本语料身份 (SNN 仅做情感演化), 语料缺失时跳过校验,
    //   避免 "corpus does not match checkpoint" (code=8) 阻断 resume;
    //   加载了语料 (--text) 时保持严格校验, 与 snn_train 语义一致。
    bool skip_corpus_check = false;

    // 调制更新间隔 (2026-08-05 引擎接线):
    //   慢时间尺度 (launch_modulatory / scaling / 情感采样) 的触发周期。
    //   默认 100 保持训练语义; 引擎对话模式可降为 10 (1.4s/更新) 加快情感响应,
    //   衰减率与事件 duration 消耗已按此参数化 (modulatory_kernels.cu)。
    int mod_update_interval = 100;

    // ==================== Task D2: BPTT 集成配置 ====================
    // bptt_enabled: true 时启用 BPTT 代理梯度训练 (主训练算法)
    //   默认 false, 由 main.cpp 根据 RunConfig.bptt_mode 设置
    //   false 时 scheduler.step() 走纯 STDP 路径 (向后兼容 e0_mode)
    bool bptt_enabled = false;
    // skip_structural_rebuild: 跳过 P3-D 结构重建 (纯 STDP 模式防 GPU hang,
    //   与 BPTT 模式拓扑保持稳定行为一致; 由 RunConfig.no_structural_rebuild 设置)
    bool skip_structural_rebuild = false;
    // skip_replay: 跳过睡眠重放 + 同周期语义评估 (2026-08-05 引擎接线):
    //   引擎对话模式不需要重放巩固; 且实测重放 kernel 之后紧邻的 llama decode
    //   会输出垃圾 logits → 采样立即 EOG (空回复, 60K 首轮复现; 15K 无重放正常)。
    //   训练模式保持默认 false (重放照常)。
    bool skip_replay = false;
    // bptt_input_mode: 0=byte (旧), 1=bpe (新)
    //   由 main.cpp 根据 RunConfig.input_mode 设置
    int  bptt_input_mode = 0;
    // 2026-08-05: 文本流注入间隔 (原编译宏 INPUT_INJECT_INTERVAL=3)
    //   由 main.cpp 根据 RunConfig.input_inject_interval 设置;
    //   长线剧本模式用 1 使文本流与每窗口事件时间同步
    int  input_inject_interval = 3;
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

    // 从 PCA 签名解码字节概率分布 (host 端, 2026-08-05 二期: SNN 记忆内容解码)
    //   反投影: rate[i] = mean_fr[i] + Σ_k sig[k]·W[i,k]   (i < N_ASSOCIATION_NEURONS_2E)
    //   logits[b] = Σ_i decode_W[i*256+b]·rate[i] → softmax → top-k 字节
    // 解码器是"单步预测器" (网络状态 → 下一字节分布), 故输出 top-k 字节而非自回归文本。
    // out_bytes/out_probs: 调用方分配 ≥k 元素; 返回实际填充数 (≤k, 失败返回 0)
    int decode_signature_top_bytes(const float* sig50,
                                   uint8_t* out_bytes, float* out_probs, int k);

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
    // 设置课程发育阶段: 填充 STDP eta 倍率 + 调质基线 + PAD 损失权重来源
    //   (由其他任务实现, 本任务只接线) — 须在 set_curriculum_mode 之前调用,
    //   main.cpp 在每处 set_curriculum_mode 前先调用 (2026-08-02 Task A)
    void set_curriculum_stage(CurriculumStage stage);

    // 每 100 步推进浓度模拟器并刷新当前块目标 (main.cpp 课程循环调用)
    //   sample: 当前课程样本 (events 用于取本块到期事件)
    //   rel: 窗口内相对步 (应为 100 倍数)
    //   base_signal: 阶段基线, GENE_MAP 列顺序 [DA,ACh,NE,5HT,GABA,Oxy]
    void advance_curriculum_target(const CurriculumSample* sample, int rel,
                                   const float base_signal[6]);
    // 复位浓度模拟器到冷启动 (conc=0, sensitivity=1), 同步刷新当前块目标
    //   (2026-08-04 修复: 原实现跨样本/窗口持续运行不重置, 慢通道 Oxy tau=500
    //   残留累积 → 目标漂移到稳态不动点, 调质 MSE 度量"稳态拟合"而非
    //   "事件→调质响应"; 样本/窗口边界必须调用, 使目标 = baseline + 本样本事件响应)
    void reset_curriculum_target();
    // 当前块模拟浓度/PAD 目标 (分段监督: N3F 每步误差与窗口末 readout 更新用,
    //   与内部注入同步推进的连续浓度; 2026-08-02 Task 8)
    const float* curriculum_target_mod_curr() const { return curriculum_target_mod_curr_; }
    const float* curriculum_target_pad_curr() const { return curriculum_target_pad_curr_; }
    // 启用课程模式: 设置目标调质 + 目标工具 + 目标 PAD + readout 学习率 + 损失权重
    //   BPTT 反向用 调质/PAD/工具 误差替代解码误差 (backward_curriculum)
    //   target_tool: 0-5 = 6 类工具, 6 = 不调用
    //   w_pad 不显式传入: 内部从 personality_profile(curriculum_stage_).bptt_loss_weight_pad
    //   取值 (初中 0.3 / 高中 0.5), 与 set_curriculum_mode 的调用时序解耦
    void set_curriculum_mode(const float target_mod[6], int target_tool,
                             const float target_pad[3],
                             float readout_lr, float w_mod, float w_tool);
    void disable_curriculum_mode();
    bool curriculum_active() const { return curriculum_mode_; }
    // 最近一次课程 BPTT loss (w_mod·MSE + w_tool·CE)
    float curriculum_last_loss() const { return curriculum_last_loss_; }

    // 冻结 BPTT 更新 (评估模式): bptt_step 只做前向 + loss 记录, 不反传不更新
    void set_bptt_freeze(bool freeze) { bptt_freeze_ = freeze; }
    bool bptt_freeze() const { return bptt_freeze_; }

    // 冻结突触权重写入 (评估模式, 2026-08-04 B3 修复):
    //   eval 期间跳过 STDP / CaMKII / scaling / 结构可塑性 (证据衰减+弱突触重置)
    //   的权重写入 — 旧实现只冻结 BPTT/readout, 突触权重仍被持续修改,
    //   评估样本间网络漂移, eval 结果非严格冻结 (n_eval×win≥1000 必触发衰减/重建)
    void set_weights_freeze(bool freeze) { weights_freeze_ = freeze; }
    bool weights_freeze() const { return weights_freeze_; }

    // ==================== N3F: 调质门控三因子在线学习 ====================
    // 设置课程突触学习算法: "bptt" (窗口重放反传) | "n3f" (三因子在线)
    //   n3f 模式: 每步课程误差 → 神经元级 eligibility → STDP 证据调制,
    //   无窗口重放/无历史缓冲, 网络可拥有任意动力学 (自发活动/噪声)
    void set_learning_rule(const std::string& rule) { learning_rule_ = rule; }
    bool n3f_mode() const { return learning_rule_ == "n3f"; }
    // N3F 每步在线学习 (在 scheduler.step() 之后由 main.cpp 调用):
    //   readout 前向 (当前帧) → 课程误差 → readout 监督 (每步)
    //   readout 权重在窗口末由 main.cpp 切换分支更新 (累计帧)
    //   loss 同步回读仅每窗口 (step % window == 0) 一次, 避免每步阻塞流水线
    //   范式切换 (B5, 2026-08-04): 不再注入 eligibility 到突触 (spec §7.2),
    //   突触塑形回归 STDP + 沙盒反馈; readout 监督路径保留
    void n3f_online_step(int current_step);

    // N3F 具身奖励 → 神经元级 eligibility (第三因子, spec §7.1, 2026-08-01)
    //   reward ∈ [-1,1]: reward>0 强化, reward<0 削弱 (uniform 广播, 生物对应 DA 系统)
    //   在 scheduler.step() 之前调用 (沙盒 feedback 后), 本次 STDP 立即生效 (无滞后)
    void n3f_embodied_step(float reward);

    // ==================== Phase 3a-F (M1/M3): 杏仁核 + HPA 皮质醇 ====================
    // 事件→杏仁核 LA 注入 + 负性事件累积 HPA 皮质醇应激
    //   (main.cpp/engine 事件 dispatch 时调用, 与 set_event_signal 并列;
    //    LA→BA 前向积分 + STDP 权重更新在 scheduler.step() 内每步推进;
    //    BA 输出→调质调制 (正性→DA↑/负性→NE↑) 在每调制更新时应用, 评估冻结时跳过)
    void amygdala_event_inject(int event_type, float intensity);
    // 读取杏仁核 BA 输出 (负性/正性组发放率 [0,1]) (日志/桥接用)
    void read_amygdala_output(float* out_neg, float* out_pos);
    // 当前 HPA 皮质醇水平 [0,1] (日志/桥接用)
    float cortisol_level() const;

    // 具身感知注入 (2026-08-01 修复):
    //   main.cpp 在 env 步计算 50 柱感知向量后调用; step() 内部在 delay_inject
    //   (清零 d_input_current) 之后、lif_adex 之前注入, 确保感知信号不被覆盖
    void set_embodied_sensory(const float sensory[50]);

    // ==================== Phase 3a-G (A): 事件→联合皮层直通注入 ====================
    // 根因修复: 事件调制对联合皮层传导 <2.3% (被文本流淹没), readout 只能学平均
    // 状态 → 事件信息必须直接进入网络内部。事件类型 k → 联合皮层固定子区域
    // 注入电流 (与文本流并行), rate 携带事件信息后 readout 才有可学信号。
    //   main.cpp/engine 事件 dispatch 时与 set_event_signal/amygdala_event_inject
    //   并列调用; step() 内 delay_inject 之后、lif_adex 之前注入, 持续
    //   EVENT_CORTEX_HOLD_STEPS 步 (仿 set_embodied_sensory 时序)
    void set_event_cortex_inject(int event_type, float intensity);

    // ==================== Phase 3a-H (M4): 脑岛内感受 ====================
    // 身体锚点: 内感受 (饥饿/温度/舒适/疲劳/疼痛) → 脑岛群体 → 调质调制
    //   (不适→NE↑, 舒适→Oxy↑)。main.cpp 每环境步 (100 SNN 步) 调 set_insula_sensory
    //   刷新 15 柱内感受 (embodied 模式); 非 embodied 模式无输入 → 脑岛静默。
    void set_insula_sensory(const float intero15[15]);
    // 读取脑岛 5 维输出 (日志/桥接用, [hunger, temp偏离, comfort, fatigue, pain])
    void read_insula_output(float out[5]);

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

    // 具身感知注入 (2026-08-01 修复): 感知向量由 main.cpp 设置, step() 内注入
    float h_embodied_sensory_[50] = {};      // host 缓存 50 柱感知向量
    bool  embodied_sensory_active_ = false;  // 是否有待注入的感知向量

    // ---- Phase 3a-G (A): 事件→联合皮层注入挂起状态 (2026-08-06) ----
    int   h_event_cortex_pending_type_ = -1; // -1 = 无挂起事件
    float h_event_cortex_pending_gain_ = 0.0f; // 当前注入电流增益 (强度已缩放)
    int   h_event_cortex_hold_left_ = 0;      // 剩余注入步数

    // ---- Phase 3a-H (M4): 脑岛内感受缓存 (2026-08-06) ----
    // 15 柱内感受 → 5 维强度 (set_insula_sensory 内换算, 持续注入非脉冲):
    //   [hunger不适, temp偏离, comfort高, fatigue, pain] ∈ [0,1]
    float h_insula_dims_[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool  insula_active_ = false;   // 是否有内感受输入 (embodied 模式)

    // ---- Phase 3a-I (M2): VTA-DA RPE 神经化缓存 (2026-08-06) ----
    // RPE 信号为调制窗口级: launch_modulatory 内更新注入强度缓存 (用本窗口
    // da_delta/prediction_error_norm), 下一窗口注入; 同时读 VTA 窗口累计发放率
    // (上一窗口注入的响应) → h_vta_rpe_ → 下窗口 STDP 第三因子 DA 叠加项
    float h_vta_rpe_inject_[2] = {0.0f, 0.0f};  // 注入强度缓存 [正RPE, 负RPE]
    float h_vta_rpe_ = 0.0f;                    // VTA 发放差 pos-neg ∈ [-1,1]

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
    // 解码器权重 host 镜像 (懒加载, N_TOTAL_NEURONS_2E×256, 2026-08-05 二期)
    std::vector<float> h_decode_weights_;
    bool h_decode_weights_loaded_ = false;
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
    float curriculum_target_pad_[3] = {0.0f, 0.0f, 0.0f};  // 目标 PAD 情感 [Pleasure, Arousal, Dominance] (2026-08-02 Task 5)
    float curriculum_readout_lr_ = 0.001f;  // readout 权重学习率
    float curriculum_w_mod_ = 1.0f;         // 调质损失权重
    float curriculum_w_tool_ = 0.3f;        // 工具损失权重 (初中 0.3)
    float curriculum_w_pad_ = 0.0f;         // PAD 情感损失权重 (2026-08-02 Task 5,
                                            //   set_curriculum_mode 内部从阶段 profile 取:
                                            //   初中 0.3 / 高中 0.5; 启蒙/成年 0 → 无贡献)
    float curriculum_last_loss_ = 0.0f;     // 最近一次课程 BPTT loss
    bool  bptt_freeze_ = false;             // 冻结 BPTT 更新 (评估模式)
    bool  weights_freeze_ = false;          // 冻结突触权重写入 (评估模式, B3)
    std::string learning_rule_ = "bptt";    // 课程突触学习算法: bptt | n3f

    // ---- Phase 3a-F (M3): HPA 皮质醇快照 (checkpoint 持久化用) ----
    // 值来自 modulatory_kernels.cu 的 g_cortisol 全局; 存此处以便作为独立
    // checkpoint section 持久化 (不塞进 SchedulerState, 避免破坏旧版加载)
    float h_cortisol_snapshot_ = 0.0f;

    // ---- Phase 3a-F (M1): 杏仁核 BA 窗口累计 (2026-08-06) ----
    // 每步在 step() 内把当前 BA 正/负性组发放率累加进来; 调制更新时
    // (launch_modulatory) 读出作为窗口内情感反应强度并清零。BA 发放窗口
    // 仅持续注入期 ~10 步, 与 mod 读取点 (100/200/...) 错位, 必须跨步累计。
    float h_amyg_ba_neg_accum_ = 0.0f;   // 窗口累计负性组发放率
    float h_amyg_ba_pos_accum_ = 0.0f;   // 窗口累计正性组发放率
    float h_amyg_ba_neg_last_ = 0.0f;    // 最近一窗口累计 (日志用)
    float h_amyg_ba_pos_last_ = 0.0f;

    // ---- 阶段参数 (仅课程模式生效, 由 set_curriculum_stage 填充) ----
    CurriculumStage curriculum_stage_ = STAGE_ENLIGHTENMENT;  // 当前课程发育阶段
    float stdp_eta_multiplier_ = 1.0f;      // STDP eta 阶段倍率 (非课程模式恒为 1.0)
    float curriculum_baseline_mod_[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                                            // 阶段调质基线 [DA, 5HT, NE, ACh, GABA, Oxy]
    bool  curriculum_baseline_active_ = false;  // 阶段基线是否生效 (仅课程模式)

    // ---- 分段监督目标 (2026-08-02 Task 5) ----
    // 当前 100 步块的模拟浓度/PAD 目标 (N3F 每步误差用, 与内部注入同步推进)
    float curriculum_target_mod_curr_[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float curriculum_target_pad_curr_[3] = {0.0f, 0.0f, 0.0f};
    CurriculumModSimulator curriculum_mod_sim_;   // 浓度模拟器 (每 100 步 advance)

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
