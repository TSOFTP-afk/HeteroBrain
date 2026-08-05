#include "scheduler.cuh"
#include "input_encoding.cuh"
#include "synapse_kernels.cuh"
#include "neuron_kernels.cuh"
#include "modulatory_kernels.cuh"
#include "bptt_trainer.cuh"  // Task E1: BPTTConfig 用于 capture/restore BPTT 状态
#include "bptt_curriculum.cuh"  // Phase 3a-D3: 课程 readout 权重 checkpoint 持久化

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace stage2e {
namespace {

constexpr char kMagic[8] = {'S','N','N','2','E','C','P','3'};
constexpr char kFooterMagic[8] = {'S','N','N','2','E','O','K','3'};
constexpr uint32_t kVersion = 3;
constexpr size_t kChunkBytes = 8U * 1024U * 1024U;

struct CheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t section_count;
    uint32_t reserved;
    uint64_t payload_bytes;
    uint64_t payload_checksum;
    uint32_t n_neurons;
    uint32_t n_synapses;
    uint32_t bio_synapse_bytes;
    uint32_t neuron_state_bytes;
};

struct DiskSection {
    char name[48];
    uint64_t bytes;
};

struct CheckpointFooter {
    char magic[8];
    uint64_t payload_checksum;
};

struct SchedulerState {
    uint32_t state_version;
    uint32_t state_bytes;
    int32_t next_step;
    uint32_t topology_seed;
    uint64_t corpus_size;
    uint64_t corpus_position;
    uint64_t corpus_fingerprint;
    uint8_t e0_ablation;
    uint8_t runtime_state_valid;
    uint8_t reserved[6];
    NetworkStats2e stats;
    int32_t delay_ring_idx;
    int32_t last_phase;
    int32_t total_steps;
    int32_t total_spikes_accum;
    int32_t inject_spikes_accum;
    int32_t min_spikes_per_step;
    int32_t max_spikes_per_step;
    int32_t total_burst_steps;
    int32_t total_single_neuron_burst_spikes;
    int64_t arrived_events_accum;
    int64_t dispatched_events_accum;
    int64_t dropped_events_accum;
    int32_t max_delay_slot_depth;
    int32_t p3_inhibitory_updates;
    int32_t p3_wm_updates;
    float p3_last_activity_drive;
    int32_t p3_kwta_updates;
    int32_t p3_kwta_active_columns;
    int32_t p3_kwta_winner_estimate;
    int32_t p3_kwta_suppressed_estimate;
    int32_t p3_kwta_target_per_column;
    int32_t p3_semantic_eval_updates;
    int32_t p3_semantic_eval_last_step;
    double p3_silhouette_score;
    double p3_js_divergence_mean;
    double p3_js_divergence_max;
    double p3_column_ratio;
    int32_t l6_total_spikes_last;
    float l6_activity_ema_mean;
    double layer_activation_delay[5];
    float layer_chi2_sig_ratio[5];
    double layer_chi2_mean[5];
    float gate_mean;
    float gate_open_ratio;
    DelayQueueCheckpointState delay_queue;
    ModulatoryRuntimeState modulatory_runtime;

    // ==================== Task E1: BPTT 训练状态 ====================
    // 注意: Task E1 扩展了 SchedulerState 大小 (新增 BPTT 字段)
    //       旧版 checkpoint (无 BPTT 字段) 的 scheduler_state section 较小, 加载时会因
    //       section_layout_matches 失败而拒绝. 这是预期的: BPTT 状态需要新字段才能恢复.
    //       若需加载旧 checkpoint, 使用 git 旧版本编译的程序.
    // BPTT 配置 (恢复时通过 init_bptt() 重建 BPTTTrainer)
    uint8_t  bptt_enabled;           // bool -> uint8_t for portable serialization
    uint8_t  bptt_input_mode;        // int -> uint8_t (0=byte, 1=bpe)
    uint8_t  reserved_bptt[2];       // 4-byte alignment
    int32_t  bptt_window_size;       // BPTT 窗口大小
    float    bptt_lr;                // 学习率
    float    bptt_grad_clip;         // 梯度裁剪
    int32_t  bptt_warmup_steps;      // warmup 步数
    float    bptt_surrogate_alpha;   // 代理梯度斜率
    float    bptt_beta;              // LIF 衰减
    float    bptt_threshold;         // 发放阈值
    // BPTT 运行时状态 (供日志连续性, 不影响训练正确性)
    int32_t  bptt_step_counter;      // 当前窗口内步数
    float    bptt_last_loss;         // 最近一次 loss
    float    bptt_last_grad_norm;    // 最近一次梯度范数
    float    bptt_current_lr;        // 当前有效学习率
    int32_t  bptt_last_target_token; // 最近一次 target token
};

struct Section {
    const char* name;
    void* pointer;
    uint64_t bytes;
    bool device;
};

uint64_t update_checksum(uint64_t value, const void* data, size_t bytes) {
    static uint32_t table[256]{};
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t x = i;
            for (int bit = 0; bit < 8; ++bit) {
                x = (x >> 1) ^ ((x & 1U) ? 0xEDB88320U : 0U);
            }
            table[i] = x;
        }
        initialized = true;
    }
    const auto* p = static_cast<const unsigned char*>(data);
    uint32_t crc = static_cast<uint32_t>(value) ^ 0xFFFFFFFFU;
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFU] ^ (crc >> 8);
    }
    return static_cast<uint64_t>(crc ^ 0xFFFFFFFFU);
}

bool write_exact(FILE* fp, const void* data, size_t bytes) {
    return bytes == 0 || std::fwrite(data, 1, bytes, fp) == bytes;
}

bool read_exact(FILE* fp, void* data, size_t bytes) {
    return bytes == 0 || std::fread(data, 1, bytes, fp) == bytes;
}

bool sync_file(FILE* fp) {
    if (std::fflush(fp) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(fp)) == 0;
#else
    return fsync(fileno(fp)) == 0;
#endif
}

void add(std::vector<Section>* sections, const char* name, void* ptr,
         uint64_t count, uint64_t element_size, bool device = true) {
    sections->push_back({name, ptr, count * element_size, device});
}

} // namespace

struct SchedulerCheckpointAccess {
    static std::vector<Section> make_sections(BioMechanismScheduler* self,
                                              SchedulerState* state);
    static SchedulerState capture_state(BioMechanismScheduler* self, int next_step,
                                        uint32_t seed);
    static bool restore_state(BioMechanismScheduler* self, const SchedulerState& state);
};

std::vector<Section> SchedulerCheckpointAccess::make_sections(
    BioMechanismScheduler* self, SchedulerState* state) {
    PersistentBuffers& b = self->alloc_->buffers();
    std::vector<Section> s;
    s.reserve(56);
    add(&s, "scheduler_state", state, 1, sizeof(*state), false);
    add(&s, "neurons", b.d_neurons, N_TOTAL_NEURONS_2E, sizeof(NeuronStateAdEx));
    add(&s, "spike_flags", b.d_spike_flags, N_TOTAL_NEURONS_2E, sizeof(bool));
    add(&s, "synapses", b.d_synapses, N_TOTAL_SYNAPSES_2E, sizeof(BioSynapse));
    add(&s, "csr_row_ptr", b.d_csr_row_ptr, N_TOTAL_NEURONS_2E + 1ULL, sizeof(int));
    add(&s, "csr_col_idx", b.d_csr_col_idx, N_TOTAL_SYNAPSES_2E, sizeof(int));
    add(&s, "weights_cache", b.d_weights_cache, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "eligibility", b.d_eligibility, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "eligibility_slow", b.d_eligibility_slow, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "synapse_alpha", b.d_synapse_alpha, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "synapse_beta", b.d_synapse_beta, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "pca_W", b.d_pca_W, (uint64_t)N_TOTAL_NEURONS_2E * PATTERN_DIM, sizeof(float));
    add(&s, "ca_snapshot", b.d_ca_snapshot, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "ca_history_sparse", b.d_ca_history_sparse,
        (uint64_t)CA_HISTORY_MAX_ACTIVE * CA_HISTORY_LEN, sizeof(float));
    add(&s, "synapse_delay", b.d_synapse_delay, N_TOTAL_SYNAPSES_2E, sizeof(uint8_t));
    add(&s, "delay_ring_indices", b.d_delay_ring_indices,
        (uint64_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY, sizeof(int));
    add(&s, "delay_ring_current", b.d_delay_ring_current,
        (uint64_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY, sizeof(float));
    add(&s, "stdp_x_pre_trace", b.d_stdp_x_pre_trace, N_TOTAL_SYNAPSES_2E, sizeof(float));
    // 闭环修复: 神经元级 eligibility (恢复训练时保留解码反传状态)
    add(&s, "neuron_eligibility", b.d_neuron_eligibility, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "camkii_activity", b.d_camkii_activity, N_TOTAL_SYNAPSES_2E, sizeof(float));
    add(&s, "input_current", b.d_input_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "nmda_current", b.d_nmda_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "inhibitory_current", b.d_inhibitory_current, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "da_concentration", b.d_da_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ach_concentration", b.d_ach_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ne_concentration", b.d_ne_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "ht5_concentration", b.d_ht5_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "gaba_concentration", b.d_gaba_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "oxytocin_concentration", b.d_oxytocin_concentration, N_TOTAL_NEURONS_2E, sizeof(float));
    add(&s, "oxytocin_receptor", b.d_oxytocin_receptor, N_TOTAL_SYNAPSES_2E, sizeof(uint8_t));
    add(&s, "hippo_indices", b.d_hippo_indices, HIPP_INDEX_SIZE, sizeof(HippoIndex));
    add(&s, "coact_trackers", b.d_coact_trackers, COACT_TRACKER_SIZE, sizeof(CoactTracker));
    add(&s, "wm_slots", b.d_wm_slots, WM_SLOTS, sizeof(WMSlot));
    add(&s, "subcolumn_fr", b.d_subcolumn_fr, W_VALUE_DIM, sizeof(float));
    add(&s, "baseline_fr", b.d_baseline_fr, W_VALUE_DIM, sizeof(float));
    add(&s, "w_pred", b.d_w_pred, (uint64_t)W_PRED_DIM * W_PRED_DIM, sizeof(float));
    add(&s, "w_value", b.d_w_value, W_VALUE_DIM, sizeof(float));
    add(&s, "pred_fr", b.d_pred_fr, W_PRED_DIM, sizeof(float));
    add(&s, "byte_histogram", b.d_byte_histogram, 256, sizeof(int));
    add(&s, "neuron_byte_counts", b.d_neuron_byte_counts,
        (uint64_t)N_TOTAL_NEURONS_2E * 256, sizeof(int));
    add(&s, "replay_injection", b.d_replay_injection, N_TOTAL_NEURONS_2E, sizeof(float));

    add(&s, "spike_counter", self->d_spike_counter_, 1, sizeof(int));
    add(&s, "single_burst_counter", self->d_single_neuron_burst_counter_, 1, sizeof(int));
    add(&s, "p3_column_spikes", self->d_p3_column_spikes_, N_COLUMNS_2E, sizeof(int));
    add(&s, "p3_kwta_stats", self->d_p3_kwta_stats_, 3, sizeof(int));
    add(&s, "p3_column_byte_responses", self->d_p3_column_byte_responses_,
        (uint64_t)N_COLUMNS_2E * 256, sizeof(int));
    add(&s, "gate_states", self->d_gate_states_, N_COLUMNS_2E, sizeof(ThalamicGateState));
    add(&s, "byte_history", self->d_byte_history_, 256, sizeof(unsigned int));
    add(&s, "gate_stats", self->d_gate_stats_, 4, sizeof(float));
    add(&s, "l6_column_spikes", self->d_l6_column_spikes_, N_COLUMNS_2E, sizeof(int));
    add(&s, "layer_byte_responses", self->d_layer_byte_responses_, 5ULL * 256, sizeof(int));
    add(&s, "layer_sig_count", self->d_layer_sig_count_, 5, sizeof(int));
    add(&s, "layer_act_count", self->d_layer_act_count_, 5, sizeof(int));
    add(&s, "layer_chi2_sum", self->d_layer_chi2_sum_, 5, sizeof(float));
    add(&s, "injections_per_byte", self->d_injections_per_byte_, 256, sizeof(float));
    // 解码权重矩阵 (Stage 2e 线性解码器: 神经活动 → 256 维字节 logits)
    // 布局 [N_TOTAL_NEURONS_2E × 256] float, 60K × 256 × 4B ≈ 58.6 MB
    // 放在 section 表末尾, 旧版 checkpoint 未含此 section 时 load 端可优雅降级
    add(&s, "decode_weights", b.d_decode_weights,
        (uint64_t)N_TOTAL_NEURONS_2E * 256, sizeof(float));
    // 课程 readout 权重 (Phase 3a-D3, 调质 N×6 + 工具 N×7)
    // 放在 section 表末尾, 旧版 checkpoint 未含时 load 端优雅降级 (随机初始化)
    add(&s, "curriculum_readout", b.d_curriculum_readout_weights,
        (uint64_t)N_TOTAL_NEURONS_2E * 6, sizeof(float));
    add(&s, "curriculum_tool_readout", b.d_curriculum_tool_weights,
        (uint64_t)N_TOTAL_NEURONS_2E * CURRICULUM_N_TOOL, sizeof(float));
    // 具身策略权重 (Phase 3a-D1, 2026-08-01), 放末尾, 旧 checkpoint 缺失时优雅降级 (随机初始化)
    add(&s, "l5_to_motor_synapses", b.d_l5_to_motor_synapses,
        (uint64_t)N_MOTOR_NEURONS * L5_TO_MOTOR_SYNAPSES_PER_NEURON, sizeof(BioSynapse));
    add(&s, "l5_to_motor_csr", b.d_l5_to_motor_csr_row_ptr,
        (uint64_t)N_MOTOR_NEURONS + 1, sizeof(int));
    add(&s, "motor_neurons", b.d_motor_neurons,
        (uint64_t)N_MOTOR_NEURONS, sizeof(NeuronStateAdEx));
    // 课程窗口累计 spike 缓冲 (窗口内平均发放率特征, resume 时延续部分窗口状态)
    add(&s, "curriculum_accum_spikes", b.d_curriculum_accum_spikes,
        (uint64_t)N_TOTAL_NEURONS_2E, sizeof(float));
    // PAD 情感 readout 权重 (Phase 3a-D3, N×3, 2026-08-02 Task 5)
    // 放在 section 表最末尾, 旧版 checkpoint 未含此节时 load 端优雅降级 (随机初始化,
    // 只重初始化 PAD 头, 不触碰已加载的 mod/tool readout 权重)
    add(&s, "curriculum_pad_readout", b.d_curriculum_pad_weights,
        (uint64_t)N_TOTAL_NEURONS_2E * 3, sizeof(float));
    // B4 修复 (2026-08-04): PCA 均值 GPU 镜像持久化 — 原实现只存 d_pca_W,
    //   resume 后 d_pca_mean_ 为构造期零值, h_mean_fr_ (零) 同步覆盖 GPU 均值,
    //   均值流形丢失. 放节表末尾保证旧 checkpoint 前缀兼容 (缺失即零, 与
    //   h_mean_fr_ 构造初值一致).
    add(&s, "pca_mean", self->d_pca_mean_, N_ASSOCIATION_NEURONS_2E, sizeof(float));
    // 2026-08-04 方案2: 浓度 readout 头权重 (6×6×4B = 144 B)
    //   放节表最末尾, 旧 checkpoint 缺失时 load 端随机初始化 (小值, 与训练初始一致)
    add(&s, "curriculum_conc_readout", b.d_curriculum_conc_weights, 6 * 6, sizeof(float));
    return s;
}

SchedulerState SchedulerCheckpointAccess::capture_state(
    BioMechanismScheduler* self, int next_step, uint32_t seed) {
    SchedulerState x{};
    x.state_version = 1;
    x.state_bytes = sizeof(x);
    x.next_step = next_step;
    x.topology_seed = seed;
    x.corpus_size = text_corpus_size();
    x.corpus_position = text_stream_position();
    x.corpus_fingerprint = text_corpus_fingerprint();
    x.e0_ablation = self->e0_ablation ? 1 : 0;
    x.stats = self->stats_;
    x.delay_ring_idx = self->delay_ring_idx_;
    x.last_phase = self->last_phase_;
    x.total_steps = self->total_steps_;
    x.total_spikes_accum = self->total_spikes_accum_;
    x.inject_spikes_accum = self->inject_spikes_accum_;
    x.min_spikes_per_step = self->min_spikes_per_step_;
    x.max_spikes_per_step = self->max_spikes_per_step_;
    x.total_burst_steps = self->total_burst_steps_;
    x.total_single_neuron_burst_spikes = self->total_single_neuron_burst_spikes_;
    x.arrived_events_accum = self->arrived_events_accum_;
    x.dispatched_events_accum = self->dispatched_events_accum_;
    x.dropped_events_accum = self->dropped_events_accum_;
    x.max_delay_slot_depth = self->max_delay_slot_depth_;
    x.p3_inhibitory_updates = self->p3_inhibitory_updates_;
    x.p3_wm_updates = self->p3_wm_updates_;
    x.p3_last_activity_drive = self->p3_last_activity_drive_;
    x.p3_kwta_updates = self->p3_kwta_updates_;
    x.p3_kwta_active_columns = self->p3_kwta_active_columns_;
    x.p3_kwta_winner_estimate = self->p3_kwta_winner_estimate_;
    x.p3_kwta_suppressed_estimate = self->p3_kwta_suppressed_estimate_;
    x.p3_kwta_target_per_column = self->p3_kwta_target_per_column_;
    x.p3_semantic_eval_updates = self->p3_semantic_eval_updates_;
    x.p3_semantic_eval_last_step = self->p3_semantic_eval_last_step_;
    x.p3_silhouette_score = self->p3_silhouette_score_;
    x.p3_js_divergence_mean = self->p3_js_divergence_mean_;
    x.p3_js_divergence_max = self->p3_js_divergence_max_;
    x.p3_column_ratio = self->p3_column_ratio_;
    x.l6_total_spikes_last = self->l6_total_spikes_last_;
    x.l6_activity_ema_mean = self->l6_activity_ema_mean_;
    std::copy_n(self->layer_activation_delay_, 5, x.layer_activation_delay);
    std::copy_n(self->layer_chi2_sig_ratio_, 5, x.layer_chi2_sig_ratio);
    std::copy_n(self->layer_chi2_mean_, 5, x.layer_chi2_mean);
    x.gate_mean = self->gate_mean_;
    x.gate_open_ratio = self->gate_open_ratio_;
    x.runtime_state_valid = export_delay_queue_state(&x.delay_queue) ? 1 : 0;
    x.modulatory_runtime = export_modulatory_runtime_state();

    // ==================== Task E1: BPTT 状态捕获 ====================
    x.bptt_enabled = self->bptt_enabled ? 1 : 0;
    x.bptt_input_mode = static_cast<uint8_t>(self->bptt_input_mode);
    x.bptt_window_size = self->bptt_window_size;
    // BPTTConfig 从 bptt_trainer_ 读取 (若已初始化)
    if (self->bptt_trainer_ != nullptr) {
        const BPTTConfig& cfg = self->bptt_trainer_->config();
        x.bptt_lr = cfg.lr;
        x.bptt_grad_clip = cfg.grad_clip;
        x.bptt_warmup_steps = cfg.warmup_steps;
        x.bptt_surrogate_alpha = cfg.surrogate_alpha;
        x.bptt_beta = cfg.beta;
        x.bptt_threshold = cfg.threshold;
    } else {
        // 未初始化, 用默认值
        x.bptt_lr = 0.01f;
        x.bptt_grad_clip = 5.0f;
        x.bptt_warmup_steps = 1000;
        x.bptt_surrogate_alpha = 4.0f;
        x.bptt_beta = 0.9f;
        x.bptt_threshold = 1.0f;
    }
    x.bptt_step_counter = self->bptt_step_counter_;
    x.bptt_last_loss = self->bptt_last_loss_;
    x.bptt_last_grad_norm = self->bptt_last_grad_norm_;
    x.bptt_current_lr = self->bptt_current_lr_;
    x.bptt_last_target_token = self->bptt_last_target_token_;

    return x;
}

bool SchedulerCheckpointAccess::restore_state(BioMechanismScheduler* self,
                                              const SchedulerState& x) {
    self->e0_ablation = x.e0_ablation != 0;
    self->stats_ = x.stats;
    self->delay_ring_idx_ = x.delay_ring_idx;
    self->last_phase_ = x.last_phase;
    self->total_steps_ = x.total_steps;
    self->total_spikes_accum_ = x.total_spikes_accum;
    self->inject_spikes_accum_ = x.inject_spikes_accum;
    self->min_spikes_per_step_ = x.min_spikes_per_step;
    self->max_spikes_per_step_ = x.max_spikes_per_step;
    self->total_burst_steps_ = x.total_burst_steps;
    self->total_single_neuron_burst_spikes_ = x.total_single_neuron_burst_spikes;
    self->arrived_events_accum_ = x.arrived_events_accum;
    self->dispatched_events_accum_ = x.dispatched_events_accum;
    self->dropped_events_accum_ = x.dropped_events_accum;
    self->max_delay_slot_depth_ = x.max_delay_slot_depth;
    self->p3_inhibitory_updates_ = x.p3_inhibitory_updates;
    self->p3_wm_updates_ = x.p3_wm_updates;
    self->p3_last_activity_drive_ = x.p3_last_activity_drive;
    self->p3_kwta_updates_ = x.p3_kwta_updates;
    self->p3_kwta_active_columns_ = x.p3_kwta_active_columns;
    self->p3_kwta_winner_estimate_ = x.p3_kwta_winner_estimate;
    self->p3_kwta_suppressed_estimate_ = x.p3_kwta_suppressed_estimate;
    self->p3_kwta_target_per_column_ = x.p3_kwta_target_per_column;
    self->p3_semantic_eval_updates_ = x.p3_semantic_eval_updates;
    self->p3_semantic_eval_last_step_ = x.p3_semantic_eval_last_step;
    self->p3_silhouette_score_ = x.p3_silhouette_score;
    self->p3_js_divergence_mean_ = x.p3_js_divergence_mean;
    self->p3_js_divergence_max_ = x.p3_js_divergence_max;
    self->p3_column_ratio_ = x.p3_column_ratio;
    self->l6_total_spikes_last_ = x.l6_total_spikes_last;
    self->l6_activity_ema_mean_ = x.l6_activity_ema_mean;
    std::copy_n(x.layer_activation_delay, 5, self->layer_activation_delay_);
    std::copy_n(x.layer_chi2_sig_ratio, 5, self->layer_chi2_sig_ratio_);
    std::copy_n(x.layer_chi2_mean, 5, self->layer_chi2_mean_);
    self->gate_mean_ = x.gate_mean;
    self->gate_open_ratio_ = x.gate_open_ratio;
    const bool delay_ok = import_delay_queue_state(x.delay_queue);
    import_modulatory_runtime_state(x.modulatory_runtime);

    // ==================== Task E1: BPTT 状态恢复 ====================
    // 注意: V/S history 不持久化 (BPTTTrainer 构造时重新分配并清零)
    //       恢复时通过 init_bptt() 重建 BPTTTrainer, 然后填充运行时状态
    self->bptt_enabled = (x.bptt_enabled != 0);
    self->bptt_input_mode = static_cast<int>(x.bptt_input_mode);
    self->bptt_window_size = x.bptt_window_size;

    // 若 checkpoint 中 BPTT 启用, 重建 BPTTTrainer
    // N3F 课程模式 (learning_rule_=="n3f"): 不重建 BPTT trainer —
    //   三因子在线学习无窗口重放/无历史缓冲, BPTT 反传会与教学信号混合干扰
    if (self->bptt_enabled && !self->n3f_mode()) {
        BPTTConfig cfg;
        cfg.window_size = x.bptt_window_size;
        cfg.lr = x.bptt_lr;
        cfg.grad_clip = x.bptt_grad_clip;
        cfg.warmup_steps = x.bptt_warmup_steps;
        cfg.surrogate_alpha = x.bptt_surrogate_alpha;
        cfg.beta = x.bptt_beta;
        cfg.threshold = x.bptt_threshold;
        // init_bptt 会 new BPTTTrainer, 分配 device 缓冲
        // (V/S history 重新分配并清零, 不恢复历史 — 下一个 BPTT 窗口重新开始)
        self->init_bptt(cfg);
        // 恢复运行时状态 (覆盖 init_bptt 设置的默认值)
        self->bptt_step_counter_ = x.bptt_step_counter;
        self->bptt_last_loss_ = x.bptt_last_loss;
        self->bptt_last_grad_norm_ = x.bptt_last_grad_norm;
        self->bptt_current_lr_ = x.bptt_current_lr;
        self->bptt_last_target_token_ = x.bptt_last_target_token;
    }

    return delay_ok;
}

namespace {

bool section_layout_matches(const std::vector<Section>& expected,
                            const std::vector<DiskSection>& actual) {
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::strncmp(expected[i].name, actual[i].name, sizeof(actual[i].name)) != 0 ||
            expected[i].bytes != actual[i].bytes) return false;
    }
    return true;
}

} // namespace

int BioMechanismScheduler::save_checkpoint(int next_step, const char* dir,
                                           uint32_t topology_seed) {
    if (!dir || !*dir || next_step < 0) return 1;
    // b.d_stdp_trace_epoch is a transient acceleration cache. Persist fully decayed traces
    // so the existing checkpoint schema remains backward compatible.
    materialize_stdp_traces(alloc_, next_step > 0 ? next_step - 1 : 0);
    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "[Checkpoint] CUDA sync failed: %s\n", cudaGetErrorString(sync_err));
        return 2;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "[Checkpoint] cannot create %s: %s\n", dir, ec.message().c_str());
        return 3;
    }

    const std::filesystem::path final_path =
        std::filesystem::path(dir) / ("ckpt_step" + std::to_string(next_step) + ".snn2e");
    const std::filesystem::path temp_path = final_path.string() + ".tmp";
    if (std::filesystem::exists(final_path, ec)) {
        std::fprintf(stderr, "[Checkpoint] refusing to overwrite %s\n", final_path.string().c_str());
        return 4;
    }
    std::filesystem::remove(temp_path, ec);

    SchedulerState state = SchedulerCheckpointAccess::capture_state(this, next_step, topology_seed);
    if (!state.runtime_state_valid) {
        std::fprintf(stderr, "[Checkpoint] failed to capture delay queue runtime state\n");
        return 5;
    }
    auto sections = SchedulerCheckpointAccess::make_sections(this, &state);
    for (const auto& section : sections) {
        if (!section.pointer && section.bytes > 0) {
            std::fprintf(stderr, "[Checkpoint] null section: %s\n", section.name);
            return 5;
        }
    }
    CheckpointHeader header{};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kVersion;
    header.header_bytes = sizeof(header);
    header.section_count = static_cast<uint32_t>(sections.size());
    header.n_neurons = N_TOTAL_NEURONS_2E;
    header.n_synapses = N_TOTAL_SYNAPSES_2E;
    header.bio_synapse_bytes = sizeof(BioSynapse);
    header.neuron_state_bytes = sizeof(NeuronStateAdEx);
    for (const auto& section : sections) header.payload_bytes += section.bytes;

    FILE* fp = std::fopen(temp_path.string().c_str(), "wb+");
    if (!fp) return 5;
    bool ok = write_exact(fp, &header, sizeof(header));
    for (const auto& section : sections) {
        DiskSection disk{};
        std::snprintf(disk.name, sizeof(disk.name), "%s", section.name);
        disk.bytes = section.bytes;
        ok = ok && write_exact(fp, &disk, sizeof(disk));
    }

    std::vector<unsigned char> chunk(kChunkBytes);
    uint64_t checksum = 0;
    for (const auto& section : sections) {
        uint64_t offset = 0;
        while (ok && offset < section.bytes) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(chunk.size(), section.bytes - offset));
            if (section.device) {
                const cudaError_t err = cudaMemcpy(chunk.data(),
                    static_cast<const char*>(section.pointer) + offset,
                    count, cudaMemcpyDeviceToHost);
                ok = err == cudaSuccess;
            } else {
                std::memcpy(chunk.data(), static_cast<const char*>(section.pointer) + offset, count);
            }
            if (ok) {
                checksum = update_checksum(checksum, chunk.data(), count);
                ok = write_exact(fp, chunk.data(), count);
            }
            offset += count;
        }
    }

    CheckpointFooter footer{};
    std::memcpy(footer.magic, kFooterMagic, sizeof(kFooterMagic));
    footer.payload_checksum = checksum;
    ok = ok && write_exact(fp, &footer, sizeof(footer));
    header.payload_checksum = checksum;
    ok = ok && std::fseek(fp, 0, SEEK_SET) == 0 && write_exact(fp, &header, sizeof(header));
    ok = ok && sync_file(fp);
    if (std::fclose(fp) != 0) ok = false;
    if (!ok) {
        std::filesystem::remove(temp_path, ec);
        std::fprintf(stderr, "[Checkpoint] write failed for step %d\n", next_step);
        return 6;
    }

    std::filesystem::rename(temp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        return 7;
    }
    std::printf("[Checkpoint] saved next_step=%d: %s (%.1f MiB, checksum=%016llx)\n",
                next_step, final_path.string().c_str(),
                header.payload_bytes / (1024.0 * 1024.0),
                static_cast<unsigned long long>(checksum));
    return 0;
}

int BioMechanismScheduler::load_checkpoint(const char* path, int* next_step,
                                           uint32_t* topology_seed) {
    if (!path || !next_step || !topology_seed) return 1;
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return 2;

    CheckpointHeader header{};
    bool ok = read_exact(fp, &header, sizeof(header));
    if (!ok || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
        header.version != kVersion || header.header_bytes != sizeof(header) ||
        header.n_neurons != N_TOTAL_NEURONS_2E ||
        header.n_synapses != N_TOTAL_SYNAPSES_2E ||
        header.bio_synapse_bytes != sizeof(BioSynapse) ||
        header.neuron_state_bytes != sizeof(NeuronStateAdEx)) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] incompatible header: %s\n", path);
        return 3;
    }

    std::vector<DiskSection> disk(header.section_count);
    ok = read_exact(fp, disk.data(), disk.size() * sizeof(DiskSection));
    SchedulerState state{};
    auto sections = SchedulerCheckpointAccess::make_sections(this, &state);
    // 兼容: 磁盘 section 可以是新 sections 的严格前缀
    //   (缺失末尾 1-2 个 section: 旧版无课程 readout / 更旧无 decode_weights;
    //    新增 4 个具身尾部节 (l5_to_motor_synapses/csr/motor_neurons/curriculum_accum_spikes)
    //    后旧 checkpoint 最多缺 4 节, 放宽上限到 6 仍可优雅降级)
    int missing_tail = 0;
    if (!ok || !section_layout_matches(sections, disk)) {
        const bool prefix_layout_ok =
            disk.size() <= sections.size() &&
            sections.size() - disk.size() <= 6;
        bool prefix_ok = prefix_layout_ok;
        for (size_t i = 0; prefix_ok && i < disk.size(); ++i) {
            if (std::strncmp(sections[i].name, disk[i].name, sizeof(disk[i].name)) != 0 ||
                sections[i].bytes != disk[i].bytes) {
                prefix_ok = false;
            }
        }
        if (!ok || !prefix_ok) {
            std::fclose(fp);
            std::fprintf(stderr, "[Checkpoint] section layout mismatch: %s\n", path);
            return 4;
        }
        missing_tail = static_cast<int>(sections.size() - disk.size());
    }
    uint64_t expected_payload_bytes = 0;
    for (size_t si = 0; si < sections.size(); ++si) {
        if (static_cast<int>(si) >= static_cast<int>(sections.size()) - missing_tail) {
            continue;  // 缺失的尾部 section 不计入 payload
        }
        if (!sections[si].pointer && sections[si].bytes > 0) {
            std::fclose(fp);
            return 4;
        }
        expected_payload_bytes += sections[si].bytes;
    }
    if (header.payload_bytes != expected_payload_bytes) {
        std::fclose(fp);
        return 4;
    }

    const long payload_offset = std::ftell(fp);
    std::vector<unsigned char> chunk(kChunkBytes);
    uint64_t checksum = 0;
    uint64_t remaining = header.payload_bytes;
    while (ok && remaining > 0) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(chunk.size(), remaining));
        ok = read_exact(fp, chunk.data(), count);
        if (ok) checksum = update_checksum(checksum, chunk.data(), count);
        remaining -= count;
    }
    CheckpointFooter footer{};
    ok = ok && read_exact(fp, &footer, sizeof(footer));
    if (!ok || checksum != header.payload_checksum ||
        std::memcmp(footer.magic, kFooterMagic, sizeof(kFooterMagic)) != 0 ||
        footer.payload_checksum != checksum) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] checksum/completion validation failed: %s\n", path);
        return 5;
    }

    if (std::fseek(fp, payload_offset, SEEK_SET) != 0 ||
        !read_exact(fp, &state, sizeof(state)) ||
        state.state_version != 1 || state.state_bytes != sizeof(state)) {
        std::fclose(fp);
        return 6;
    }
    if (!skip_corpus_check &&
        (state.corpus_size != text_corpus_size() ||
         state.corpus_fingerprint != text_corpus_fingerprint())) {
        std::fclose(fp);
        std::fprintf(stderr, "[Checkpoint] corpus does not match checkpoint\n");
        return 8;
    }

    if (std::fseek(fp, payload_offset, SEEK_SET) != 0) {
        std::fclose(fp);
        return 6;
    }
    std::vector<std::string> missing_names;
    for (size_t si = 0; si < sections.size(); ++si) {
        const auto& section = sections[si];
        if (static_cast<int>(si) >= static_cast<int>(sections.size()) - missing_tail) {
            // 缺失的尾部 section: 跳过读取, 循环后按类型初始化
            missing_names.emplace_back(section.name);
            continue;
        }
        uint64_t offset = 0;
        while (ok && offset < section.bytes) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(chunk.size(), section.bytes - offset));
            ok = read_exact(fp, chunk.data(), count);
            if (ok && section.device) {
                const cudaError_t err = cudaMemcpy(
                    static_cast<char*>(section.pointer) + offset,
                    chunk.data(), count, cudaMemcpyHostToDevice);
                ok = err == cudaSuccess;
            } else if (ok) {
                std::memcpy(static_cast<char*>(section.pointer) + offset, chunk.data(), count);
            }
            offset += count;
        }
    }
    // 缺失 section 的初始化策略 (按类型):
    //   decode_weights: 零初始化 (原兼容行为)
    //   curriculum readout: 随机小值 (与训练初始一致, 支持续训重新学习)
    PersistentBuffers& pb = alloc_->buffers();
    for (const auto& name : missing_names) {
        if (name == "decode_weights") {
            const cudaError_t err = cudaMemset(pb.d_decode_weights, 0,
                (size_t)N_TOTAL_NEURONS_2E * 256 * sizeof(float));
            if (err != cudaSuccess) ok = false;
            std::printf("[Checkpoint] 警告: decode_weights section 未找到, 用零初始化\n");
        } else if (name == "curriculum_readout" || name == "curriculum_tool_readout") {
            launch_curriculum_readout_init(pb, 0.01f, 42u);
            std::printf("[Checkpoint] 警告: curriculum readout 未找到, 用随机小值初始化\n");
        } else if (name == "curriculum_pad_readout") {
            // 只重初始化 PAD 头 (2026-08-02 Task 5):
            //   旧 checkpoint 通常含 mod/tool 节但缺 PAD 节; 全量 launch_curriculum_readout_init
            //   会重随机化已加载的 mod/tool 权重 → 必须用 PAD-only init
            launch_curriculum_pad_readout_init(pb, 0.01f, 42u);
            std::printf("[Checkpoint] 警告: curriculum_pad_readout 未找到, 用随机小值初始化 (仅 PAD 头)\n");
        } else if (name == "curriculum_conc_readout") {
            // 2026-08-04 方案2: 只初始化浓度头 (6×6), 不触碰 mod/tool/pad 权重
            launch_curriculum_conc_readout_init(pb, 0.01f, 42u);
            std::printf("[Checkpoint] 警告: curriculum_conc_readout 未找到, 用随机小值初始化 (仅浓度头)\n");
        }
    }
    std::fclose(fp);
    if (!ok || state.state_version != 1 || state.state_bytes != sizeof(state)) return 7;
    if (!set_text_stream_position(static_cast<size_t>(state.corpus_position))) return 9;
    if (!state.runtime_state_valid || !SchedulerCheckpointAccess::restore_state(this, state)) {
        return 10;
    }
    set_e0_ablation(e0_ablation);

    // B4 修复 (2026-08-04): resume 后同步 GPU PCA 基矩阵/均值回 CPU 镜像
    //   根因: h_pca_W_/h_mean_fr_ 是 CPU 端 Oja 镜像 (构造时随机/零值), 不随
    //   checkpoint 恢复; 下一 PCA_SYNC_INTERVAL 周期 (launch_pca_update_cpu)
    //   会用随机/零镜像覆盖 GPU d_pca_W → PCA 流形破坏 (签名/海马/WM 检索).
    //   恢复后立即 D2H 回填镜像, 与 GPU 保持一致. 旧 checkpoint 缺 pca_mean
    //   节时 d_pca_mean_ 保持构造期零值, 回填 h_mean_fr_ 也为零, 行为一致.
    if (d_pca_mean_ && !h_pca_W_.empty() && !h_mean_fr_.empty() && pb.d_pca_W) {
        const size_t pca_w_bytes =
            (size_t)N_ASSOCIATION_NEURONS_2E * PCA_N_COMPONENTS * sizeof(float);
        const cudaError_t err_w = cudaMemcpy(h_pca_W_.data(), pb.d_pca_W,
                                             pca_w_bytes, cudaMemcpyDeviceToHost);
        const cudaError_t err_m = cudaMemcpy(h_mean_fr_.data(), d_pca_mean_,
                                             (size_t)N_ASSOCIATION_NEURONS_2E * sizeof(float),
                                             cudaMemcpyDeviceToHost);
        if (err_w != cudaSuccess || err_m != cudaSuccess) {
            std::fprintf(stderr, "[Checkpoint] PCA 镜像回填失败: %s / %s\n",
                         cudaGetErrorString(err_w), cudaGetErrorString(err_m));
            return 11;
        }
    }
    *next_step = state.next_step;
    *topology_seed = state.topology_seed;
    // Loaded checkpoints contain eager traces materialized after the previous
    // step. Seed the transient lazy epochs without changing the file format.
    reset_stdp_trace_epochs(alloc_, *next_step > 0 ? *next_step - 1 : 0);
    std::printf("[Checkpoint] resumed next_step=%d from %s\n", *next_step, path);
    return 0;
}

int BioMechanismScheduler::prune_checkpoints(const char* dir, int keep_latest) {
    if (!dir || keep_latest <= 0) return 0;
    std::error_code ec;
    struct NumberedCheckpoint {
        long long step;
        std::filesystem::path path;
    };
    std::vector<NumberedCheckpoint> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) return 1;
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && name.rfind("ckpt_step", 0) == 0 &&
            entry.path().extension() == ".snn2e") {
            const std::string digits = name.substr(9, name.size() - 9 - 6);
            try {
                size_t consumed = 0;
                const long long step = std::stoll(digits, &consumed);
                if (consumed == digits.size()) files.push_back({step, entry.path()});
            } catch (const std::exception&) {
                continue;
            }
        }
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return a.step > b.step;
    });
    for (size_t i = static_cast<size_t>(keep_latest); i < files.size(); ++i) {
        std::filesystem::remove(files[i].path, ec);
        if (ec) return 2;
    }
    return 0;
}

} // namespace stage2e
