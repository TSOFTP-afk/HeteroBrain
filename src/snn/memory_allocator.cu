// =============================================================================
// Stage 2e 显存分配器实现 (P0)
// =============================================================================

#include "memory_allocator.cuh"
#include <cstdio>

namespace stage2e {

template<typename T>
T* MemoryAllocator::alloc(size_t count, const char* name, size_t* accum_bytes) {
    size_t bytes = count * sizeof(T);
    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[Stage2e] CUDA 分配失败: %s, 需要 %zu 字节 (%.2f MB)\n",
                name, bytes, bytes / (1024.0 * 1024.0));
        fprintf(stderr, "         错误: %s\n", cudaGetErrorString(err));
        allocation_failed_ = true;
        return nullptr;
    }
    // 清零 (避免 NaN 传播)
    cudaMemset(ptr, 0, bytes);
    *accum_bytes += bytes;
    vram_used_ += bytes;
    if (vram_used_ > vram_peak_) vram_peak_ = vram_used_;
    printf("  [+]. %-32s %10zu × %2zuB = %8.2f MB (累计 %7.2f MB)\n",
           name, count, sizeof(T), bytes / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0));
    return ptr;
}

size_t MemoryAllocator::allocate_all() {
    allocation_failed_ = false;
    printf("[Stage2e P0] 开始分配 GPU 显存...\n");
    printf("  %-34s %10s %8s %10s %10s\n",
           "Buffer", "Count", "Size", "Bytes", "Total MB");
    printf("  %-34s %10s %8s %10s %10s\n",
           "------", "-----", "----", "-----", "--------");

    size_t total = 0;

    // --- 神经元 ---
    d_bufs_.d_neurons           = alloc<NeuronStateAdEx>(N_TOTAL_NEURONS_2E, "d_neurons", &total);
    d_bufs_.d_spike_flags       = alloc<bool>(N_TOTAL_NEURONS_2E, "d_spike_flags", &total);

    // --- 突触结构 (CSR) ---
    d_bufs_.d_synapses          = alloc<BioSynapse>(N_TOTAL_SYNAPSES_2E, "d_synapses", &total);
    d_bufs_.d_csr_row_ptr       = alloc<int>(N_TOTAL_NEURONS_2E + 1, "d_csr_row_ptr", &total);
    d_bufs_.d_csr_col_idx       = alloc<int>(N_TOTAL_SYNAPSES_2E, "d_csr_col_idx", &total);
    d_bufs_.d_weights_cache     = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_weights_cache", &total);
    d_bufs_.d_eligibility       = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_eligibility", &total);
    d_bufs_.d_eligibility_slow  = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_eligibility_slow", &total);

    // --- PSW 概率突触权重 (10.7M × 4B × 2 = 85.6 MB) ---
    d_bufs_.d_synapse_alpha     = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_synapse_alpha (PSW)", &total);
    d_bufs_.d_synapse_beta      = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_synapse_beta (PSW)", &total);

    // --- v3 强化 A: PCA 反投影矩阵 (55K × 50 × 4B = 11 MB) ---
    d_bufs_.d_pca_W             = alloc<float>((size_t)N_TOTAL_NEURONS_2E * PATTERN_DIM,
                                               "d_pca_W (55K×50)", &total);

    // --- v3 强化 G: NMDA 钙浓度快照 ---
    d_bufs_.d_ca_snapshot       = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_ca_snapshot", &total);
    d_bufs_.d_ca_history_sparse = alloc<float>((size_t)CA_HISTORY_MAX_ACTIVE * CA_HISTORY_LEN,
                                               "d_ca_history_sparse", &total);

    // --- v4 强化 I: 突触传导延迟 ---
    d_bufs_.d_synapse_delay     = alloc<uint8_t>(N_TOTAL_SYNAPSES_2E, "d_synapse_delay", &total);
    // 延迟环形队列
    d_bufs_.d_delay_ring_indices = alloc<int>((size_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY,
                                              "d_delay_ring_indices", &total);
    d_bufs_.d_delay_ring_current = alloc<float>((size_t)DELAY_STEPS_MAX * DELAY_RING_SLOT_CAPACITY,
                                                "d_delay_ring_current", &total);

    // --- v4 强化 J: STDP x_pre trace ---
    d_bufs_.d_stdp_x_pre_trace  = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_stdp_x_pre_trace", &total);
    d_bufs_.d_stdp_trace_epoch  = alloc<int>(N_TOTAL_SYNAPSES_2E, "d_stdp_trace_epoch (lazy)", &total);

    // --- eligibility 闭环修复: 神经元级 eligibility (60K × 4B = 240KB) ---
    d_bufs_.d_neuron_eligibility = alloc<float>(N_TOTAL_NEURONS_2E, "d_neuron_eligibility", &total);

    // --- v4 强化 K: CaMKII activity ---
    d_bufs_.d_camkii_activity   = alloc<float>(N_TOTAL_SYNAPSES_2E, "d_camkii_activity", &total);

    // --- 输入电流 ---
    d_bufs_.d_input_current     = alloc<float>(N_TOTAL_NEURONS_2E, "d_input_current", &total);
    d_bufs_.d_nmda_current      = alloc<float>(N_TOTAL_NEURONS_2E, "d_nmda_current", &total);
    d_bufs_.d_nmda_post_state   = alloc<float2>(N_TOTAL_NEURONS_2E, "d_nmda_post_state", &total);
    d_bufs_.d_inhibitory_current = alloc<float>(N_TOTAL_NEURONS_2E, "d_inhibitory_current", &total);

    // --- 调质浓度 (Phase 3a: 6 维) ---
    d_bufs_.d_da_concentration      = alloc<float>(N_TOTAL_NEURONS_2E, "d_da_concentration", &total);
    d_bufs_.d_ach_concentration     = alloc<float>(N_TOTAL_NEURONS_2E, "d_ach_concentration", &total);
    d_bufs_.d_ne_concentration      = alloc<float>(N_TOTAL_NEURONS_2E, "d_ne_concentration", &total);
    d_bufs_.d_ht5_concentration     = alloc<float>(N_TOTAL_NEURONS_2E, "d_ht5_concentration", &total);
    d_bufs_.d_gaba_concentration    = alloc<float>(N_TOTAL_NEURONS_2E, "d_gaba_concentration", &total);
    d_bufs_.d_oxytocin_concentration= alloc<float>(N_TOTAL_NEURONS_2E, "d_oxytocin_concentration", &total);
    // Phase 3a: 催产素受体密度 (按突触索引, 仅 SocialBonding 子集非零)
    d_bufs_.d_oxytocin_receptor     = alloc<uint8_t>(N_TOTAL_SYNAPSES_2E, "d_oxytocin_receptor", &total);

    // --- v3 强化 C: 海马体索引 ---
    d_bufs_.d_hippo_indices     = alloc<HippoIndex>(HIPP_INDEX_SIZE, "d_hippo_indices", &total);
    // Task 3: 海马 LRU 游标 + 已填充计数 + top-K 索引缓冲 (1 + 1 + 200 int = 808B)
    //   alloc() 已 cudaMemset 为 0: write_cursor=0, filled_count=0, top_k 全 0
    d_bufs_.d_hippo_write_cursor = alloc<int>(1, "d_hippo_write_cursor", &total);
    d_bufs_.d_hippo_filled_count = alloc<int>(1, "d_hippo_filled_count", &total);
    d_bufs_.d_hippo_top_k        = alloc<int>(HIPP_REPLAY_BATCH, "d_hippo_top_k", &total);

    // --- v3 强化 D: 共激活跟踪器 ---
    d_bufs_.d_coact_trackers    = alloc<CoactTracker>(COACT_TRACKER_SIZE, "d_coact_trackers", &total);
    // Task 6: tracker 条目计数 (append-only; alloc 已 cudaMemset 为 0)
    d_bufs_.d_tracker_count     = alloc<int>(1, "d_tracker_count", &total);

    // --- v3 强化 E: 工作记忆槽位 ---
    d_bufs_.d_wm_slots          = alloc<WMSlot>(WM_SLOTS, "d_wm_slots", &total);

    // --- Task 9: WM 完整闭环缓冲 (LRU 游标 + 前额叶输入电流) ---
    d_bufs_.d_wm_write_cursor   = alloc<int>(1, "d_wm_write_cursor (LRU)", &total);
    d_bufs_.d_prefrontal_input  = alloc<float>(N_PREFRONTAL_NEURONS,
                                                "d_prefrontal_input (5K)", &total);

    // --- DA 价值函数相关 ---
    d_bufs_.d_subcolumn_fr      = alloc<float>(W_VALUE_DIM, "d_subcolumn_fr", &total);
    d_bufs_.d_baseline_fr       = alloc<float>(W_VALUE_DIM, "d_baseline_fr", &total);
    d_bufs_.d_w_pred            = alloc<float>((size_t)W_PRED_DIM * W_PRED_DIM, "d_w_pred (200×200)", &total);
    d_bufs_.d_w_value           = alloc<float>(W_VALUE_DIM, "d_w_value", &total);
    d_bufs_.d_pred_fr           = alloc<float>(W_PRED_DIM, "d_pred_fr", &total);
    // 上一步亚柱发放率 (W_pred 完整矩阵预测输入, 200 × 4B = 800 字节)
    d_bufs_.d_subcol_fr_prev    = alloc<float>(W_VALUE_DIM, "d_subcol_fr_prev", &total);

    // --- 字节直方图 (NE 用) ---
    d_bufs_.d_byte_histogram    = alloc<int>(256, "d_byte_histogram", &total);

    // --- 卡方检验: 神经元×字节发放计数 (55K × 256 = 14.08M int32 = 56 MB) ---
    d_bufs_.d_neuron_byte_counts = alloc<int>(N_TOTAL_NEURONS_2E * 256,
                                              "d_neuron_byte_counts (55K×256)", &total);

    // --- 重放注入缓冲 ---
    d_bufs_.d_replay_injection  = alloc<float>(N_TOTAL_NEURONS_2E, "d_replay_injection", &total);

    // -----------------------------------------------------------------
    // 语言运动皮层 (Stage 2e "语言运动皮层"基础设施)
    // 新增显存约 79 MB (主要项: d_decode_weights 58.6 MB + d_l5_to_motor_synapses 19 MB)
    // -----------------------------------------------------------------
    // 运动皮层神经元状态 (5K × 56B = 0.28 MB) + 脉冲标志 (5K × 1B = 5 KB)
    d_bufs_.d_motor_neurons        = alloc<NeuronStateAdEx>(N_MOTOR_NEURONS,
                                                            "d_motor_neurons", &total);
    d_bufs_.d_motor_spike_flags    = alloc<bool>(N_MOTOR_NEURONS,
                                                 "d_motor_spike_flags", &total);

    // 线性解码器权重矩阵 (60K × 256 × 4B = 58.6 MB, 最大项)
    d_bufs_.d_decode_weights       = alloc<float>((size_t)N_TOTAL_NEURONS_2E * 256,
                                                  "d_decode_weights (60K×256)", &total);
    // 解码 logits / 误差 (256 × 4B = 1 KB)
    d_bufs_.d_decode_logits        = alloc<float>(256, "d_decode_logits", &total);
    d_bufs_.d_decode_error         = alloc<float>(256, "d_decode_error", &total);
    // 预测字节 (1 × 4B, host-readable)
    d_bufs_.d_decode_predicted_byte = alloc<int>(1, "d_decode_predicted_byte", &total);

    // 课程训练调质 readout 层 (Phase 3a-D3, 60K×6×4B = 1.4 MB)
    d_bufs_.d_curriculum_readout_weights = alloc<float>((size_t)N_TOTAL_NEURONS_2E * 6,
                                                        "d_curriculum_readout (60K×6)", &total);
    d_bufs_.d_curriculum_logits  = alloc<float>(6, "d_curriculum_logits", &total);
    d_bufs_.d_curriculum_error   = alloc<float>(6, "d_curriculum_error", &total);
    d_bufs_.d_curriculum_tool_weights = alloc<float>((size_t)N_TOTAL_NEURONS_2E * 7,
                                                     "d_curriculum_tool_readout (60K×7)", &total);
    d_bufs_.d_curriculum_tool_logits  = alloc<float>(7, "d_curriculum_tool_logits", &total);
    d_bufs_.d_curriculum_tool_error   = alloc<float>(7, "d_curriculum_tool_error", &total);
    // PAD 情感 readout (2026-08-02 Task 5, 60K×3×4B = 0.7 MB)
    d_bufs_.d_curriculum_pad_weights = alloc<float>((size_t)N_TOTAL_NEURONS_2E * 3,
                                                    "d_curriculum_pad_readout (60K×3)", &total);
    d_bufs_.d_curriculum_pad_logits  = alloc<float>(3, "d_curriculum_pad_logits", &total);
    d_bufs_.d_curriculum_pad_error   = alloc<float>(3, "d_curriculum_pad_error", &total);
    // 课程窗口累计 spike 平均发放率 (60K×4B = 0.23 MB)
    d_bufs_.d_curriculum_accum_spikes = alloc<float>(N_TOTAL_NEURONS_2E,
                                                     "d_curriculum_accum_spikes (60K)", &total);
    // 2026-08-04 方案2: 浓度 readout 头 (6×6×4B = 144 B)
    d_bufs_.d_curriculum_conc_weights = alloc<float>(6 * 6,
                                                     "d_curriculum_conc_readout (6×6)", &total);
    // 2026-08-01 spec §7.9: 目标调质 + PAD + loss 归约缓冲 (消除静态懒分配)
    //   [0..6) = 调质目标, [6..9) = PAD 目标 (Task 5 扩展)
    d_bufs_.d_curriculum_target = alloc<float>(9, "d_curriculum_target (6 mod + 3 pad)", &total);
    d_bufs_.d_curriculum_loss   = alloc<float>(1, "d_curriculum_loss", &total);

    // L5 → 运动皮层稀疏 CSR 突触
    // 突触数 = N_MOTOR_NEURONS × L5_TO_MOTOR_SYNAPSES_PER_NEURON = 5000 × 50 = 250,000
    // BioSynapse 80B × 250K = 19 MB
    const size_t n_l5_motor_syn = (size_t)N_MOTOR_NEURONS * L5_TO_MOTOR_SYNAPSES_PER_NEURON;
    d_bufs_.d_l5_to_motor_synapses     = alloc<BioSynapse>(n_l5_motor_syn,
                                                            "d_l5_to_motor_synapses", &total);
    d_bufs_.d_l5_to_motor_weights      = alloc<float>(n_l5_motor_syn,
                                                       "d_l5_to_motor_weights", &total);
    d_bufs_.d_l5_to_motor_csr_row_ptr  = alloc<int>(N_MOTOR_NEURONS + 1,
                                                    "d_l5_to_motor_csr_row_ptr", &total);
    d_bufs_.d_l5_to_motor_csr_col_idx  = alloc<int>(n_l5_motor_syn,
                                                    "d_l5_to_motor_csr_col_idx", &total);

    // -----------------------------------------------------------------
    // 杏仁核情感学习核心 (Phase 3a-F, M1, 2026-08-06)
    // W 500×500×4B = 1MB + 状态 ~20KB
    // -----------------------------------------------------------------
    d_bufs_.d_amyg_v_la      = alloc<float>(N_AMYGDALA_LA, "d_amyg_v_la", &total);
    d_bufs_.d_amyg_v_ba      = alloc<float>(N_AMYGDALA_BA, "d_amyg_v_ba", &total);
    d_bufs_.d_amyg_spike_la  = alloc<bool>(N_AMYGDALA_LA, "d_amyg_spike_la", &total);
    d_bufs_.d_amyg_spike_ba  = alloc<bool>(N_AMYGDALA_BA, "d_amyg_spike_ba", &total);
    d_bufs_.d_amyg_input_la  = alloc<float>(N_AMYGDALA_LA, "d_amyg_input_la", &total);
    d_bufs_.d_amyg_input_ba  = alloc<float>(N_AMYGDALA_BA, "d_amyg_input_ba", &total);
    d_bufs_.d_amyg_w_la_ba   = alloc<float>((size_t)N_AMYGDALA_LA * N_AMYGDALA_BA,
                                            "d_amyg_w_la_ba (500×500)", &total);
    d_bufs_.d_amyg_trace_la  = alloc<float>(N_AMYGDALA_LA, "d_amyg_trace_la", &total);
    d_bufs_.d_amyg_trace_ba  = alloc<float>(N_AMYGDALA_BA, "d_amyg_trace_ba", &total);

    // -----------------------------------------------------------------
    // 脑岛内感受模块 (Phase 3a-H, M4, 2026-08-06) — 独立小模块
    // -----------------------------------------------------------------
    d_bufs_.d_insula_v      = alloc<float>(N_INSULA_NEURONS, "d_insula_v", &total);
    d_bufs_.d_insula_spike  = alloc<bool>(N_INSULA_NEURONS, "d_insula_spike", &total);
    d_bufs_.d_insula_input  = alloc<float>(N_INSULA_NEURONS, "d_insula_input", &total);
    d_bufs_.d_insula_accum  = alloc<unsigned int>(5, "d_insula_accum", &total);
    // VTA-DA 模块 (M2)
    d_bufs_.d_vta_v         = alloc<float>(N_VTA_NEURONS, "d_vta_v", &total);
    d_bufs_.d_vta_spike     = alloc<bool>(N_VTA_NEURONS, "d_vta_spike", &total);
    d_bufs_.d_vta_input     = alloc<float>(N_VTA_NEURONS, "d_vta_input", &total);
    d_bufs_.d_vta_accum     = alloc<unsigned int>(2, "d_vta_accum", &total);

    printf("  %-34s %10s %8s %10s %10s\n",
           "------", "-----", "----", "-----", "--------");
    printf("[Stage2e P0] 持久显存分配完成: %.2f MB (%.2f GB)\n",
           vram_used_ / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0 * 1024.0));
    printf("[Stage2e P0] v4 设计目标: 1242 MB, 实际: %.2f MB, 偏差: %+.2f MB\n",
           vram_used_ / (1024.0 * 1024.0),
           vram_used_ / (1024.0 * 1024.0) - 1242.0);

    if (allocation_failed_) {
        fprintf(stderr, "[Stage2e P0] 一个或多个持久缓冲分配失败\n");
        return 0;
    }
    return vram_used_;
}

void MemoryAllocator::free_all() {
    if (vram_used_ == 0) return;

    printf("[Stage2e P0] 释放 GPU 显存 (%.2f MB)...\n",
           vram_used_ / (1024.0 * 1024.0));

    #define FREE_PTR(p) do { if (p) { cudaFree(p); p = nullptr; } } while(0)

    FREE_PTR(d_bufs_.d_neurons);
    FREE_PTR(d_bufs_.d_spike_flags);
    FREE_PTR(d_bufs_.d_synapses);
    FREE_PTR(d_bufs_.d_csr_row_ptr);
    FREE_PTR(d_bufs_.d_csr_col_idx);
    FREE_PTR(d_bufs_.d_weights_cache);
    FREE_PTR(d_bufs_.d_eligibility);
    FREE_PTR(d_bufs_.d_eligibility_slow);
    FREE_PTR(d_bufs_.d_synapse_alpha);
    FREE_PTR(d_bufs_.d_synapse_beta);
    FREE_PTR(d_bufs_.d_pca_W);
    FREE_PTR(d_bufs_.d_ca_snapshot);
    FREE_PTR(d_bufs_.d_ca_history_sparse);
    FREE_PTR(d_bufs_.d_synapse_delay);
    FREE_PTR(d_bufs_.d_delay_ring_indices);
    FREE_PTR(d_bufs_.d_delay_ring_current);
    FREE_PTR(d_bufs_.d_stdp_x_pre_trace);
    FREE_PTR(d_bufs_.d_stdp_trace_epoch);
    FREE_PTR(d_bufs_.d_neuron_eligibility);
    FREE_PTR(d_bufs_.d_camkii_activity);
    FREE_PTR(d_bufs_.d_input_current);
    FREE_PTR(d_bufs_.d_nmda_current);
    FREE_PTR(d_bufs_.d_nmda_post_state);
    FREE_PTR(d_bufs_.d_inhibitory_current);
    FREE_PTR(d_bufs_.d_da_concentration);
    FREE_PTR(d_bufs_.d_ach_concentration);
    FREE_PTR(d_bufs_.d_ne_concentration);
    FREE_PTR(d_bufs_.d_ht5_concentration);
    FREE_PTR(d_bufs_.d_gaba_concentration);
    FREE_PTR(d_bufs_.d_oxytocin_concentration);
    FREE_PTR(d_bufs_.d_oxytocin_receptor);
    FREE_PTR(d_bufs_.d_hippo_indices);
    // Task 3: 海马 LRU 游标 + filled_count + top-K 缓冲释放
    FREE_PTR(d_bufs_.d_hippo_write_cursor);
    FREE_PTR(d_bufs_.d_hippo_filled_count);
    FREE_PTR(d_bufs_.d_hippo_top_k);
    FREE_PTR(d_bufs_.d_coact_trackers);
    FREE_PTR(d_bufs_.d_tracker_count);
    FREE_PTR(d_bufs_.d_wm_slots);
    // Task 9: WM 完整闭环缓冲释放
    FREE_PTR(d_bufs_.d_wm_write_cursor);
    FREE_PTR(d_bufs_.d_prefrontal_input);
    FREE_PTR(d_bufs_.d_subcolumn_fr);
    FREE_PTR(d_bufs_.d_baseline_fr);
    FREE_PTR(d_bufs_.d_w_pred);
    FREE_PTR(d_bufs_.d_w_value);
    FREE_PTR(d_bufs_.d_pred_fr);
    FREE_PTR(d_bufs_.d_subcol_fr_prev);
    FREE_PTR(d_bufs_.d_byte_histogram);
    FREE_PTR(d_bufs_.d_neuron_byte_counts);
    FREE_PTR(d_bufs_.d_replay_injection);

    // 语言运动皮层缓冲区释放
    FREE_PTR(d_bufs_.d_motor_neurons);
    FREE_PTR(d_bufs_.d_motor_spike_flags);
    FREE_PTR(d_bufs_.d_decode_weights);
    FREE_PTR(d_bufs_.d_decode_logits);
    FREE_PTR(d_bufs_.d_decode_error);
    FREE_PTR(d_bufs_.d_decode_predicted_byte);
    FREE_PTR(d_bufs_.d_curriculum_readout_weights);
    FREE_PTR(d_bufs_.d_curriculum_logits);
    FREE_PTR(d_bufs_.d_curriculum_error);
    FREE_PTR(d_bufs_.d_curriculum_tool_weights);
    FREE_PTR(d_bufs_.d_curriculum_tool_logits);
    FREE_PTR(d_bufs_.d_curriculum_tool_error);
    // 2026-08-02 Task 5: PAD 情感 readout 缓冲释放
    FREE_PTR(d_bufs_.d_curriculum_pad_weights);
    FREE_PTR(d_bufs_.d_curriculum_pad_logits);
    FREE_PTR(d_bufs_.d_curriculum_pad_error);
    FREE_PTR(d_bufs_.d_curriculum_accum_spikes);
    FREE_PTR(d_bufs_.d_curriculum_conc_weights);
    FREE_PTR(d_bufs_.d_curriculum_target);
    FREE_PTR(d_bufs_.d_curriculum_loss);
    FREE_PTR(d_bufs_.d_l5_to_motor_synapses);
    FREE_PTR(d_bufs_.d_l5_to_motor_weights);
    FREE_PTR(d_bufs_.d_l5_to_motor_csr_row_ptr);
    FREE_PTR(d_bufs_.d_l5_to_motor_csr_col_idx);
    // 杏仁核缓冲释放
    FREE_PTR(d_bufs_.d_amyg_v_la);
    FREE_PTR(d_bufs_.d_amyg_v_ba);
    FREE_PTR(d_bufs_.d_amyg_spike_la);
    FREE_PTR(d_bufs_.d_amyg_spike_ba);
    FREE_PTR(d_bufs_.d_amyg_input_la);
    FREE_PTR(d_bufs_.d_amyg_input_ba);
    FREE_PTR(d_bufs_.d_amyg_w_la_ba);
    FREE_PTR(d_bufs_.d_amyg_trace_la);
    FREE_PTR(d_bufs_.d_amyg_trace_ba);
    // 脑岛缓冲释放
    FREE_PTR(d_bufs_.d_insula_v);
    FREE_PTR(d_bufs_.d_insula_spike);
    FREE_PTR(d_bufs_.d_insula_input);
    FREE_PTR(d_bufs_.d_insula_accum);
    // VTA 缓冲释放
    FREE_PTR(d_bufs_.d_vta_v);
    FREE_PTR(d_bufs_.d_vta_spike);
    FREE_PTR(d_bufs_.d_vta_input);
    FREE_PTR(d_bufs_.d_vta_accum);

    #undef FREE_PTR

    vram_used_ = 0;
}

bool MemoryAllocator::check_budget() const {
    if (vram_used_ > budget_bytes_) {
        fprintf(stderr, "[Stage2e] 显存超预算! 已用 %.2f MB > 上限 %.2f MB\n",
                vram_used_ / (1024.0 * 1024.0),
                budget_bytes_ / (1024.0 * 1024.0));
        return false;
    }
    if (vram_used_ > budget_bytes_ * 9 / 10) {
        fprintf(stderr, "[Stage2e] 警告: 显存接近预算上限 (%.2f MB / %.2f MB)\n",
                vram_used_ / (1024.0 * 1024.0),
                budget_bytes_ / (1024.0 * 1024.0));
    }
    return true;
}

void MemoryAllocator::print_budget_report() const {
    printf("\n========== Stage 2e v4 显存预算报告 ==========\n");
    printf("  持久分配:       %.2f MB\n", vram_used_ / (1024.0 * 1024.0));
    printf("  峰值 (含临时):  %.2f MB\n", vram_peak_ / (1024.0 * 1024.0));
    printf("  v4 预算目标:    1332 MB\n");
    printf("  配置预算:       %.2f GB (%s)\n",
           budget_bytes_ / (1024.0 * 1024.0 * 1024.0),
           vram_used_ <= budget_bytes_ ? "OK" : "OVER");
    printf("  余量:           %.2f MB (%.1f%%)\n",
           budget_bytes_ > vram_used_ ? (budget_bytes_ - vram_used_) / (1024.0 * 1024.0) : 0.0,
           budget_bytes_ > vram_used_
               ? (budget_bytes_ - vram_used_) * 100.0 / budget_bytes_ : 0.0);
    printf("==============================================\n\n");

    // 查询 GPU 实际显存
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("  GPU 总显存:     %.2f GB\n", total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  GPU 可用显存:   %.2f GB\n", free_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  本进程占用:     %.2f GB (%.1f%%)\n",
           (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0),
           (total_bytes - free_bytes) * 100.0 / total_bytes);
}

} // namespace stage2e
