#ifndef SNN_STAGE2E_PERSONALITY_PROFILES_H
#define SNN_STAGE2E_PERSONALITY_PROFILES_H

// =============================================================================
// 三阶段人格参数表 (Personality Profiles)
// =============================================================================
// 对应 spec: docs/developmental-training-master-spec.md §5.4/§5.5/§6.1
//
// 每个发育阶段对应一组人格参数, 用于:
//   - PSW 突触先验注入 (personality_loader 使用)
//   - 调质基线注入 (modulatory_kernels 使用)
//   - BPTT 课程损失权重 (bptt_curriculum 使用)
//
// 阶段:
//   STAGE_ENLIGHTENMENT  启蒙期 (Stage 0, 纯 STDP, 无监督)
//   STAGE_MIDDLE_SCHOOL  初中期 (Stage 1a, STDP+BPTT, 学业+社交)
//   STAGE_HIGH_SCHOOL    高中期 (Stage 1b, STDP+BPTT, 复杂认知+情感)
//   STAGE_ADULT          成年期 (Stage 2, 纯 STDP, 用户个性化)
// =============================================================================

namespace stage2e {

enum CurriculumStage {
    STAGE_ENLIGHTENMENT = 0,
    STAGE_MIDDLE_SCHOOL = 1,
    STAGE_HIGH_SCHOOL   = 2,
    STAGE_ADULT         = 3,
    STAGE_COUNT         = 4
};

// 6 维调质基线索引 (与 modulatory_kernels 的 conc 通道顺序一致)
enum ModChannel {
    MOD_DA  = 0,
    MOD_5HT = 1,
    MOD_NE  = 2,
    MOD_ACH = 3,
    MOD_GABA = 4,
    MOD_OXY = 5
};

struct PersonalityProfile {
    CurriculumStage stage;
    const char* name;                 // 阶段名 (用于日志/检查点)

    // === PSW 突触先验 ===
    float psw_alpha_beta;             // α+β 置信度目标 (初中 0.3 / 高中 1.0 / 成年 2.0)
    float stdp_eta_multiplier;        // STDP 学习率倍率 (启蒙 3.0x / 初中 1.5x / 高中 1.0x / 成年 0.3x)

    // === BPTT 课程损失权重 (spec §5.3) ===
    bool  bptt_enabled;               // 是否启用 BPTT
    float bptt_loss_weight_mod;       // 调质轨迹损失权重 (初中 1.0 / 高中 0.7)
    float bptt_loss_weight_pad;       // PAD 情感损失权重   (初中 0.3 / 高中 0.5)
    float bptt_loss_weight_tool;      // 工具调用损失权重   (初中 0.0 / 高中 0.5)

    // === 调质基线 (6 维, 与 ModChannel 顺序一致) ===
    float baseline_mod[6];            // [DA, 5HT, NE, ACh, GABA, Oxy]
};

// 三阶段参数表 (spec §7 对照表)
inline const PersonalityProfile& personality_profile(CurriculumStage stage) {
    static const PersonalityProfile kProfiles[STAGE_COUNT] = {
        // --- 启蒙期 (Stage 0): 纯 STDP + 事件驱动, 无监督 ---
        {
            STAGE_ENLIGHTENMENT, "enlightenment",
            /*psw_alpha_beta=*/0.1f,
            /*stdp_eta_multiplier=*/3.0f,
            /*bptt_enabled=*/false,
            /*loss_mod=*/1.0f, /*loss_pad=*/0.0f, /*loss_tool=*/0.0f,
            /*baseline_mod=*/{0.25f, 0.10f, 0.30f, 0.30f, 0.15f, 0.15f},
        },
        // --- 初中期 (Stage 1a): STDP + BPTT, 基础认知 + 情感反应 ---
        {
            STAGE_MIDDLE_SCHOOL, "middle_school",
            /*psw_alpha_beta=*/0.3f,
            /*stdp_eta_multiplier=*/1.5f,
            /*bptt_enabled=*/true,
            /*loss_mod=*/1.0f, /*loss_pad=*/0.3f, /*loss_tool=*/0.3f,
            /*baseline_mod=*/{0.22f, 0.15f, 0.25f, 0.28f, 0.18f, 0.20f},
        },
        // --- 高中期 (Stage 1b): STDP + BPTT, 自我认同 + 价值观 ---
        {
            STAGE_HIGH_SCHOOL, "high_school",
            /*psw_alpha_beta=*/1.0f,
            /*stdp_eta_multiplier=*/1.0f,
            /*bptt_enabled=*/true,
            /*loss_mod=*/0.7f, /*loss_pad=*/0.5f, /*loss_tool=*/0.5f,
            /*baseline_mod=*/{0.20f, 0.18f, 0.22f, 0.25f, 0.22f, 0.22f},
        },
        // --- 成年期 (Stage 2): 纯 STDP 在线学习, 用户个性化 ---
        {
            STAGE_ADULT, "adult",
            /*psw_alpha_beta=*/2.0f,
            /*stdp_eta_multiplier=*/0.3f,
            /*bptt_enabled=*/false,
            /*loss_mod=*/0.0f, /*loss_pad=*/0.0f, /*loss_tool=*/0.0f,
            /*baseline_mod=*/{0.18f, 0.22f, 0.20f, 0.25f, 0.28f, 0.20f},
        },
    };
    int idx = static_cast<int>(stage);
    if (idx < 0 || idx >= STAGE_COUNT) idx = 0;
    return kProfiles[idx];
}

// 从字符串解析阶段 (返回 false 表示未知)
inline bool parse_curriculum_stage(const char* name, CurriculumStage* out) {
    if (!name || !out) return false;
    for (int i = 0; i < STAGE_COUNT; ++i) {
        const PersonalityProfile& p = personality_profile(static_cast<CurriculumStage>(i));
        if (strcmp(name, p.name) == 0) {
            *out = static_cast<CurriculumStage>(i);
            return true;
        }
    }
    return false;
}

} // namespace stage2e

#endif // SNN_STAGE2E_PERSONALITY_PROFILES_H
