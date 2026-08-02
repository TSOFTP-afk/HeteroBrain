#ifndef SNN_STAGE2E_GENE_EVENT_MAP_H
#define SNN_STAGE2E_GENE_EVENT_MAP_H

#include <algorithm>
#include "event_types.h"

// =============================================================================
// Phase 3a-C1: 基因硬编码映射表 — 事件类型 → 6 维调质增量
// =============================================================================
// 详见设计文档 §4
//
// GENE_MAP_BASE: intensity=0、修饰符全默认时的基准增量
//   6 维: [DA, ACh, NE, 5HT, GABA, Oxy]
//   duration_s: 事件持续秒数 (C1 仅用 pulse 型, duration 控制衰减)
//
// 生物学依据: Schultz 1997 (DA), LeDoux 2000 (5HT/NE), Kosfeld 2005 (Oxy)
// =============================================================================

namespace stage2e {

struct GeneMapEntry {
    float da_delta;
    float ach_delta;
    float ne_delta;
    float ht5_delta;
    float gaba_delta;
    float oxy_delta;
    float duration_s;
};

// intensity=0 时的基准映射 (设计文档 §4.1)
static const GeneMapEntry GENE_MAP_BASE[EVT_COUNT] = {
    // EVT_FOOD_TASTY:    DA↑, 5HT略降 (满足感)
    { 0.40f,  0.10f,  0.05f, -0.05f,  0.00f,  0.02f,  0.5f},
    // EVT_FOOD_BLAND:    微弱 DA
    { 0.05f,  0.00f,  0.00f,  0.00f,  0.00f,  0.00f,  0.2f},
    // EVT_THREAT_PHYSICAL: 5HT↑+NE↑ (杏仁核应激)
    {-0.20f,  0.30f,  0.60f,  0.40f,  0.10f, -0.05f,  2.0f},
    // EVT_THREAT_SOCIAL: 5HT↑+NE↑ (社交应激)
    {-0.15f,  0.20f,  0.45f,  0.35f,  0.05f, -0.10f,  3.0f},
    // EVT_PRAISE:        DA↑+Oxy↑ (社交接纳)
    { 0.25f,  0.10f,  0.15f, -0.05f,  0.00f,  0.20f,  1.0f},
    // EVT_CRITICISM:     5HT↑+DA↓ (社交疼痛)
    {-0.10f,  0.05f,  0.20f,  0.25f,  0.00f, -0.15f,  2.0f},
    // EVT_SOCIAL_BOND:   Oxy↑为主 (依恋)
    { 0.10f,  0.05f, -0.05f,  0.05f,  0.05f,  0.35f,  5.0f},
    // EVT_SOCIAL_LOSS:   5HT↑+Oxy↓ (哀伤)
    {-0.15f,  0.00f,  0.10f,  0.30f,  0.00f, -0.25f,  8.0f},
    // EVT_ACHIEVEMENT:   DA↑强烈 (目标达成)
    { 0.50f,  0.15f,  0.20f, -0.10f,  0.00f,  0.05f,  1.5f},
    // EVT_NOVELTY:       ACh↑+DA↑ (惊奇)
    { 0.15f,  0.40f,  0.10f,  0.00f,  0.00f,  0.00f,  2.0f},
    // EVT_QUESTION:      知识性问题 (ACh↑↑ 认知警觉 + NE↑↑ 任务警觉 + DA↓ 非奖赏探索)
    //   与 novelty 区分: novelty = ACh+DA (好奇/奖赏预期), question = ACh+NE (认知任务警觉)
    //   NE 0.45 vs novelty 0.10 → 4.5 倍区分度, 驱动工具调用 readout 的特征信号
    { 0.10f,  0.45f,  0.45f,  0.05f,  0.05f,  0.05f,  2.0f},
};

// 应用修饰符 + intensity 调制 (纯函数, 设计文档 §4.2)
inline GeneMapEntry apply_modifiers(GeneMapEntry base, int modifier_flags, int intensity) {
    GeneMapEntry result = base;
    // intensity 调制: scale = max(0.05, 1.0 + intensity * 0.02)
    float scale = std::max(0.05f, 1.0f + intensity * 0.02f);
    result.da_delta   *= scale;
    result.ach_delta  *= scale;
    result.ne_delta   *= scale;
    result.ht5_delta  *= scale;
    result.gaba_delta *= scale;
    result.oxy_delta  *= scale;
    // publicity=public: Oxy×1.5 + NE×1.2
    if (modifier_flags & MOD_PUBLIC) {
        result.oxy_delta *= 1.5f;
        result.ne_delta  *= 1.2f;
    }
    // authority=authority: DA×1.3 + 5HT×1.2
    if (modifier_flags & MOD_AUTHORITY) {
        result.da_delta  *= 1.3f;
        result.ht5_delta *= 1.2f;
    }
    // temporal=sustained: duration×3
    if (modifier_flags & MOD_SUSTAINED) {
        result.duration_s *= 3.0f;
    }
    return result;
}

// 事件类型 → 默认修饰符 flags (与 Python generate_curriculum_data.py EVENT_DEFAULT_MOD 一致)
//   public → MOD_PUBLIC (Oxy×1.5 + NE×1.2) | authority → MOD_AUTHORITY (DA×1.3 + 5HT×1.2)
//   sustained → MOD_SUSTAINED (duration×3)
inline int event_default_modifier_flags(EventType t) {
    switch (t) {
        case EVT_THREAT_PHYSICAL: return MOD_AUTHORITY;
        case EVT_THREAT_SOCIAL:
        case EVT_PRAISE:
        case EVT_CRITICISM:       return MOD_PUBLIC | MOD_AUTHORITY;
        case EVT_SOCIAL_BOND:
        case EVT_SOCIAL_LOSS:     return MOD_PUBLIC | MOD_SUSTAINED;
        case EVT_ACHIEVEMENT:     return MOD_PUBLIC;
        default:                  return 0;  // food_tasty/food_bland/novelty/question = private/peer/momentary
    }
}

} // namespace stage2e

#endif // SNN_STAGE2E_GENE_EVENT_MAP_H
