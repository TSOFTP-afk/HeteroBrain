#ifndef SNN_STAGE2E_EVENT_TYPES_H
#define SNN_STAGE2E_EVENT_TYPES_H

// =============================================================================
// Phase 3a-C1: 事件类型枚举 (10 主类型 × 4 修饰符维度)
// =============================================================================
// 详见 docs/superpowers/specs/2026-07-30-event-driven-modulator-injection-design.md §3
//
// 10 主类型按进化意义分类: 食物/威胁/社交/成就/新奇
// 4 修饰符维度: publicity / authority / temporal / intensity
// =============================================================================

#include <string>

namespace stage2e {

enum EventType {
    EVT_FOOD_TASTY = 0,
    EVT_FOOD_BLAND,
    EVT_THREAT_PHYSICAL,
    EVT_THREAT_SOCIAL,
    EVT_PRAISE,
    EVT_CRITICISM,
    EVT_SOCIAL_BOND,
    EVT_SOCIAL_LOSS,
    EVT_ACHIEVEMENT,
    EVT_NOVELTY,
    EVT_QUESTION,   // 知识性问题 (课程训练: 遇到不会的 → 调用工具)
    EVT_COUNT
};

// 修饰符位域 (可用位或组合)
enum EventModifier {
    MOD_PRIVATE   = 0,
    MOD_PUBLIC    = 1 << 0,  // publicity=public: Oxy×1.5 + NE×1.2
    MOD_PEER      = 0,
    MOD_AUTHORITY = 1 << 1,  // authority=authority: DA×1.3 + 5HT×1.2
    MOD_MOMENTARY = 0,
    MOD_SUSTAINED = 1 << 2,  // temporal=sustained: duration×3
};

// 将字符串映射到 EventType, 失败返回 EVT_COUNT
inline EventType event_type_from_string(const char* s) {
    if (!s) return EVT_COUNT;
    std::string str(s);
    if (str == "food_tasty")       return EVT_FOOD_TASTY;
    if (str == "food_bland")       return EVT_FOOD_BLAND;
    if (str == "threat_physical")  return EVT_THREAT_PHYSICAL;
    if (str == "threat_social")    return EVT_THREAT_SOCIAL;
    if (str == "praise")           return EVT_PRAISE;
    if (str == "criticism")        return EVT_CRITICISM;
    if (str == "social_bond")      return EVT_SOCIAL_BOND;
    if (str == "social_loss")      return EVT_SOCIAL_LOSS;
    if (str == "achievement")      return EVT_ACHIEVEMENT;
    if (str == "novelty")          return EVT_NOVELTY;
    if (str == "question")         return EVT_QUESTION;
    return EVT_COUNT;
}

} // namespace stage2e

#endif // SNN_STAGE2E_EVENT_TYPES_H
