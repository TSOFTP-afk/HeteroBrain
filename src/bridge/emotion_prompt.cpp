#include "emotion_prompt.h"

#include <cstdio>

namespace vita {
namespace bridge {

namespace {

// PAD → 中文情绪词 (阈值 0.33, 与 [-1,1] 范围对应)
const char* pleasure_word(float p) {
    if (p > 0.33f)  return "愉悦";
    if (p < -0.33f) return "低落";
    return "平静";
}

const char* arousal_word(float a) {
    if (a > 0.33f)  return "兴奋";
    if (a < -0.33f) return "困倦";
    return "平稳";
}

const char* dominance_word(float d) {
    if (d > 0.33f)  return "主动";
    if (d < -0.33f) return "顺从";
    return "中立";
}

std::string fmt2(float v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return std::string(buf);
}

}  // namespace

std::string build_mood_description(const EmotionState& s) {
    return std::string("当前情感: ") + pleasure_word(s.pleasure) + "、" +
           arousal_word(s.arousal) + "、" + dominance_word(s.dominance) +
           " (愉悦 " + fmt2(s.pleasure) + ", 唤醒 " + fmt2(s.arousal) +
           ", 主导 " + fmt2(s.dominance) + ", 共情 " + fmt2(s.empathy_level) + ")";
}

std::string build_system_prompt_snippet(const EmotionState& s) {
    // 描述性语言而非数值标签: 1B 小模型对数值不敏感, 且"【】/数字"风格
    // 会被模型模仿复读 (实测复读 system prompt 原文)。
    // 2026-08-05: 主语明确为"你的…"(助手自身情绪基调), 避免模型把情感
    // 片段当成"用户的心境"来描述 → 角色错乱 ("你/我"不分)。
    const char* emp = s.empathy_level < 0.33f ? "低"
                    : s.empathy_level < 0.66f ? "中" : "高";
    std::string out = std::string("你的情绪基调：") + pleasure_word(s.pleasure) + "、" +
                      arousal_word(s.arousal) + "、" + dominance_word(s.dominance) + "。";
    out += std::string("你的共情倾向：") + emp + "。";
    return out;
}

}  // namespace bridge
}  // namespace vita
