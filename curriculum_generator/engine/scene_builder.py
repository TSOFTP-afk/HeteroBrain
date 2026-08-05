"""场景组合引擎：把"学生生活"拆成 领域×主题×关系×强度档×基调 维度，
按主题事件骨架程序化生成 500-1000+ 个场景模板 (写入 kb/scenes.json, 可查可改)。

场景模板 = 一个 (stage, domain, theme, relation, level, tone) 组合 + 事件槽位池。
事件槽位 (cause→effect→consequence→resolution) 来自主题骨架, 经关系过滤 + 强度档缩放。
"""
import json
import os
from typing import Dict, List, Optional, Tuple

_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kb")

# =============================================================================
# 维度定义
# =============================================================================
DOMAINS = ["academic", "social", "family", "interest", "daily", "health", "environment", "self"]

# 事件类型全集 (与 kb/events.json + C++ event_types.h 对齐)
_EVT_EXTRA = {"food_tasty", "food_bland", "threat_physical"}  # 启用 C++ 已支持但此前未用的类型

RELATIONS = [
    # (id, 中文, 关系倾向的事件类型白名单; None = 不过滤)
    ("teacher",    "老师",     {"praise", "criticism", "achievement", "novelty", "question"} | _EVT_EXTRA),
    ("parent",     "父母",     {"praise", "criticism", "achievement", "novelty", "question", "social_bond"} | _EVT_EXTRA),
    ("peer",       "同学",     {"praise", "criticism", "achievement", "social_bond", "social_loss", "threat_social", "question"} | _EVT_EXTRA),
    ("best_friend", "好友",    {"praise", "achievement", "social_bond", "social_loss", "threat_social", "criticism", "novelty"} | _EVT_EXTRA),
    ("stranger",   "陌生人",   {"threat_social", "novelty", "praise", "criticism", "social_loss", "social_bond"} | _EVT_EXTRA),
    ("self",       "自我",     {"novelty", "achievement", "question", "criticism", "social_loss", "praise"} | _EVT_EXTRA),
    ("sibling",    "兄弟姐妹", {"praise", "criticism", "achievement", "novelty", "question",
                                "social_bond", "social_loss", "threat_social"} | _EVT_EXTRA),
    ("online",     "网友",     {"praise", "criticism", "achievement", "novelty", "question",
                                "social_bond", "social_loss", "threat_social"} | _EVT_EXTRA),
    ("relative",   "亲戚",     {"praise", "criticism", "achievement", "novelty", "question",
                                "social_bond", "social_loss", "threat_social"} | _EVT_EXTRA),
    ("counselor",  "心理老师", {"praise", "criticism", "achievement", "novelty", "question",
                                "social_bond", "social_loss", "threat_social"} | _EVT_EXTRA),
]

LEVELS = [
    ("mild",    "轻度",   0.7),
    ("medium",  "中度",   1.0),
    ("severe",  "重度",   1.3),
    ("extreme", "极重度", 1.6),
]

TONES = ["pos", "neg", "mixed"]

# 主题骨架: (领域, 主题, 中文主题词, 工具) → 基调骨架
# 骨架槽位: cause/effect/consequence/resolution, 每槽 1-2 个候选 (事件类型, 强度区间)
#   tool: 0-5 = 6 类工具 (0=生成器 1=计算器 2=草稿 3=记忆检索 4=知识查询 5=时钟), 6=不调用
THEMES: Dict[Tuple[str, str], Dict] = {
    # ===== 学业 =====
    ("academic", "exam"):      {"name": "考试", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 30, 50)],
                  "consequence": [("praise", 20, 35)], "resolution": [("social_bond", 10, 25)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("social_bond", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 20, 40)], "resolution": [("social_bond", 10, 25)]}}},
    ("academic", "quiz"):      {"name": "测验", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "class"):     {"name": "课堂", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "homework"):  {"name": "作业", "tool": 2, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "competition"): {"name": "竞赛", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 20, 35)], "resolution": [("social_bond", 10, 25)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("academic", "study"):     {"name": "自习", "tool": 3, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 社交 =====
    ("social", "friendship"):  {"name": "友谊", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 25)]},
        "mixed": {"cause": [("social_loss", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "conflict"):    {"name": "矛盾", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -30, -20)],
                  "consequence": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -30, -20)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("social", "rejection"):   {"name": "拒绝", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -35, -25)], "effect": [("social_loss", -25, -15)]},
        "mixed": {"cause": [("threat_social", -35, -25)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 25)]}}},
    ("social", "acceptance"):  {"name": "接纳", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("social", "rumor"):       {"name": "谣言", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 30, 45)], "resolution": [("social_bond", 10, 25)]}}},
    ("social", "farewell"):    {"name": "离别", "tool": 6, "tones": {
        "mixed": {"cause": [("social_loss", -30, -20)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 家庭 =====
    ("family", "communication"): {"name": "沟通", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("achievement", 10, 20)]}}},
    ("family", "expectation"): {"name": "期望", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 25, 40)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("family", "support"):     {"name": "支持", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("family", "conflict"):    {"name": "冲突", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("criticism", -30, -20)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_bond", 20, 35)]}}},
    # ===== 兴趣 =====
    ("interest", "hobby"):     {"name": "特长", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "game"):      {"name": "游戏", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "club"):      {"name": "社团", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)]}}},
    ("interest", "sports"):    {"name": "运动", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("social_bond", 10, 25)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 日常 =====
    ("daily", "late"):         {"name": "迟到", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("daily", "forget"):       {"name": "遗忘", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -20, -10)]}}},
    ("daily", "chores"):       {"name": "家务", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("daily", "weather"):      {"name": "天气", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    # ===== 健康 =====
    ("health", "fatigue"):     {"name": "疲劳", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -20, -10)], "effect": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 10, 20)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("health", "sickness"):    {"name": "生病", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_physical", -25, -15), ("threat_social", -20, -10)],
                  "effect": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "sleep"):       {"name": "睡眠", "tool": 6, "tones": {
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 环境 =====
    ("environment", "noise"):  {"name": "噪音", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "crowd"):  {"name": "拥挤", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 15, 30)]}}},
    ("environment", "pressure"): {"name": "压力", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 20, 40)], "resolution": [("social_bond", 10, 25)]}}},
    # ===== 自我 =====
    ("self", "growth"):        {"name": "成长", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]}}},
    ("self", "doubt"):         {"name": "自我怀疑", "tool": 6, "tones": {
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 15, 30)]}}},
    ("self", "goal"):          {"name": "目标", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 25, 40)]}}},
    # ===== 知识链 (工具调用) =====
    ("academic", "math"):      {"name": "数学题", "tool": 1, "tones": {
        "pos":   {"cause": [("question", 20, 35)], "effect": [("achievement", 25, 40)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("academic", "recall"):    {"name": "回忆知识点", "tool": 3, "tones": {
        "pos":   {"cause": [("question", 20, 35)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "writing"):   {"name": "写作", "tool": 2, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "expression"): {"name": "表达", "tool": 0, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 30)],
                  "consequence": [("praise", 15, 30)]}}},
    ("academic", "knowledge"): {"name": "查知识", "tool": 4, "tones": {
        "pos":   {"cause": [("question", 20, 40)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("daily", "planning"):     {"name": "时间规划", "tool": 5, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)]}}},
    # ===== 学业·扩充 =====
    ("academic", "ranking"):   {"name": "排名波动", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 30, 50)],
                  "consequence": [("praise", 20, 35)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -20, -10)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 25, 40)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "tutoring"):  {"name": "补课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "cheating"):  {"name": "作弊", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -25, -15)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 15, 30)]}}},
    ("academic", "scholarship"): {"name": "奖学金", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 30, 45)], "effect": [("praise", 25, 40)],
                  "consequence": [("social_bond", 10, 25)]}}},
    ("academic", "puzzle"):    {"name": "难题攻克", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 20, 35)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 25, 40)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "recitation"): {"name": "背诵", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "experiment"): {"name": "实验课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "speech"):    {"name": "演讲", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 20, 35)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 25, 40)], "resolution": [("social_bond", 10, 20)]}}},
    ("academic", "group_task"): {"name": "小组作业", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -20, -10)]}}},
    ("academic", "reading"):   {"name": "课外阅读", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 社交·扩充 =====
    ("social", "crush"):       {"name": "好感", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("achievement", 10, 20)]}}},
    ("social", "jealousy"):    {"name": "嫉妒", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -20, -10)]},
        "mixed": {"cause": [("social_loss", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "help"):        {"name": "求助", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 20, 35)],
                  "consequence": [("achievement", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("social", "gift"):        {"name": "礼物", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)]}}},
    ("social", "apology"):     {"name": "道歉", "tool": 6, "tones": {
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("praise", 10, 20)]},
        "pos":   {"cause": [("social_loss", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)]}}},
    ("social", "reconcile"):   {"name": "和解", "tool": 6, "tones": {
        "mixed": {"cause": [("social_loss", -30, -20)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_bond", 20, 35)]},
        "pos":   {"cause": [("social_loss", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "clique"):      {"name": "小团体", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -20, -10)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 15, 30)]},
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("social", "online"):      {"name": "网络社交", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("novelty", 20, 35)], "effect": [("threat_social", -30, -20)],
                  "consequence": [("social_loss", -20, -10)]}}},
    ("social", "idol"):        {"name": "偶像", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("social", "outcast"):     {"name": "被孤立", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -35, -25)], "effect": [("social_loss", -30, -20)],
                  "consequence": [("criticism", -20, -10)]},
        "mixed": {"cause": [("threat_social", -35, -25)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("achievement", 10, 25)]}}},
    ("social", "secret"):      {"name": "秘密分享", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "joke"):        {"name": "玩笑", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    # ===== 家庭·扩充 =====
    ("family", "move"):        {"name": "搬家", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("social_loss", -30, -20)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("achievement", 10, 25)]}}},
    ("family", "reunion"):     {"name": "团聚", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("family", "family_change"): {"name": "家庭变故", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -30, -20)],
                  "consequence": [("criticism", -20, -10)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "allowance"):   {"name": "零花钱", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -15, -5)]}}},
    ("family", "punishment"):  {"name": "惩罚", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -20, -10)]},
        "mixed": {"cause": [("criticism", -30, -20)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "reward"):      {"name": "奖励", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 25, 40)], "effect": [("praise", 25, 40)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("family", "sibling"):     {"name": "手足", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("family", "grandparents"): {"name": "隔代亲情", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 兴趣·扩充 =====
    ("interest", "fan"):       {"name": "追星", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "anime"):     {"name": "动漫", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "craft"):     {"name": "手工", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "instrument"): {"name": "乐器", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("interest", "drawing"):   {"name": "绘画", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "programming"): {"name": "编程", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 25, 40)]}}},
    ("interest", "books"):     {"name": "阅读", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "collection"): {"name": "收藏", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 日常·扩充 =====
    ("daily", "lost"):         {"name": "丢失物品", "tool": 6, "tones": {
        "neg":   {"cause": [("social_loss", -20, -10)], "effect": [("criticism", -20, -10)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("daily", "accident"):     {"name": "小意外", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("threat_physical", -25, -15), ("threat_social", -25, -15)],
                  "effect": [("social_loss", -20, -10)]}}},
    ("daily", "shopping"):     {"name": "购物", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("daily", "dining"):       {"name": "聚餐", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 15, 30), ("social_bond", 15, 30)],
                  "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("daily", "festival"):     {"name": "节日", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("daily", "phone"):        {"name": "手机使用", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 15, 30)]},
        "neg":   {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("daily", "pet"):          {"name": "宠物", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 健康·扩充 =====
    ("health", "myopia"):      {"name": "近视", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "injury"):      {"name": "受伤", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_physical", -30, -20), ("threat_social", -25, -15)],
                  "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "checkup"):     {"name": "体检", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "insomnia"):    {"name": "失眠", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -20, -10)], "effect": [("social_loss", -15, -5)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("health", "exercise"):    {"name": "锻炼", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 环境·扩充 =====
    ("environment", "heat"):   {"name": "高温", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_physical", -25, -15), ("threat_social", -20, -10)],
                  "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("environment", "seat"):   {"name": "座位调整", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("achievement", 10, 20)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("environment", "classroom"): {"name": "教室环境", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "schedule"): {"name": "作息变动", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("criticism", -20, -10)], "effect": [("social_loss", -15, -5)]}}},
    ("environment", "commute"): {"name": "通勤", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -15, -5)]}}},
    # ===== 自我·扩充 =====
    ("self", "resolve"):       {"name": "立志", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("self", "reflect"):       {"name": "反思", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 10, 20)]}}},
    ("self", "diary"):         {"name": "日记", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("self", "solitude"):      {"name": "独处", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 20)]}}},
    ("self", "choice"):        {"name": "选择", "tool": 6, "tones": {
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]},
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("self", "persist"):       {"name": "坚持", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 25, 40)],
                  "consequence": [("social_bond", 10, 20)]}}},
    # ===== 学业·扩充二 =====
    ("academic", "morning_reading"): {"name": "早读", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "evening_study"): {"name": "晚自习", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "correction"): {"name": "订正错题", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "preview"):    {"name": "预习", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "review"):     {"name": "复习", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "notes"):      {"name": "笔记", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "new_textbook"): {"name": "发新书", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("academic", "exam_anxiety"): {"name": "考前焦虑", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -20, -10)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "score_return"): {"name": "发成绩", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 20, 35)]},
        "neg":   {"cause": [("novelty", 15, 30)], "effect": [("criticism", -30, -20)],
                  "consequence": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 25, 40)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "quick_answer"): {"name": "课堂抢答", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "procrastination"): {"name": "作业拖延", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("academic", "quality_class"): {"name": "素质课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 10, 20)]}}},
    ("academic", "class_election"): {"name": "班干部竞选", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 20, 35)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("achievement", 25, 40)], "resolution": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]}}},
    ("academic", "peer_tutoring"): {"name": "辅导同学", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "reading_report"): {"name": "读书分享", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 社交·扩充二 =====
    ("social", "class_activity"): {"name": "班级活动", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "class_meeting"): {"name": "班会", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("social", "teamwork"):    {"name": "团队合作", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]}}},
    ("social", "nickname"):    {"name": "起外号", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("social", "borrowing"):   {"name": "借东西", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "comfort"):     {"name": "安慰", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("social_loss", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "encourage"):   {"name": "鼓励", "tool": 6, "tones": {
        "pos":   {"cause": [("praise", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("social", "trust"):       {"name": "信任", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "gossip"):      {"name": "八卦", "tool": 6, "tones": {
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "rival"):       {"name": "竞争对手", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("achievement", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "penpal"):      {"name": "笔友", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "volunteer"):   {"name": "志愿服务", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("social", "deskmate"):    {"name": "同桌", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "group_photo"): {"name": "合影", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "invitation"):  {"name": "邀约", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]}}},
    # ===== 家庭·扩充二 =====
    ("family", "family_dinner"): {"name": "家庭聚餐", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "parent_busy"): {"name": "父母忙碌", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_trip"): {"name": "家庭旅行", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "parents_meeting"): {"name": "家长会", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_decision"): {"name": "家庭决策", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 10, 20)]}}},
    ("family", "caregiving"):  {"name": "照顾家人", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "family_illness"): {"name": "家人生病", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_heirloom"): {"name": "家庭传统", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "sibling_share"): {"name": "手足分享", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_budget"): {"name": "家庭开支", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("criticism", -20, -10)], "effect": [("social_loss", -15, -5)]}}},
    # ===== 兴趣·扩充二 =====
    ("interest", "music"):     {"name": "音乐", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "photography"): {"name": "摄影", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "cooking"):   {"name": "烹饪", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 10, 25), ("novelty", 15, 30)],
                  "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 10, 20)]}}},
    ("interest", "gardening"): {"name": "种植", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "modeling"):  {"name": "模型制作", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "chess"):     {"name": "棋类", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("interest", "dance"):     {"name": "舞蹈", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "calligraphy"): {"name": "书法", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "travel"):    {"name": "旅行", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 10, 25)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "movies"):    {"name": "电影", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 日常·扩充二 =====
    ("daily", "breakfast"):    {"name": "早餐", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("daily", "commute_bus"):  {"name": "公交出行", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("daily", "uniform"):      {"name": "校服", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("daily", "haircut"):      {"name": "理发", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("daily", "rainy_day"):    {"name": "雨天", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 20)]},
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("daily", "power_outage"): {"name": "停电", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("daily", "packages"):     {"name": "快递", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("daily", "red_envelope"): {"name": "红包", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("social_bond", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    # ===== 健康·扩充二 =====
    ("health", "headache"):    {"name": "头痛", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "stomachache"): {"name": "胃痛", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "allergy"):     {"name": "过敏", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("health", "dental"):      {"name": "看牙", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("health", "cold"):        {"name": "感冒", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]}}},
    ("health", "eye_fatigue"): {"name": "用眼疲劳", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -15, -5)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("criticism", -15, -5)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 环境·扩充二 =====
    ("environment", "season_change"): {"name": "换季", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 20)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("environment", "classroom_swap"): {"name": "换教室", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 20)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("environment", "renovation"): {"name": "学校装修", "tool": 6, "tones": {
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("criticism", -15, -5)], "resolution": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("environment", "canteen"): {"name": "食堂", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 10, 25), ("novelty", 10, 25)],
                  "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("food_bland", -15, -5), ("novelty", 10, 25)],
                  "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "dormitory"): {"name": "宿舍", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("environment", "lighting"): {"name": "照明", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]}}},
    # ===== 自我·扩充二 =====
    ("self", "self_discipline"): {"name": "自律", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 25, 40)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("self", "habit"):         {"name": "习惯养成", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "emotion_control"): {"name": "情绪管理", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("self", "gratitude"):     {"name": "感恩", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("self", "forgiveness"):   {"name": "原谅", "tool": 6, "tones": {
        "pos":   {"cause": [("social_loss", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "courage"):       {"name": "勇气", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "patience"):      {"name": "耐心", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("criticism", -15, -5)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "dream"):         {"name": "梦想", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 25, 40)]}}},
    # ===== 学业·扩充三 =====
    ("academic", "attendance"): {"name": "全勤", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "dictation"):  {"name": "听写", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "oral_english"): {"name": "英语口语", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "science_fair"): {"name": "科技节", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "olympiad"):   {"name": "奥赛", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 20, 35)], "resolution": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("criticism", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 25, 40)]}}},
    ("academic", "paper_review"): {"name": "试卷讲评", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "library"):    {"name": "图书馆", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("academic", "school_newspaper"): {"name": "校报投稿", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "subject_switch"): {"name": "换课", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("achievement", 10, 20)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("academic", "homework_check"): {"name": "作业检查", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 15, 30)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("academic", "book_borrow"): {"name": "借书", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "teacher_visit"): {"name": "家访", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 社交·扩充三 =====
    ("social", "birthday"):    {"name": "生日", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 15, 30), ("novelty", 15, 30)],
                  "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "graduation"):  {"name": "毕业", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("social_loss", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "first_meet"):  {"name": "初次见面", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "self_intro"):  {"name": "自我介绍", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("social", "small_talk"):  {"name": "闲聊", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "compliment"):  {"name": "称赞他人", "tool": 6, "tones": {
        "pos":   {"cause": [("praise", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("praise", 15, 30)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "defend"):      {"name": "仗义执言", "tool": 6, "tones": {
        "pos":   {"cause": [("threat_social", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "share_food"):  {"name": "分享食物", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "play_together"): {"name": "一起游戏", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 10, 25)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "watch_match"): {"name": "一起看比赛", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]},
        "neg":   {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "outing"):      {"name": "郊游", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "neighbor"):    {"name": "邻居", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 20)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 家庭·扩充三 =====
    ("family", "family_video"): {"name": "视频通话", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "home_tutoring"): {"name": "父母辅导作业", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -25, -15)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("family", "parent_cooking"): {"name": "父母做饭", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "family_photo"): {"name": "全家福", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "parents_quarrel"): {"name": "父母争吵", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "parent_trip"):  {"name": "父母出差", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_rules"): {"name": "家规", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_news"):  {"name": "家庭喜讯", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]}}},
    ("family", "family_wish"):  {"name": "家庭愿望", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("family", "home_alone"):   {"name": "独自在家", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]}}},
    # ===== 兴趣·扩充三 =====
    ("interest", "singing"):   {"name": "唱歌", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "web_novel"): {"name": "网络小说", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("interest", "cosplay"):   {"name": "动漫角色扮演", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "basketball"): {"name": "篮球", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_loss", -15, -5)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "swimming"):  {"name": "游泳", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "running"):   {"name": "跑步", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)], "resolution": [("praise", 10, 20)]}}},
    ("interest", "cycling"):   {"name": "骑行", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "hiking"):    {"name": "远足", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "board_game"): {"name": "桌游", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("interest", "comics"):    {"name": "漫画", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 10, 25)]}}},
    # ===== 日常·扩充三 =====
    ("daily", "classroom_duty"): {"name": "值日", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)], "resolution": [("praise", 10, 20)]}}},
    ("daily", "school_bus"):   {"name": "校车", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("daily", "umbrella"):     {"name": "借伞", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("daily", "lunch"):        {"name": "午饭", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 10, 25), ("novelty", 10, 25)],
                  "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("daily", "snack"):        {"name": "零食", "tool": 6, "tones": {
        "pos":   {"cause": [("food_tasty", 10, 25), ("novelty", 10, 25)],
                  "effect": [("achievement", 10, 20)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("daily", "laundry"):      {"name": "洗衣服", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("daily", "cleaning"):     {"name": "大扫除", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("daily", "nap"):          {"name": "午休", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 健康·扩充三 =====
    ("health", "fever"):       {"name": "发烧", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_physical", -25, -15), ("threat_social", -25, -15)],
                  "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "cough"):       {"name": "咳嗽", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -15, -5)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -15, -5)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "sprain"):      {"name": "扭伤", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]}}},
    ("health", "medicine"):    {"name": "吃药", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 10, 25)], "effect": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("health", "hospital"):    {"name": "住院", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_physical", -35, -25), ("threat_social", -30, -20)],
                  "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "vaccination"): {"name": "打疫苗", "tool": 6, "tones": {
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 环境·扩充三 =====
    ("environment", "air_quality"): {"name": "空气质量", "tool": 6, "tones": {
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("criticism", -15, -5)], "resolution": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "temperature"): {"name": "降温", "tool": 6, "tones": {
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("criticism", -15, -5)], "resolution": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 20)]}}},
    ("environment", "wind"):    {"name": "大风", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("environment", "snow"):    {"name": "下雪", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("achievement", 10, 25)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("environment", "road"):    {"name": "修路", "tool": 6, "tones": {
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("criticism", -15, -5)], "resolution": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "playground"): {"name": "操场", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("social_bond", 10, 20)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 自我·扩充三 =====
    ("self", "self_awareness"): {"name": "自我认识", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("self", "breakthrough"):  {"name": "突破极限", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 30, 45)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "facing_failure"): {"name": "面对失败", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -20, -10)]},
        "mixed": {"cause": [("criticism", -30, -20)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 10, 20)], "resolution": [("social_bond", 10, 20)]}}},
    ("self", "success_experience"): {"name": "成功体验", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 25, 40)], "effect": [("praise", 20, 35)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("self", "comparison"):    {"name": "与他人比较", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 10, 20)], "resolution": [("social_bond", 10, 20)]}}},
    ("self", "perfectionism"): {"name": "完美主义", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "stress_relief"): {"name": "解压", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "inner_peace"):   {"name": "内心平静", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    # ===== 学业·扩充四 =====
    ("academic", "physics_class"): {"name": "物理课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "english_class"): {"name": "英语课", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "pe_class"):   {"name": "体育课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "computer_class"): {"name": "电脑课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("academic", "art_class"):  {"name": "美术课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "history_class"): {"name": "历史课", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "answer_question"): {"name": "回答问题", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "board_work"): {"name": "黑板演算", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "penmanship"): {"name": "书写", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("academic", "study_group"): {"name": "学习小组", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 15, 30)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("academic", "knowledge_contest"): {"name": "知识竞赛", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 25, 40)],
                  "consequence": [("praise", 20, 35)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 25, 40)]}}},
    ("academic", "self_test"):  {"name": "自测", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    # ===== 社交·扩充四 =====
    ("social", "new_friend"):  {"name": "交新朋友", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("social", "old_friend"):  {"name": "老友重逢", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "accompany"):   {"name": "陪伴", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "listening"):   {"name": "倾听", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "advice"):      {"name": "征求意见", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("social", "opinion"):     {"name": "表达观点", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("praise", 10, 20)]}}},
    ("social", "disagreement"): {"name": "意见分歧", "tool": 6, "tones": {
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_bond", 20, 35)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("social", "mediation"):   {"name": "调解", "tool": 6, "tones": {
        "pos":   {"cause": [("threat_social", -25, -15)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "misunderstanding"): {"name": "误会", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "being_heard"): {"name": "被倾听", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("social_loss", -20, -10)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("social", "decline_invite"): {"name": "婉拒邀约", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_bond", 15, 30)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -15, -5)]}}},
    ("social", "friend_success"): {"name": "朋友成功", "tool": 6, "tones": {
        "pos":   {"cause": [("praise", 15, 30)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("achievement", 20, 35)], "effect": [("criticism", -20, -10)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 家庭·扩充四 =====
    ("family", "family_memories"): {"name": "家庭回忆", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "parent_health"): {"name": "父母健康", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "grandparent_health"): {"name": "祖辈健康", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("family", "family_plan"):  {"name": "家庭计划", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("social_bond", 10, 20)]}}},
    ("family", "transfer_school"): {"name": "转学", "tool": 6, "tones": {
        "neg":   {"cause": [("social_loss", -30, -20)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("social_loss", -30, -20)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("family", "family_reading"): {"name": "亲子阅读", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "family_sports"): {"name": "亲子运动", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("family", "night_talk"):  {"name": "睡前谈心", "tool": 6, "tones": {
        "pos":   {"cause": [("social_bond", 20, 35)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 兴趣·扩充四 =====
    ("interest", "lego"):      {"name": "积木", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "origami"):   {"name": "折纸", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "stamp"):     {"name": "集邮", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("interest", "astronomy"): {"name": "观星", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "aquarium"):  {"name": "养鱼", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("social_bond", 10, 25)],
                  "consequence": [("achievement", 10, 20)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "baking"):    {"name": "烘焙", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("interest", "sewing"):    {"name": "缝纫", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 15, 30)]}}},
    ("interest", "woodwork"):  {"name": "木工", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    # ===== 日常·扩充四 =====
    ("daily", "morning_routine"): {"name": "晨间洗漱", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("daily", "new_clothes"):  {"name": "新衣服", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("achievement", 10, 20)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("daily", "glasses"):      {"name": "配眼镜", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 15, 30)], "resolution": [("praise", 10, 20)]}}},
    ("daily", "backpack"):     {"name": "书包", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 20)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 10, 25)]}}},
    ("daily", "school_supplies"): {"name": "买文具", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 20)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("daily", "elevator"):     {"name": "电梯", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]},
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("daily", "walk_school"):  {"name": "步行上学", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("daily", "table_etiquette"): {"name": "餐桌礼仪", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 15, 30)], "effect": [("praise", 10, 25)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 健康·扩充四 =====
    ("health", "nosebleed"):   {"name": "流鼻血", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "sunstroke"):   {"name": "中暑", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("social_loss", -15, -5)],
                  "consequence": [("criticism", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "weight"):      {"name": "体重变化", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("criticism", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "growth_pain"): {"name": "生长痛", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "toothache"):   {"name": "牙痛", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("health", "skin_problem"): {"name": "皮肤问题", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -20, -10)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_loss", -10, -5)]},
        "mixed": {"cause": [("threat_social", -20, -10)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    # ===== 环境·扩充四 =====
    ("environment", "exam_room"): {"name": "考场环境", "tool": 6, "tones": {
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]},
        "neg":   {"cause": [("threat_social", -25, -15)], "effect": [("criticism", -20, -10)],
                  "consequence": [("social_loss", -10, -5)]}}},
    ("environment", "window"):  {"name": "通风", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 10, 25)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "green_campus"): {"name": "校园绿化", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 10, 25)],
                  "consequence": [("social_bond", 10, 20)], "resolution": [("praise", 10, 20)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 10, 20)]}}},
    ("environment", "school_animals"): {"name": "校园小动物", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("environment", "assembly"): {"name": "集会", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 10, 25)], "effect": [("threat_social", -20, -10)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("environment", "campus_event"): {"name": "校园活动", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("social_bond", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("achievement", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("social_bond", 15, 30)]}}},
    # ===== 自我·扩充四 =====
    ("self", "mood"):          {"name": "心情波动", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]},
        "neg":   {"cause": [("criticism", -25, -15)], "effect": [("social_loss", -20, -10)],
                  "consequence": [("threat_social", -15, -5)]},
        "mixed": {"cause": [("novelty", 15, 30)], "effect": [("criticism", -20, -10)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
    ("self", "loneliness"):    {"name": "孤独感", "tool": 6, "tones": {
        "neg":   {"cause": [("threat_social", -30, -20)], "effect": [("social_loss", -25, -15)],
                  "consequence": [("criticism", -15, -5)]},
        "mixed": {"cause": [("threat_social", -30, -20)], "effect": [("achievement", 20, 35)],
                  "consequence": [("social_bond", 15, 30)]}}},
    ("self", "confidence"):    {"name": "自信", "tool": 6, "tones": {
        "pos":   {"cause": [("achievement", 20, 35)], "effect": [("praise", 15, 30)],
                  "consequence": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("criticism", -20, -10)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "shyness"):       {"name": "害羞", "tool": 6, "tones": {
        "mixed": {"cause": [("threat_social", -25, -15)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "pos":   {"cause": [("novelty", 10, 25)], "effect": [("achievement", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "anger"):         {"name": "愤怒", "tool": 6, "tones": {
        "neg":   {"cause": [("criticism", -30, -20)], "effect": [("threat_social", -25, -15)],
                  "consequence": [("social_loss", -20, -10)]},
        "mixed": {"cause": [("criticism", -30, -20)], "effect": [("social_bond", 15, 30)],
                  "consequence": [("praise", 10, 20)]}}},
    ("self", "curiosity"):     {"name": "好奇心", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("self", "imagination"):   {"name": "想象力", "tool": 6, "tones": {
        "pos":   {"cause": [("novelty", 20, 35)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 25)], "resolution": [("social_bond", 10, 20)]},
        "mixed": {"cause": [("novelty", 20, 35)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)]}}},
    ("self", "learning_style"): {"name": "学习方法", "tool": 6, "tones": {
        "pos":   {"cause": [("question", 15, 30)], "effect": [("achievement", 20, 35)],
                  "consequence": [("praise", 10, 20)]},
        "mixed": {"cause": [("question", 15, 30)], "effect": [("criticism", -15, -5)],
                  "consequence": [("achievement", 20, 35)], "resolution": [("praise", 10, 20)]}}},
}

TONE_CN = {"pos": "积极", "neg": "消极", "mixed": "波折"}


def _scale_range(irange: Tuple[int, int], level_scale: float) -> Tuple[int, int]:
    lo, hi = irange
    if lo >= 0:
        return (int(round(lo * level_scale)), int(round(hi * level_scale)))
    return (int(round(lo * level_scale)), int(round(hi * level_scale)))


def _filter_by_relation(events: Dict[str, List], relation_whitelist: Optional[set]) -> Dict[str, List]:
    """按关系过滤槽位事件 (保持槽位顺序与完整性; 过滤后槽位为空则跳过)"""
    if not relation_whitelist:
        return events
    out: Dict[str, List] = {}
    for role, cands in events.items():
        keep = [c for c in cands if c[0] in relation_whitelist]
        if keep:
            out[role] = keep
    return out


def build_scene_library(stage: str, min_tones: Optional[List[str]] = None) -> List[dict]:
    """维度组合 → 场景模板库 (500-1000+)。

    stage: middle_school | high_school (高中强制部分主题/关系变体, 数据语义略强)
    每个模板: {scene_id, stage, domain, theme, relation, level, tone, desc,
              events: [ {type, intensity_range, role}, ... ], tool}
    """
    scenes: List[dict] = []
    for (domain, theme), theme_def in THEMES.items():
        tones = list(theme_def["tones"].keys())
        if min_tones:
            tones = [t for t in tones if t in min_tones]
        for tone in tones:
            skeleton = theme_def["tones"][tone]
            for rel_id, rel_cn, whitelist in RELATIONS:
                # 知识链 (tool!=6) 关系限定: 工具场景都是自我/泛化情境
                if theme_def["tool"] != 6 and rel_id not in ("self", "peer", "best_friend"):
                    continue
                for lvl_id, lvl_cn, lvl_scale in LEVELS:
                    rel_events = _filter_by_relation(skeleton, whitelist)
                    if not rel_events:
                        continue
                    slots = []
                    for role in ("cause", "effect", "consequence", "resolution"):
                        cands = rel_events.get(role)
                        if not cands:
                            continue
                        slots.append({
                            "role": role,
                            "candidates": [
                                {"type": t, "intensity_range": _scale_range((lo, hi), lvl_scale)}
                                for t, lo, hi in cands
                            ],
                        })
                    if len(slots) < 2:
                        continue
                    scene_id = f"{domain}_{theme}_{rel_id}_{lvl_id}_{tone}"
                    desc = f"{rel_cn}情境·{theme_def['name']}·{TONE_CN[tone]}({lvl_cn})"
                    scenes.append({
                        "scene_id": scene_id,
                        "stage": stage,
                        "domain": domain,
                        "theme": theme,
                        "relation": rel_id,
                        "level": lvl_id,
                        "tone": tone,
                        "desc": desc,
                        "events": slots,
                        "tool": theme_def["tool"],
                    })
    return scenes


def save_scene_library(scenes: List[dict], path: str = None) -> str:
    p = path or os.path.join(_KB_DIR, "scenes.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"stage": "combined", "scene_count": len(scenes), "scenes": scenes},
                  f, ensure_ascii=False, indent=1)
    return p


def load_scene_library(path: str = None) -> List[dict]:
    p = path or os.path.join(_KB_DIR, "scenes.json")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenes"]
