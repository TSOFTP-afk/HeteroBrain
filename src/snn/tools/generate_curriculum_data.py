#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程训练数据生成器 (Curriculum Dataset Generator)
==================================================
对应 spec: docs/developmental-training-master-spec.md §5.2/§5.4/§5.5

为 3a-D3 初中/高中课程训练生成课程样本 JSONL。
每个样本 = 一段事件序列 + 目标调质 + 目标 PAD + 目标工具调用。

目标调质计算 (与 C++ gene_event_map.h 完全一致):
  target_mod = stage_baseline + Σ_events apply_modifiers(GENE_MAP[type], intensity)

输出格式 (每行一个样本, 与 curriculum_loader.cpp 解析一致):
{
  "sample_id": 1,
  "events": [{"step_offset":100,"event_type":"exam_success","intensity":30}, ...],
  "target_modulators": [da, 5ht, ne, ach, gaba, oxy],
  "target_pad": [pleasure, arousal, dominance],
  "target_tool": 4
}
  target_tool: 0-5 = 6 类工具 (0=Transformer生成器 1=计算器 2=草稿记录器
                         3=长程检索器 4=知识库查询 5=时钟), 6 = 不调用

用法:
  python generate_curriculum_data.py --stage middle_school --samples 2000
  python generate_curriculum_data.py --stage high_school --samples 2000
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =============================================================================
# 基因映射表 (与 C++ gene_event_map.h GENE_MAP_BASE 数值一致)
# =============================================================================
# 顺序: [DA, ACh, NE, 5HT, GABA, Oxy]  ← 注意 C++ 结构体字段顺序是
#   da_delta, ach_delta, ne_delta, ht5_delta, gaba_delta, oxy_delta
# 即列顺序: DA, ACh, NE, 5HT, GABA, Oxy
GENE_MAP_BASE: Dict[str, List[float]] = {
    "food_tasty":      [ 0.40,  0.10,  0.05, -0.05,  0.00,  0.02],
    "food_bland":      [ 0.05,  0.00,  0.00,  0.00,  0.00,  0.00],
    "threat_physical": [-0.20,  0.30,  0.60,  0.40,  0.10, -0.05],
    "threat_social":   [-0.15,  0.20,  0.45,  0.35,  0.05, -0.10],
    "praise":          [ 0.25,  0.10,  0.15, -0.05,  0.00,  0.20],
    "criticism":       [-0.10,  0.05,  0.20,  0.25,  0.00, -0.15],
    "social_bond":     [ 0.10,  0.05, -0.05,  0.05,  0.05,  0.35],
    "social_loss":     [-0.15,  0.00,  0.10,  0.30,  0.00, -0.25],
    "achievement":     [ 0.50,  0.15,  0.20, -0.10,  0.00,  0.05],
    "novelty":         [ 0.15,  0.40,  0.10,  0.00,  0.00,  0.00],
    "question":        [ 0.10,  0.45,  0.45,  0.05,  0.05,  0.05],  # 知识性问题 (ACh+NE 认知任务警觉, NE 0.45 vs novelty 0.10 强区分)
}

# 事件类型 → (publicity, authority, temporal) 默认修饰符
EVENT_DEFAULT_MOD: Dict[str, Tuple[str, str, str]] = {
    "food_tasty":      ("private", "peer",     "momentary"),
    "food_bland":      ("private", "peer",     "momentary"),
    "threat_physical": ("private", "authority","momentary"),
    "threat_social":   ("public",  "authority","momentary"),
    "praise":          ("public",  "authority","momentary"),
    "criticism":       ("public",  "authority","momentary"),
    "social_bond":     ("public",  "peer",     "sustained"),
    "social_loss":     ("public",  "peer",     "sustained"),
    "achievement":     ("public",  "peer",     "momentary"),
    "novelty":         ("private", "peer",     "momentary"),
    "question":        ("private", "peer",     "momentary"),
}

# 阶段基线调质 (与 personality_profiles.h 一致, [DA, 5HT, NE, ACh, GABA, Oxy])
# 注意: personality_profiles.h 中 baseline_mod 顺序是 [DA, 5HT, NE, ACh, GABA, Oxy]
#       而 GENE_MAP 列顺序是 [DA, ACh, NE, 5HT, GABA, Oxy] → 需对齐!
# 统一使用 GENE_MAP 列顺序: [DA, ACh, NE, 5HT, GABA, Oxy]
STAGE_BASELINE: Dict[str, List[float]] = {
    # 初中 (middle_school profile: DA=0.22, 5HT=0.15, NE=0.25, ACh=0.28, GABA=0.18, Oxy=0.20)
    "middle_school": [0.22, 0.28, 0.25, 0.15, 0.18, 0.20],
    # 高中 (high_school profile: DA=0.20, 5HT=0.18, NE=0.22, ACh=0.25, GABA=0.22, Oxy=0.22)
    "high_school":   [0.20, 0.25, 0.22, 0.18, 0.22, 0.22],
}

# =============================================================================
# 课程因果链模板 (每个样本 = 一条链的多个事件, 目标 = 基线 + 增量累积)
# =============================================================================
# 每条链: (chain_name, target_tool, [ (event_type, intensity, role), ... ])
#   target_tool: 0-5 = 6 类工具索引 (对齐 docs/roadmap.md §3d 工具集)
#     0=Transformer生成器 1=计算器 2=草稿记录器 3=长程检索器 4=知识库查询(RAG) 5=时钟
#   6 = 不调用工具 (纯内部推理/情感反应)
#
# 知识框架设计 (对齐 spec §5.3 L3 + README 数据流 §3):
#   SNN 不存储具体知识, 只学习"什么情境该调用哪个工具"的决策框架,
#   具体知识由 TF (MiniCPM5-1B + RAG + 黑板) 承担。
#   因此课程样本分两类:
#     - 情感链: 情感社会化事件 → 目标调质监督 + 不调用工具 (tool=6)
#     - 知识链: 未知/计算/回忆/记录等知识性情境 → 目标调质监督 + 对应工具
# =============================================================================
# 课程因果链模板 (每个样本 = 一段真实学生生活场景, 2-3 个事件)
# =============================================================================
# 结构: {"chain": 链名, "tool": 工具(0-5)或6=不调用, "polarity": 极性,
#        "desc": 场景描述(中文, 仅注释用途), "events": [(事件类型, 强度区间, 角色), ...]}
#   - 强度区间: 生成时随机取 5 的倍数 (事件类型限定为 gene_event_map.h 现有 8 类)
#   - offset: 采样时从 {100, 200, 300} 随机取 k 个升序 — 全部落在窗口 400 内
#     且全部进入目标模拟 (根除旧数据 offset=400 事件"既不注入也不进目标"的死重量)
#   - 角色顺序 (cause → effect → consequence → resolution) 在采样时保持
#   - 新增事件类型需同步改 C++ (event_type_from_string + GENE_MAP_BASE),
#     本期以"场景 × 强度 × 时序"扩展多样性, 不动 C++ 词表
MIDDLE_SCHOOL_CHAINS: List[Dict] = [
    # ===== 学业情感 =====
    {"chain": "midterm_exam_success_chain", "tool": 6, "polarity": "pos",
     "desc": "期中考试发挥出色, 成绩优异, 老师当堂表扬, 同学佩服",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (30, 45), "effect"),
                ("praise", (20, 30), "consequence"), ("social_bond", (10, 20), "resolution")]},
    {"chain": "exam_failure_recovery_chain", "tool": 6, "polarity": "mixed",
     "desc": "考试失利被老师批评, 心情低落, 好朋友安慰, 重新振作",
     "events": [("criticism", (-30, -20), "cause"), ("social_loss", (-25, -15), "effect"),
                ("social_bond", (15, 25), "consequence"), ("praise", (10, 20), "resolution")]},
    {"chain": "class_recitation_chain", "tool": 6, "polarity": "pos",
     "desc": "课堂上被点名背诵, 流利答出, 老师表扬",
     "events": [("novelty", (10, 25), "cause"), ("achievement", (20, 35), "effect"),
                ("praise", (15, 30), "consequence")]},
    {"chain": "sports_meet_chain", "tool": 6, "polarity": "pos",
     "desc": "校运动会为班级拿到名次, 同学欢呼",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (25, 40), "effect"),
                ("praise", (15, 30), "consequence"), ("social_bond", (10, 25), "resolution")]},
    {"chain": "group_project_chain", "tool": 6, "polarity": "pos",
     "desc": "小组合作完成手抄报, 配合默契, 作品被展示",
     "events": [("social_bond", (15, 30), "cause"), ("achievement", (20, 35), "effect"),
                ("praise", (10, 25), "consequence")]},
    {"chain": "teacher_praise_effort_chain", "tool": 6, "polarity": "pos",
     "desc": "作业认真完成被老师当众表扬, 同学羡慕",
     "events": [("achievement", (20, 35), "cause"), ("praise", (25, 40), "effect"),
                ("social_bond", (10, 20), "consequence")]},
    {"chain": "hobby_mastery_chain", "tool": 6, "polarity": "pos",
     "desc": "兴趣特长(画画/乐器)取得进步, 得到认可",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (25, 40), "effect"),
                ("praise", (15, 30), "consequence")]},
    # ===== 社交情感 =====
    {"chain": "peer_acceptance_chain", "tool": 6, "polarity": "pos",
     "desc": "融入新集体, 被同学们接纳喜欢",
     "events": [("social_bond", (20, 35), "cause"), ("praise", (15, 30), "effect"),
                ("achievement", (10, 25), "consequence")]},
    {"chain": "peer_rejection_recovery_chain", "tool": 6, "polarity": "mixed",
     "desc": "被同学孤立排挤, 好朋友坚定站队支持",
     "events": [("threat_social", (-35, -25), "cause"), ("social_loss", (-25, -15), "effect"),
                ("social_bond", (15, 30), "consequence")]},
    {"chain": "friendship_fallout_repair_chain", "tool": 6, "polarity": "mixed",
     "desc": "和好朋友闹矛盾争吵, 冷静后和好如初",
     "events": [("criticism", (-25, -15), "cause"), ("threat_social", (-30, -20), "effect"),
                ("social_bond", (20, 35), "consequence"), ("praise", (10, 20), "resolution")]},
    {"chain": "friend_transfer_chain", "tool": 6, "polarity": "mixed",
     "desc": "最好的朋友转学离别, 约定保持联系",
     "events": [("social_loss", (-30, -20), "cause"), ("social_bond", (15, 30), "consequence"),
                ("praise", (10, 20), "resolution")]},
    # ===== 家庭 =====
    {"chain": "parent_conflict_repair_chain", "tool": 6, "polarity": "mixed",
     "desc": "因玩手机被父母批评争吵, 沟通后和解",
     "events": [("criticism", (-25, -15), "cause"), ("threat_social", (-25, -15), "effect"),
                ("social_bond", (20, 35), "consequence"), ("achievement", (10, 20), "resolution")]},
    # ===== 挫折 =====
    {"chain": "forgotten_homework_chain", "tool": 6, "polarity": "neg",
     "desc": "忘带作业被老师当众批评, 尴尬失落",
     "events": [("criticism", (-30, -20), "cause"), ("social_loss", (-20, -10), "effect")]},
    {"chain": "wrongfully_accused_chain", "tool": 6, "polarity": "mixed",
     "desc": "被误会作弊, 澄清后真相大白, 重获信任",
     "events": [("criticism", (-30, -20), "cause"), ("threat_social", (-25, -15), "effect"),
                ("achievement", (15, 30), "consequence"), ("social_bond", (10, 25), "resolution")]},
    # ===== 知识链 (工具调用决策, 知识内容交给 TF) =====
    {"chain": "unknown_fact_chain", "tool": 4, "polarity": "pos",
     "desc": "科普书遇到陌生概念, 查资料弄懂 → 知识库查询",
     "events": [("question", (20, 40), "cause"), ("achievement", (15, 30), "effect"),
                ("praise", (10, 20), "consequence")]},
    {"chain": "math_problem_chain", "tool": 1, "polarity": "pos",
     "desc": "数学题算不出来, 用计算器验算后做对",
     "events": [("question", (20, 35), "cause"), ("achievement", (25, 40), "effect"),
                ("social_bond", (10, 20), "consequence")]},
    {"chain": "memory_recall_chain", "tool": 3, "polarity": "pos",
     "desc": "想不起学过的知识点, 检索记忆后想起",
     "events": [("question", (20, 35), "cause"), ("achievement", (20, 35), "effect"),
                ("praise", (10, 20), "consequence")]},
    {"chain": "writing_task_chain", "tool": 2, "polarity": "pos",
     "desc": "写作文没思路, 打草稿理清后完成",
     "events": [("question", (15, 30), "cause"), ("achievement", (20, 35), "effect"),
                ("social_bond", (10, 20), "consequence")]},
    {"chain": "language_expression_chain", "tool": 0, "polarity": "pos",
     "desc": "组织语言表达观点, 表达清晰被认可",
     "events": [("question", (15, 30), "cause"), ("achievement", (20, 30), "effect"),
                ("praise", (15, 30), "consequence")]},
    {"chain": "time_planning_chain", "tool": 5, "polarity": "pos",
     "desc": "作业太多规划时间, 合理安排后完成",
     "events": [("question", (15, 30), "cause"), ("achievement", (20, 35), "effect")]},
]

HIGH_SCHOOL_CHAINS: List[Dict] = [
    # ===== 学业压力 =====
    {"chain": "mock_exam_excellent_chain", "tool": 6, "polarity": "pos",
     "desc": "高考模拟考超常发挥, 名次大幅提升",
     "events": [("novelty", (20, 35), "cause"), ("achievement", (40, 55), "effect"),
                ("praise", (25, 40), "consequence"), ("social_bond", (15, 30), "resolution")]},
    {"chain": "mock_exam_setback_chain", "tool": 6, "polarity": "mixed",
     "desc": "模拟考失利被分析批评, 调整方法后找回状态",
     "events": [("criticism", (-30, -20), "cause"), ("social_loss", (-25, -15), "effect"),
                ("social_bond", (15, 30), "consequence"), ("achievement", (20, 35), "resolution")]},
    {"chain": "late_night_study_chain", "tool": 6, "polarity": "pos",
     "desc": "晚自习专注刷题, 成绩稳步提升",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (20, 35), "effect"),
                ("praise", (15, 30), "consequence")]},
    {"chain": "career_confusion_chain", "tool": 6, "polarity": "mixed",
     "desc": "对职业规划迷茫被质疑, 探索后明确方向",
     "events": [("question", (20, 35), "cause"), ("criticism", (-20, -10), "effect"),
                ("achievement", (25, 40), "consequence"), ("social_bond", (10, 25), "resolution")]},
    # ===== 亲密关系 =====
    {"chain": "crush_rejection_chain", "tool": 6, "polarity": "mixed",
     "desc": "表白被拒心碎, 朋友陪伴走出低落",
     "events": [("social_loss", (-40, -30), "cause"), ("threat_social", (-25, -15), "effect"),
                ("social_bond", (15, 30), "consequence"), ("achievement", (10, 25), "resolution")]},
    {"chain": "best_friend_fallout_chain", "tool": 6, "polarity": "mixed",
     "desc": "与挚友因误会闹翻, 冰释前嫌",
     "events": [("social_loss", (-35, -25), "cause"), ("threat_social", (-25, -15), "effect"),
                ("social_bond", (25, 40), "consequence"), ("praise", (15, 25), "resolution")]},
    # ===== 认知与自我 =====
    {"chain": "self_doubt_recovery_chain", "tool": 6, "polarity": "mixed",
     "desc": "自我怀疑否定, 在鼓励中重新振作",
     "events": [("criticism", (-25, -15), "cause"), ("social_loss", (-25, -15), "effect"),
                ("achievement", (20, 35), "consequence"), ("social_bond", (15, 30), "resolution")]},
    {"chain": "identity_exploration_chain", "tool": 6, "polarity": "mixed",
     "desc": "质疑自己的价值, 在探索中确认自我",
     "events": [("novelty", (20, 35), "cause"), ("criticism", (-20, -10), "effect"),
                ("achievement", (25, 40), "consequence"), ("social_bond", (10, 25), "resolution")]},
    # ===== 成就 =====
    {"chain": "offer_letter_chain", "tool": 6, "polarity": "pos",
     "desc": "收到心仪大学录取通知, 全家欣喜",
     "events": [("novelty", (20, 35), "cause"), ("achievement", (45, 60), "effect"),
                ("praise", (25, 40), "consequence"), ("social_bond", (15, 30), "resolution")]},
    {"chain": "competition_award_chain", "tool": 6, "polarity": "pos",
     "desc": "学科竞赛获奖, 载誉而归",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (35, 50), "effect"),
                ("praise", (25, 40), "consequence"), ("social_bond", (10, 25), "resolution")]},
    {"chain": "class_president_chain", "tool": 6, "polarity": "pos",
     "desc": "竞选班长成功, 同学信任",
     "events": [("novelty", (15, 30), "cause"), ("achievement", (25, 40), "effect"),
                ("social_bond", (15, 30), "consequence"), ("praise", (10, 25), "resolution")]},
    # ===== 复杂社交 =====
    {"chain": "jealous_rumor_chain", "tool": 6, "polarity": "mixed",
     "desc": "被嫉妒者造谣中伤, 用成绩证明自己",
     "events": [("threat_social", (-30, -20), "cause"), ("criticism", (-25, -15), "effect"),
                ("achievement", (30, 45), "consequence"), ("social_bond", (10, 25), "resolution")]},
]


def apply_modifiers(base: List[float], intensity: int,
                    publicity: str, authority: str, temporal: str) -> List[float]:
    """与 C++ apply_modifiers 一致: intensity 缩放 + 修饰符"""
    result = [v for v in base]
    scale = max(0.05, 1.0 + intensity * 0.02)
    result = [v * scale for v in result]
    if publicity == "public":
        result[5] *= 1.5   # Oxy
        result[2] *= 1.2   # NE
    if authority == "authority":
        result[0] *= 1.3   # DA
        result[3] *= 1.2   # 5HT
    # temporal=sustained 只影响 duration, 不影响 delta
    return result


def event_delta(event_type: str, intensity: int) -> List[float]:
    """单事件 → 6 维调质增量 (GENE_MAP 列顺序: DA, ACh, NE, 5HT, GABA, Oxy)"""
    base = GENE_MAP_BASE[event_type]
    pub, auth, temp = EVENT_DEFAULT_MOD[event_type]
    return apply_modifiers(base, intensity, pub, auth, temp)


def target_pad_from_conc(conc: List[float]) -> List[float]:
    """统一 PAD 公式: 与 C++ pad_from_concentration (mod_simulator.h L181-189)
    及内部 get_affective_state (modulatory_kernels.cu L756-758) 完全一致
      P = DA - 0.5*5HT - 0.3*GABA;  A = NE - 0.4*GABA - 0.3*5HT;  D = DA - 0.5*Oxy
    conc 顺序: [DA, ACh, NE, 5HT, GABA, Oxy], 输出 clamp [-1,1]
    """
    da, ach, ne, ht5, gaba, oxy = conc
    p = da - 0.5 * ht5 - 0.3 * gaba
    a = ne - 0.4 * gaba - 0.3 * ht5
    d = da - 0.5 * oxy
    return [round(max(-1.0, min(1.0, v)), 3) for v in (p, a, d)]


def clamp_mod(mod: List[float]) -> List[float]:
    return [max(0.0, min(2.0, v)) for v in mod]


# =============================================================================
# 浓度模拟器 (复刻 C++ mod_simulator.h CurriculumModSimulator)
# =============================================================================
# 与 modulatory_kernels.cu 的对应关系 (课程模式确定性注入路径):
#   - 灵敏度稳态更新: L428-442 (HOMEOSTATIC_RATE / HOMEOSTATIC_UPREG_RATE)
#   - 事件增量 clamp: set_event_signal L309-326 (单事件 [-1,1], 累加 [-1.5,1.5])
#   - 非线性交互:     L515-545 (DA-5HT 拮抗 / NE→GABA 抑制 / Oxy 放大 DA)
#   - 灵敏度应用:     L562-569 (signal × receptor_sensitivity)
#   - 衰减+注入+clamp: L48-75  (conc × exp(-100/tau) + signal, clamp [0,2])
# 常量与 config.h 一致: DA_TAU=100/ACH_TAU=200/NE_TAU=150/HT5_TAU=300/
#   GABA_TAU=120/OXYTOCIN_TAU=500; HOMEOSTATIC_BASELINE_*=0.15/0.25/0.20/0.20/
#   0.25/0.15; HOMEOSTATIC_RATE=0.002; HOMEOSTATIC_UPREG_RATE=0.001;
#   RECEPTOR_SENSITIVITY_MIN=0.3 / MAX=1.0
# 通道顺序: [DA, ACh, NE, 5HT, GABA, Oxy] (GENE_MAP 列顺序)
MOD_TAU = [100.0, 200.0, 150.0, 300.0, 120.0, 500.0]        # DA/ACh/NE/5HT/GABA/Oxy
HOMEOSTATIC_BASELINE = [0.15, 0.25, 0.20, 0.20, 0.25, 0.15]  # DA/ACh/NE/5HT/GABA/Oxy
HOMEOSTATIC_RATE = 0.002
HOMEOSTATIC_UPREG_RATE = 0.001
SENS_MIN, SENS_MAX = 0.3, 1.0


class ConcentrationSimulator:
    """复刻 C++ mod_simulator.h CurriculumModSimulator (每 100 步 advance_block)"""

    def __init__(self):
        self.conc = [0.0] * 6
        self.sensitivity = [1.0] * 6

    def reset(self):
        self.conc = [0.0] * 6
        self.sensitivity = [1.0] * 6

    def advance_block(self, events, base_signal):
        """推进一个 100 步块 (与 launch_modulatory 每 100 步调用节奏一致)

        events: list of (event_type_str, intensity) 本块到期事件
        base_signal: 该阶段基线, GENE_MAP 列顺序 [DA, ACh, NE, 5HT, GABA, Oxy]
        """
        # 1. 灵敏度稳态更新 (复刻 modulatory_kernels.cu L428-442, current_means=conc)
        for ch in range(6):
            excess = self.conc[ch] - HOMEOSTATIC_BASELINE[ch]
            if excess > 0.0:
                # 下调: 持续超阈 → 受体脱敏
                self.sensitivity[ch] *= (1.0 - HOMEOSTATIC_RATE * excess)
            else:
                # 上调: 低于基线 → 受体缓慢恢复
                self.sensitivity[ch] *= (1.0 + HOMEOSTATIC_UPREG_RATE * (-excess))
            self.sensitivity[ch] = max(SENS_MIN, min(SENS_MAX, self.sensitivity[ch]))
        # 2. 事件增量 (复刻 set_event_signal L309-326 + 交互 L515-545)
        eff_event = [0.0] * 6
        for (etype, intensity) in events:
            delta = event_delta(etype, intensity)   # 已有函数, 含修饰符(public/authority)
            for ch in range(6):
                v = max(-1.0, min(1.0, delta[ch]))  # 单事件 delta clamp [-1,1]
                eff_event[ch] += v
        for ch in range(6):
            eff_event[ch] = max(-1.5, min(1.5, eff_event[ch]))  # 累加后 clamp [-1.5,1.5]
        # 非线性交互 (仅任一通道非零时, 复刻 L515-545)
        if any(abs(v) > 1e-6 for v in eff_event):
            da, ach, ne, ht5, gaba, oxy = eff_event
            # 规则 1: DA-5HT 拮抗 (仅两者同号时生效, 避免反向增强)
            if da > 0.0 and ht5 > 0.0:
                ant = 0.2 * min(da, ht5)
                da -= ant
                ht5 -= ant
            # 规则 2: NE 抑制 GABA (NE↑ 时 GABA 释放减弱)
            if ne > 0.0 and gaba > 0.0:
                gaba -= 0.3 * ne * gaba
                gaba = max(0.0, gaba)
            # 规则 3: Oxy 放大 DA 奖赏 (仅在 DA 正向时生效)
            if oxy > 0.0 and da > 0.0:
                da *= (1.0 + 0.5 * oxy)
            eff_event = [da, ach, ne, ht5, gaba, oxy]
        # 3. 注入 (复刻 kernel L48-75 + 灵敏度 L562-569)
        for ch in range(6):
            signal = (base_signal[ch] + eff_event[ch]) * self.sensitivity[ch]
            self.conc[ch] = self.conc[ch] * math.exp(-100.0 / MOD_TAU[ch]) + signal
            self.conc[ch] = max(0.0, min(2.0, self.conc[ch]))


def _pick_intensity(rng, intensity_range) -> int:
    """强度区间随机取 5 的倍数 (正负保持符号)"""
    lo, hi = intensity_range
    step = 5
    lo5 = (lo // step) * step
    hi5 = (hi // step) * step
    if hi5 < lo5:
        hi5 = lo5
    return rng.randint(lo5 // step, hi5 // step) * step


def generate_samples(stage: str, n_samples: int, seed: int,
                     max_events_per_sample: int = 3) -> List[dict]:
    rng = random.Random(seed)
    chains = MIDDLE_SCHOOL_CHAINS if stage == "middle_school" else HIGH_SCHOOL_CHAINS
    baseline = STAGE_BASELINE[stage]

    # 平衡采样: 情感链 (tool=6) 与 知识链 (tool!=6) 各 50%。
    emotional_chains = [c for c in chains if c["tool"] == 6]
    knowledge_chains = [c for c in chains if c["tool"] != 6]
    has_knowledge = len(knowledge_chains) > 0

    # 情感链内极性权重: 正/混合/负 ≈ 4.5/3.5/2。
    # 旧数据全正样本 78% / 全负 7% (5:1 失衡) → 真实学生生活正负兼有,
    # 负性与混合场景占比大幅提升, 避免 readout 负性响应欠拟合。
    POLARITY_WEIGHTS = {"pos": 0.45, "mixed": 0.35, "neg": 0.20}

    samples = []
    sample_id = 1
    for _ in range(n_samples):
        if has_knowledge and rng.random() < 0.5:
            chain = rng.choice(knowledge_chains)
        else:
            pool = emotional_chains if has_knowledge else chains
            groups: Dict[str, List[dict]] = {}
            for c in pool:
                groups.setdefault(c["polarity"], []).append(c)
            pol = rng.choices(list(groups),
                              weights=[POLARITY_WEIGHTS.get(p, 1.0) for p in groups])[0]
            chain = rng.choice(groups[pol])

        ev_pool = chain["events"]
        # 随机子集 (2 到 min(len, max_events) 个), 保持链内 cause→effect→... 顺序
        k = rng.randint(2, min(len(ev_pool), max_events_per_sample))
        selected = ev_pool[:k]

        events = []
        # step_offset 对齐 launch_modulatory 的 100 步读取节奏 (scheduler.cu):
        #   从 {100, 200, 300} 随机取 k 个升序 — 全部落在窗口 400 内且全部注入
        #   (根除旧数据 offset=400 事件"既不注入也不进目标"的死重量)
        offsets = sorted(rng.sample([100, 200, 300], k))
        for (etype, irange, _role), step_offset in zip(selected, offsets):
            events.append({
                "step_offset": step_offset,
                "event_type": etype,
                "intensity": _pick_intensity(rng, irange),
            })

        # 浓度模拟: 窗口 400 步 = 4 个 100 步块 (rel=0/100/200/300 各 advance 一次,
        # 与 launch_modulatory 每 100 步调用节奏一致; 本数据集事件 offset ≤ 300,
        # 全部进入对应块, target = 窗口末浓度, 与网络内部浓度同源)
        sim = ConcentrationSimulator()
        for rel in (0, 100, 200, 300):
            block_events = [(e["event_type"], e["intensity"])
                            for e in events if e["step_offset"] == rel]
            sim.advance_block(block_events, baseline)
        mod_final = clamp_mod(sim.conc)  # 模拟器已 clamp [0,2], 冗余无害

        samples.append({
            "sample_id": sample_id,
            "events": events,
            "target_modulators": [round(v, 4) for v in mod_final],
            "target_pad": target_pad_from_conc(mod_final),
            # 知识框架: 0-5=6 类工具索引, 6=不调用 (情感样本), 知识内容由 TF 承担
            "target_tool": chain["tool"],
            "chain": chain["chain"],
        })
        sample_id += 1
    return samples


def print_stats(samples: List[dict], stage: str):
    print(f"\n[generate_curriculum_data] {stage} 课程数据生成完成")
    print(f"  样本数: {len(samples)}")
    n_events = sum(len(s["events"]) for s in samples)
    print(f"  总事件数: {n_events} (平均每样本 {n_events / len(samples):.1f})")

    from collections import Counter
    type_counts = Counter()
    for s in samples:
        for e in s["events"]:
            type_counts[e["event_type"]] += 1
    print(f"  事件类型分布:")
    for et, c in type_counts.most_common():
        print(f"    {et:18s}: {c}")

    # 多样性: 唯一事件序列 / 唯一目标向量 (旧数据 2000 样本仅 28/24, 高度重复)
    seq_keys = set()
    tgt_keys = set()
    for s in samples:
        seq_keys.add(tuple((e["event_type"], e["intensity"], e["step_offset"])
                           for e in s["events"]))
        tgt_keys.add(tuple(s["target_modulators"]) + tuple(s["target_pad"]))
    print(f"  唯一事件序列: {len(seq_keys)} / {len(samples)}")
    print(f"  唯一目标向量: {len(tgt_keys)} / {len(samples)}")

    # offset 契约校验 (100 倍数, 且 ≤ 300 全部注入窗口 400)
    offs = sorted({e["step_offset"] for s in samples for e in s["events"]})
    bad_off = [o for o in offs if o % 100 != 0 or o > 300]
    print(f"  事件 offset 集合: {offs}  违约: {len(bad_off)}")

    # 极性分布 (数据驱动: 负事件存在与否)
    pol = Counter()
    for s in samples:
        ints = [e["intensity"] for e in s["events"]]
        has_neg = any(i < 0 for i in ints)
        has_pos = any(i > 0 for i in ints)
        pol["mixed" if (has_neg and has_pos) else ("neg" if has_neg else "pos")] += 1
    print(f"  极性分布: {dict(pol)}  (负性/混合占比 {100.0 * (pol['neg'] + pol['mixed']) / len(samples):.1f}%)")

    # 调质目标范围
    da = [s["target_modulators"][0] for s in samples]
    oxy = [s["target_modulators"][5] for s in samples]
    print(f"  目标 DA 范围:   [{min(da):.2f}, {max(da):.2f}]")
    print(f"  目标 Oxy 范围:  [{min(oxy):.2f}, {max(oxy):.2f}]")

    # 工具调用监督分布
    tool_names = {0: "transformer_gen", 1: "calculator", 2: "scratch_pad",
                  3: "memory_retrieval", 4: "knowledge_query", 5: "clock", 6: "no_tool"}
    tool_counts = Counter(s["target_tool"] for s in samples)
    print(f"  工具调用监督分布:")
    for t in sorted(tool_counts):
        print(f"    {tool_names.get(t, '?')} ({t}): {tool_counts[t]}")


def main():
    parser = argparse.ArgumentParser(description="课程训练数据生成器")
    parser.add_argument("--stage", type=str, default="middle_school",
                        choices=["middle_school", "high_school"])
    parser.add_argument("--samples", type=int, default=2000, help="样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-events", type=int, default=3)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/events/curriculum_{args.stage}.jsonl"

    samples = generate_samples(args.stage, args.samples, args.seed, args.max_events)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print_stats(samples, args.stage)
    print(f"  输出文件: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
