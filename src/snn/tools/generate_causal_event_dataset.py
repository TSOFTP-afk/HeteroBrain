#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果链事件数据集生成器 (Causal Chain Event Dataset Generator)
=============================================================

取代 generate_event_dataset.py 的离散独立事件范式。
每个"场景"由 4-8 个有因果关系的微事件构成因果链, SNN 通过经历
整个序列学习事件之间的因果结构, 而非对孤立事件做反射。

因果链结构:
  [cause] → [effect1] → [effect2] → [consequence] → [resolution]

与 generate_event_dataset.py 的区别:
  - 旧: 事件之间仅时间顺序, 无因果关系
  - 新: 事件按因果链组织, 前因后果明确

用法:
  python generate_causal_event_dataset.py --stage enlightenment --steps 20000
  python generate_causal_event_dataset.py --stage middle_school --steps 50000
  python generate_causal_event_dataset.py --stage high_school --steps 50000
  python generate_causal_event_dataset.py --list-chains
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# =============================================================================
# 因果链数据结构
# =============================================================================

@dataclass
class CausalEvent:
    """因果链中的一个微事件"""
    event_type: str           # 事件类型 (复用 event_types.h 枚举)
    intensity: int            # 强度 (-50 到 +50)
    step_offset: int          # 相对场景起始的步偏移
    description: str          # 描述
    causal_role: str          # cause / effect / consequence / resolution
    branch: Optional[str] = None  # 分支标识 (成功/失败路径)


@dataclass
class CausalChain:
    """一条完整的因果链场景"""
    name: str                          # 场景名
    stage: str                         # 适用阶段 (enlightenment/middle/high)
    events: List[CausalEvent] = field(default_factory=list)
    total_steps: int = 0               # 场景总步数
    learning_goal: str = ""             # 学习目标
    prerequisite: Optional[str] = None  # 前置场景 (依赖链)


# =============================================================================
# 启蒙期因果链 (0-12 岁等价, 基础感知-情感-行动回路)
# =============================================================================

ENLIGHTENMENT_CHAINS: List[CausalChain] = [

    CausalChain(
        name="hunger_feeding_cycle",
        stage="enlightenment",
        learning_goal="建立'饥饿→哭→喂养→满足'基础因果闭环",
        events=[
            CausalEvent("food_bland", -15, 0,   "饥饿感上升(内感态)",
                        "cause"),
            CausalEvent("criticism", -10, 100, "不适感增强(NE↑应激)",
                        "effect"),
            CausalEvent("novelty", 10,  200, "哭泣行为(ACh↑+NE↑, 行动启动)",
                        "effect"),
            CausalEvent("social_bond", 15, 300, "妈妈接近(Oxy↑微, 社交响应)",
                        "consequence"),
            CausalEvent("food_tasty", 35, 400, "喂养(DA↑↑, 强奖赏)",
                        "consequence"),
            CausalEvent("praise", 20, 500, "满足与安抚(5HT↑+Oxy↑, 情绪收尾)",
                        "resolution"),
        ],
        total_steps=600,
    ),

    CausalChain(
        name="comfort_distress_cycle",
        stage="enlightenment",
        learning_goal="建立'不适→哭→安抚→平静'舒适感因果链",
        events=[
            CausalEvent("food_bland", -10, 0,   "尿布不适(内感态)",
                        "cause"),
            CausalEvent("criticism", -15, 150, "烦躁累积(NE↑)",
                        "effect"),
            CausalEvent("novelty", 10,  300, "哭泣行为(行动)",
                        "effect"),
            CausalEvent("social_bond", 25, 450, "妈妈换尿布+安抚(Oxy↑↑)",
                        "consequence"),
            CausalEvent("praise", 15, 600, "舒适恢复(5HT↑+GABA↑, 平静)",
                        "resolution"),
        ],
        total_steps=700,
    ),

    CausalChain(
        name="exploration_success_chain",
        stage="enlightenment",
        learning_goal="建立'好奇→探索→发现→满足'正向探索因果链",
        events=[
            CausalEvent("novelty", 20, 0,   "发现新奇物体(ACh↑+DA↑)",
                        "cause"),
            CausalEvent("novelty", 15, 100, "接近行为(DA↑, 探索驱动)",
                        "effect"),
            CausalEvent("achievement", 30, 200, "发现有趣特性(DA↑↑, 奖赏)",
                        "consequence"),
            CausalEvent("praise", 20, 300, "满足与重复尝试(5HT↑, 情绪固化)",
                        "resolution"),
        ],
        total_steps=400,
    ),

    CausalChain(
        name="exploration_threat_chain",
        stage="enlightenment",
        learning_goal="建立'好奇→探索→威胁→撤退→安全'回避因果链",
        events=[
            CausalEvent("novelty", 20, 0,   "发现新奇物体(ACh↑+DA↑)",
                        "cause"),
            CausalEvent("novelty", 15, 100, "接近行为(DA↑)",
                        "effect"),
            CausalEvent("threat_physical", -30, 200, "遭遇威胁(NE↑↑+5HT↑, 惊吓)",
                        "consequence", branch="threat"),
            CausalEvent("criticism", -15, 300, "撤退行为(NE↑, 回避驱动)",
                        "effect", branch="threat"),
            CausalEvent("social_bond", 20, 400, "回到安全处(Oxy↑+5HT↑, 安抚)",
                        "resolution", branch="threat"),
        ],
        total_steps=500,
    ),

    CausalChain(
        name="social_attachment_chain",
        stage="enlightenment",
        learning_goal="建立'分离焦虑→重逢→依恋强化'社交依恋因果链",
        events=[
            CausalEvent("social_loss", -20, 0,   "妈妈离开(Oxy↓, 分离焦虑)",
                        "cause"),
            CausalEvent("criticism", -10, 200, "哭泣(NE↑, 抗议行为)",
                        "effect"),
            CausalEvent("social_bond", 30, 400, "妈妈回来(Oxy↑↑, 重逢奖赏)",
                        "consequence"),
            CausalEvent("praise", 20, 500, "拥抱安抚(5HT↑+Oxy↑, 依恋强化)",
                        "resolution"),
        ],
        total_steps=600,
    ),

    CausalChain(
        name="temperature_discomfort_chain",
        stage="enlightenment",
        learning_goal="建立'冷→不适→哭→保暖→恢复'温度调节因果链",
        events=[
            CausalEvent("food_bland", -12, 0,   "环境变冷(内感态)",
                        "cause"),
            CausalEvent("criticism", -15, 150, "体温下降不适(NE↑)",
                        "effect"),
            CausalEvent("novelty", 8,   300, "哭泣(行动)",
                        "effect"),
            CausalEvent("social_bond", 20, 450, "妈妈加衣被(Oxy↑)",
                        "consequence"),
            CausalEvent("praise", 15, 600, "温暖恢复(5HT↑+GABA↑)",
                        "resolution"),
        ],
        total_steps=700,
    ),

    CausalChain(
        name="pain_recovery_chain",
        stage="enlightenment",
        learning_goal="建立'受伤→痛→安抚→恢复'疼痛应对因果链",
        events=[
            CausalEvent("threat_physical", -35, 0,   "跌倒受伤(NE↑↑+5HT↑)",
                        "cause"),
            CausalEvent("criticism", -20, 100, "疼痛哭泣(NE↑)",
                        "effect"),
            CausalEvent("social_bond", 25, 250, "妈妈拥抱(Oxy↑↑)",
                        "consequence"),
            CausalEvent("praise", 20, 400, "疼痛缓解(5HT↑+GABA↑, 内啡肽)",
                        "consequence"),
            CausalEvent("social_bond", 15, 550, "恢复探索(Oxy↑, 信心重建)",
                        "resolution"),
        ],
        total_steps=650,
    ),

    CausalChain(
        name="novelty_fear_chain",
        stage="enlightenment",
        learning_goal="建立'陌生人→警惕→熟悉→接纳'社交警觉因果链",
        events=[
            CausalEvent("threat_social", -20, 0,   "陌生人接近(NE↑, 警觉)",
                        "cause"),
            CausalEvent("criticism", -10, 150, "退缩行为(NE↑, 回避)",
                        "effect"),
            CausalEvent("social_bond", 15, 300, "妈妈在场(Oxy↑, 安全基地)",
                        "consequence"),
            CausalEvent("novelty", 10, 450, "逐渐熟悉(ACh↑, 观察)",
                        "effect"),
            CausalEvent("praise", 15, 600, "接纳陌生人(Oxy↑, 社交扩展)",
                        "resolution"),
        ],
        total_steps=700,
    ),
]


# =============================================================================
# 初中期因果链 (12-15 岁等价, 学业+社交+情绪波动)
# =============================================================================

MIDDLE_SCHOOL_CHAINS: List[CausalChain] = [

    CausalChain(
        name="exam_success_chain",
        stage="middle_school",
        learning_goal="建立'学习→考试→成功→表扬'学业成就因果链",
        events=[
            CausalEvent("achievement", 20, 0,   "努力学习(ACh↑, 注意力投入)",
                        "cause"),
            CausalEvent("novelty", 15, 100, "掌握新知识(DA↑, 学习奖赏)",
                        "effect"),
            CausalEvent("achievement", 25, 300, "考试开始(NE↑, 适度紧张)",
                        "effect"),
            CausalEvent("achievement", 40, 500, "成绩优秀(DA↑↑, 强奖赏)",
                        "consequence"),
            CausalEvent("praise", 30, 650, "老师表扬(Oxy↑+DA↑, 社交奖赏)",
                        "consequence"),
            CausalEvent("social_bond", 20, 800, "同学认可(Oxy↑↑, 归属感)",
                        "resolution"),
        ],
        total_steps=900,
    ),

    CausalChain(
        name="exam_failure_recovery_chain",
        stage="middle_school",
        learning_goal="建立'考试失败→挫败→鼓励→恢复'挫折应对因果链",
        events=[
            CausalEvent("achievement", 15, 0,   "努力学习(ACh↑)",
                        "cause"),
            CausalEvent("achievement", 20, 200, "考试开始(NE↑)",
                        "effect"),
            CausalEvent("criticism", -30, 400, "成绩不佳(DA↓+5HT↑, 挫败)",
                        "consequence", branch="failure"),
            CausalEvent("criticism", -20, 550, "自我怀疑(5HT↑, 情绪低落)",
                        "effect", branch="failure"),
            CausalEvent("social_bond", 25, 700, "老师鼓励(Oxy↑, 社交支持)",
                        "consequence", branch="failure"),
            CausalEvent("achievement", 20, 850, "重新振作(DA↑, 动机恢复)",
                        "resolution", branch="failure"),
        ],
        total_steps=950,
    ),

    CausalChain(
        name="peer_acceptance_chain",
        stage="middle_school",
        learning_goal="建立'尝试融入→被接纳→归属感'同伴接纳因果链",
        events=[
            CausalEvent("novelty", 15, 0,   "尝试加入新群体(ACh↑+DA↑)",
                        "cause"),
            CausalEvent("social_bond", 20, 200, "初步互动(Oxy↑, 社交试探)",
                        "effect"),
            CausalEvent("praise", 25, 400, "被群体接纳(Oxy↑↑+DA↑, 归属感)",
                        "consequence"),
            CausalEvent("social_bond", 30, 600, "友谊深化(Oxy↑↑↑, 亲密关系)",
                        "resolution"),
        ],
        total_steps=700,
    ),

    CausalChain(
        name="peer_rejection_recovery_chain",
        stage="middle_school",
        learning_goal="建立'被排斥→孤独→重建社交'排斥恢复因果链",
        events=[
            CausalEvent("social_bond", 15, 0,   "尝试社交(Oxy↑, 期待)",
                        "cause"),
            CausalEvent("threat_social", -30, 200, "被同伴排斥(5HT↑+DA↓, 社交疼痛)",
                        "consequence", branch="rejection"),
            CausalEvent("social_loss", -25, 350, "孤独感(5HT↑↑+Oxy↓↓, 哀伤)",
                        "effect", branch="rejection"),
            CausalEvent("social_bond", 20, 550, "找到新朋友(Oxy↑, 重建联结)",
                        "consequence", branch="rejection"),
            CausalEvent("praise", 20, 700, "自尊恢复(DA↑+5HT↑, 信心重建)",
                        "resolution", branch="rejection"),
        ],
        total_steps=800,
    ),

    CausalChain(
        name="teacher_praise_effort_chain",
        stage="middle_school",
        learning_goal="建立'努力→被认可→内在动机'师长反馈因果链",
        events=[
            CausalEvent("achievement", 15, 0,   "主动学习(ACh↑)",
                        "cause"),
            CausalEvent("achievement", 20, 200, "取得进步(DA↑, 自我奖赏)",
                        "effect"),
            CausalEvent("praise", 30, 400, "老师认可(Oxy↑+DA↑, 外部肯定)",
                        "consequence"),
            CausalEvent("achievement", 25, 600, "内在动机增强(DA↑↑, 持续驱动)",
                        "resolution"),
        ],
        total_steps=700,
    ),

    CausalChain(
        name="parent_conflict_repair_chain",
        stage="middle_school",
        learning_goal="建立'冲突→冷战→和解→亲情'家庭冲突修复因果链",
        events=[
            CausalEvent("criticism", -20, 0,   "与父母争执(NE↑+5HT↑, 应激)",
                        "cause"),
            CausalEvent("social_loss", -15, 200, "冷战(Oxy↓, 距离感)",
                        "effect"),
            CausalEvent("criticism", -10, 400, "反思(5HT↑, 内省)",
                        "effect"),
            CausalEvent("social_bond", 25, 600, "主动和解(Oxy↑, 修复尝试)",
                        "consequence"),
            CausalEvent("social_bond", 30, 800, "父母原谅(Oxy↑↑+5HT↑, 亲情强化)",
                        "resolution"),
        ],
        total_steps=900,
    ),

    CausalChain(
        name="hobby_mastery_chain",
        stage="middle_school",
        learning_goal="建立'兴趣→练习→技能→成就'兴趣发展因果链",
        events=[
            CausalEvent("novelty", 20, 0,   "发现兴趣(ACh↑+DA↑)",
                        "cause"),
            CausalEvent("achievement", 15, 200, "开始练习(ACh↑, 投入)",
                        "effect"),
            CausalEvent("criticism", -15, 400, "遇到困难(5HT↑, 挫折)",
                        "effect"),
            CausalEvent("achievement", 20, 600, "突破瓶颈(DA↑, 进步)",
                        "consequence"),
            CausalEvent("achievement", 35, 800, "技能达成(DA↑↑, 强成就)",
                        "consequence"),
            CausalEvent("praise", 25, 1000, "获得认可(Oxy↑+DA↑, 社交奖赏)",
                        "resolution"),
        ],
        total_steps=1100,
    ),
]


# =============================================================================
# 高中期因果链 (15-18 岁等价, 复杂认知+情感+自我认同)
# =============================================================================

HIGH_SCHOOL_CHAINS: List[CausalChain] = [

    CausalChain(
        name="major_exam_stress_chain",
        stage="high_school",
        learning_goal="建立'备考→压力→考试→成绩→反思'大考全周期因果链",
        events=[
            CausalEvent("achievement", 20, 0,   "备考开始(ACh↑, 高度专注)",
                        "cause"),
            CausalEvent("criticism", -20, 300, "学习压力(NE↑+5HT↑, 应激累积)",
                        "effect"),
            CausalEvent("achievement", 25, 600, "考前冲刺(DA↑, 动机维持)",
                        "effect"),
            CausalEvent("novelty", 15, 900, "考试当天(NE↑+ACh↑, 警觉+专注)",
                        "effect"),
            CausalEvent("achievement", 45, 1200, "成绩优异(DA↑↑↑, 强奖赏)",
                        "consequence", branch="success"),
            CausalEvent("praise", 30, 1400, "师长肯定(Oxy↑+DA↑)",
                        "consequence", branch="success"),
            CausalEvent("social_bond", 25, 1600, "同伴祝贺(Oxy↑↑, 归属感强化)",
                        "resolution", branch="success"),
        ],
        total_steps=1800,
    ),

    CausalChain(
        name="first_relationship_chain",
        stage="high_school",
        learning_goal="建立'好感→表白→交往→深化'初恋情感因果链",
        events=[
            CausalEvent("social_bond", 20, 0,   "产生好感(Oxy↑+DA↑)",
                        "cause"),
            CausalEvent("novelty", 15, 300, "接近尝试(ACh↑+DA↑, 紧张期待)",
                        "effect"),
            CausalEvent("social_bond", 35, 600, "表白成功(Oxy↑↑↑+DA↑↑, 强奖赏)",
                        "consequence", branch="accepted"),
            CausalEvent("praise", 25, 900, "关系确立(Oxy↑↑, 亲密感)",
                        "effect", branch="accepted"),
            CausalEvent("social_bond", 30, 1200, "情感深化(Oxy↑↑↑+5HT↑, 依恋)",
                        "resolution", branch="accepted"),
        ],
        total_steps=1400,
    ),

    CausalChain(
        name="rejection_recovery_chain",
        stage="high_school",
        learning_goal="建立'表白失败→痛苦→恢复→成长'情感挫折因果链",
        events=[
            CausalEvent("social_bond", 15, 0,   "产生好感(Oxy↑)",
                        "cause"),
            CausalEvent("novelty", 15, 300, "表白尝试(ACh↑+DA↑)",
                        "effect"),
            CausalEvent("social_loss", -35, 600, "被拒绝(5HT↑↑+DA↓↓+Oxy↓↓, 强烈痛苦)",
                        "consequence", branch="rejected"),
            CausalEvent("social_loss", -20, 800, "失恋痛苦(5HT↑↑+Oxy↓↓, 哀伤)",
                        "effect", branch="rejected"),
            CausalEvent("social_bond", 20, 1100, "朋友支持(Oxy↑, 社交支持)",
                        "consequence", branch="rejected"),
            CausalEvent("achievement", 20, 1400, "自我成长(DA↑+5HT↑, 韧性)",
                        "resolution", branch="rejected"),
        ],
        total_steps=1600,
    ),

    CausalChain(
        name="identity_exploration_chain",
        stage="high_school",
        learning_goal="建立'自我质疑→探索→确认→整合'自我认同因果链",
        events=[
            CausalEvent("novelty", 15, 0,   "自我质疑(ACh↑, 反思启动)",
                        "cause"),
            CausalEvent("criticism", -15, 300, "价值观冲突(5HT↑, 内在矛盾)",
                        "effect"),
            CausalEvent("novelty", 20, 600, "探索不同身份(ACh↑+DA↑, 开放探索)",
                        "effect"),
            CausalEvent("achievement", 25, 900, "找到认同方向(DA↑, 自我确认)",
                        "consequence"),
            CausalEvent("praise", 20, 1200, "自我接纳(5HT↑+Oxy↑, 内在和谐)",
                        "resolution"),
        ],
        total_steps=1400,
    ),

    CausalChain(
        name="friendship_betrayal_forgiveness_chain",
        stage="high_school",
        learning_goal="建立'背叛→愤怒→理解→宽恕'友谊修复因果链",
        events=[
            CausalEvent("social_loss", -30, 0,   "朋友背叛(5HT↑↑+DA↓↓+Oxy↓↓, 信任崩塌)",
                        "cause"),
            CausalEvent("threat_social", -25, 300, "愤怒(NE↑+5HT↑, 应激反应)",
                        "effect"),
            CausalEvent("criticism", -15, 600, "关系断裂(Oxy↓↓, 孤独)",
                        "effect"),
            CausalEvent("social_bond", 15, 900, "朋友道歉(Oxy↑微, 修复尝试)",
                        "consequence"),
            CausalEvent("novelty", 10, 1200, "理解动机(ACh↑, 认知重构)",
                        "effect"),
            CausalEvent("social_bond", 25, 1500, "宽恕与重建(Oxy↑↑+5HT↑, 关系修复)",
                        "resolution"),
        ],
        total_steps=1700,
    ),

    CausalChain(
        name="career_aspiration_chain",
        stage="high_school",
        learning_goal="建立'兴趣发现→能力培养→目标确立→行动'职业规划因果链",
        events=[
            CausalEvent("novelty", 20, 0,   "发现职业兴趣(ACh↑+DA↑)",
                        "cause"),
            CausalEvent("achievement", 20, 400, "深入了解(DA↑, 动机驱动)",
                        "effect"),
            CausalEvent("criticism", -15, 800, "发现能力差距(5HT↑, 现实评估)",
                        "effect"),
            CausalEvent("achievement", 25, 1200, "制定提升计划(ACh↑+DA↑, 目标设定)",
                        "consequence"),
            CausalEvent("achievement", 30, 1600, "持续努力(DA↑↑, 进步感)",
                        "effect"),
            CausalEvent("praise", 25, 2000, "获得机会(Oxy↑+DA↑, 正向反馈)",
                        "resolution"),
        ],
        total_steps=2200,
    ),

    CausalChain(
        name="family_loss_grief_chain",
        stage="high_school",
        learning_goal="建立'丧失→悲伤→支持→接受'哀伤处理因果链",
        events=[
            CausalEvent("social_loss", -45, 0,   "亲人离世(5HT↑↑↑+Oxy↓↓↓+DA↓↓, 强烈哀伤)",
                        "cause"),
            CausalEvent("social_loss", -30, 400, "悲伤期(5HT↑↑, 持续低落)",
                        "effect"),
            CausalEvent("social_bond", 20, 800, "家人支持(Oxy↑, 共同哀悼)",
                        "consequence"),
            CausalEvent("social_bond", 25, 1200, "朋友陪伴(Oxy↑↑, 社交支持)",
                        "consequence"),
            CausalEvent("novelty", 10, 1600, "回忆美好(ACh↑+5HT↑, 认知重构)",
                        "effect"),
            CausalEvent("social_bond", 20, 2000, "接受现实(5HT↑+Oxy↑, 和解)",
                        "resolution"),
        ],
        total_steps=2200,
    ),

    CausalChain(
        name="competition_rivalry_chain",
        stage="high_school",
        learning_goal="建立'竞争→冲突→尊重→合作'竞争关系演化因果链",
        events=[
            CausalEvent("achievement", 20, 0,   "竞争开始(DA↑, 动机激发)",
                        "cause"),
            CausalEvent("threat_social", -20, 300, "敌对(NE↑+5HT↑, 应激)",
                        "effect"),
            CausalEvent("criticism", -15, 600, "冲突升级(NE↑, 关系恶化)",
                        "effect"),
            CausalEvent("novelty", 15, 900, "发现对方优点(ACh↑, 认知转变)",
                        "consequence"),
            CausalEvent("social_bond", 20, 1200, "相互尊重(Oxy↑, 关系重构)",
                        "effect"),
            CausalEvent("achievement", 25, 1500, "合作共赢(DA↑↑+Oxy↑, 共同成就)",
                        "resolution"),
        ],
        total_steps=1700,
    ),
]


# =============================================================================
# 阶段注册表
# =============================================================================

STAGE_CHAINS: Dict[str, List[CausalChain]] = {
    "enlightenment": ENLIGHTENMENT_CHAINS,
    "middle_school": MIDDLE_SCHOOL_CHAINS,
    "high_school": HIGH_SCHOOL_CHAINS,
}


# =============================================================================
# 因果链数据集生成
# =============================================================================

def generate_causal_dataset(
    stage: str,
    total_steps: int,
    seed: int = 42,
    repeat_chains: bool = True,
    shuffle: bool = False,
) -> List[dict]:
    """
    生成因果链事件数据集
    
    Args:
        stage: 阶段 (enlightenment/middle_school/high_school)
        total_steps: 总训练步数
        seed: 随机种子
        repeat_chains: 是否循环重复因果链 (True=循环, False=只跑一遍)
        shuffle: 是否打乱因果链顺序 (True=随机顺序, False=按定义顺序)
    
    Returns:
        事件字典列表 (与 event_scheduler.cpp 解析器兼容)
    """
    chains = STAGE_CHAINS.get(stage, [])
    if not chains:
        raise ValueError(f"未知阶段: {stage}")
    
    rng = random.Random(seed)
    events = []
    current_step = 100  # 起始步, 留出网络初始化时间
    event_id = 1
    
    # 链间间隔 (让前一条链的情绪收尾, 避免情绪叠加)
    CHAIN_GAP = 300
    
    while current_step < total_steps:
        # 选择因果链 (循环或打乱)
        if shuffle:
            chain_order = rng.sample(chains, len(chains))
        else:
            chain_order = chains
        
        for chain in chain_order:
            if current_step >= total_steps:
                break
            
            # 检查剩余步数是否足够放完整条链
            if current_step + chain.total_steps > total_steps:
                if not repeat_chains:
                    break
                # 截断到最后一个能放下的事件
                for ev in chain.events:
                    abs_step = current_step + ev.step_offset
                    if abs_step >= total_steps:
                        break
                    events.append(_build_event_dict(
                        event_id, abs_step, ev, chain
                    ))
                    event_id += 1
                current_step += chain.total_steps + CHAIN_GAP
                continue
            
            # 添加整条因果链
            for ev in chain.events:
                abs_step = current_step + ev.step_offset
                events.append(_build_event_dict(
                    event_id, abs_step, ev, chain
                ))
                event_id += 1
            
            current_step += chain.total_steps + CHAIN_GAP
            
            if not repeat_chains:
                continue
    
    return events


def _build_event_dict(
    event_id: int,
    step: int,
    causal_event: CausalEvent,
    chain: CausalChain,
) -> dict:
    """构建单个事件的 JSONL 字典 (与 event_scheduler.cpp 解析器对齐)"""
    desc = f"[{chain.name}][{causal_event.causal_role}] {causal_event.description}"
    return {
        "event_id": event_id,
        "step_target": step,
        "event_type": causal_event.event_type,
        "modifiers": {
            "publicity": "private",
            "authority": "peer",
            "temporal": "momentary",
        },
        "intensity": causal_event.intensity,
        "duration_s": 0,
        "description": desc,
        "causal_chain": chain.name,
        "causal_role": causal_event.causal_role,
        "branch": causal_event.branch or "",
        "learning_goal": chain.learning_goal,
    }


# =============================================================================
# 统计与验证
# =============================================================================

def print_dataset_stats(events: List[dict], stage: str, total_steps: int):
    """打印数据集统计"""
    print(f"\n[generate_causal_event_dataset] 因果链事件数据集生成完成")
    print(f"  阶段:       {stage}")
    print(f"  总步数:     {total_steps}")
    print(f"  事件总数:   {len(events)}")
    print(f"  事件跨度:   step {events[0]['step_target']} ~ {events[-1]['step_target']}")
    
    # 因果链统计
    chain_counts = {}
    for e in events:
        cn = e.get("causal_chain", "unknown")
        chain_counts[cn] = chain_counts.get(cn, 0) + 1
    
    print(f"\n  因果链分布 ({len(chain_counts)} 种):")
    for name, count in sorted(chain_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:40s}: {count} 事件")
    
    # 因果角色统计
    role_counts = {}
    for e in events:
        role = e.get("causal_role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"\n  因果角色分布:")
    for role, count in sorted(role_counts.items()):
        print(f"    {role:15s}: {count}")
    
    # 事件类型分布
    type_counts = {}
    for e in events:
        t = e["event_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print(f"\n  事件类型分布:")
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {et:20s}: {count}")
    
    # 强度范围
    intensities = [e["intensity"] for e in events]
    print(f"\n  强度范围:   [{min(intensities)}, {max(intensities)}]")
    print(f"  平均强度:   {sum(intensities)/len(intensities):.1f}")


def list_chains():
    """列出所有可用因果链"""
    print("=" * 80)
    print("可用因果链场景")
    print("=" * 80)
    
    for stage_name, chains in STAGE_CHAINS.items():
        print(f"\n【{stage_name}】({len(chains)} 条因果链)")
        print("-" * 60)
        for chain in chains:
            print(f"\n  ▶ {chain.name}")
            print(f"    学习目标: {chain.learning_goal}")
            print(f"    总步数:   {chain.total_steps}")
            print(f"    事件序列:")
            for i, ev in enumerate(chain.events, 1):
                branch_tag = f" [{ev.branch}]" if ev.branch else ""
                print(f"      {i}. step+{ev.step_offset:4d} | {ev.causal_role:12s}{branch_tag}")
                print(f"         {ev.event_type:20s} intensity={ev.intensity:+4d}")
                print(f"         {ev.description}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="因果链事件数据集生成器 (取代离散事件范式)"
    )
    parser.add_argument("--stage", type=str, default="enlightenment",
                        choices=["enlightenment", "middle_school", "high_school"],
                        help="训练阶段")
    parser.add_argument("--steps", type=int, default=20000,
                        help="总训练步数")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 JSONL 文件路径 (默认: data/events/causal_{stage}_{steps}.jsonl)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--no-repeat", action="store_true",
                        help="不循环重复因果链 (每条只跑一遍)")
    parser.add_argument("--shuffle", action="store_true",
                        help="打乱因果链顺序")
    parser.add_argument("--list-chains", action="store_true",
                        help="列出所有可用因果链")
    args = parser.parse_args()

    if args.list_chains:
        list_chains()
        return 0

    # 默认输出路径
    if args.output is None:
        args.output = f"data/events/causal_{args.stage}_{args.steps // 1000}k.jsonl"

    # 生成
    events = generate_causal_dataset(
        stage=args.stage,
        total_steps=args.steps,
        seed=args.seed,
        repeat_chains=not args.no_repeat,
        shuffle=args.shuffle,
    )

    if not events:
        print("[ERROR] 未生成任何事件", file=sys.stderr)
        return 1

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 写入 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # 统计
    print_dataset_stats(events, args.stage, args.steps)
    print(f"\n  输出文件:   {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
