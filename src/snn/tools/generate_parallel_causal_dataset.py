#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行因果链事件数据集生成器 (Parallel Causal Chain Dataset Generator)
=====================================================================

扩展 generate_causal_event_dataset.py, 支持多条因果链在同一时间窗口内并行运行。

设计动机:
  现实事件是允许叠加发生的:
  - 你考试时可能同时肚子饿
  - 你被批评时可能同时身体不适
  - 你社交成功时可能同时家庭冲突
  
  串行因果链无法训练 SNN 处理"多因果链并行"的真实复杂性。
  本生成器通过并行调度多条因果链, 让 SNN 学习多线索叠加响应。

并行调度规则:
  1. 同类链不并行 (避免 hunger + hunger 混淆)
  2. 跨类链可并行 (hunger + exploration + social 叠加)
  3. 生存类链优先级最高 (身体威胁 > 社交 > 认知)
  4. 调质信号叠加 (同一时刻多个事件 → 调质响应相加)
  5. 冲突处理 (DA↑ + DA↓ 同时发生 → 净效应)

用法:
  python generate_parallel_causal_dataset.py --stage enlightenment --steps 20000 --parallel 2
  python generate_parallel_causal_dataset.py --stage middle_school --steps 50000 --parallel 3
  python generate_parallel_causal_dataset.py --list-combinations
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# 复用 generate_causal_event_dataset.py 的因果链定义
from generate_causal_event_dataset import (
    ENLIGHTENMENT_CHAINS,
    MIDDLE_SCHOOL_CHAINS,
    HIGH_SCHOOL_CHAINS,
    CausalChain,
    CausalEvent,
    STAGE_CHAINS,
)


# =============================================================================
# 因果链分类 (用于并行调度规则)
# =============================================================================

CHAIN_CATEGORY = {
    # 启蒙期
    "hunger_feeding_cycle":           "survival",      # 生存类
    "comfort_distress_cycle":         "survival",
    "temperature_discomfort_chain":   "survival",
    "pain_recovery_chain":            "survival",
    "exploration_success_chain":      "exploration",   # 探索类
    "exploration_threat_chain":       "exploration",
    "novelty_fear_chain":             "exploration",
    "social_attachment_chain":        "social",        # 社交类

    # 初中期
    "exam_success_chain":              "academic",      # 学业类
    "exam_failure_recovery_chain":     "academic",
    "teacher_praise_effort_chain":     "academic",
    "hobby_mastery_chain":            "academic",
    "peer_acceptance_chain":           "social",
    "peer_rejection_recovery_chain":   "social",
    "parent_conflict_repair_chain":    "social",

    # 高中期
    "major_exam_stress_chain":         "academic",
    "first_relationship_chain":        "social",
    "rejection_recovery_chain":       "social",
    "identity_exploration_chain":      "cognitive",     # 认知类
    "friendship_betrayal_forgiveness_chain": "social",
    "career_aspiration_chain":        "cognitive",
    "family_loss_grief_chain":        "social",
    "competition_rivalry_chain":      "social",
}

# 类别优先级 (高优先级先调度, 数值越小优先级越高)
CATEGORY_PRIORITY = {
    "survival":   0,    # 生存类最高
    "social":     1,
    "academic":   2,
    "exploration": 3,
    "cognitive":  4,
}


# =============================================================================
# 并行链组合规则 (哪些链可以同时运行)
# =============================================================================

# 推荐的并行组合 (stage, parallel_count) -> List[List[chain_names]]
# 这些组合经过设计, 避免逻辑冲突
RECOMMENDED_COMBINATIONS: Dict[Tuple[str, int], List[List[str]]] = {

    # 启蒙期 2 链并行: 生存 + 社交 (饥饿同时被关注)
    ("enlightenment", 2): [
        ["hunger_feeding_cycle", "social_attachment_chain"],
        ["comfort_distress_cycle", "exploration_success_chain"],
        ["pain_recovery_chain", "social_attachment_chain"],
        ["temperature_discomfort_chain", "novelty_fear_chain"],
        ["exploration_threat_chain", "social_attachment_chain"],
        ["hunger_feeding_cycle", "exploration_success_chain"],
    ],

    # 启蒙期 3 链并行: 生存 + 社交 + 探索
    ("enlightenment", 3): [
        ["hunger_feeding_cycle", "social_attachment_chain", "exploration_success_chain"],
        ["comfort_distress_cycle", "pain_recovery_chain", "novelty_fear_chain"],
        ["temperature_discomfort_chain", "social_attachment_chain", "exploration_threat_chain"],
    ],

    # 初中期 2 链并行: 学业 + 社交 (考试同时被排斥)
    ("middle_school", 2): [
        ["exam_success_chain", "peer_acceptance_chain"],
        ["exam_failure_recovery_chain", "parent_conflict_repair_chain"],
        ["hobby_mastery_chain", "peer_rejection_recovery_chain"],
        ["teacher_praise_effort_chain", "parent_conflict_repair_chain"],
        ["exam_success_chain", "hobby_mastery_chain"],
        ["peer_rejection_recovery_chain", "hobby_mastery_chain"],
    ],

    # 初中期 3 链并行: 学业 + 社交 + 家庭
    ("middle_school", 3): [
        ["exam_success_chain", "peer_acceptance_chain", "parent_conflict_repair_chain"],
        ["exam_failure_recovery_chain", "peer_rejection_recovery_chain", "hobby_mastery_chain"],
        ["teacher_praise_effort_chain", "peer_acceptance_chain", "parent_conflict_repair_chain"],
    ],

    # 高中期 2 链并行: 学业 + 情感 (大考同时初恋)
    ("high_school", 2): [
        ["major_exam_stress_chain", "first_relationship_chain"],
        ["rejection_recovery_chain", "identity_exploration_chain"],
        ["career_aspiration_chain", "competition_rivalry_chain"],
        ["family_loss_grief_chain", "identity_exploration_chain"],
        ["friendship_betrayal_forgiveness_chain", "major_exam_stress_chain"],
        ["major_exam_stress_chain", "rejection_recovery_chain"],
    ],

    # 高中期 3 链并行: 学业 + 情感 + 认知
    ("high_school", 3): [
        ["major_exam_stress_chain", "first_relationship_chain", "identity_exploration_chain"],
        ["career_aspiration_chain", "competition_rivalry_chain", "family_loss_grief_chain"],
        ["rejection_recovery_chain", "identity_exploration_chain", "friendship_betrayal_forgiveness_chain"],
    ],
}


# =============================================================================
# 并行调度核心
# =============================================================================

@dataclass
class ParallelSlot:
    """一个并行槽位: 跟踪一条正在运行的因果链"""
    chain: CausalChain
    start_step: int
    next_event_idx: int = 0
    
    @property
    def is_finished(self) -> bool:
        return self.next_event_idx >= len(self.chain.events)
    
    def next_event_step(self) -> Optional[int]:
        """返回下一个事件的绝对步数, 若链已结束返回 None"""
        if self.is_finished:
            return None
        ev = self.chain.events[self.next_event_idx]
        return self.start_step + ev.step_offset
    
    def consume_next(self) -> Optional[CausalEvent]:
        """消费下一个事件, 返回该事件"""
        if self.is_finished:
            return None
        ev = self.chain.events[self.next_event_idx]
        self.next_event_idx += 1
        return ev


def select_parallel_combination(
    stage: str,
    parallel_count: int,
    rng: random.Random,
    used_combinations: set,
) -> List[CausalChain]:
    """
    选择一组可并行的因果链
    
    规则:
      1. 优先从推荐组合中选 (未用过的优先)
      2. 回退到自动组合: 按类别分散选择
    """
    key = (stage, parallel_count)
    
    # 尝试推荐组合
    if key in RECOMMENDED_COMBINATIONS:
        combos = RECOMMENDED_COMBINATIONS[key]
        # 优先未用过的
        unused = [c for c in combos if tuple(sorted(c)) not in used_combinations]
        if unused:
            chosen_names = rng.choice(unused)
            used_combinations.add(tuple(sorted(chosen_names)))
        else:
            chosen_names = rng.choice(combos)
        
        # 根据名字查找链
        chains_by_name = {c.name: c for c in STAGE_CHAINS[stage]}
        return [chains_by_name[n] for n in chosen_names if n in chains_by_name]
    
    # 自动组合: 按类别分散
    all_chains = STAGE_CHAINS[stage]
    by_category = defaultdict(list)
    for c in all_chains:
        cat = CHAIN_CATEGORY.get(c.name, "other")
        by_category[cat].append(c)
    
    # 按优先级排序类别
    sorted_cats = sorted(by_category.keys(), 
                         key=lambda c: CATEGORY_PRIORITY.get(c, 99))
    
    chosen = []
    for cat in sorted_cats:
        if len(chosen) >= parallel_count:
            break
        if by_category[cat]:
            chosen.append(rng.choice(by_category[cat]))
            by_category[cat].remove(chosen[-1])
    
    return chosen


# =============================================================================
# 正态分布驱动的动态并行调度 (核心重构 v2)
# =============================================================================
#
# 理论依据: 中心极限定理 (Central Limit Theorem)
#   - 现实中"同一时刻发生的事件数"是多个独立因素叠加的结果
#   - 根据CLT, 这类叠加量趋向正态分布 N(μ, σ²)
#   - 参考: https://m.kepuchina.cn/tuwendetail?id=676277
#   - 参考: https://blog.csdn.net/WuLex/article/details/157073705
#
# 正态分布特性 (68-95-99.7 法则):
#   μ ± 1σ: 68% 的时刻
#   μ ± 2σ: 95% 的时刻
#   μ ± 3σ: 99.7% 的时刻
#
# 设计原则:
#   1. 并行深度服从正态分布 N(μ_parallel, σ_parallel)
#   2. 每个时间窗口(如 500 步)采样一次"目标并行深度"
#   3. 通过反馈控制让实际活跃链数收敛到目标
#   4. 无硬上限 (理论上 0 到 ∞ 都可能, 但 3σ 外概率 < 0.3%)
#   5. 每条链仍独立异步调度, 时间维度保留随机性


# 阶段默认参数 (μ, σ) - 可通过 CLI 覆盖
# 启蒙期: 均值 1.5 (婴儿事件简单, 同时发生少)
# 初中期: 均值 2.5 (学业+社交叠加增多)
# 高中期: 均值 3.5 (复杂多线索)
STAGE_PARALLEL_PARAMS = {
    "enlightenment": (1.5, 0.8),   # μ=1.5, σ=0.8 → 主要 0-3
    "middle_school": (2.5, 1.0),   # μ=2.5, σ=1.0 → 主要 1-4
    "high_school":   (3.5, 1.2),   # μ=3.5, σ=1.2 → 主要 1-6
}


def generate_parallel_dataset(
    stage: str,
    total_steps: int,
    parallel_count: int = 0,       # 已弃用, 0=正态分布模式
    seed: int = 42,
    chain_gap: int = 300,
    lambda_new_chain: float = 0.003,  # 启动概率基准 (保留兼容)
    max_concurrent: int = 0,           # 0=无硬上限, 用正态分布尾部自然衰减
    min_idle_steps: int = 100,         # 最少空闲步数
    mu_parallel: float = 0.0,          # 正态分布均值, 0=用阶段默认
    sigma_parallel: float = 0.0,       # 正态分布标准差, 0=用阶段默认
    target_window: int = 500,           # 目标深度采样窗口步数
) -> List[dict]:
    """
    正态分布驱动的动态并行因果链数据集生成

    并行深度服从正态分布 N(μ, σ):
      - 每个时间窗口采样一次"目标并行深度"
      - 通过反馈控制让活跃链数趋向目标
      - 无硬上限, 极端值 (μ±3σ外) 概率 < 0.3%
      - 理论依据: 中心极限定理

    Args:
        stage: 训练阶段
        total_steps: 总训练步数
        mu_parallel: 正态分布均值 (0=用阶段默认)
        sigma_parallel: 正态分布标准差 (0=用阶段默认)
        target_window: 目标深度重采样窗口步数

    Returns:
        事件字典列表 (按 step_target 排序)
    """
    rng = random.Random(seed)
    all_chains = STAGE_CHAINS.get(stage, [])
    if not all_chains:
        raise ValueError(f"未知阶段: {stage}")

    # 应用阶段默认正态参数
    if mu_parallel <= 0 or sigma_parallel <= 0:
        mu_parallel, sigma_parallel = STAGE_PARALLEL_PARAMS.get(
            stage, (2.0, 1.0)
        )

    active_slots: List[ParallelSlot] = []
    events: List[dict] = []
    event_id = 1

    # 记录最近使用过的链 (避免立刻重复同一条)
    recent_chain_names: List[str] = []
    RECENT_WINDOW = 2

    # 统计并行深度直方图 (供事后分析)
    parallel_depth_histogram = defaultdict(int)

    # 当前时间窗口的目标并行深度 (正态采样)
    current_target_depth = max(0, int(round(rng.gauss(mu_parallel, sigma_parallel))))
    window_start_step = 100

    # 时间推进主循环
    current_step = 100  # 起始步, 留出网络初始化时间
    last_chain_end_step = 0  # 上一条链结束步数

    while current_step < total_steps:
        # === 0. 每个窗口重新采样目标并行深度 (正态分布) ===
        if current_step - window_start_step >= target_window:
            current_target_depth = max(
                0, int(round(rng.gauss(mu_parallel, sigma_parallel)))
            )
            window_start_step = current_step

        # === 1. 清理已完成的链 ===
        active_slots = [s for s in active_slots if not s.is_finished]
        current_depth = len(active_slots)

        # === 2. 根据正态分布目标决定是否启动新链 ===
        # 反馈控制: 当活跃链数 < 目标深度时, 提高启动概率
        #           当活跃链数 >= 目标深度时, 降低启动概率
        depth_deficit = current_target_depth - current_depth

        if depth_deficit > 0:
            # 需要更多链: 启动概率与缺口成正比
            # 每步概率 = base_rate * (1 + deficit)
            start_prob = min(0.05, 0.01 * (1 + depth_deficit))

            # 冷却期检查 (除非目标深度>0且当前无链)
            in_cooldown = (
                current_step < last_chain_end_step + min_idle_steps
                and current_depth > 0
            )

            if not in_cooldown or current_depth == 0:
                if rng.random() < start_prob:
                    # 选择一条未在 recent 内的链
                    candidates = [
                        c for c in all_chains
                        if c.name not in recent_chain_names[-RECENT_WINDOW:]
                    ]
                    if not candidates:
                        candidates = all_chains

                    new_chain = rng.choice(candidates)
                    new_slot = ParallelSlot(chain=new_chain, start_step=current_step)
                    active_slots.append(new_slot)
                    recent_chain_names.append(new_chain.name)
                    current_depth = len(active_slots)

        # === 3. 消费当前步所有应该触发的事件 ===
        simultaneous_events = []

        for slot in active_slots:
            if slot.is_finished:
                continue
            next_step = slot.next_event_step()
            if next_step == current_step:
                ev = slot.consume_next()
                simultaneous_events.append((slot.chain, ev, slot))

        # === 4. 记录事件 (附带并行上下文) ===
        for chain, ev, slot in simultaneous_events:
            events.append(_build_parallel_event_dict(
                event_id, current_step, ev, chain,
                parallel_context=[s.chain.name for s in active_slots],
                simultaneous_count=len(simultaneous_events),
            ))
            event_id += 1

        # === 5. 统计当前步的并行深度 ===
        # 统计所有时刻的活跃链数 (无论是否有事件触发)
        parallel_depth_histogram[current_depth] += 1

        # === 6. 更新 last_chain_end_step ===
        for slot in active_slots:
            if slot.is_finished:
                end_step = slot.start_step + slot.chain.total_steps
                if end_step > last_chain_end_step:
                    last_chain_end_step = end_step

        # === 7. 时间推进 ===
        if active_slots:
            current_step += 1
        else:
            # 没有活跃链: 快进到冷却结束
            cooldown_end = last_chain_end_step + min_idle_steps
            if current_step < cooldown_end:
                current_step = cooldown_end
            else:
                current_step += 1
    
    # 按 step_target 排序 (虽然生成时已基本有序, 但并行链可能导致乱序)
    events.sort(key=lambda e: e["step_target"])
    
    # 重新编号 event_id
    for i, e in enumerate(events, 1):
        e["event_id"] = i
    
    # 保存统计供后续 print 使用
    generate_parallel_dataset._last_histogram = parallel_depth_histogram
    
    return events


def _build_parallel_event_dict(
    event_id: int,
    step: int,
    causal_event: CausalEvent,
    chain: CausalChain,
    parallel_context: List[str],
    simultaneous_count: int,
) -> dict:
    """构建并行事件字典 (扩展因果链元信息)"""
    desc = (
        f"[{chain.name}][{causal_event.causal_role}]"
        f"{' [并行×'+str(simultaneous_count)+']' if simultaneous_count > 1 else ''}"
        f" {causal_event.description}"
    )
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
        # === 并行训练元信息 (新增) ===
        "parallel_context": parallel_context,      # 同时运行的所有链名
        "simultaneous_count": simultaneous_count,  # 当前步同时触发的事件数
        "parallel_depth": len(parallel_context),   # 并行深度
    }


# =============================================================================
# 调质叠加分析
# =============================================================================

# 基因映射表 (从 generate_event_dataset.py 复制, 用于分析叠加效应)
GENE_MAP_BASE = {
    "food_tasty":      {"DA": +0.40, "ACh": +0.10, "NE": +0.05, "5HT": -0.05, "GABA":  0.00, "Oxy": +0.02},
    "food_bland":      {"DA": +0.05, "ACh":  0.00, "NE":  0.00, "5HT":  0.00, "GABA":  0.00, "Oxy":  0.00},
    "threat_physical": {"DA": -0.20, "ACh": +0.30, "NE": +0.60, "5HT": +0.40, "GABA": +0.10, "Oxy": -0.05},
    "threat_social":   {"DA": -0.15, "ACh": +0.20, "NE": +0.45, "5HT": +0.35, "GABA": +0.05, "Oxy": -0.10},
    "praise":          {"DA": +0.25, "ACh": +0.10, "NE": +0.15, "5HT": -0.05, "GABA":  0.00, "Oxy": +0.20},
    "criticism":        {"DA": -0.10, "ACh": +0.05, "NE": +0.20, "5HT": +0.25, "GABA":  0.00, "Oxy": -0.15},
    "social_bond":      {"DA": +0.10, "ACh": +0.05, "NE": -0.05, "5HT": +0.05, "GABA": +0.05, "Oxy": +0.35},
    "social_loss":      {"DA": -0.15, "ACh":  0.00, "NE": +0.10, "5HT": +0.30, "GABA":  0.00, "Oxy": -0.25},
    "achievement":      {"DA": +0.50, "ACh": +0.15, "NE": +0.20, "5HT": -0.10, "GABA":  0.00, "Oxy": +0.05},
    "novelty":          {"DA": +0.15, "ACh": +0.40, "NE": +0.10, "5HT":  0.00, "GABA":  0.00, "Oxy":  0.00},
}


def analyze_modulator_superposition(events: List[dict]) -> List[dict]:
    """
    分析同时触发事件的调质叠加效应
    
    找出 simultaneous_count > 1 的时刻, 计算调质净效应,
    标注冲突情况 (如 DA↑ + DA↓ 同时发生)
    """
    # 按步数分组
    by_step = defaultdict(list)
    for e in events:
        by_step[e["step_target"]].append(e)
    
    superposition_events = []
    
    for step, evts in sorted(by_step.items()):
        if len(evts) < 2:
            continue
        
        # 计算调质叠加
        net_effect = {"DA": 0, "5HT": 0, "NE": 0, "ACh": 0, "GABA": 0, "Oxy": 0}
        conflicts = []
        
        for e in evts:
            gene = GENE_MAP_BASE.get(e["event_type"], {})
            for mod, delta in gene.items():
                net_effect[mod] += delta * (e["intensity"] / 30.0)  # 归一化到 intensity=30
            
            # 检测冲突 (同一调质有正有负)
        
        # 检测每个调质的方向冲突
        for mod in net_effect:
            positive = False
            negative = False
            for e in evts:
                gene = GENE_MAP_BASE.get(e["event_type"], {})
                delta = gene.get(mod, 0) * (e["intensity"] / 30.0)
                if delta > 0.01: positive = True
                if delta < -0.01: negative = True
            if positive and negative:
                conflicts.append(mod)
        
        superposition_events.append({
            "step": step,
            "events": [e["event_type"] for e in evts],
            "chains": [e["causal_chain"] for e in evts],
            "net_effect": net_effect,
            "conflicts": conflicts,
            "is_conflict": len(conflicts) > 0,
        })
    
    return superposition_events


# =============================================================================
# 统计与输出
# =============================================================================

def print_parallel_stats(events: List[dict], stage: str, parallel_count: int, total_steps: int):
    """打印并行数据集统计"""
    mode_label = (
        f"概率驱动动态 (λ={parallel_count})" if parallel_count == 0
        else f"固定并行×{parallel_count} (旧模式)"
    )
    print(f"\n[generate_parallel_causal_dataset] 动态并行因果链数据集生成完成")
    print(f"  阶段:           {stage}")
    print(f"  调度模式:       {mode_label}")
    print(f"  总步数:         {total_steps}")
    print(f"  事件总数:       {len(events)}")
    print(f"  事件跨度:       step {events[0]['step_target']} ~ {events[-1]['step_target']}")
    
    # === 动态并行深度直方图 (核心新增) ===
    # 显示每个事件触发时刻的"当时有多少条链在并行"
    depth_counts = defaultdict(int)
    for e in events:
        depth_counts[e.get("parallel_depth", 1)] += 1
    
    print(f"\n  === 动态并行深度分布 (每个事件触发时的活跃链数) ===")
    total_events = len(events)
    for depth, count in sorted(depth_counts.items()):
        pct = count / total_events * 100
        bar = "█" * int(pct / 2)
        print(f"    {depth} 链并行: {count:4d} 事件 ({pct:5.1f}%) {bar}")
    
    # 真实并行触发时刻分布 (基于 generate_parallel_dataset._last_histogram)
    histogram = getattr(generate_parallel_dataset, "_last_histogram", None)
    if histogram:
        total_moments = sum(histogram.values())
        print(f"\n  === 真实并行触发时刻分布 (共 {total_moments} 个事件时刻) ===")
        for depth, count in sorted(histogram.items()):
            pct = count / total_moments * 100
            bar = "█" * int(pct / 2)
            print(f"    {depth} 链并行时刻: {count:4d} ({pct:5.1f}%) {bar}")
    
    # 同时触发事件数分布 (同一 step 触发多个事件)
    sim_counts = defaultdict(int)
    for e in events:
        sim_counts[e.get("simultaneous_count", 1)] += 1
    
    print(f"\n  === 同一 step 同时触发事件数分布 ===")
    for sim, count in sorted(sim_counts.items()):
        pct = count / total_events * 100
        bar = "█" * int(pct / 2)
        print(f"    ×{sim} 同时触发: {count:4d} 事件 ({pct:5.1f}%) {bar}")
    
    # 因果链统计
    chain_counts = defaultdict(int)
    for e in events:
        chain_counts[e.get("causal_chain", "unknown")] += 1
    
    print(f"\n  因果链分布 ({len(chain_counts)} 种):")
    for name, count in sorted(chain_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:42s}: {count}")
    
    # 调质叠加分析
    superpos = analyze_modulator_superposition(events)
    print(f"\n  调质叠加分析:")
    print(f"    同时触发时刻: {len(superpos)}")
    
    if superpos:
        conflict_count = sum(1 for s in superpos if s["is_conflict"])
        print(f"    其中有冲突:   {conflict_count} ({conflict_count/len(superpos)*100:.1f}%)")
        
        # 冲突示例
        conflict_examples = [s for s in superpos if s["is_conflict"]][:3]
        if conflict_examples:
            print(f"\n    冲突示例 (前 3 个):")
            for s in conflict_examples:
                print(f"      step {s['step']}: {s['events']} → 冲突调质: {s['conflicts']}")
                net = {k: round(v, 3) for k, v in s["net_effect"].items() if abs(v) > 0.01}
                print(f"        净效应: {net}")
    
    # 事件类型分布
    type_counts = defaultdict(int)
    for e in events:
        type_counts[e["event_type"]] += 1
    
    print(f"\n  事件类型分布:")
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {et:20s}: {count}")
    
    # 强度统计
    intensities = [e["intensity"] for e in events]
    print(f"\n  强度范围:       [{min(intensities)}, {max(intensities)}]")
    print(f"  平均强度:       {sum(intensities)/len(intensities):.1f}")


def list_combinations():
    """列出所有推荐的并行组合"""
    print("=" * 80)
    print("推荐的并行因果链组合")
    print("=" * 80)
    
    for (stage, parallel_count), combos in sorted(RECOMMENDED_COMBINATIONS.items()):
        print(f"\n【{stage}】并行深度 = {parallel_count} ({len(combos)} 种组合)")
        print("-" * 60)
        for i, combo in enumerate(combos, 1):
            print(f"  {i}. " + " + ".join(combo))


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="并行因果链事件数据集生成器 (支持多链同时运行)"
    )
    parser.add_argument("--stage", type=str, default="enlightenment",
                        choices=["enlightenment", "middle_school", "high_school"],
                        help="训练阶段")
    parser.add_argument("--steps", type=int, default=20000,
                        help="总训练步数")
    parser.add_argument("--parallel", type=int, default=0,
                        help="并行模式: 0=正态分布驱动(推荐), 1=纯串行, 2/3=固定并行(已弃用)")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="正态分布均值 μ (0=用阶段默认: 启蒙1.5/初中2.5/高中3.5)")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="正态分布标准差 σ (0=用阶段默认: 启蒙0.8/初中1.0/高中1.2)")
    parser.add_argument("--target-window", type=int, default=500,
                        help="目标深度重采样窗口步数 (每 N 步重新正态采样)")
    parser.add_argument("--min-idle", type=int, default=100,
                        help="链结束后的最小冷却步数")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 JSONL 文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--list-combinations", action="store_true",
                        help="列出推荐的并行组合")
    args = parser.parse_args()

    if args.list_combinations:
        list_combinations()
        return 0

    # 默认输出路径
    if args.output is None:
        if args.parallel == 0:
            args.output = (
                f"data/events/dynamic_"
                f"{args.stage}_{args.steps // 1000}k.jsonl"
            )
        else:
            args.output = (
                f"data/events/parallel{args.parallel}_"
                f"{args.stage}_{args.steps // 1000}k.jsonl"
            )

    # 生成
    if args.parallel == 1:
        # 串行模式: 回退到原生成器
        print("[INFO] parallel=1, 使用串行模式 (回退到 generate_causal_event_dataset)")
        from generate_causal_event_dataset import generate_causal_dataset
        events = generate_causal_dataset(
            stage=args.stage, total_steps=args.steps, seed=args.seed
        )
    elif args.parallel == 0:
        # 正态分布驱动模式 (推荐)
        stage_defaults = STAGE_PARALLEL_PARAMS.get(args.stage, (2.0, 1.0))
        mu = args.mu if args.mu > 0 else stage_defaults[0]
        sigma = args.sigma if args.sigma > 0 else stage_defaults[1]
        print(f"[INFO] parallel=0, 使用正态分布驱动模式")
        print(f"  μ_parallel    = {mu} (阶段默认: {stage_defaults[0]})")
        print(f"  σ_parallel    = {sigma} (阶段默认: {stage_defaults[1]})")
        print(f"  target_window = {args.target_window}")
        print(f"  min_idle      = {args.min_idle}")
        print(f"  理论分布: 68% 在 [{mu-sigma:.1f}, {mu+sigma:.1f}]")
        print(f"            95% 在 [{mu-2*sigma:.1f}, {mu+2*sigma:.1f}]")
        print(f"            99.7% 在 [{mu-3*sigma:.1f}, {mu+3*sigma:.1f}]")
        events = generate_parallel_dataset(
            stage=args.stage,
            total_steps=args.steps,
            parallel_count=0,  # 触发正态模式
            seed=args.seed,
            mu_parallel=mu,
            sigma_parallel=sigma,
            target_window=args.target_window,
            min_idle_steps=args.min_idle,
        )
    else:
        # 旧版固定并行模式 (保留兼容)
        print(f"[WARN] 固定 parallel={args.parallel} 模式已弃用, 建议用 --parallel 0")
        events = generate_parallel_dataset(
            stage=args.stage,
            total_steps=args.steps,
            parallel_count=args.parallel,
            seed=args.seed,
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
    print_parallel_stats(events, args.stage, args.parallel, args.steps)
    print(f"\n  输出文件:       {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
