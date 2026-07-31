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
  "events": [{"step_offset":0,"event_type":"exam_success","intensity":30}, ...],
  "target_modulators": [da, 5ht, ne, ach, gaba, oxy],
  "target_pad": [pleasure, arousal, dominance],
  "target_tool": -1
}

用法:
  python generate_curriculum_data.py --stage middle_school --samples 2000
  python generate_curriculum_data.py --stage high_school --samples 2000
"""

import argparse
import json
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
# 每条链: (chain_name, [ (event_type, intensity, role), ... ])
MIDDLE_SCHOOL_CHAINS: List[Tuple[str, List[Tuple[str, int, str]]]] = [
    # 学业
    ("exam_success_chain", [
        ("novelty",    20, "cause"),      # 考试开始
        ("achievement", 40, "effect"),     # 成绩好
        ("praise",     25, "consequence"), # 老师表扬
        ("social_bond", 15, "resolution"), # 同伴认可
    ]),
    ("exam_failure_recovery_chain", [
        ("criticism",  -25, "cause"),     # 考试失败
        ("social_loss",-20, "effect"),    # 失落
        ("social_bond", 20, "consequence"),# 同伴安慰
        ("praise",     15, "resolution"),  # 恢复自信
    ]),
    ("teacher_praise_effort_chain", [
        ("achievement", 30, "cause"),
        ("praise",      30, "effect"),
        ("social_bond", 15, "consequence"),
    ]),
    ("hobby_mastery_chain", [
        ("novelty",     25, "cause"),
        ("achievement", 35, "effect"),
        ("praise",      20, "consequence"),
    ]),
    # 社交
    ("peer_acceptance_chain", [
        ("social_bond", 25, "cause"),
        ("praise",      20, "effect"),
        ("achievement", 15, "consequence"),
    ]),
    ("peer_rejection_recovery_chain", [
        ("threat_social", -30, "cause"),
        ("social_loss",   -20, "effect"),
        ("social_bond",    20, "consequence"),
        ("praise",         15, "resolution"),
    ]),
    ("parent_conflict_repair_chain", [
        ("criticism",    -20, "cause"),
        ("threat_social",-25, "effect"),
        ("social_bond",   25, "consequence"),
        ("achievement",   15, "resolution"),
    ]),
]

HIGH_SCHOOL_CHAINS: List[Tuple[str, List[Tuple[str, int, str]]]] = [
    # 学业压力
    ("major_exam_stress_chain", [
        ("novelty",     30, "cause"),      # 大考开始
        ("achievement", 50, "effect"),     # 成绩优异
        ("praise",      30, "consequence"),# 老师表扬
        ("social_bond", 20, "resolution"), # 同伴认可
    ]),
    # 亲密关系
    ("first_relationship_chain", [
        ("social_bond", 35, "cause"),
        ("achievement", 25, "effect"),
        ("praise",      20, "consequence"),
        ("social_bond", 30, "resolution"),
    ]),
    ("rejection_recovery_chain", [
        ("social_loss",   -35, "cause"),   # 表白失败
        ("threat_social", -20, "effect"),
        ("social_bond",    20, "consequence"), # 朋友支持
        ("achievement",    15, "resolution"),  # 自我成长
    ]),
    # 认知探索
    ("identity_exploration_chain", [
        ("novelty",      30, "cause"),     # 自我质疑
        ("criticism",    -15, "effect"),
        ("achievement",   30, "consequence"), # 确认自我
        ("social_bond",   20, "resolution"),
    ]),
    ("career_aspiration_chain", [
        ("novelty",      25, "cause"),
        ("achievement",   35, "effect"),
        ("praise",        25, "consequence"),
        ("social_bond",   15, "resolution"),
    ]),
    # 复杂社交
    ("friendship_betrayal_forgiveness_chain", [
        ("social_loss",   -30, "cause"),   # 背叛
        ("threat_social", -25, "effect"),
        ("social_bond",    25, "consequence"), # 原谅
        ("praise",         20, "resolution"),
    ]),
    ("competition_rivalry_chain", [
        ("threat_social", -25, "cause"),
        ("achievement",    40, "effect"),  # 赢得竞争
        ("praise",         25, "consequence"),
        ("social_bond",    15, "resolution"),
    ]),
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


def target_pad_from_mod(mod: List[float]) -> List[float]:
    """PAD 情感状态 = 调质的启发式映射 (与 get_affective_state 风格类似)"""
    da, ach, ne, ht5, gaba, oxy = mod
    pleasure   = max(-1.0, min(1.0,  0.5 * da - 0.3 * ne + 0.4 * oxy - 0.3 * ht5))
    arousal    = max(-1.0, min(1.0,  0.3 * ne + 0.4 * ach))
    dominance  = max(-1.0, min(1.0,  0.3 * da - 0.2 * ht5 + 0.2 * oxy))
    return [round(pleasure, 3), round(arousal, 3), round(dominance, 3)]


def clamp_mod(mod: List[float]) -> List[float]:
    return [max(0.0, min(2.0, v)) for v in mod]


def generate_samples(stage: str, n_samples: int, seed: int,
                     max_events_per_sample: int = 4) -> List[dict]:
    rng = random.Random(seed)
    chains = MIDDLE_SCHOOL_CHAINS if stage == "middle_school" else HIGH_SCHOOL_CHAINS
    baseline = STAGE_BASELINE[stage]

    samples = []
    sample_id = 1
    for _ in range(n_samples):
        chain_name, chain_events = rng.choice(chains)
        # 随机子集 (至少 2 个事件), 保持链内顺序
        k = rng.randint(2, min(len(chain_events), max_events_per_sample))
        selected = chain_events[:k]

        events = []
        mod_accum = [v for v in baseline]
        # step_offset 对齐 launch_modulatory 的 100 步读取节奏:
        #   launch_modulatory 仅在每个 100 倍数步读取 h_event_signal,
        #   事件只有落在 100 倍数步才能被注入调质系统 (见 scheduler.cu)
        #   因此 offset 取 100/200/300/400 (窗口起点须为 100 倍数, 窗口 ≥ 400)
        for i, (etype, intensity, role) in enumerate(selected):
            step_offset = 100 * (i + 1)
            events.append({
                "step_offset": step_offset,
                "event_type": etype,
                "intensity": intensity,
            })
            delta = event_delta(etype, intensity)
            mod_accum = [a + b for a, b in zip(mod_accum, delta)]

        mod_final = clamp_mod(mod_accum)
        samples.append({
            "sample_id": sample_id,
            "events": events,
            "target_modulators": [round(v, 4) for v in mod_final],
            "target_pad": target_pad_from_mod(mod_final),
            "target_tool": -1,   # 初中/启蒙不训工具, 高中工具留 3e
            "chain": chain_name,
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

    # 调质目标范围
    da = [s["target_modulators"][0] for s in samples]
    oxy = [s["target_modulators"][5] for s in samples]
    print(f"  目标 DA 范围:   [{min(da):.2f}, {max(da):.2f}]")
    print(f"  目标 Oxy 范围:  [{min(oxy):.2f}, {max(oxy):.2f}]")


def main():
    parser = argparse.ArgumentParser(description="课程训练数据生成器")
    parser.add_argument("--stage", type=str, default="middle_school",
                        choices=["middle_school", "high_school"])
    parser.add_argument("--samples", type=int, default=2000, help="样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-events", type=int, default=4)
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
