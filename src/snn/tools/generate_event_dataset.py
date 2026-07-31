#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-C1: 离散事件 → 连续情绪变化 数据集生成器
=====================================================
生成 events.jsonl 事件流文件, 供 snn_train.exe --event-stream 使用。

事件格式 (与 event_scheduler.cpp 解析器对齐):
  {"event_id":1, "step_target":500, "event_type":"food_tasty",
   "modifiers":{"publicity":"public","authority":"peer","temporal":"momentary"},
   "intensity":30, "duration_s":0, "description":"吃到巧克力"}

基因映射表 (与 gene_event_map.h GENE_MAP_BASE 对齐):
  food_tasty:       DA↑0.40  ACh↑0.10  NE↑0.05  5HT↓0.05  GABA±0   Oxy↑0.02
  food_bland:       DA↑0.05  (其余近零)
  threat_physical:  DA↓0.20  ACh↑0.30  NE↑0.60  5HT↑0.40  GABA↑0.10  Oxy↓0.05
  threat_social:    DA↓0.15  ACh↑0.20  NE↑0.45  5HT↑0.35  GABA↑0.05  Oxy↓0.10
  praise:           DA↑0.25  ACh↑0.10  NE↑0.15  5HT↓0.05  GABA±0   Oxy↑0.20
  criticism:        DA↓0.10  ACh↑0.05  NE↑0.20  5HT↑0.25  GABA±0   Oxy↓0.15
  social_bond:      DA↑0.10  ACh↑0.05  NE↓0.05  5HT↑0.05  GABA↑0.05  Oxy↑0.35
  social_loss:      DA↓0.15  NE↑0.10  5HT↑0.30  Oxy↓0.25
  achievement:      DA↑0.50  ACh↑0.15  NE↑0.20  5HT↓0.10  Oxy↑0.05
  novelty:          DA↑0.15  ACh↑0.40  NE↑0.10

用法:
  python generate_event_dataset.py --steps 5000  --output data/events/events_5k.jsonl
  python generate_event_dataset.py --steps 100000 --output data/events/events_100k.jsonl --mode random
  python generate_event_dataset.py --list-scenarios
"""

import argparse
import json
import os
import random
import sys

# =============================================================================
# 事件类型定义 (与 event_types.h EventType 枚举对齐)
# =============================================================================
EVENT_TYPES = [
    "food_tasty",
    "food_bland",
    "threat_physical",
    "threat_social",
    "praise",
    "criticism",
    "social_bond",
    "social_loss",
    "achievement",
    "novelty",
]

# 修饰符选项
PUBLICITY_OPTIONS = ["private", "public"]
AUTHORITY_OPTIONS = ["peer", "authority"]
TEMPORAL_OPTIONS = ["momentary", "sustained"]


# =============================================================================
# 预定义场景: 5K 步 "一天生活" 事件序列
# 每个事件间隔约 500 步, 覆盖全 10 种事件类型 + 修饰符组合
# =============================================================================
SCENARIO_5K_DAILY_LIFE = [
    # (step, event_type, publicity, authority, temporal, intensity, description)
    (300,   "food_tasty",      "private", "peer",     "momentary",  30,  "早晨吃到喜欢的早餐"),
    (700,   "novelty",         "private", "peer",     "momentary",  25,  "路上看到新奇的事物"),
    (1100,  "criticism",       "public",  "authority", "momentary",  20,  "会议上被上司批评"),
    (1500,  "achievement",     "public",  "authority", "momentary",  40,  "完成一个重要项目"),
    (1900,  "food_bland",      "private", "peer",     "momentary", -10,  "午餐味道很淡"),
    (2300,  "social_bond",     "private", "peer",     "sustained",  15,  "与好友长时间聊天"),
    (2700,  "threat_physical", "private", "peer",     "momentary", -30,  "差点被车撞到"),
    (3100,  "praise",          "public",  "authority", "momentary",  20,  "公开受到表扬"),
    (3500,  "social_loss",     "private", "peer",     "momentary",  20,  "朋友搬去另一个城市"),
    (3900,  "food_tasty",      "private", "peer",     "momentary",  35,  "晚餐吃到了美食"),
    (4300,  "threat_social",   "public",  "peer",     "momentary", -20,  "社交场合被排斥"),
    (4700,  "social_bond",     "private", "peer",     "sustained",  10,  "与家人共度夜晚"),
]

# 10K 步扩展场景: 在 5K 基础上增加更多事件 + 更极端强度
SCENARIO_10K_EXTENDED = SCENARIO_5K_DAILY_LIFE + [
    (5200,  "achievement",     "public",  "authority", "momentary",  45,  "获得年度最佳员工"),
    (5600,  "novelty",         "private", "peer",     "momentary",  30,  "尝试新的爱好"),
    (6000,  "criticism",       "private", "peer",     "momentary",  10,  "自我反思发现不足"),
    (6400,  "food_tasty",      "public",  "peer",     "momentary",  20,  "和朋友聚餐"),
    (6800,  "threat_physical", "private", "peer",     "sustained", -40,  "持续的身体不适"),
    (7200,  "praise",          "private", "peer",     "momentary",  15,  "收到朋友的赞美"),
    (7600,  "social_loss",     "private", "peer",     "sustained",  30,  "宠物走丢了"),
    (8000,  "social_bond",     "public",  "peer",     "sustained",  20,  "参加社区活动"),
    (8400,  "achievement",     "private", "peer",     "momentary",  25,  "学会了一项新技能"),
    (8800,  "food_bland",      "private", "peer",     "momentary", -15,  "出差吃到难吃的飞机餐"),
    (9200,  "threat_social",   "private", "authority", "momentary", -25,  "被上级约谈"),
    (9600,  "novelty",         "public",  "peer",     "momentary",  35,  "参观新展览"),
]


# =============================================================================
# JSONL 事件构建
# =============================================================================
def build_event(event_id, step, event_type, publicity="private",
                authority="peer", temporal="momentary",
                intensity=0, duration_s=0, description=""):
    """构建单个事件的 JSONL 行 (与 event_scheduler.cpp 解析器对齐)。"""
    event = {
        "event_id": event_id,
        "step_target": step,
        "event_type": event_type,
        "modifiers": {
            "publicity": publicity,
            "authority": authority,
            "temporal": temporal,
        },
        "intensity": intensity,
        "duration_s": duration_s,
        "description": description,
    }
    return event


def event_to_jsonl(event):
    """将事件转为 JSONL 行字符串。"""
    return json.dumps(event, ensure_ascii=False)


# =============================================================================
# 生成模式
# =============================================================================
def generate_from_scenario(scenario, total_steps):
    """从预定义场景生成事件列表。"""
    events = []
    for eid, (step, etype, pub, auth, temp, inten, desc) in enumerate(scenario, 1):
        if step >= total_steps:
            break
        events.append(build_event(eid, step, etype, pub, auth, temp, inten, 0, desc))
    return events


def generate_random(total_steps, seed=42, min_interval=300, max_interval=800):
    """随机生成事件序列, 覆盖全 10 种类型 + 随机修饰符。"""
    rng = random.Random(seed)
    events = []
    step = rng.randint(200, 500)
    eid = 1
    while step < total_steps:
        etype = rng.choice(EVENT_TYPES)
        pub = rng.choice(PUBLICITY_OPTIONS)
        auth = rng.choice(AUTHORITY_OPTIONS)
        temp = rng.choice(TEMPORAL_OPTIONS)
        # 强度: 偏向中等, 偶尔极端
        intensity = rng.randint(-40, 40)
        if rng.random() < 0.15:
            intensity = rng.randint(-50, -40) if rng.random() < 0.5 else rng.randint(40, 50)
        desc = f"随机事件_{etype}_step{step}"
        events.append(build_event(eid, step, etype, pub, auth, temp, intensity, 0, desc))
        eid += 1
        step += rng.randint(min_interval, max_interval)
    return events


def generate_mixed(total_steps, seed=42):
    """混合模式: 前半用预定义场景, 后半用随机生成。"""
    half = total_steps // 2
    events = generate_from_scenario(SCENARIO_10K_EXTENDED, total_steps)
    if half < len(events):
        events = events[:0]  # 清空, 场景已覆盖
    # 随机补充后半段
    rng = random.Random(seed)
    start_step = max(e["step_target"] for e in events) + 500 if events else half
    eid = len(events) + 1
    step = start_step
    while step < total_steps:
        etype = rng.choice(EVENT_TYPES)
        pub = rng.choice(PUBLICITY_OPTIONS)
        auth = rng.choice(AUTHORITY_OPTIONS)
        temp = rng.choice(TEMPORAL_OPTIONS)
        intensity = rng.randint(-35, 35)
        desc = f"补充事件_{etype}_step{step}"
        events.append(build_event(eid, step, etype, pub, auth, temp, intensity, 0, desc))
        eid += 1
        step += rng.randint(300, 700)
    return events


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a-C1 事件数据集生成器 (events.jsonl)"
    )
    parser.add_argument("--steps", type=int, default=5000,
                        help="总训练步数 (决定事件时间跨度, 默认 5000)")
    parser.add_argument("--output", "-o", type=str, default="data/events/events.jsonl",
                        help="输出 JSONL 文件路径")
    parser.add_argument("--mode", type=str, default="scenario",
                        choices=["scenario", "random", "mixed"],
                        help="生成模式: scenario=预定义场景, random=随机, mixed=混合")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (random/mixed 模式)")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="列出可用预定义场景")
    args = parser.parse_args()

    if args.list_scenarios:
        print("可用预定义场景:")
        print(f"  SCENARIO_5K_DAILY_LIFE  : {len(SCENARIO_5K_DAILY_LIFE)} 个事件, 覆盖 5K 步")
        print(f"  SCENARIO_10K_EXTENDED   : {len(SCENARIO_10K_EXTENDED)} 个事件, 覆盖 10K 步")
        print("\n事件类型 (10 种):")
        for i, et in enumerate(EVENT_TYPES):
            print(f"  {i}: {et}")
        return

    # 选择生成模式
    if args.mode == "scenario":
        if args.steps <= 5000:
            events = generate_from_scenario(SCENARIO_5K_DAILY_LIFE, args.steps)
        else:
            events = generate_from_scenario(SCENARIO_10K_EXTENDED, args.steps)
    elif args.mode == "random":
        events = generate_random(args.steps, seed=args.seed)
    else:
        events = generate_mixed(args.steps, seed=args.seed)

    if not events:
        print("[WARN] 未生成任何事件, 请检查 --steps 参数", file=sys.stderr)
        return 1

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 写入 JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for evt in events:
            f.write(event_to_jsonl(evt) + "\n")

    # 统计
    type_counts = {}
    for evt in events:
        t = evt["event_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"[generate_event_dataset] 已生成 {len(events)} 个事件 → {args.output}")
    print(f"  总步数:     {args.steps}")
    print(f"  生成模式:   {args.mode}")
    print(f"  事件跨度:   step {events[0]['step_target']} ~ {events[-1]['step_target']}")
    print(f"  事件类型分布:")
    for et in EVENT_TYPES:
        c = type_counts.get(et, 0)
        if c > 0:
            print(f"    {et:20s}: {c}")
    print(f"  强度范围:   [{min(e['intensity'] for e in events)}, {max(e['intensity'] for e in events)}]")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
