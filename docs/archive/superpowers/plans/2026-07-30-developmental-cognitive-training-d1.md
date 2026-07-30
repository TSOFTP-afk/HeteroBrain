# Phase 3a-D1: 发育式认知训练 — 新生儿期最小可行验证 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现新生儿期(0-1月)发育式认知训练数据管线，端到端验证因果链序列能否驱动 SNN 调质响应。

**Architecture:** 模板库定义场景骨架(含内置叙事文本) → 数据集组装器生成 JSONL 事件流 + 叙事文本 → BPE/字节转换器生成 token 流 → snn_train 端到端训练 → 验证器检查因果链完整性。D1 阶段不依赖 LLM，模板内置叙事；D2 阶段再接入 LLM 增强。

**Tech Stack:** Python 3 (数据生成), CUDA SNN (训练), JSONL (事件流), BPE/byte (token 编码)

**Spec:** `docs/superpowers/specs/2026-07-30-developmental-cognitive-training-design.md`

---

## 文件结构

| 文件 | 职责 | 创建/修改 |
|------|------|----------|
| `src/snn/tools/developmental_templates.py` | 新生儿期(0-1月)5个场景骨架 + 内置叙事 | 创建 |
| `src/snn/tools/generate_developmental_dataset.py` | 模板 → developmental_events.jsonl + narrative_text.txt | 创建 |
| `src/snn/tools/narrative_to_bpe.py` | 叙事文本 → token 流(.bin), 支持 byte fallback | 创建 |
| `src/snn/tools/narrative_generator.py` | LLM 叙事增强(可选, D1 阶段为 stub) | 创建 |
| `src/snn/tools/validate_developmental.py` | 因果链完整性 + 格式验证 | 创建 |
| `data/developmental/` | 输出目录 | 创建 |

**关键设计决策:**
- D1 模板内置叙事文本, 不依赖 LLM API (避免外部依赖)
- BPE 转换器支持 byte fallback (若 BPE tokenizer 不可用则用 UTF-8 字节)
- step_target 计算公式: `age_months * 2000 + scene_index * 400 + segment.step_offset`

---

### Task 1: 新生儿期场景模板库

**Files:**
- Create: `src/snn/tools/developmental_templates.py`

- [ ] **Step 1: 编写模板库**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-D1: 新生儿期 (0-1月) 发育场景模板库
================================================
5 个核心场景, 每场景含因果链 + 内置叙事 + 事件配置。
D1 阶段不依赖 LLM, 模板直接提供叙事文本。

场景列表:
  1. hunger_feeding:  饥饿→哭→喂养→满足 (因果链原型)
  2. warmth_safety:   寒冷→哭→包裹→温暖 (温度调节)
  3. startle_recover: 惊吓→哭→安抚→平静 (惊吓反射)
  4. sleep_wake:      困倦→哭→哄睡→入睡 (睡眠周期)
  5. discomfort_change: 不适→哭→换尿布→舒适 (身体护理)

时间映射: 1月龄 = 2000步, 每场景约400步 (含间隔)
"""

# 发育阶段定义
DEVELOPMENTAL_STAGES = {
    "neonatal": {"age_months": (0, 1), "narrative_style": "minimal"},
    "infant":   {"age_months": (2, 12), "narrative_style": "short_phrase"},
    "toddler":  {"age_months": (13, 36), "narrative_style": "short_sentence"},
    "preschool":{"age_months": (48, 72), "narrative_style": "full_sentence"},
    "school":   {"age_months": (84, 144), "narrative_style": "paragraph"},
    "teen":     {"age_months": (156, 216), "narrative_style": "complex"},
}

# 修饰符默认值
DEFAULT_MODIFIERS = {"publicity": "private", "authority": "authority", "temporal": "momentary"}

# 新生儿期 (0-1月) 场景模板
# 每个场景: 4-6 个微事件构成因果链
NEONATAL_SCENES = [
    {
        "scene_id": "neonatal_001_hunger",
        "age_months": 0,
        "scene_type": "hunger_feeding",
        "learning_goal": "basic_causality_cry_response",
        "causal_chain": ["hunger", "cry", "mother_response", "feeding", "satisfaction"],
        "narrative_segments": [
            {"text": "饿了",     "event_type": "food_bland",      "intensity": -25, "step_offset": 0},
            {"text": "哭",       "event_type": "threat_physical", "intensity": -10, "step_offset": 50},
            {"text": "妈妈来",   "event_type": "social_bond",     "intensity": 8,   "step_offset": 100},
            {"text": "喂奶",     "event_type": "food_tasty",      "intensity": 30,  "step_offset": 150},
            {"text": "饱了",     "event_type": "achievement",     "intensity": 12,  "step_offset": 200},
        ],
        "scene_duration_steps": 300,
    },
    {
        "scene_id": "neonatal_002_warmth",
        "age_months": 0,
        "scene_type": "warmth_safety",
        "learning_goal": "temperature_regulation",
        "causal_chain": ["cold", "cry", "wrapping", "warmth", "comfort"],
        "narrative_segments": [
            {"text": "冷",       "event_type": "threat_physical", "intensity": -15, "step_offset": 0},
            {"text": "哭",       "event_type": "threat_physical", "intensity": -10, "step_offset": 50},
            {"text": "裹好",     "event_type": "social_bond",     "intensity": 8,   "step_offset": 100},
            {"text": "暖了",     "event_type": "food_tasty",      "intensity": 20,  "step_offset": 150},
            {"text": "舒服",     "event_type": "social_bond",     "intensity": 10,  "step_offset": 200},
        ],
        "scene_duration_steps": 300,
    },
    {
        "scene_id": "neonatal_003_startle",
        "age_months": 0,
        "scene_type": "startle_recover",
        "learning_goal": "startle_reflex_recovery",
        "causal_chain": ["startle", "cry", "comfort", "soothing", "calm"],
        "narrative_segments": [
            {"text": "响声",     "event_type": "threat_physical", "intensity": -30, "step_offset": 0},
            {"text": "吓哭",     "event_type": "threat_physical", "intensity": -20, "step_offset": 40},
            {"text": "抱起",     "event_type": "social_bond",     "intensity": 10,  "step_offset": 80},
            {"text": "轻拍",     "event_type": "social_bond",     "intensity": 8,   "step_offset": 120},
            {"text": "静了",     "event_type": "achievement",     "intensity": 5,   "step_offset": 160},
        ],
        "scene_duration_steps": 250,
    },
    {
        "scene_id": "neonatal_004_sleep",
        "age_months": 1,
        "scene_type": "sleep_wake",
        "learning_goal": "sleep_cycle_regulation",
        "causal_chain": ["sleepiness", "fuss", "rocking", "falling_asleep", "rest"],
        "narrative_segments": [
            {"text": "困了",     "event_type": "food_bland",      "intensity": -10, "step_offset": 0},
            {"text": "闹",       "event_type": "threat_physical", "intensity": -8,  "step_offset": 50},
            {"text": "摇",       "event_type": "social_bond",     "intensity": 8,   "step_offset": 100},
            {"text": "睡了",     "event_type": "achievement",     "intensity": 15,  "step_offset": 150},
            {"text": "安稳",     "event_type": "social_bond",     "intensity": 10,  "step_offset": 200},
        ],
        "scene_duration_steps": 300,
    },
    {
        "scene_id": "neonatal_005_discomfort",
        "age_months": 1,
        "scene_type": "discomfort_change",
        "learning_goal": "discomfort_relief_causality",
        "causal_chain": ["discomfort", "cry", "change", "relief", "comfort"],
        "narrative_segments": [
            {"text": "不适",     "event_type": "threat_physical", "intensity": -18, "step_offset": 0},
            {"text": "哭",       "event_type": "threat_physical", "intensity": -12, "step_offset": 50},
            {"text": "换",       "event_type": "social_bond",     "intensity": 8,   "step_offset": 100},
            {"text": "干净",     "event_type": "food_tasty",      "intensity": 18,  "step_offset": 150},
            {"text": "舒服",     "event_type": "social_bond",     "intensity": 10,  "step_offset": 200},
        ],
        "scene_duration_steps": 300,
    },
]


def get_scenes_for_age(age_months):
    """返回指定月龄的所有场景模板。"""
    if 0 <= age_months <= 1:
        return NEONATAL_SCENES
    return []


def get_all_scenes(age_min=0, age_max=1):
    """返回指定月龄范围内的所有场景。"""
    all_scenes = []
    for age in range(age_min, age_max + 1):
        all_scenes.extend(get_scenes_for_age(age))
    return all_scenes


def get_narrative_length(age_months):
    """返回指定月龄的叙事长度(字数)。"""
    if age_months <= 1:   return 2  # 新生儿: 1-3字/段
    if age_months <= 12:  return 5  # 婴儿: 2-5字/段
    if age_months <= 36:  return 7  # 幼儿: 3-7字/段
    if age_months <= 72:  return 15 # 学前: 完整句
    if age_months <= 144: return 30 # 学龄: 短段落
    return 50  # 青少年: 复杂叙事
```

- [ ] **Step 2: 验证模板库可导入**

Run: `python -c "from src.snn.tools.developmental_templates import NEONATAL_SCENES, get_all_scenes; scenes = get_all_scenes(0, 1); print(f'{len(scenes)} scenes'); assert len(scenes) == 5; s0 = scenes[0]; print(f'scene_id={s0[\"scene_id\"]}, segments={len(s0[\"narrative_segments\"])}'); assert len(s0['narrative_segments']) == 5; print('OK')"`
Expected: `5 scenes` + `scene_id=neonatal_001_hunger, segments=5` + `OK`

- [ ] **Step 3: 提交**

```bash
git add src/snn/tools/developmental_templates.py
git commit -m "feat(snn): add neonatal developmental scene templates (Phase 3a-D1 Task 1)"
```

---

### Task 2: 发育数据集组装器

**Files:**
- Create: `src/snn/tools/generate_developmental_dataset.py`

**职责:** 从模板库生成两个输出文件:
1. `developmental_events.jsonl` — 展平的微事件流, 供 `--event-stream` 使用
2. `narrative_text.txt` — 拼接的叙事文本, 供 `narrative_to_bpe.py` 使用

**step_target 计算公式:**
```
step_target = age_months * STEPS_PER_MONTH + scene_index * SCENE_INTERVAL + segment.step_offset
```
其中 `STEPS_PER_MONTH = 2000`, `SCENE_INTERVAL = 400` (场景间隔, 含休息期)

- [ ] **Step 1: 编写数据集组装器**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-D1: 发育数据集组装器
===============================
从模板库生成 developmental_events.jsonl + narrative_text.txt

输出:
  developmental_events.jsonl  — 供 snn_train --event-stream
  narrative_text.txt          — 供 narrative_to_bpe.py

用法:
  python generate_developmental_dataset.py --age-min 0 --age-max 1 --output-dir data/developmental
"""

import argparse
import json
import os
import sys

# 将当前文件所在目录加入 path, 以便导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from developmental_templates import get_all_scenes, DEFAULT_MODIFIERS

# 时间映射常量
STEPS_PER_MONTH = 2000
SCENE_INTERVAL = 400  # 场景间隔步数 (含休息期)


def compute_step_target(age_months, scene_index, step_offset):
    """计算微事件的绝对 step_target。"""
    base = age_months * STEPS_PER_MONTH
    scene_base = scene_index * SCENE_INTERVAL
    return base + scene_base + step_offset


def build_event_record(scene, segment_idx, segment, step_target):
    """构建单个微事件的 JSONL 记录 (与 event_scheduler.cpp 解析器对齐)。"""
    return {
        "scene_id": scene["scene_id"],
        "age_months": scene["age_months"],
        "segment_idx": segment_idx,
        "step_target": step_target,
        "event_type": segment["event_type"],
        "intensity": segment["intensity"],
        "modifiers": DEFAULT_MODIFIERS,
        "description": segment["text"],
        "learning_goal": scene["learning_goal"],
    }


def generate_dataset(age_min, age_max, output_dir):
    """生成发育数据集。"""
    scenes = get_all_scenes(age_min, age_max)
    if not scenes:
        print(f"[WARN] 无 {age_min}-{age_max} 月龄的场景模板", file=sys.stderr)
        return 1

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    events_path = os.path.join(output_dir, "developmental_events.jsonl")
    narrative_path = os.path.join(output_dir, "narrative_text.txt")

    all_events = []
    narrative_lines = []

    for scene_idx, scene in enumerate(scenes):
        scene_narrative = []
        for seg_idx, segment in enumerate(scene["narrative_segments"]):
            step_target = compute_step_target(
                scene["age_months"], scene_idx, segment["step_offset"]
            )
            evt = build_event_record(scene, seg_idx, segment, step_target)
            all_events.append(evt)
            scene_narrative.append(segment["text"])
        # 叙事文本: 场景内用句号连接, 场景间用空行分隔
        narrative_lines.append("。".join(scene_narrative) + "。")

    # 按 step_target 排序事件
    all_events.sort(key=lambda e: e["step_target"])

    # 写入 JSONL
    with open(events_path, "w", encoding="utf-8") as f:
        for evt in all_events:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    # 写入叙事文本 (场景间空行分隔)
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(narrative_lines) + "\n")

    # 统计
    total_steps = (age_max + 1) * STEPS_PER_MONTH
    print(f"[generate_developmental_dataset] 已生成数据集 → {output_dir}")
    print(f"  月龄范围:   {age_min}-{age_max} 月")
    print(f"  场景数:     {len(scenes)}")
    print(f"  微事件数:   {len(all_events)}")
    print(f"  总步数:     {total_steps}")
    print(f"  事件文件:   {events_path}")
    print(f"  叙事文件:   {narrative_path}")

    # 打印事件类型分布
    type_counts = {}
    for evt in all_events:
        t = evt["event_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  事件类型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:20s}: {c}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a-D1 发育数据集组装器"
    )
    parser.add_argument("--age-min", type=int, default=0,
                        help="最小月龄 (默认 0)")
    parser.add_argument("--age-max", type=int, default=1,
                        help="最大月龄 (默认 1)")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="data/developmental",
                        help="输出目录 (默认 data/developmental)")
    args = parser.parse_args()
    return generate_dataset(args.age_min, args.age_max, args.output_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
```

- [ ] **Step 2: 运行组装器生成新生儿期数据集**

Run: `python src\snn\tools\generate_developmental_dataset.py --age-min 0 --age-max 1 --output-dir data\developmental`
Expected: 输出 `5 场景数`, `25 微事件数` (5场景×5段), `4000 总步数`, 生成 `developmental_events.jsonl` 和 `narrative_text.txt`

- [ ] **Step 3: 验证 JSONL 格式与 EventScheduler 兼容**

Run: `python -c "import json; lines=open('data/developmental/developmental_events.jsonl',encoding='utf-8').readlines(); evt=json.loads(lines[0]); assert 'step_target' in evt and 'event_type' in evt and 'intensity' in evt, 'missing fields'; print(f'first event: step={evt[\"step_target\"]} type={evt[\"event_type\"]} desc={evt[\"description\"]}'); print(f'total events: {len(lines)}'); assert len(lines)==25; print('OK')"`
Expected: `first event: step=0 type=food_bland desc=饿了` + `total events: 25` + `OK`

- [ ] **Step 4: 提交**

```bash
git add src/snn/tools/generate_developmental_dataset.py data/developmental/
git commit -m "feat(snn): add developmental dataset assembler (Phase 3a-D1 Task 2)"
```

---

### Task 3: 叙事 BPE/字节转换器

**Files:**
- Create: `src/snn/tools/narrative_to_bpe.py`

**职责:** 将 `narrative_text.txt` 转换为 `narrative_tokens.bin` (int32 流), 供 `--bpe-data` 或 `--input-mode byte` 使用。
支持两种模式:
- `byte` 模式: UTF-8 字节流 (无需 tokenizer, D1 默认)
- `bpe` 模式: BPE token 流 (需 tokenizer, 若可用)

- [ ] **Step 1: 编写 BPE/字节转换器**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-D1: 叙事文本 → token 流转换器
=========================================
将 narrative_text.txt 转换为 narrative_tokens.bin (int32 流)

支持模式:
  byte: UTF-8 字节流 (默认, 无外部依赖)
  bpe:  BPE token 流 (需 tokenizer, 若可用)

用法:
  python narrative_to_bpe.py --input data/developmental/narrative_text.txt --output data/developmental/narrative_tokens.bin --mode byte
"""

import argparse
import os
import struct
import sys


def text_to_byte_stream(text):
    """将文本转为 UTF-8 字节流 (int32 数组, 每个元素是一个字节 0-255)。"""
    byte_data = text.encode("utf-8")
    return list(byte_data)


def text_to_bpe_stream(text, tokenizer=None):
    """将文本转为 BPE token 流 (int32 数组)。"""
    if tokenizer is None:
        # Fallback: 用字节模式
        return text_to_byte_stream(text)
    tokens = tokenizer.encode(text)
    return tokens


def write_token_stream(tokens, output_path):
    """将 token 流写入 .bin 文件 (int32 little-endian)。"""
    with open(output_path, "wb") as f:
        for token in tokens:
            f.write(struct.pack("<i", token))
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a-D1 叙事文本 → token 流转换器"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入叙事文本文件路径")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="输出 token 流 .bin 文件路径")
    parser.add_argument("--mode", type=str, default="byte",
                        choices=["byte", "bpe"],
                        help="编码模式: byte (默认) 或 bpe")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if args.mode == "byte":
        tokens = text_to_byte_stream(text)
        mode_desc = "UTF-8 字节流"
    else:
        # BPE 模式: 尝试加载 tokenizer, 失败则 fallback 到 byte
        try:
            # 尝试导入现有 BPE 工具 (如果可用)
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prepare_bpe_data import load_tokenizer
            tokenizer = load_tokenizer()
            tokens = text_to_bpe_stream(text, tokenizer)
            mode_desc = "BPE token 流"
        except Exception as e:
            print(f"[WARN] BPE tokenizer 不可用 ({e}), 回退到 byte 模式", file=sys.stderr)
            tokens = text_to_byte_stream(text)
            mode_desc = "UTF-8 字节流 (BPE fallback)"

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    count = write_token_stream(tokens, args.output)

    print(f"[narrative_to_bpe] 转换完成")
    print(f"  输入:     {args.input}")
    print(f"  输出:     {args.output}")
    print(f"  模式:     {mode_desc}")
    print(f"  文本长度: {len(text)} 字符")
    print(f"  token 数: {count}")
    print(f"  文件大小: {os.path.getsize(args.output)} 字节")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行转换器生成 token 流**

Run: `python src\snn\tools\narrative_to_bpe.py --input data\developmental\narrative_text.txt --output data\developmental\narrative_tokens.bin --mode byte`
Expected: 输出 `模式: UTF-8 字节流`, `token 数: <N>` (N = 叙事文本字节数), 生成 `narrative_tokens.bin`

- [ ] **Step 3: 验证 .bin 文件格式正确**

Run: `python -c "import struct, os; data=open('data/developmental/narrative_tokens.bin','rb').read(); assert len(data)%4==0, 'file size not multiple of 4'; tokens=[struct.unpack('<i', data[i:i+4])[0] for i in range(0, min(20,len(data)), 4)]; print(f'first tokens: {tokens}'); assert all(0<=t<=255 for t in tokens), 'token out of byte range'; print(f'total tokens: {len(data)//4}'); print('OK')"`
Expected: `first tokens: [...]` (0-255 范围) + `total tokens: <N>` + `OK`

- [ ] **Step 4: 提交**

```bash
git add src/snn/tools/narrative_to_bpe.py data/developmental/narrative_tokens.bin
git commit -m "feat(snn): add narrative to BPE/byte stream converter (Phase 3a-D1 Task 3)"
```

---

### Task 4: 发育验证器

**Files:**
- Create: `src/snn/tools/validate_developmental.py`

**职责:** 验证发育数据集的因果链完整性 + 格式合法性 + 训练日志分析

- [ ] **Step 1: 编写发育验证器**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-D1: 发育数据集验证器
=============================
验证内容:
  1. JSONL 格式合法性 (字段完整、类型正确)
  2. 因果链完整性 (每场景的微事件数 ≥ 4, step_offset 递增)
  3. step_target 全局递增
  4. 事件类型与 gene_event_map.h 对齐
  5. 训练日志分析 (可选, 分析 snn_train 输出)

用法:
  python validate_developmental.py --events data/developmental/developmental_events.jsonl
  python validate_developmental.py --events data/developmental/developmental_events.jsonl --log run_dev.log
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# 合法事件类型 (与 event_types.h EventType 枚举对齐)
VALID_EVENT_TYPES = {
    "food_tasty", "food_bland", "threat_physical", "threat_social",
    "praise", "criticism", "social_bond", "social_loss",
    "achievement", "novelty",
}


def load_events(path):
    """加载 events.jsonl。"""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                evt = json.loads(line)
                evt["_line_num"] = line_num
                events.append(evt)
            except json.JSONDecodeError as e:
                print(f"  [FAIL] line {line_num}: JSON 解析错误: {e}")
    return events


def validate_format(events):
    """验证 JSONL 格式合法性。"""
    passed = 0
    failed = 0
    for evt in events:
        errors = []
        if "step_target" not in evt:
            errors.append("缺少 step_target")
        if "event_type" not in evt:
            errors.append("缺少 event_type")
        elif evt["event_type"] not in VALID_EVENT_TYPES:
            errors.append(f"未知事件类型: {evt['event_type']}")
        if "intensity" not in evt:
            errors.append("缺少 intensity")
        elif not isinstance(evt["intensity"], (int, float)) or not (-50 <= evt["intensity"] <= 50):
            errors.append(f"intensity 超出 [-50,50]: {evt['intensity']}")
        if "scene_id" not in evt:
            errors.append("缺少 scene_id")

        if errors:
            failed += 1
            for e in errors:
                print(f"  [FAIL] line {evt['_line_num']}: {e}")
        else:
            passed += 1
    return passed, failed


def validate_causal_chain(events):
    """验证因果链完整性: 每场景微事件数 ≥ 4, step_offset 递增。"""
    scenes = defaultdict(list)
    for evt in events:
        scenes[evt.get("scene_id", "")].append(evt)

    passed = 0
    failed = 0
    for scene_id, evts in scenes.items():
        evts.sort(key=lambda e: e.get("segment_idx", 0))
        errors = []
        if len(evts) < 4:
            errors.append(f"微事件数 {len(evts)} < 4 (因果链不完整)")
        # 检查 segment_idx 递增
        for i in range(1, len(evts)):
            if evts[i].get("segment_idx", 0) <= evts[i-1].get("segment_idx", 0):
                errors.append(f"segment_idx 未递增: {evts[i-1].get('segment_idx')} → {evts[i].get('segment_idx')}")
        # 检查 step_target 递增
        for i in range(1, len(evts)):
            if evts[i].get("step_target", 0) <= evts[i-1].get("step_target", 0):
                errors.append(f"step_target 未递增: {evts[i-1].get('step_target')} → {evts[i].get('step_target')}")

        if errors:
            failed += 1
            for e in errors:
                print(f"  [FAIL] scene {scene_id}: {e}")
        else:
            passed += 1
            print(f"  [PASS] scene {scene_id}: {len(evts)} 个微事件, 因果链完整")

    return passed, failed


def validate_step_order(events):
    """验证全局 step_target 递增。"""
    prev = -1
    for evt in events:
        step = evt.get("step_target", 0)
        if step < prev:
            print(f"  [FAIL] line {evt['_line_num']}: step_target {step} < 前一个 {prev}")
            return False
        prev = step
    print(f"  [PASS] 全局 step_target 递增 ({len(events)} 个事件)")
    return True


def analyze_training_log(log_path):
    """分析训练日志中的事件触发和调质响应。"""
    if not os.path.exists(log_path):
        print(f"  [WARN] 日志文件不存在: {log_path}")
        return

    events_triggered = 0
    affective_lines = 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "[Event]" in line:
                events_triggered += 1
            if "[Affective]" in line:
                affective_lines += 1

    print(f"  日志分析:")
    print(f"    事件触发次数: {events_triggered}")
    print(f"    情感状态采样: {affective_lines}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a-D1 发育数据集验证器"
    )
    parser.add_argument("--events", type=str, required=True,
                        help="developmental_events.jsonl 文件路径")
    parser.add_argument("--log", type=str, default="",
                        help="训练日志文件路径 (可选, 用于分析事件触发)")
    args = parser.parse_args()

    if not os.path.exists(args.events):
        print(f"[ERROR] 事件文件不存在: {args.events}", file=sys.stderr)
        return 1

    print(f"=" * 60)
    print(f"Phase 3a-D1 发育数据集验证")
    print(f"=" * 60)

    events = load_events(args.events)
    print(f"\n[1] 格式验证: {len(events)} 个事件")
    fmt_pass, fmt_fail = validate_format(events)
    print(f"  格式: {fmt_pass} PASS, {fmt_fail} FAIL")

    print(f"\n[2] 因果链完整性验证:")
    chain_pass, chain_fail = validate_causal_chain(events)
    print(f"  因果链: {chain_pass} PASS, {chain_fail} FAIL")

    print(f"\n[3] 全局 step_target 顺序验证:")
    step_ok = validate_step_order(events)

    if args.log:
        print(f"\n[4] 训练日志分析:")
        analyze_training_log(args.log)

    all_ok = (fmt_fail == 0) and (chain_fail == 0) and step_ok
    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"[RESULT] ALL PASS — 发育数据集验证通过")
        return 0
    else:
        print(f"[RESULT] FAIL — 发育数据集验证未通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行验证器检查数据集**

Run: `python src\snn\tools\validate_developmental.py --events data\developmental\developmental_events.jsonl`
Expected: `[1] 格式验证: 25 个事件` + `格式: 25 PASS, 0 FAIL` + 5 个 scene PASS + `[RESULT] ALL PASS`

- [ ] **Step 3: 提交**

```bash
git add src/snn/tools/validate_developmental.py
git commit -m "feat(snn): add developmental dataset validator (Phase 3a-D1 Task 4)"
```

---

### Task 5: LLM 叙事增强器 (Stub)

**Files:**
- Create: `src/snn/tools/narrative_generator.py`

**职责:** D1 阶段为 stub (直接返回模板内置叙事), D2 阶段接入 LLM API。接口已定义, 实现留 D2。

- [ ] **Step 1: 编写 LLM 叙事生成器 stub**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-D1: LLM 叙事生成器 (Stub)
====================================
D1 阶段: 直接返回模板内置叙事 (stub)
D2 阶段: 接入 LLM API 生成年龄适配的自然语言叙事

接口:
  generate_narrative(scene_skeleton) -> list[str]
  返回每个 causal_chain 环节对应的叙事文本列表

用法:
  python narrative_generator.py --check  # 验证 stub 可用
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from developmental_templates import NEONATAL_SCENES


def generate_narrative(scene_skeleton):
    """
    生成场景叙事文本列表。

    D1 (stub): 直接返回模板内置的 narrative_segments 文本。
    D2 (LLM):  调用 LLM 按年龄生成自然语言叙事, 缓存到 data/developmental/narrative_cache/

    Args:
        scene_skeleton: 场景模板 dict, 含 narrative_segments

    Returns:
        list[str]: 每个 causal_chain 环节对应的叙事文本
    """
    # D1 stub: 直接返回模板内置文本
    return [seg["text"] for seg in scene_skeleton["narrative_segments"]]


def generate_narrative_llm(scene_skeleton, age_months):
    """
    D2 阶段: 调用 LLM 生成年龄适配叙事 (占位, D2 实现)。

    Prompt 策略:
      - 新生儿: 1-3 字/段
      - 婴儿: 2-5 字/段
      - 幼儿: 3-7 字/段
      - 学前: 完整句
      - 学龄: 短段落
      - 青少年: 复杂叙事
    """
    raise NotImplementedError("LLM 叙事生成将在 Phase 3a-D2 实现")


def main():
    """验证 stub 可用性。"""
    print("[narrative_generator] D1 stub 验证")
    for scene in NEONATAL_SCENES:
        narratives = generate_narrative(scene)
        print(f"  {scene['scene_id']}: {narratives}")
    print(f"\n[OK] stub 可用, {len(NEONATAL_SCENES)} 个场景的内置叙事已就绪")
    print("[INFO] LLM 增强将在 Phase 3a-D2 实现")


if __name__ == "__main__":
    main()

