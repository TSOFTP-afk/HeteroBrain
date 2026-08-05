"""规律性组合引擎：场景模板 → 因果约束的事件序列 → 样本。

采样规则 (确定性 + 随机化):
  1. 场景采样: 知识链/情感链各半; 情感链内 正/混合/负 按权重平衡
  2. 槽位采样: cause/effect 必选, consequence/resolution 可选, 保持角色顺序
  3. 事件采样: 每槽随机选候选事件 + 区间内随机强度 (5 的倍数)
  4. offset: 从 {100,200,300} 随机取 k 个升序 (全部注入窗口 400 且进目标)
  5. 目标: 浓度模拟器 4 块推进 → 窗口末浓度 + PAD
"""
import random
from typing import List, Optional

from .simulator import ConcentrationSimulator, EventKB, target_pad_from_conc
from .scene_builder import TONE_CN

OFFSET_POOL = [100, 200, 300]      # 窗口 400 内, 全部注入且进目标 (无死重量)
OFFSET_POOL_FULL = [100, 200, 300, 400]  # 预留: 若用窗口 500 可启用

# 情感链内极性权重 (真实学生生活正负兼有)
POLARITY_WEIGHTS = {"pos": 0.45, "mixed": 0.35, "neg": 0.20}


def pick_intensity(rng: random.Random, irange) -> int:
    """强度区间随机取 5 的倍数 (正负保持符号)"""
    lo, hi = irange
    step = 5
    lo5 = (lo // step) * step
    hi5 = (hi // step) * step
    if hi5 < lo5:
        hi5 = lo5
    return rng.randint(lo5 // step, hi5 // step) * step


def sample_scene(rng: random.Random, scenes: List[dict],
                 tool_weights: Optional[dict] = None) -> dict:
    """按 知识/情感 各半 + 情感内极性权重 采样场景模板"""
    emotional = [s for s in scenes if s["tool"] == 6]
    knowledge = [s for s in scenes if s["tool"] != 6]
    has_knowledge = bool(knowledge)
    if has_knowledge and rng.random() < 0.5:
        return rng.choice(knowledge)
    pool = emotional if has_knowledge else scenes
    groups: dict = {}
    for s in pool:
        groups.setdefault(s["tone"], []).append(s)
    pol = rng.choices(list(groups),
                      weights=[POLARITY_WEIGHTS.get(t, 1.0) for t in groups])[0]
    return rng.choice(groups[pol])


def build_event_chain(rng: random.Random, scene: dict,
                      max_events: int = 3) -> List[dict]:
    """场景模板 → 事件序列 (保持 cause→effect→... 角色顺序)"""
    slots = scene["events"]
    # 槽位选择: 前 2 个 (cause/effect) 必选, 之后按概率附加 (保持顺序)
    n_min = min(2, len(slots))
    n_max = min(len(slots), max_events)
    k = rng.randint(n_min, n_max)
    chosen = slots[:k]
    events = []
    for slot in chosen:
        cand = rng.choice(slot["candidates"])
        events.append({
            "step_offset": 0,  # 由下方统一分配
            "event_type": cand["type"],
            "intensity": pick_intensity(rng, cand["intensity_range"]),
            "role": slot["role"],
        })
    # offset 随机升序分配 (保持因果顺序)
    offsets = sorted(rng.sample(OFFSET_POOL, k))
    for ev, off in zip(events, offsets):
        ev["step_offset"] = off
    return events


def generate_sample(rng: random.Random, scene: dict, simulator: ConcentrationSimulator,
                    baseline: List[float], sample_id: int, max_events: int = 3) -> dict:
    """单个样本: 场景 → 事件链 → 模拟器目标 → 输出 (与 curriculum_loader.cpp 格式一致)"""
    events = build_event_chain(rng, scene, max_events)
    simulator.reset()
    for rel in (0, 100, 200, 300):
        block = [(e["event_type"], e["intensity"]) for e in events if e["step_offset"] == rel]
        simulator.advance_block(block, baseline)
    mod_final = [max(0.0, min(2.0, v)) for v in simulator.conc]
    return {
        "sample_id": sample_id,
        "events": [{k: e[k] for k in ("step_offset", "event_type", "intensity")}
                   for e in events],
        "target_modulators": [round(v, 4) for v in mod_final],
        "target_pad": target_pad_from_conc(mod_final),
        "target_tool": scene["tool"],
        "chain": scene["scene_id"],
        "scene_desc": scene["desc"],
    }


def generate_samples(scenes: List[dict], event_kb: EventKB, stage_baseline: dict,
                     n_samples: int, seed: int, max_events: int = 3,
                     llm_chain_fn=None, use_llm_frac: float = 0.0) -> List[dict]:
    """批量生成样本。

    llm_chain_fn: 可选, callable(scene) -> List[dict] 事件链 (MiniCPM 生成);
                  返回 None 或非法链时回退到规则引擎。
    use_llm_frac: LLM 生成占比 (0-1), 剩余走规则引擎。
    """
    rng = random.Random(seed)
    stage = scenes[0]["stage"] if scenes else "middle_school"
    baseline = stage_baseline[stage]
    sim = ConcentrationSimulator(event_kb)

    samples = []
    for i in range(n_samples):
        scene = sample_scene(rng, scenes)
        events = None
        if llm_chain_fn is not None and rng.random() < use_llm_frac:
            events = llm_chain_fn(scene)
        if not events:
            events = build_event_chain(rng, scene, max_events)
        sim.reset()
        for rel in (0, 100, 200, 300):
            block = [(e["event_type"], e["intensity"]) for e in events if e["step_offset"] == rel]
            sim.advance_block(block, baseline)
        mod_final = [max(0.0, min(2.0, v)) for v in sim.conc]
        samples.append({
            "sample_id": i + 1,
            "events": [{k: e[k] for k in ("step_offset", "event_type", "intensity")}
                       for e in events],
            "target_modulators": [round(v, 4) for v in mod_final],
            "target_pad": target_pad_from_conc(mod_final),
            "target_tool": scene["tool"],
            "chain": scene["scene_id"],
            "scene_desc": scene["desc"],
        })
    return samples
