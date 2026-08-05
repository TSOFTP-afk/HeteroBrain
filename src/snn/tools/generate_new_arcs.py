#!/usr/bin/env python3
"""生成 3 条新增故事线 jsonl (值日冲突/奥赛集训/家庭风波)
============================================================
每条线独立连续模拟 (arc 边界冷启动, 段间不 reset), 与 transfer_semester 同格式。
输出: data/events/curriculum_new_arcs.jsonl (仅新线)
"""

import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_serial_curriculum import simulate_segment, BASELINE
from longarc_stories import NEW_ARCS
from generate_curriculum_data import (
    ConcentrationSimulator, target_pad_from_conc, clamp_mod,
)


def generate_arc(arc):
    sim = ConcentrationSimulator()
    samples, prev = [], None
    for seg_idx, seg in enumerate(arc["segments"]):
        conc = simulate_segment(sim, seg["events"])
        pad = target_pad_from_conc(conc)
        causal_links = []
        if prev is not None:
            causal_links.append({"prev_segment": prev,
                                 "relation": "跨窗口状态延续"})
        samples.append({
            "sample_id": seg_idx + 1,
            "arc_id": arc["arc_id"],
            "segment_idx": seg_idx,
            "prev_segment": prev,
            "time_label": seg["time_label"],
            "causal_links": causal_links,
            "events": seg["events"],
            "target_modulators": [round(v, 4) for v in conc],
            "target_pad": pad,
            "target_tool": 6,
        })
        prev = seg_idx
    return samples


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                            "events", "curriculum_new_arcs.jsonl")
    all_samples, sample_id = [], 1
    for arc in NEW_ARCS:
        for s in generate_arc(arc):
            s["sample_id"] = sample_id
            all_samples.append(s)
            sample_id += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计: 正负平衡
    from collections import Counter
    etype = Counter()
    neg_types = {"criticism", "threat_social", "social_loss", "threat_physical"}
    pos_types = {"praise", "achievement", "social_bond", "food_tasty", "novelty"}
    neg = pos = 0
    for s in all_samples:
        for e in s["events"]:
            etype[e["event_type"]] += 1
            if e["event_type"] in neg_types:
                neg += 1
            elif e["event_type"] in pos_types:
                pos += 1

    print(f"已保存 {out_path}: {len(all_samples)} 段 / {len(NEW_ARCS)} 条线")
    print(f"事件类型: {dict(etype)}")
    print(f"正负比: 正 {pos} / 负 {neg} = {pos/neg:.2f}:1")
    # 与旧长线对比
    old = [json.loads(l) for l in open(
        os.path.join(os.path.dirname(out_path), "curriculum_long_arc_50.jsonl"),
        encoding="utf-8")]
    o_neg = o_pos = 0
    for s in old:
        for e in s["events"]:
            if e["event_type"] in neg_types:
                o_neg += 1
            elif e["event_type"] in pos_types:
                o_pos += 1
    print(f"原长线(53段)正负比: {o_pos/o_neg:.2f}:1 → 合并后 "
          f"{(o_pos+pos)/(o_neg+neg):.2f}:1")


if __name__ == "__main__":
    main()
