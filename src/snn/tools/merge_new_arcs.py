#!/usr/bin/env python3
"""合并训练库: 53 段长线 + 151 段新线 → 统一 jsonl + 净化文本流
============================================================
输入:
  data/events/curriculum_long_arc_50.jsonl  (转学适应线 53 段)
  data/events/curriculum_new_arcs.jsonl     (值日/奥赛/家庭 151 段)
输出:
  data/events/curriculum_all.jsonl          (合并 204 段, 训练用)
  data/scripts/story_text_all.txt           (合并净化文本流, SNN 输入用)
用法:
  python merge_new_arcs.py
"""

import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_serial_curriculum import sanitize_first_person, LONG_ARC
from longarc_stories import NEW_ARCS

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


def load_jsonl(rel):
    p = os.path.join(DATA, "events", rel)
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    old = load_jsonl("curriculum_long_arc_50.jsonl")
    new = load_jsonl("curriculum_new_arcs.jsonl")

    # ---- 1. 合并 jsonl: 全局 sample_id 重排 ----
    merged = []
    for s in old + new:
        s = dict(s)
        s["sample_id"] = len(merged) + 1
        merged.append(s)
    out_json = os.path.join(DATA, "events", "curriculum_all.jsonl")
    with open(out_json, "w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"合并 jsonl → {out_json}: {len(merged)} 段 "
          f"(旧 {len(old)} + 新 {len(new)})")

    # ---- 2. 合并净化文本流 (长线在前, 新线按序在后) ----
    def text_lines(arcs_with_segs):
        lines, n_rewrite = [], 0
        for arc, samples in arcs_with_segs:
            for s in samples:
                seg = arc["segments"][s["segment_idx"]]
                pure = sanitize_first_person(seg["snn_view"])
                if pure != seg["snn_view"]:
                    n_rewrite += 1
                lines.append(f"[{seg['time_label']}]")
                lines.append(pure)
                lines.append("")
        return lines, n_rewrite

    arcs_with_segs = [(LONG_ARC, old)]
    for arc in NEW_ARCS:
        arcs_with_segs.append((arc, [s for s in new if s["arc_id"] == arc["arc_id"]]))
    lines, n_rewrite = text_lines(arcs_with_segs)

    out_txt = os.path.join(DATA, "scripts", "story_text_all.txt")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    n_bytes = os.path.getsize(out_txt)
    n_chars = sum(len(l) for l in lines)
    print(f"净化文本流 → {out_txt} (重写 {n_rewrite}/{len(merged)} 段)")
    print(f"  {n_chars} 字符 / {n_bytes} 字节 (约 {len(merged)} 段 × 400 步 = "
          f"{len(merged)*400} 步, 消耗 {len(merged)*400/20*1000} 字节 → "
          f"覆盖 {n_bytes/(len(merged)*400/20*1000)*100:.0f}%)")

    # ---- 3. 统计 ----
    from collections import Counter
    neg_types = {"criticism", "threat_social", "social_loss", "threat_physical"}
    pos_types = {"praise", "achievement", "social_bond", "food_tasty", "novelty"}
    etype, neg, pos = Counter(), 0, 0
    for s in merged:
        for e in s["events"]:
            etype[e["event_type"]] += 1
            neg += e["event_type"] in neg_types
            pos += e["event_type"] in pos_types
    print(f"合并后事件: {dict(etype)}")
    print(f"合并后正负比: 正 {pos} / 负 {neg} = {pos/neg:.2f}:1")


if __name__ == "__main__":
    main()
