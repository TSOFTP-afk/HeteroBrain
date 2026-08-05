"""输出校验与统计: 数据契约 + 多样性报告。"""
import collections
from typing import List


def validate_samples(samples: List[dict], valid_events: set,
                     max_offset: int = 300) -> dict:
    """校验样本集合, 返回 {ok, errors[], stats{}}"""
    errors = []
    offs = set()
    pol = collections.Counter()
    for s in samples:
        evs = s["events"]
        if not (2 <= len(evs) <= 4):
            errors.append(f"sample {s['sample_id']}: 事件数 {len(evs)} 不在 [2,4]")
        prev = -1
        for e in evs:
            if e["event_type"] not in valid_events:
                errors.append(f"sample {s['sample_id']}: 未知事件类型 {e['event_type']}")
            if e["step_offset"] % 100 != 0 or e["step_offset"] > max_offset:
                errors.append(f"sample {s['sample_id']}: offset {e['step_offset']} 违约")
            if e["step_offset"] <= prev:
                errors.append(f"sample {s['sample_id']}: offset 非严格递增")
            prev = e["step_offset"]
            offs.add(e["step_offset"])
        ints = [e["intensity"] for e in evs]
        has_neg = any(i < 0 for i in ints)
        has_pos = any(i > 0 for i in ints)
        pol["mixed" if (has_neg and has_pos) else ("neg" if has_neg else "pos")] += 1
        for tgt in ("target_modulators", "target_pad"):
            for v in s[tgt]:
                if v != v:  # NaN
                    errors.append(f"sample {s['sample_id']}: {tgt} NaN")
    # 多样性
    seq_keys = set()
    tgt_keys = set()
    for s in samples:
        seq_keys.add(tuple((e["event_type"], e["intensity"], e["step_offset"])
                           for e in s["events"]))
        tgt_keys.add(tuple(s["target_modulators"]) + tuple(s["target_pad"]))
    stats = {
        "n_samples": len(samples),
        "unique_sequences": len(seq_keys),
        "unique_targets": len(tgt_keys),
        "offsets": sorted(offs),
        "polarity": dict(pol),
        "negative_containing_frac": round(
            100.0 * (pol["neg"] + pol["mixed"]) / len(samples), 1) if samples else 0.0,
    }
    return {"ok": not errors, "errors": errors[:20], "stats": stats}


def print_report(report: dict):
    st = report["stats"]
    print(f"  样本数: {st['n_samples']}")
    print(f"  唯一事件序列: {st['unique_sequences']} / {st['n_samples']}")
    print(f"  唯一目标向量: {st['unique_targets']} / {st['n_samples']}")
    print(f"  offset 集合: {st['offsets']}")
    print(f"  极性分布: {st['polarity']}  (负性/混合占比 {st['negative_containing_frac']}%)")
    if report["ok"]:
        print("  契约校验: OK")
    else:
        print(f"  契约校验: {len(report['errors'])} 处错误 (前 20):")
        for e in report["errors"]:
            print(f"    - {e}")
