# -*- coding: utf-8 -*-
"""
验证: 事件类型组合(出现集合)是否唯一确定 chain → 决定分类器特征设计
"""
import json
import collections

DATA = r"f:\thetrueai\data\events\curriculum_middle_school.jsonl"

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]

# chain -> 事件类型集合 -> 工具
sig = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for r in rows:
    et = tuple(sorted({e["event_type"] for e in r["events"]}))
    sig[r["chain"]][et][r["target_tool"]] += 1

print(f"{'chain':<32}{'事件类型集合':<70}{'工具'}")
print("-" * 120)
for ch in sorted(sig):
    for et, tc in sorted(sig[ch].items()):
        print(f"{ch:<32}{str(et):<70}{dict(tc)}")

# 集合唯一性: 不同 chain 是否共享相同事件类型集合
sig_by_et = collections.defaultdict(list)
for ch, m in sig.items():
    for et in m:
        sig_by_et[et].append(ch)
print("\n=== 事件类型集合 → chain 唯一性 ===")
uniq = 0
for et, chs in sorted(sig_by_et.items()):
    tag = "唯一" if len(chs) == 1 else f"冲突({len(chs)}个chain)"
    if len(chs) == 1:
        uniq += 1
    print(f"  {str(et):<70} → {chs}  {tag}")
print(f"唯一集合: {uniq}/{len(sig_by_et)}")
