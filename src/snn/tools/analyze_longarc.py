#!/usr/bin/env python3
"""分析长线训练数据与 CSV 趋势"""
import csv, json, statistics, sys

# ============ 1. CSV 趋势 ============
raw = open('build/snn/longarc_10k.csv', encoding='utf-8-sig').readlines()
data = [l for l in raw if not l.startswith('#')]
rows = list(csv.DictReader(data))
print(f'CSV 行数: {len(rows)} (step 0-{rows[-1]["step"]})')

def f(r, k): return float(r[k])

print('\n=== 5000 步窗口聚合 (浓度/PAD 均值) ===')
print(f'{"step":>6} {"DA":>6} {"Oxy":>6} {"5HT":>6} {"P":>7} {"A":>7} {"D":>7}')
for s in range(0, 50001, 5000):
    seg = [r for r in rows if s <= f(r, 'step') < s + 5000]
    if not seg:
        continue
    avg = {k: statistics.mean(f(r, k) for r in seg)
           for k in ['da', 'oxy', 'ht5', 'pleasure', 'arousal', 'dominance']}
    print(f'{s:>6} {avg["da"]:6.3f} {avg["oxy"]:6.3f} {avg["ht5"]:6.3f} '
          f'{avg["pleasure"]:7.3f} {avg["arousal"]:7.3f} {avg["dominance"]:7.3f}')

# ============ 2. 长线数据统计 ============
samples = [json.loads(l) for l in open('data/events/curriculum_long_arc_50.jsonl', encoding='utf-8')]
print(f'\n=== 长线数据 (curriculum_long_arc_50.jsonl) ===')
print(f'样本数: {len(samples)}')

from collections import Counter
etype_cnt = Counter()
par_cnt = 0
intens_hist = Counter()
for s in samples:
    offs = [e['step_offset'] for e in s['events']]
    if len(offs) != len(set(offs)):
        par_cnt += 1
    for e in s['events']:
        etype_cnt[e['event_type']] += 1
        intens_hist[e['intensity']] += 1
print(f'事件总数: {sum(etype_cnt.values())}, 含并行事件段: {par_cnt}/{len(samples)}')
print(f'事件类型分布: {dict(etype_cnt)}')
print(f'强度值分布: {dict(sorted(intens_hist.items()))}')

# 目标调质范围
mods = [s['target_modulators'] for s in samples]
print(f'\n目标调质范围 (每通道 min-max):')
for i, name in enumerate(['DA', 'ACh', 'NE', '5HT', 'GABA', 'Oxy']):
    vals = [m[i] for m in mods]
    print(f'  {name}: {min(vals):.3f} ~ {max(vals):.3f}')

# ============ 3. 与旧数据对比 ============
old = [json.loads(l) for l in open('data/events/curriculum_middle_school.jsonl', encoding='utf-8')]
print(f'\n=== 对比: 旧课程数据 (curriculum_middle_school.jsonl) ===')
print(f'样本数: {len(old)}')
old_events = sum(len(s['events']) for s in old)
new_events = sum(len(s['events']) for s in samples)
print(f'每样本事件数: 旧 {old_events/len(old):.2f} vs 长线 {new_events/len(samples):.2f}')
old_mods = [s['target_modulators'] for s in old]
print(f'旧目标调质范围 (每通道 min-max):')
for i, name in enumerate(['DA', 'ACh', 'NE', '5HT', 'GABA', 'Oxy']):
    vals = [m[i] for m in old_mods]
    print(f'  {name}: {min(vals):.3f} ~ {max(vals):.3f}')

# 弧线多样性: PAD P 极差
def pad_p_range(mods_):
    ps = [m[2] * 0 - 1 for m in mods_[:1]]  # placeholder
    return None
new_p = [s['target_pad'][0] for s in samples]
old_p = [s['target_pad'][0] for s in old]
print(f'\nPAD P 范围: 长线 {min(new_p):.3f}~{max(new_p):.3f} (跨度 {max(new_p)-min(new_p):.3f})')
print(f'PAD P 范围: 旧    {min(old_p):.3f}~{max(old_p):.3f} (跨度 {max(old_p)-min(old_p):.3f})')
