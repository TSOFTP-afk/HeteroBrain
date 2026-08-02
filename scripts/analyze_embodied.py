import re, statistics

path = r'C:\Users\26455\AppData\Local\Temp\trae-agent-toolhost\jobs\job-5f3fb1faa7de490db3cdc24940ac984b\output.log'
rows = []
pat = re.compile(r'EMBODIED step=(\d+)\] hunger=([\d.-]+) temp=([\d.-]+) comfort=([\d.-]+) fatigue=([\d.-]+) aro=([\d.-]+) \| cry=([\d.-]+) gaze=([\d.-]+) appr=([\d.-]+) avd=([\d.-]+) intr=([\d.-]+) \| threat=([\d.-]+) novel=([\d.-]+) reward=([\d.-]+)')
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    for ln in f:
        m = pat.search(ln)
        if m:
            rows.append([float(x) if i>0 else int(x) for i,x in enumerate(m.groups())])

print('EMBODIED samples:', len(rows))
if not rows:
    raise SystemExit

bins = [
    ('0-1K  ', [r for r in rows if r[0] < 1000]),
    ('1-2K  ', [r for r in rows if 1000 <= r[0] < 2000]),
    ('2-3K  ', [r for r in rows if r[0] >= 2000]),
]
print(f'{"段":8s} {"cry":>6s} {"gaze":>6s} {"appr":>6s} {"avd":>6s} {"intr":>6s} {"reward":>8s}')
for label, rs in bins:
    if not rs: continue
    def mean(i): return statistics.mean(r[i] for r in rs)
    print(f'{label:8s} {mean(6):6.3f} {mean(7):6.3f} {mean(8):6.3f} {mean(9):6.3f} {mean(10):6.3f} {mean(13):8.4f}')

diffs = []
for r in rows:
    acts = r[6:11]
    diffs.append(max(acts) - min(acts))
print('\n行为分化度 (max-min): mean=%.4f max=%.4f' % (statistics.mean(diffs), max(diffs)))

# 饥饿→cry 关联 (应学到: 饿→cry)
for thr in [0.5, 0.3, 0.1]:
    sub = [c for h, c in [(r[1], r[6]) for r in rows] if h > thr]
    if sub:
        print(f'hunger>{thr}: n={len(sub)} cry均值={statistics.mean(sub):.3f}')
sub = [c for h, c in [(r[1], r[6]) for r in rows] if h <= 0.1]
print(f'hunger<=0.1: n={len(sub)} cry均值={statistics.mean(sub):.3f}' if sub else '')
