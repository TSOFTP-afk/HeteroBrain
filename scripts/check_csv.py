import statistics

path = r'F:\thetrueai\run_curriculum_bptt_scratch_20k.csv'
rows = []
cols = None
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith('step,'))
cols = lines[hdr_idx].strip().split(',')
for ln in lines[hdr_idx+1:]:
    if not ln.strip():
        continue
    parts = ln.strip().split(',')
    if not parts[0].strip().isdigit():
        continue
    rows.append(dict(zip(cols, parts)))

print('rows:', len(rows))
state_cols = ['da','ach','ne','ht5','gaba','oxy','pleasure','arousal','dominance','temp_delta','empathy']
def by_step(r): return int(r['step'])
rows_sorted = sorted(rows, key=by_step)

def seg(rs, label):
    print(f'[{label}] n={len(rs)}')
    for c in state_cols:
        vals = [float(r[c]) for r in rs if r.get(c) not in (None, '')]
        if not vals: continue
        m = statistics.mean(vals)
        print(f'  {c:10s} mean={m:8.4f}', end='')
    print()

seg([r for r in rows_sorted if int(r['step']) < 5000], '0-5K ')
seg([r for r in rows_sorted if 15000 <= int(r['step']) < 20000], '15-20K')

print('\n--- 权重/脉冲 (BPTT 无重建, 权重仅 BPTT 反传修改) ---')
for label, lo, hi in [('0-1K',0,1000),('9-10K',9000,10000),('19-20K',19000,20000)]:
    rs = [r for r in rows_sorted if lo <= int(r['step']) < hi]
    if not rs: continue
    wm  = statistics.mean([float(r['weight_mean']) for r in rs])
    wmm = statistics.mean([float(r['weight_abs_mean']) for r in rs])
    sp  = statistics.mean([float(r['spikes']) for r in rs])
    print(f'[{label}] weight_mean={wm:.5f} w_abs={wmm:.5f} spikes/step={sp:.0f}')
