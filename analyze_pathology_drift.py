"""分析正常场景的调质数值是否落入病理区域 (病理滑移风险)."""
import csv
import numpy as np
from collections import defaultdict

# 病理类场景名单
PATHOLOGY_SCENARIOS = {
    "addiction_cocaine", "addiction_withdrawal", "depression_helplessness",
    "fear_conditioning", "fear_extinction", "meditation_compassion",
    "schizophrenia_positive", "schizophrenia_negative",
    "bipolar_mania", "bipolar_depression",
    "anxiety_disorder", "ocd_checking",
    "adhd_inattentive", "autism_social_deficit",
    "alcohol_intoxication", "alcohol_withdrawal",
}

with open("data/neuroscience_traces/validation_report_with_pathology.csv",
          encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# 分组: 正常 vs 病理
normal_rows = [r for r in rows if r["scenario"] not in PATHOLOGY_SCENARIOS]
pathology_rows = [r for r in rows if r["scenario"] in PATHOLOGY_SCENARIOS]

print("=" * 75)
print("病理滑移风险分析 (正常场景 vs 病理场景调质数值分布对比)")
print("=" * 75)
print(f"  正常场景样本: {len(normal_rows)} (59 场景 × 21 强度)")
print(f"  病理场景样本: {len(pathology_rows)} (16 场景 × 3 强度)")
print()

# ===== 1. 定义病理阈值 (基于病理场景的 P5/P95) =====
print("--- 1. 病理区域阈值定义 (基于病理场景分布) ---")
print()
print(f"{'调质':6s} {'病理min':>10s} {'病理P5':>10s} {'病理P25':>10s} {'病理P50':>10s} {'病理P75':>10s} {'病理P95':>10s} {'病理max':>10s}")
print("-" * 78)

pathology_thresholds = {}
for ch in ["DA", "5HT", "NE", "ACh", "GABA", "Oxy"]:
    # 用 mean 浓度做对比 (反映持续水平而非瞬时峰值)
    key = f"conc_{ch}_mean"
    path_vals = [float(r[key]) for r in pathology_rows if key in r and r[key]]
    norm_vals = [float(r[key]) for r in normal_rows if key in r and r[key]]
    if not path_vals:
        continue
    path_arr = np.array(path_vals)
    norm_arr = np.array(norm_vals)
    p5 = np.percentile(path_arr, 5)
    p25 = np.percentile(path_arr, 25)
    p50 = np.percentile(path_arr, 50)
    p75 = np.percentile(path_arr, 75)
    p95 = np.percentile(path_arr, 95)
    pathology_thresholds[ch] = {
        "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        "min": float(path_arr.min()), "max": float(path_arr.max()),
        "norm_p50": float(np.percentile(norm_arr, 50)),
        "norm_p95": float(np.percentile(norm_arr, 95)),
    }
    print(f"{ch:6s} {path_arr.min():10.3f} {p5:10.3f} {p25:10.3f} {p50:10.3f} {p75:10.3f} {p95:10.3f} {path_arr.max():10.3f}")

print()
print("--- 2. 正常场景均值浓度分布 ---")
print()
print(f"{'调质':6s} {'正常min':>10s} {'正常P5':>10s} {'正常P25':>10s} {'正常P50':>10s} {'正常P75':>10s} {'正常P95':>10s} {'正常max':>10s}")
print("-" * 78)
for ch in ["DA", "5HT", "NE", "ACh", "GABA", "Oxy"]:
    key = f"conc_{ch}_mean"
    vals = [float(r[key]) for r in normal_rows if key in r and r[key]]
    if not vals:
        continue
    arr = np.array(vals)
    print(f"{ch:6s} {arr.min():10.3f} {np.percentile(arr,5):10.3f} {np.percentile(arr,25):10.3f} {np.percentile(arr,50):10.3f} {np.percentile(arr,75):10.3f} {np.percentile(arr,95):10.3f} {arr.max():10.3f}")

# ===== 3. 病理滑移风险检测 =====
print()
print("=" * 75)
print("--- 3. 病理滑移风险检测 (正常场景均值落入病理 P25-P75 区间) ---")
print("=" * 75)
print()
print("判定规则:")
print("  - 高风险: 正常场景 mean > 病理 P75 (持续偏高)")
print("  - 中风险: 正常场景 mean > 病理 P50 (高于病理中位数)")
print("  - 低风险: 正常场景 mean > 病理 P25 (接近病理区)")
print("  - 安全:   正常场景 mean < 病理 P25")
print()

risk_cases = []
for ch in ["DA", "5HT", "NE", "ACh", "GABA", "Oxy"]:
    key = f"conc_{ch}_mean"
    if ch not in pathology_thresholds:
        continue
    thr = pathology_thresholds[ch]
    print(f"  [{ch}] 病理 P25={thr['p25']:.3f}  P50={thr['p50']:.3f}  P75={thr['p75']:.3f}  正常 P95={thr['norm_p95']:.3f}")

    high_risk = []
    mid_risk = []
    low_risk = []
    for r in normal_rows:
        if key not in r or not r[key]:
            continue
        val = float(r[key])
        if val > thr["p75"]:
            high_risk.append((r["scenario"], r["intensity_str"], val))
        elif val > thr["p50"]:
            mid_risk.append((r["scenario"], r["intensity_str"], val))
        elif val > thr["p25"]:
            low_risk.append((r["scenario"], r["intensity_str"], val))

    print(f"       高风险: {len(high_risk):3d} 个子数据集 (mean > 病理P75)")
    print(f"       中风险: {len(mid_risk):3d} 个子数据集 (mean > 病理P50)")
    print(f"       低风险: {len(low_risk):3d} 个子数据集 (mean > 病理P25)")
    if high_risk:
        print(f"       高风险案例 (前 5):")
        for sc, i, v in high_risk[:5]:
            print(f"         {sc}/{i}: {ch} mean={v:.3f} (>P75={thr['p75']:.3f})")
        for case in high_risk:
            risk_cases.append((ch, "高", *case))
    print()

# ===== 4. 总结 =====
print("=" * 75)
print("--- 4. 总结 ---")
print("=" * 75)
print()
print(f"  总风险案例数: {len(risk_cases)}")
if risk_cases:
    print(f"  其中高风险: {sum(1 for r in risk_cases if r[1]=='高')}")
    print()
    print("  按场景统计风险案例:")
    sc_count = defaultdict(int)
    for ch, level, sc, i, v in risk_cases:
        sc_count[sc] += 1
    for sc, cnt in sorted(sc_count.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sc:35s}: {cnt} 个调质超标")
else:
    print("  无风险案例")
print()
print("  说明:")
print("  - 'mean > 病理 P75' 意味着该正常场景的调质平均浓度")
print("    超过了 75% 的病理场景, 存在滑向病理状态的风险")
print("  - 需要检查这些场景的动力学是否会在更长时窗下漂移")
