"""精确病理滑移风险分析 (按病理类型分组 + 恢复能力检测).

核心问题:
  1. 病理场景包含方向相反的调质变化 (depression=低DA, addiction=高DA)
     混在一起做 P75 会导致阈值失真
  2. 真正的"病理滑移"不是"平均浓度高", 而是:
        a. 刺激结束后无法恢复到基线 (恢复能力失效)
        b. 调质模式持续异常 (而非瞬态峰值)

本脚本改进:
  1. 按"病理方向"分组定义阈值
  2. 检查每个正常场景的"恢复期"调质是否回到基线 ±20%
"""
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, "src/snn/tools")
from affective_dynamics_sim import (
    signals_from_traces, run_dynamics, compute_affective_state,
    CHANNEL_ORDER,
)


# ===== 1. 病理方向分组 (基于神经科学) =====
PATHOLOGY_GROUPS = {
    "高DA类": {
        "scenarios": ["addiction_cocaine", "schizophrenia_positive", "bipolar_mania"],
        "channels": ["DA"],
        "direction": "high",
        "description": "DA 过度活跃 (成瘾/精分阳性/躁狂)",
    },
    "低DA类": {
        "scenarios": ["depression_helplessness", "schizophrenia_negative",
                       "adhd_inattentive", "addiction_withdrawal"],
        "channels": ["DA"],
        "direction": "low",
        "description": "DA 不足 (抑郁/精分阴性/ADHD/戒断)",
    },
    "高5HT类": {
        "scenarios": ["anxiety_disorder", "ocd_checking", "fear_conditioning"],
        "channels": ["5HT"],
        "direction": "high",
        "description": "5HT 失调升高 (焦虑/强迫/恐惧)",
    },
    "低Oxy类": {
        "scenarios": ["autism_social_deficit", "alcohol_withdrawal"],
        "channels": ["Oxy"],
        "direction": "low",
        "description": "Oxy 不足 (自闭/酒精戒断)",
    },
    "高NE类": {
        "scenarios": ["anxiety_disorder", "ptsd_reexperiencing" if False else "alcohol_withdrawal"],
        "channels": ["NE"],
        "direction": "high",
        "description": "NE 过度活跃 (焦虑/戒断)",
    },
}

NORMAL_SCENARIOS_EXCLUDED = set()
for g in PATHOLOGY_GROUPS.values():
    NORMAL_SCENARIOS_EXCLUDED.update(g["scenarios"])


def load_npz_traces(npz_path: Path) -> dict:
    """加载 NPZ 并适配 key."""
    data = np.load(npz_path, allow_pickle=True)
    traces = {}
    key_map = {"DA": "da_trace", "5HT": "ht5_trace", "NE": "ne_trace",
               "ACh": "ach_trace", "Oxy": "oxy_trace"}
    for new_key, old_key in key_map.items():
        if new_key in data.files:
            traces[old_key] = data[new_key]
    return traces


def compute_recovery_ratio(npz_path: Path, sim_dt_ms: float = 1.0) -> dict:
    """计算恢复期/基线期 的浓度比值.

    恢复能力指标:
      - ratio = recovery_mean / baseline_mean
      - ratio ≈ 1.0: 完美恢复
      - ratio > 1.5: 恢复不全 (病理滑移风险)
      - ratio < 0.5: 过度抑制

    基线期: 前 20% 时长
    恢复期: 后 20% 时长
    """
    traces = load_npz_traces(npz_path)
    if len(traces) < 4:
        return {}

    first_trace = next(iter(traces.values()))
    n_data = len(first_trace)
    n_steps = n_data * 10  # 100Hz → 1000Hz
    t_grid_ms = np.arange(n_steps, dtype=np.float64) * sim_dt_ms

    signals = signals_from_traces(traces, t_grid_ms,
                                   fs_data_hz=100.0, sim_dt_ms=sim_dt_ms)
    conc = run_dynamics(signals, sim_dt_ms)

    n = n_steps
    baseline_end = int(n * 0.2)
    recovery_start = int(n * 0.8)

    result = {}
    for ch in CHANNEL_ORDER:
        if ch not in conc:
            continue
        c = conc[ch]
        baseline_mean = float(c[:baseline_end].mean())
        recovery_mean = float(c[recovery_start:].mean())
        if baseline_mean > 1e-6:
            ratio = recovery_mean / baseline_mean
        else:
            ratio = 0.0 if recovery_mean < 1e-6 else 99.0
        result[ch] = {
            "baseline": baseline_mean,
            "recovery": recovery_mean,
            "ratio": ratio,
        }
    return result


def main():
    # 加载验证报告
    with open("data/neuroscience_traces/validation_report_with_pathology.csv",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    normal_rows = [r for r in rows if r["scenario"] not in NORMAL_SCENARIOS_EXCLUDED]

    print("=" * 78)
    print("精确病理滑移风险分析 v2 (按病理方向分组)")
    print("=" * 78)
    print()

    # ===== 1. 按病理方向定义阈值 =====
    print("--- 1. 按病理方向分组的阈值定义 ---")
    print()
    print(f"{'病理组':12s} {'调质':6s} {'方向':6s} {'正常P50':>10s} {'正常P95':>10s} {'病理P50':>10s} {'病理P95':>10s} {'病理max':>10s}")
    print("-" * 78)

    group_thresholds = {}
    for group_name, group_info in PATHOLOGY_GROUPS.items():
        ch = group_info["channels"][0]
        direction = group_info["direction"]
        key = f"conc_{ch}_mean"

        path_vals = [float(r[key]) for r in rows
                      if r["scenario"] in group_info["scenarios"] and key in r and r[key]]
        norm_vals = [float(r[key]) for r in normal_rows
                      if key in r and r[key]]

        if not path_vals:
            continue

        path_arr = np.array(path_vals)
        norm_arr = np.array(norm_vals)

        thr = {
            "channel": ch,
            "direction": direction,
            "path_P50": float(np.percentile(path_arr, 50)),
            "path_P95": float(np.percentile(path_arr, 95)),
            "path_max": float(path_arr.max()),
            "norm_P50": float(np.percentile(norm_arr, 50)),
            "norm_P95": float(np.percentile(norm_arr, 95)),
        }
        group_thresholds[group_name] = thr
        print(f"{group_name:12s} {ch:6s} {direction:6s} {thr['norm_P50']:10.3f} {thr['norm_P95']:10.3f} {thr['path_P50']:10.3f} {thr['path_P95']:10.3f} {thr['path_max']:10.3f}")

    # ===== 2. 滑移风险检测 (方向感知) =====
    print()
    print("=" * 78)
    print("--- 2. 方向感知的滑移风险检测 ---")
    print("=" * 78)
    print()
    print("判定规则 (方向感知):")
    print("  高风险 (HIGH): 正常场景 mean 超过病理 P95 且方向一致")
    print("  中风险 (MED):  正常场景 mean 超过病理 P50 且方向一致")
    print("  安全 (SAFE):   未超过病理 P50 或方向相反")
    print()

    total_high = 0
    total_med = 0
    risk_by_scenario = defaultdict(list)

    for group_name, thr in group_thresholds.items():
        ch = thr["channel"]
        direction = thr["direction"]
        key = f"conc_{ch}_mean"

        print(f"  [{group_name}] ({ch} {direction})")
        print(f"    病理 P50={thr['path_P50']:.3f}  P95={thr['path_P95']:.3f}  正常 P95={thr['norm_P95']:.3f}")

        high_cases = []
        med_cases = []
        for r in normal_rows:
            if key not in r or not r[key]:
                continue
            val = float(r[key])

            if direction == "high":
                if val > thr["path_P95"]:
                    high_cases.append((r["scenario"], r["intensity_str"], val))
                elif val > thr["path_P50"]:
                    med_cases.append((r["scenario"], r["intensity_str"], val))
            else:  # low
                if val < thr["path_P50"] * 0.5:  # 低于病理中位数的一半
                    high_cases.append((r["scenario"], r["intensity_str"], val))
                elif val < thr["path_P50"] * 0.8:
                    med_cases.append((r["scenario"], r["intensity_str"], val))

        print(f"    高风险: {len(high_cases):3d} 个子数据集")
        print(f"    中风险: {len(med_cases):3d} 个子数据集")
        if high_cases:
            print(f"    高风险案例 (前 5):")
            for sc, i, v in high_cases[:5]:
                print(f"      {sc}/{i}: {ch}={v:.3f} (>病理P95={thr['path_P95']:.3f})")
        total_high += len(high_cases)
        total_med += len(med_cases)
        for sc, i, v in high_cases:
            risk_by_scenario[sc].append((group_name, ch, v))
        print()

    # ===== 3. 恢复能力检测 (抽样) =====
    print("=" * 78)
    print("--- 3. 恢复能力检测 (抽样 20 个高风险场景) ---")
    print("=" * 78)
    print()
    print("指标: ratio = 恢复期均值 / 基线期均值")
    print("  ratio ≈ 1.0: 完美恢复")
    print("  ratio > 1.5: 恢复不全 (滑移风险)")
    print("  ratio < 0.5: 过度抑制")
    print()

    # 找出高风险场景做恢复能力检测
    high_risk_scenarios = sorted(risk_by_scenario.keys(),
                                  key=lambda s: -len(risk_by_scenario[s]))[:10]
    if high_risk_scenarios:
        print(f"  对 {len(high_risk_scenarios)} 个高风险场景抽样检测恢复能力:")
        print()
        print(f"  {'场景':30s} {'强度':8s} {'DA_ratio':>10s} {'5HT_ratio':>10s} {'NE_ratio':>10s} {'Oxy_ratio':>10s} {'判定':8s}")
        print("  " + "-" * 76)

        datasets_dir = Path("data/neuroscience_traces/datasets")
        recovery_fail_count = 0
        for sc in high_risk_scenarios:
            npz = datasets_dir / sc / "i+00" / "traces.npz"
            if not npz.exists():
                continue
            rec = compute_recovery_ratio(npz)
            if not rec:
                continue
            da_r = rec.get("DA", {}).get("ratio", 0)
            ht_r = rec.get("5HT", {}).get("ratio", 0)
            ne_r = rec.get("NE", {}).get("ratio", 0)
            oxy_r = rec.get("Oxy", {}).get("ratio", 0)

            # 判定: 任一调质 ratio > 1.5 或 < 0.5 视为恢复不全
            ratios = [da_r, ht_r, ne_r, oxy_r]
            fails = [r for r in ratios if r > 1.5 or r < 0.5]
            verdict = "恢复失败" if fails else "正常恢复"
            if fails:
                recovery_fail_count += 1
            print(f"  {sc:30s} {'i+00':8s} {da_r:10.2f} {ht_r:10.2f} {ne_r:10.2f} {oxy_r:10.2f} {verdict:8s}")

        print()
        print(f"  恢复失败场景数: {recovery_fail_count}/{len(high_risk_scenarios)}")

    # ===== 4. 总结 =====
    print()
    print("=" * 78)
    print("--- 4. 总结 ---")
    print("=" * 78)
    print()
    print(f"  方向感知高风险案例: {total_high}")
    print(f"  方向感知中风险案例: {total_med}")
    print()
    if total_high > 100:
        print("  ⚠️  发现大量高风险案例")
        print("     这表明部分正常场景的调质数值确实落入了病理区域")
        print("     建议:")
        print("     1. 检查这些场景的刺激参数 (peak_df/poisson_rate) 是否过高")
        print("     2. 在 CUDA SNN 中加入'病理状态检测器' (持续超阈值告警)")
        print("     3. 考虑为每个调质加入'稳态补偿机制' (受体上调/下调)")
    elif total_high > 0:
        print("  ⚠️  发现少量高风险案例")
        print("     可能是极端强度 (i+50) 导致, 建议降低 strong 强度的 peak_df 上限")
    else:
        print("  ✅ 无高风险案例, 正常场景不会滑向病理")


if __name__ == "__main__":
    main()
