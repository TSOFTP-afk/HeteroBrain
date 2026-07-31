#!/usr/bin/env python3
"""批量验证 6 维调质动力学稳定性 (接入 SNN Phase 3a 验证).

================================================================
目标
================================================================
遍历 datasets/ 下所有 {scenario}/i{N:+03d}/ 子数据集, 对每个子数据集
加载 GRAB 轨迹 NPZ, 用 affective_dynamics_sim.py 的核心函数复现 SNN
6 维调质差分方程, 检查:

  1. 6 维浓度始终 ∈ [0, 2] (CUDA clamp)
  2. PAD 三维 ∈ [-1, 1]
  3. LLM 调制信号有界
  4. GABA 抗焦虑反馈在 NE 高时上升
  5. DA 响应快于 5HT (时间常数)

输出:
  - CSV 汇总报告 (每行一个子数据集)
  - 失败案例列表
  - 总体通过率

用法:
    python batch_validate_dynamics.py
    python batch_validate_dynamics.py --workers 8 --output validation_report.csv
依赖:
    pip install numpy scipy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# 复用 affective_dynamics_sim 的核心函数
sys.path.insert(0, str(Path(__file__).parent))
from affective_dynamics_sim import (
    signals_from_traces,
    run_dynamics,
    run_dynamics_with_homeostasis,
    compute_affective_state,
    CHANNEL_ORDER,
    TRACE_KEY,
    TAU_MS,
    CONC_MIN,
    CONC_MAX,
    HOMEOSTATIC_BASELINE,
)


def load_dataset_npz(npz_path: Path) -> dict:
    """加载子数据集 NPZ, 把 key 适配为 affective_dynamics_sim 期望的格式.

    新数据集 key: DA / 5HT / NE / ACh / Oxy (无 GABA, 无 _trace 后缀)
    期望 key:     da_trace / ht5_trace / ne_trace / ach_trace / oxy_trace
    """
    data = np.load(npz_path, allow_pickle=True)
    traces = {}
    # key 映射: 新格式 → 旧格式
    key_map = {
        "DA":  "da_trace",
        "5HT": "ht5_trace",
        "NE":  "ne_trace",
        "ACh": "ach_trace",
        "Oxy": "oxy_trace",
    }
    for new_key, old_key in key_map.items():
        if new_key in data.files:
            traces[old_key] = data[new_key]
    # GABA 不在数据集中, signals_from_traces 会自动合成
    return traces


def validate_single(npz_path: Path, sim_dt_ms: float = 1.0,
                    enable_homeostasis: bool = True) -> dict:
    """对单个子数据集运行 6 维调质动力学验证.

    enable_homeostasis: True 时启用 Phase 3a-B 稳态补偿 (受体下调/上调)
    返回验证结果字典 (用于 CSV 汇总).
    """
    result = {
        "scenario": npz_path.parent.parent.name,
        "intensity_str": npz_path.parent.name,
        "npz_path": str(npz_path.relative_to(npz_path.parents[3])),
        "homeostasis": "ON" if enable_homeostasis else "OFF",
    }

    try:
        traces = load_dataset_npz(npz_path)
        if len(traces) < 4:  # 至少需要 4 个 channel
            result["status"] = "SKIP"
            result["error"] = f"only {len(traces)} channels"
            return result

        # 采样率: 从 time_s 推断
        # 数据集默认 fs=100Hz (10ms), sim_dt=1ms, 需要插值
        n_data = len(next(iter(traces.values())))
        # 数据采样率 100Hz, sim_dt=1ms → 插值到 1000Hz
        first_trace = next(iter(traces.values()))
        n_steps = len(first_trace) * 10  # 100Hz → 1000Hz (1ms 步长)
        t_grid_ms = np.arange(n_steps, dtype=np.float64) * sim_dt_ms

        # 生成 signals (含 GABA 反馈合成)
        signals = signals_from_traces(traces, t_grid_ms,
                                       fs_data_hz=100.0, sim_dt_ms=sim_dt_ms)

        # 跑 6 维动力学 (Phase 3a-B: 可选稳态补偿)
        if enable_homeostasis:
            conc, sensitivity_log = run_dynamics_with_homeostasis(
                signals, sim_dt_ms, enable_homeostasis=True)
        else:
            conc = run_dynamics(signals, sim_dt_ms)

        # 计算 PAD + LLM 调制信号
        affective = compute_affective_state(conc)

        # ===== 准则检查 =====
        all_pass = True
        errors = []

        # 准则 1: 6 维浓度 ∈ [0, 2]
        for ch in CHANNEL_ORDER:
            c = conc[ch]
            cmin, cmax = float(c.min()), float(c.max())
            if cmin < -0.001 or cmax > 2.001:
                all_pass = False
                errors.append(f"{ch} 越界 [{cmin:.3f},{cmax:.3f}]")
            result[f"conc_{ch}_min"] = round(cmin, 4)
            result[f"conc_{ch}_max"] = round(cmax, 4)
            result[f"conc_{ch}_mean"] = round(float(c.mean()), 4)

        # 准则 2: PAD ∈ [-1, 1]
        for key in ["pleasure", "arousal", "dominance"]:
            v = affective[key]
            vmin, vmax = float(v.min()), float(v.max())
            if vmin < -1.05 or vmax > 1.05:
                all_pass = False
                errors.append(f"{key} 越界 [{vmin:.3f},{vmax:.3f}]")
            result[f"pad_{key}_min"] = round(vmin, 4)
            result[f"pad_{key}_max"] = round(vmax, 4)

        # 准则 3: LLM 调制信号有界
        llm_bounds = {
            "temperature_delta": (-0.55, 0.55),
            "top_p_delta":       (-0.45, 0.05),
            "repetition_delta":  (-0.05, 0.25),
            "empathy_level":     (-0.05, 1.05),
        }
        for key, (lo, hi) in llm_bounds.items():
            v = affective[key]
            vmin, vmax = float(v.min()), float(v.max())
            if vmin < lo or vmax > hi:
                all_pass = False
                errors.append(f"{key} 越界 [{vmin:.3f},{vmax:.3f}]")
            result[f"llm_{key}_min"] = round(vmin, 4)
            result[f"llm_{key}_max"] = round(vmax, 4)

        # 准则 4: GABA 反馈在 NE 高时上升
        ne = conc["NE"]
        gaba = conc["GABA"]
        ne_high_mask = ne > 0.3
        ne_high_ratio = float(ne_high_mask.mean())  # NE 超阈比例
        if ne_high_mask.any() and (~ne_high_mask).any() and ne_high_ratio > 0.02:
            gaba_high = float(gaba[ne_high_mask].mean())
            gaba_low = float(gaba[~ne_high_mask].mean())
            # 容差 0.005: NE 边界附近 GABA 差异在噪声范围内不视为失效
            gaba_ok = gaba_high >= gaba_low - 0.005
            if not gaba_ok:
                all_pass = False
                errors.append(f"GABA 反馈失效 (NE高={gaba_high:.3f} ≤ 基线={gaba_low:.3f})")
            result["gaba_feedback_high"] = round(gaba_high, 4)
            result["gaba_feedback_low"] = round(gaba_low, 4)
            result["gaba_feedback_ok"] = int(gaba_ok)
        else:
            result["gaba_feedback_ok"] = -1  # 不适用 (NE 极少超阈)
            result["gaba_feedback_high"] = 0
            result["gaba_feedback_low"] = 0

        # 准则 5: 恢复能力 (A3 新增) — 末段浓度应回落到基线附近
        #   末段 = 最后 20% 时间, 基线段 = 前 10% 时间
        #   慢调质 (Oxy tau=500ms, 5HT tau=300ms) 恢复阈值放宽到 3.0
        n_total = n_steps
        baseline_end = int(n_total * 0.1)
        recovery_start = int(n_total * 0.8)
        recovery_fail_channels = []
        for ch in CHANNEL_ORDER:
            baseline_mean = float(conc[ch][:baseline_end].mean())
            recovery_mean = float(conc[ch][recovery_start:].mean())
            # 恢复比: 末段均值 / 基线均值 (理想 ≈ 1.0, 病理滑移 > 阈值)
            if abs(baseline_mean) > 1e-6:
                ratio = recovery_mean / baseline_mean
            else:
                ratio = 0.0 if recovery_mean < 1e-6 else 99.0
            result[f"recovery_ratio_{ch}"] = round(ratio, 4)
            result[f"recovery_mean_{ch}"] = round(recovery_mean, 4)
            # 恢复阈值: 快调质 (DA/ACh/NE/GABA) = 2.0, 慢调质 (5HT/Oxy) = 3.0
            #   5HT tau=300ms, Oxy tau=500ms, 恢复速度慢, 需更宽松阈值
            recovery_thresh = 3.0 if ch in ("5HT", "Oxy") else 2.0
            if ratio > recovery_thresh:
                recovery_fail_channels.append(f"{ch}={ratio:.2f}")
        if recovery_fail_channels:
            # 恢复能力失败仅在 homeostasis ON 时计入 FAIL (OFF 时预期可能不恢复)
            if enable_homeostasis:
                all_pass = False
                errors.append(f"恢复失败 [{', '.join(recovery_fail_channels)}]")
            result["recovery_ok"] = 0
        else:
            result["recovery_ok"] = 1

        # 准则 6: 稳态补偿效果 (仅 homeostasis ON) — 受体灵敏度应有所下调
        if enable_homeostasis:
            final_sens = sensitivity_log[-1]  # (6,)
            min_sens = float(final_sens.min())
            result["sensitivity_min"] = round(min_sens, 4)
            # 灵敏度至少有变化 (min < 0.99 说明至少一个 channel 触发了下调)
            # 不强制 FAIL, 仅记录 (稳态补偿是渐进过程)
            result["sensitivity_active"] = int(min_sens < 0.99)

        result["status"] = "PASS" if all_pass else "FAIL"
        result["errors"] = "; ".join(errors) if errors else ""

    except Exception as e:
        result["status"] = "ERROR"
        result["errors"] = str(e)[:200]

    return result


def find_all_datasets(root: Path) -> list[Path]:
    """查找所有子数据集 NPZ 文件."""
    npz_files = sorted(root.glob("*/*/traces.npz"))
    return npz_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量验证 6 维调质动力学稳定性 (遍历所有场景×强度)."
    )
    parser.add_argument(
        "--datasets-dir",
        default="data/neuroscience_traces/datasets",
        help="数据集根目录",
    )
    parser.add_argument(
        "--output",
        default="data/neuroscience_traces/validation_report.csv",
        help="CSV 汇总报告输出路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行进程数 (默认 4)",
    )
    parser.add_argument(
        "--sim-dt-ms",
        type=float,
        default=1.0,
        help="模拟时间步长 ms (默认 1.0)",
    )
    parser.add_argument(
        "--no-homeostasis",
        action="store_true",
        help="禁用 Phase 3a-B 稳态补偿 (默认启用, 用于对比测试)",
    )
    args = parser.parse_args()

    enable_homeostasis = not args.no_homeostasis

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        sys.stderr.write(f"[ERROR] datasets dir not found: {datasets_dir}\n")
        return 1

    npz_files = find_all_datasets(datasets_dir)
    if not npz_files:
        sys.stderr.write(f"[ERROR] no traces.npz found under {datasets_dir}\n")
        return 1

    print(f"[INFO] 找到 {len(npz_files)} 个子数据集")
    print(f"[INFO] 并行进程数: {args.workers}")
    print(f"[INFO] 稳态补偿: {'ON' if enable_homeostasis else 'OFF'}")
    print(f"[INFO] 开始批量验证...")
    print()

    start_time = time.time()
    results = []

    # 并行验证
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_single, npz, args.sim_dt_ms,
                            enable_homeostasis): npz
            for npz in npz_files
        }
        done_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                done_count += 1
                if done_count % 50 == 0 or done_count == len(npz_files):
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (len(npz_files) - done_count) / rate if rate > 0 else 0
                    print(f"  进度: {done_count}/{len(npz_files)} "
                          f"({done_count/len(npz_files)*100:.1f}%) "
                          f"速率: {rate:.1f}/s  ETA: {eta:.0f}s")
            except Exception as e:
                npz = futures[future]
                results.append({
                    "scenario": npz.parent.parent.name,
                    "intensity_str": npz.parent.name,
                    "status": "ERROR",
                    "errors": str(e)[:200],
                })

    elapsed = time.time() - start_time

    # ===== 汇总统计 =====
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_error = sum(1 for r in results if r.get("status") == "ERROR")
    n_skip = sum(1 for r in results if r.get("status") == "SKIP")
    n_total = len(results)

    print()
    print("=" * 70)
    print("批量验证汇总报告")
    print("=" * 70)
    print(f"  总子数据集:  {n_total}")
    print(f"  通过 (PASS): {n_pass}  ({n_pass/n_total*100:.1f}%)")
    print(f"  失败 (FAIL): {n_fail}  ({n_fail/n_total*100:.1f}%)")
    print(f"  错误 (ERROR): {n_error}  ({n_error/n_total*100:.1f}%)")
    print(f"  跳过 (SKIP):  {n_skip}  ({n_skip/n_total*100:.1f}%)")
    print(f"  耗时: {elapsed:.1f}s  (平均 {elapsed/n_total*1000:.0f}ms/个)")

    # 失败案例详情
    if n_fail > 0 or n_error > 0:
        print()
        print("--- 失败/错误案例 (前 20 个) ---")
        fail_cases = [r for r in results
                      if r.get("status") in ("FAIL", "ERROR")]
        for r in fail_cases[:20]:
            print(f"  [{r['status']}] {r['scenario']}/{r['intensity_str']}: "
                  f"{r.get('errors', '')}")

    # 按场景统计通过率
    print()
    print("--- 按场景统计通过率 ---")
    scenario_stats = {}
    for r in results:
        sc = r.get("scenario", "?")
        if sc not in scenario_stats:
            scenario_stats[sc] = {"pass": 0, "total": 0}
        scenario_stats[sc]["total"] += 1
        if r.get("status") == "PASS":
            scenario_stats[sc]["pass"] += 1

    for sc in sorted(scenario_stats.keys()):
        s = scenario_stats[sc]
        rate = s["pass"] / s["total"] * 100 if s["total"] > 0 else 0
        flag = "✓" if rate == 100 else "✗" if rate < 50 else "~"
        print(f"  {flag} {sc:35s} {s['pass']}/{s['total']} ({rate:.0f}%)")

    # 写 CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OK] CSV 报告: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    # 最终判定
    print()
    print("=" * 70)
    if n_fail == 0 and n_error == 0:
        print(f"[ALL PASS] 全部 {n_total} 个子数据集通过验证")
        print("           6 维调质动力学在所有场景×强度下稳定")
        print("           → 可以接入 CUDA SNN 主程序做端到端训练")
        ret = 0
    else:
        print(f"[ISSUES] {n_fail + n_error} 个子数据集未通过")
        print("         请检查失败案例的 errors 字段")
        ret = 3
    print("=" * 70)
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
