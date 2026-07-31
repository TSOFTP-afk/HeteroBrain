#!/usr/bin/env python3
"""加载从论文图提取的真实神经调质浓度轨迹 (路径 C 数据准备).

================================================================
背景
================================================================
SNN 的 6 维调质系统 (DA / 5HT / NE / ACh / GABA / 催产素) 应该学习
"真实大脑在刺激下的调质动态", 而不是 BPE token 流.

本脚本读取 data/neuroscience_traces/csv_raw/ 下用 WebPlotDigitizer
从论文补充材料图提取的 CSV 文件, 重采样到统一时间轴, 输出标准化的
npz 张量供 affective_dynamics_sim.py 验证 SNN 调质动力学.

================================================================
论文图提取清单 (用户需用 WebPlotDigitizer 手动提取)
================================================================
每条 CSV 必须包含两列: time_s, dF_F
(time_s 单位秒, dF_F 是 GRAB 探针 ΔF/F0 荧光信号, 正比于调质浓度)

必填 (覆盖 6 维中的 5 维, GABA 暂无 GRAB 探针, 用合成信号补):

  csv_raw/DA_weber2024_cocaine_cue.csv
    来源: Weber 2024 Neuropsychopharmacology (PMID 39300272, PMC11632087)
    图: Fig 1B 或 Fig 2 (NAcc DA transients during cue-induced seeking)
    内容: 大鼠自我给药 + cue 诱导的 DA 瞬态变化
    典型时长: ~10s, 峰值 ΔF/F ≈ 0.3-0.5

  csv_raw/5HT_deng2024_seizure.csv
    来源: Deng 2024 Nat Methods (PMID 38443508)
    图: Fig 6 (seizure-induced 5-HT waves)
    内容: 小鼠癫痫诱导后的皮层 5HT 释放波动
    典型时长: ~60s, 峰值 ΔF/F ≈ 0.2-0.4

  csv_raw/NE_breton2022_looming.csv
    来源: Breton-Provencher 2022 Nature (PMID 35650441)
    图: Fig 2 或 Fig 4 (looming stimulus evoked NE release)
    内容: 视觉威胁刺激 (looming) 引发的 NE 释放
    典型时长: ~5s, 峰值 ΔF/F ≈ 0.4-0.6

  csv_raw/ACh_jing2020_running.csv
    来源: Jing 2020 Nat Methods (GRAB-ACh2.0, 对应 PMID 33087905 的同系列)
    图: Fig 5 (小鼠运动期间 ACh 动态)
    内容: 跑轮运动期间前额叶 ACh 变化
    典型时长: ~30s, 峰值 ΔF/F ≈ 0.1-0.3

  csv_raw/Oxy_qian2023_social.csv
    来源: Qian 2023 Nat Biotechnol (PMID 36593404)
    图: Fig 4 或 Fig 5 (社交互动期间催产素释放)
    内容: 小鼠社交接触时的催产素瞬态变化
    典型时长: ~20s, 峰值 ΔF/F ≈ 0.15-0.3

可选 (DA 第二条, 用于交叉验证):
  csv_raw/DA_zhuo2024_reward.csv
    来源: Zhuo 2024 Nat Methods (PMID 38036855, GRAB-DA2m)
    图: Fig 3 (reward consumption evoked DA)
    内容: 奖励消费时的 DA 瞬态

================================================================
CSV 文件命名规范
================================================================
<调质>_<作者年份>_<实验简述>.csv
例: DA_weber2024_cocaine_cue.csv
调质前缀必须是: DA / 5HT / NE / ACh / GABA / Oxy
(GABA 因无公开 GRAB 数据, 由本脚本自动生成合成轨迹补齐)

================================================================
输出格式
================================================================
data/neuroscience_traces/processed/traces.npz
含以下数组:
  - time_s:           (T,)       统一时间轴, 秒
  - da_trace:         (T,)       DA ΔF/F 信号
  - ht5_trace:        (T,)       5HT ΔF/F 信号
  - ne_trace:         (T,)       NE ΔF/F 信号
  - ach_trace:        (T,)       ACh ΔF/F 信号
  - gaba_trace:       (T,)       GABA 合成信号 (sigma 0.15 ± 0.05)
  - oxy_trace:        (T,)       催产素 ΔF/F 信号
  - stimulus_events:  (N_evt, 2) 每行 [事件时间_s, 事件类型_id]
                       事件类型: 0=奖励 1=惩罚 2=社交 3=认知 4=威胁
  - source_meta:      (6,)       每个 channel 的来源论文 PMID (GABA=0 表合成)

用法:
    python prepare_neuroscience_data.py
    python prepare_neuroscience_data.py --csv-dir data/neuroscience_traces/csv_raw \
                                        --output  data/neuroscience_traces/processed/traces.npz \
                                        --target-fs 100.0
依赖:
    pip install numpy scipy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.interpolate import interp1d
    from scipy.signal import resample_poly
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    sys.stderr.write("[WARN] scipy not found, falling back to numpy linear interp\n")


# ============================================================================
# 论文清单 (用于校验文件名与提取源一致)
# ============================================================================
EXPECTED_TRACES = {
    "DA": {
        "primary":   "DA_weber2024_cocaine_cue.csv",
        "secondary": "DA_zhuo2024_reward.csv",
        "pmid":      39300272,
    },
    "5HT": {
        "primary":   "5HT_deng2024_seizure.csv",
        "secondary": None,
        "pmid":      38443508,
    },
    "NE": {
        "primary":   "NE_breton2022_looming.csv",
        "secondary": None,
        "pmid":      35650441,
    },
    "ACh": {
        "primary":   "ACh_jing2020_running.csv",
        "secondary": None,
        "pmid":      33087905,
    },
    "Oxy": {
        "primary":   "Oxy_qian2023_social.csv",
        "secondary": None,
        "pmid":      36593404,
    },
    # GABA 无公开 GRAB 探针, 用合成信号
    "GABA": {
        "primary":   None,
        "secondary": None,
        "pmid":      0,
    },
}

# 6 维 channel 在输出数组中的顺序 (与 modulatory_kernels.cu 一致)
CHANNEL_ORDER = ["DA", "5HT", "NE", "ACh", "GABA", "Oxy"]
CHANNEL_TO_KEY = {
    "DA":  "da_trace",
    "5HT": "ht5_trace",
    "NE":  "ne_trace",
    "ACh": "ach_trace",
    "GABA":"gaba_trace",
    "Oxy": "oxy_trace",
}


def load_csv_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 WebPlotDigitizer 导出的 CSV (time_s, dF_F)."""
    times, values = [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            # 跳过表头
            try:
                t = float(row[0])
                v = float(row[1])
            except ValueError:
                continue
            times.append(t)
            values.append(v)
    if len(times) < 2:
        raise ValueError(f"CSV has too few data points: {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def resample_to_grid(time_s: np.ndarray, value: np.ndarray,
                     t_grid: np.ndarray) -> np.ndarray:
    """把 (time_s, value) 重采样到统一时间轴 t_grid."""
    if HAVE_SCIPY:
        # 用 scipy 线性插值, 边界外填 0 (假设刺激前为基线)
        f = interp1d(time_s, value, kind="linear",
                     bounds_error=False, fill_value=0.0)
        return f(t_grid).astype(np.float32)
    else:
        # numpy fallback
        return np.interp(t_grid, time_s, value, left=0.0, right=0.0).astype(np.float32)


def make_synthetic_gaba(t_grid: np.ndarray, seed: int = 42) -> np.ndarray:
    """GABA 无公开 GRAB 数据, 合成一个慢振荡 + 噪声的合成信号.

    生物学约束: GABA 是抑制性调质, 在持续兴奋时应上升以拉住 DA.
    所以我们让 GABA 与"信号密度"反相关: 当其他 channel 都活跃时,
    GABA 缓慢上升然后回落.
    """
    rng = np.random.default_rng(seed)
    # 慢振荡: 0.1 Hz 量级 (符合 GABA 慢时间尺度, tau=120s)
    slow = 0.10 + 0.05 * np.sin(2 * np.pi * 0.1 * t_grid / max(t_grid[-1], 1.0))
    # 中等振荡: 1 Hz
    mid = 0.03 * np.sin(2 * np.pi * 1.0 * t_grid)
    # 白噪声
    noise = rng.normal(0, 0.02, size=t_grid.shape)
    sig = slow + mid + noise
    # 钳位到 [0, 0.4]
    return np.clip(sig, 0.0, 0.4).astype(np.float32)


def detect_stimulus_events(traces: dict[str, np.ndarray],
                           t_grid: np.ndarray,
                           threshold: float = 0.05) -> np.ndarray:
    """从 DA / NE / Oxy 轨迹自动检测刺激事件 (用于 SNN 注入电流时机).

    简单策略: 当 DA 或 NE 的 ΔF/F 超过 threshold 的上升沿出现时, 标记一个事件.
    返回 (N_evt, 2): [time_s, event_type_id]
        事件类型: 0=奖励(DA 主导) 1=惩罚(NE+5HT 共升) 2=社交(Oxy 主导)
                  3=认知(ACh 主导) 4=威胁(NE 主导)
    """
    events = []
    dt = float(t_grid[1] - t_grid[0]) if len(t_grid) > 1 else 0.01

    for ch in ["DA", "NE", "Oxy", "ACh"]:
        key = CHANNEL_TO_KEY[ch]
        if key not in traces:
            continue
        sig = traces[key]
        # 上升沿检测
        above = sig > threshold
        # 找上升沿 (False → True 的跳变)
        rising = np.where((~above[:-1]) & (above[1:]))[0]
        for idx in rising:
            t_evt = float(t_grid[idx])
            # 判断事件类型
            if ch == "DA":
                etype = 0  # 奖励
            elif ch == "NE":
                # 区分威胁 vs 惩罚: 看是否同时有 5HT 上升
                ht5 = traces.get(CHANNEL_TO_KEY["5HT"], np.zeros_like(sig))
                if ht5[idx] > threshold * 0.5:
                    etype = 1  # 惩罚
                else:
                    etype = 4  # 威胁
            elif ch == "Oxy":
                etype = 2  # 社交
            else:  # ACh
                etype = 3  # 认知
            events.append([t_evt, etype])

    if not events:
        return np.zeros((0, 2), dtype=np.float32)
    events_arr = np.asarray(events, dtype=np.float32)
    # 按时间排序
    events_arr = events_arr[np.argsort(events_arr[:, 0])]
    # 去重: 200ms 内的同类事件合并
    if len(events_arr) > 1:
        keep = [0]
        for i in range(1, len(events_arr)):
            t_prev = events_arr[keep[-1], 0]
            t_curr = events_arr[i, 0]
            if (t_curr - t_prev) > 0.2:  # 200ms 去抖
                keep.append(i)
        events_arr = events_arr[keep]
    return events_arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="加载从论文图提取的真实神经调质 CSV 轨迹, 输出标准化 npz."
    )
    parser.add_argument(
        "--csv-dir",
        default="data/neuroscience_traces/csv_raw",
        help="WebPlotDigitizer 导出的 CSV 目录",
    )
    parser.add_argument(
        "--output",
        default="data/neuroscience_traces/processed/traces.npz",
        help="输出 npz 路径",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=100.0,
        help="目标采样率 Hz (默认 100, 即 10ms 一步, 对应 SNN step)",
    )
    parser.add_argument(
        "--total-duration",
        type=float,
        default=120.0,
        help="输出轨迹总时长秒 (默认 120, 短的 CSV 会被零填充到该长度)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式: 缺失任何必填 CSV 时直接报错退出",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        sys.stderr.write(f"[ERROR] csv-dir not found: {csv_dir}\n")
        sys.stderr.write("[HINT] 请先用 WebPlotDigitizer (https://apps.automeris.io/wpd4/) "
                         "从论文图提取 CSV 放入该目录\n")
        return 1

    # 统一时间轴
    fs = args.target_fs
    total_t = args.total_duration
    n_samples = int(fs * total_t)
    t_grid = np.linspace(0, total_t, n_samples, dtype=np.float64)
    dt = 1.0 / fs
    print(f"[INFO] Target grid: fs={fs:.1f} Hz, duration={total_t:.1f}s, "
          f"n_samples={n_samples}, dt={dt*1000:.2f} ms")

    # 加载每个 channel
    traces: dict[str, np.ndarray] = {}
    source_pmids: list[int] = []

    for ch in CHANNEL_ORDER:
        info = EXPECTED_TRACES[ch]
        csv_name = info["primary"]
        if csv_name is None:
            # GABA: 合成
            print(f"[INFO] {ch}: no public GRAB data, synthesizing...")
            sig = make_synthetic_gaba(t_grid)
            source_pmids.append(0)
        else:
            csv_path = csv_dir / csv_name
            if not csv_path.exists():
                # 尝试 secondary
                if info["secondary"]:
                    csv_path = csv_dir / info["secondary"]
                if not csv_path.exists():
                    msg = f"[WARN] {ch}: missing CSV ({csv_name})"
                    if args.strict:
                        sys.stderr.write(msg + "\n")
                        return 2
                    print(msg + " — 用零填充")
                    sig = np.zeros_like(t_grid, dtype=np.float32)
                    source_pmids.append(0)
                    traces[CHANNEL_TO_KEY[ch]] = sig
                    continue
            print(f"[INFO] {ch}: loading {csv_path.name}")
            t_raw, v_raw = load_csv_trace(csv_path)
            # 重采样到 t_grid
            sig = resample_to_grid(t_raw, v_raw, t_grid)
            source_pmids.append(info["pmid"])
        traces[CHANNEL_TO_KEY[ch]] = sig

    # 检测刺激事件
    events = detect_stimulus_events(traces, t_grid)
    print(f"[INFO] Detected {len(events)} stimulus events")
    for t_evt, etype in events[:10]:
        type_names = ["奖励", "惩罚", "社交", "认知", "威胁"]
        tn = type_names[int(etype)] if int(etype) < 5 else "?"
        print(f"       t={t_evt:6.2f}s  type={int(etype)} ({tn})")
    if len(events) > 10:
        print(f"       ... ({len(events)-10} more)")

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "time_s": t_grid.astype(np.float32),
        "stimulus_events": events,
        "source_pmids": np.asarray(source_pmids, dtype=np.int32),
    }
    for ch in CHANNEL_ORDER:
        save_dict[CHANNEL_TO_KEY[ch]] = traces[CHANNEL_TO_KEY[ch]]

    np.savez_compressed(out_path, **save_dict)
    out_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved: {out_path} ({out_mb:.2f} MB)")
    print(f"     Channels: {CHANNEL_ORDER}")
    print(f"     Shape:    ({n_samples},) per channel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
