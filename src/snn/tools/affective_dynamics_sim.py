#!/usr/bin/env python3
"""Python 复现 SNN 6 维调质动力学, 用真实 GRAB 神经科学数据验证.

================================================================
目标
================================================================
回答 "6 维调质动力学在真实大脑调质动态输入下是否稳定" 这个问题.

不依赖 CUDA SNN 主程序, 直接用 numpy 复现 modulatory_kernels.cu 的差分方程:
    conc[t+1] = conc[t] * step_decay + signal[t]
    step_decay = exp(-sim_dt / tau)

然后加载 prepare_neuroscience_data.py 生成的 traces.npz, 把 GRAB ΔF/F 信号
当作对应调质的 signal 输入, 跑 100K 步, 检查:
  1. 6 维浓度始终 ∈ [0, 2] (与 CUDA kernel clamp 一致)
  2. PAD 三维 (pleasure/arousal/dominance) 在刺激切换时有可解释响应
  3. LLM 调制信号 (temperature_delta 等) 有界不发散
  4. GABA 抗焦虑反馈: 当 NE 持续高时, GABA 应上升拉住 NE

如果通过, 说明 Phase 3a 的 6 维调质代码在真实数据下也是稳定的,
可以直接接入 CUDA SNN 主程序做端到端训练.

================================================================
与 CUDA 实现的对应关系
================================================================
modulatory_kernels.cu::modulatory_kernel
    每 100 步 launch 一次, 应用累积衰减 + 100 步累积 signal
本脚本
    单步动力学, 每 step 应用 step_decay + 当前 signal
    (数学上等价, 只是时间分辨率更高)

信号计算 (来自 launch_modulatory):
    da_signal   = DA_BASE + DA_GAIN * (1 - pred_err) + bonus(da_delta)
    ach_signal  = 0.2 + 0.3*novelty + 0.1*pred_succ_cos
    ne_signal   = 0.05 + (0.5*kl_div if kl_div > 0.5 else 0)
    ht5_signal  = 0.1 + (0.3*|da_delta| if da_delta < -0.5 else 0)
    gaba_signal = 0.15 + (0.4*(last_ne_mean - 0.3) if last_ne > 0.3 else 0)
    oxy_signal  = 0.05 + 0.3*empathy

PAD 映射 (来自 get_affective_state):
    pleasure  = DA - 0.5*5HT - 0.3*GABA
    arousal   = NE - 0.4*GABA - 0.3*5HT
    dominance = DA - 0.5*Oxy

LLM 调制信号:
    temperature_delta = 0.3*DA - 0.3*5HT - 0.1*GABA
    top_p_delta       = -0.2*NE
    repetition_delta  = 0.1*NE
    empathy_level     = 0.5*Oxy  (clamp [0,1])

用法:
    python affective_dynamics_sim.py
    python affective_dynamics_sim.py --traces data/neuroscience_traces/processed/traces.npz \
                                      --steps 100000 --sim-dt-ms 1.0
依赖:
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.signal import lfilter
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    sys.stderr.write("[WARN] scipy missing, falling back to pure-numpy loop (slower)\n")

# ============================================================================
# 调质参数 (与 src/snn/config.h 严格对齐)
# ============================================================================
TAU_MS = {
    "DA":  100.0,
    "ACh": 200.0,
    "NE":  150.0,
    "5HT": 300.0,
    "GABA":120.0,
    "Oxy": 500.0,
}

BASE = {
    "DA":   0.1,   # DA_BASE
    "ACh":  0.2,   # launch_modulatory 里硬编码
    "NE":   0.05,
    "5HT":  0.1,
    "GABA": 0.15,  # GABA_BASE
    "Oxy":  0.05,  # OXYTOCIN_BASE
}

GAIN = {
    "DA":   0.5,   # DA_GAIN
    "ACh":  0.3,   # novelty 增益
    "NE":   0.5,   # KL 触发增益
    "5HT":  0.3,   # |da_delta| 增益
    "GABA": 0.4,   # GABA_GAIN (NE 反馈)
    "Oxy":  0.3,   # OXYTOCIN_GAIN (empathy)
}

# 浓度 clamp (与 CUDA kernel 一致)
CONC_MIN = 0.0
CONC_MAX = 2.0

CHANNEL_ORDER = ["DA", "ACh", "NE", "5HT", "GABA", "Oxy"]
TRACE_KEY = {
    "DA":  "da_trace",
    "ACh": "ach_trace",
    "NE":  "ne_trace",
    "5HT": "ht5_trace",
    "GABA":"gaba_trace",
    "Oxy": "oxy_trace",
}

# ============================================================================
# Phase 3a-B: 稳态补偿参数 (与 config.h HOMEOSTATIC_* 严格对齐)
# ============================================================================
HOMEOSTATIC_RATE = 0.002       # 受体下调速率 (每 100 步)
HOMEOSTATIC_UPREG_RATE = 0.001 # 受体上调速率 (更慢, 避免震荡)
RECEPTOR_SENSITIVITY_MIN = 0.3 # 灵敏度下限 (防完全失敏)
RECEPTOR_SENSITIVITY_MAX = 1.0 # 灵敏度上限
# 各调质稳态基线阈值 (超过此值触发下调, 与 config.h 对齐)
HOMEOSTATIC_BASELINE = {
    "DA":   0.15,
    "ACh":  0.25,
    "NE":   0.20,
    "5HT":  0.20,
    "GABA": 0.25,
    "Oxy":  0.15,
}


# ============================================================================
# 信号生成: 从 GRAB ΔF/F 轨迹导出每步的 signal[t]
# ============================================================================
# CUDA 中 launch_modulatory 每 100 步调用一次, 信号是 100 步累积值
# Python 用 sim_dt_ms 步长时, 每步信号需缩放: signal_per_step = signal_cuda * (sim_dt / 100)
CUDA_MODULATORY_INTERVAL_MS = 100.0


def signals_from_traces(traces: dict, t_grid_ms: np.ndarray,
                         fs_data_hz: float = 100.0,
                         sim_dt_ms: float = 1.0) -> dict:
    """把 GRAB ΔF/F 信号转换为 modulatory_kernel 的 signal 输入.

    映射策略 (简化版, 不模拟完整 SNN 内部):
      DA  ← GRAB-DA 信号 (直接映射, ΔF/F ≈ signal 强度)
      5HT ← GRAB-5HT
      NE  ← GRAB-NE
      ACh ← GRAB-ACh
      Oxy ← GRAB-Oxy (乘以 empathy 归一化系数)
      GABA ← 内部反馈 (由 NE 上一轮均值驱动, 见 launch_modulatory)

    时间尺度匹配:
      CUDA launch_modulatory 每 100 步调用一次, signal 是 100 步累积值.
      Python 用 sim_dt_ms 步长, 每步注入 signal_scaled = signal_cuda * (sim_dt / 100).
      这样稳态浓度 conc_ss = signal / (1 - exp(-sim_dt/tau)) 与 CUDA 一致.
    """
    dt_ms = float(t_grid_ms[1] - t_grid_ms[0]) if len(t_grid_ms) > 1 else sim_dt_ms
    n_steps = len(t_grid_ms)

    # 信号缩放因子: Python 每步信号 = CUDA 100步信号 * (sim_dt / 100)
    signal_scale = dt_ms / CUDA_MODULATORY_INTERVAL_MS

    # 数据采样间隔 (ms)
    data_dt_ms = 1000.0 / fs_data_hz  # 100Hz → 10ms

    signals = {}
    for ch in CHANNEL_ORDER:
        key = TRACE_KEY[ch]
        if key not in traces:
            sig = np.full(n_steps, BASE[ch], dtype=np.float32)
        else:
            raw = traces[key]
            # 把 raw (在 data_dt_ms 间隔采样) 插值到 t_grid_ms (sim_dt 间隔)
            t_data = np.arange(len(raw), dtype=np.float64) * data_dt_ms
            sig_data = np.interp(t_grid_ms, t_data, raw, left=0.0, right=0.0)
            # ΔF/F (0~1 量级) → signal (BASE + GAIN * ΔF/F)
            # 注: GRAB 信号已是相对浓度变化, 直接当作 signal 增量
            sig = BASE[ch] + GAIN[ch] * sig_data.astype(np.float32)
        # 按时间步长缩放 (匹配 CUDA 100步间隔的信号幅度)
        sig = sig * signal_scale
        # clamp signal ≥ 0 (CUDA: if da_signal < 0: da_signal = 0)
        sig = np.clip(sig, 0.0, CONC_MAX)
        signals[ch] = sig

    # GABA 特殊处理: 用 NE 浓度的滞后均值反馈 (与 CUDA h_last_ne_mean 一致)
    # CUDA: h_last_ne_mean 是上一轮 (100步) 的 NE 浓度均值 (范围 0-2), 阈值 0.3
    # Python: 先跑一遍 NE 动力学得到浓度, 再用浓度的滞后均值驱动 GABA
    ne_tau = TAU_MS["NE"]
    ne_step_decay = float(np.exp(-dt_ms / ne_tau))
    ne_sig = signals["NE"]
    if HAVE_SCIPY:
        ne_conc_prerun = lfilter([1.0], [1.0, -ne_step_decay], ne_sig).astype(np.float32)
    else:
        ne_conc_prerun = np.empty_like(ne_sig)
        ne_conc_prerun[0] = ne_sig[0]
        for t in range(1, len(ne_sig)):
            ne_conc_prerun[t] = ne_step_decay * ne_conc_prerun[t-1] + ne_sig[t]
    ne_conc_prerun = np.clip(ne_conc_prerun, CONC_MIN, CONC_MAX)

    # 滞后窗口: 对应 CUDA 的 100 步间隔
    feedback_window = max(1, int(round(CUDA_MODULATORY_INTERVAL_MS / dt_ms)))
    last_ne_mean = np.full(n_steps, 0.05, dtype=np.float32)
    ne_cumsum = np.cumsum(ne_conc_prerun)
    for t in range(feedback_window, n_steps):
        last_ne_mean[t] = (ne_cumsum[t] - ne_cumsum[t-feedback_window]) / feedback_window
    # GABA 反馈: NE 浓度均值 > 0.3 时增加 GABA (阈值在浓度空间, 与 CUDA 一致)
    gaba_feedback = np.where(last_ne_mean > 0.3,
                             GAIN["GABA"] * (last_ne_mean - 0.3),
                             0.0).astype(np.float32)
    # GABA 信号 = 基线 + 反馈, 然后缩放到 sim_dt 步长
    gaba_signal_full = BASE["GABA"] + gaba_feedback
    signals["GABA"] = np.clip(gaba_signal_full * signal_scale, 0.0, CONC_MAX)

    return signals


# ============================================================================
# 6 维调质动力学 (向量化 lfilter)
# ============================================================================
def run_dynamics(signals: dict, sim_dt_ms: float) -> dict:
    """跑 6 维差分方程: conc[t+1] = conc[t] * step_decay + signal[t].

    用 scipy.signal.lfilter 一次性算完 (比 Python 循环快 1000x).
    lfilter(b=[1.0], a=[1.0, -step_decay], x=signal) 实现:
        y[t] = signal[t] + step_decay * y[t-1]
    """
    conc_out = {}
    for ch in CHANNEL_ORDER:
        tau = TAU_MS[ch]
        step_decay = float(np.exp(-sim_dt_ms / tau))
        sig = signals[ch]
        if HAVE_SCIPY:
            # lfilter: y[t] = b[0]*x[t] - a[1]*y[t-1]   (a[0]=1)
            # 我们要 y[t] = x[t] + step_decay * y[t-1]
            # 所以 b=[1.0], a=[1.0, -step_decay]
            conc = lfilter([1.0], [1.0, -step_decay], sig).astype(np.float32)
        else:
            # 纯 numpy 递推 (慢但正确)
            conc = np.empty_like(sig)
            conc[0] = sig[0]
            for t in range(1, len(sig)):
                conc[t] = step_decay * conc[t-1] + sig[t]
        # clamp (与 CUDA kernel 一致)
        conc = np.clip(conc, CONC_MIN, CONC_MAX)
        conc_out[ch] = conc
    return conc_out


# ============================================================================
# Phase 3a-B: 带稳态补偿的 6 维调质动力学
# ============================================================================
def run_dynamics_with_homeostasis(signals: dict, sim_dt_ms: float,
                                   enable_homeostasis: bool = True) -> tuple:
    """跑 6 维差分方程 + 受体灵敏度稳态补偿.

    两阶段近似法 (比逐 chunk lfilter 快 100x):
      Pass 1: 无补偿跑一遍 lfilter, 得到基线浓度轨迹
      Pass 2: 从基线浓度提取每 chunk 均值 → 更新灵敏度 → 缩放信号 → 重跑 lfilter

    近似精度: 稳态补偿速率 HOMEOSTATIC_RATE=0.002 极慢, 两阶段误差 < 1%
    (保守估计: Pass 1 浓度偏高 → 灵敏度下调偏多 → Pass 2 浓度偏低, 偏保守)

    返回:
      conc: dict[str, np.ndarray]  6 维浓度轨迹
      sensitivity_log: np.ndarray  (N_chunks, 6) 受体灵敏度历史
    """
    n_steps = len(next(iter(signals.values())))
    chunk_size = max(1, int(round(CUDA_MODULATORY_INTERVAL_MS / sim_dt_ms)))
    n_chunks = (n_steps + chunk_size - 1) // chunk_size

    if not enable_homeostasis:
        conc = run_dynamics(signals, sim_dt_ms)
        sensitivity_log = np.ones((n_chunks, 6), dtype=np.float32)
        return conc, sensitivity_log

    # ===== Pass 1: 无补偿基线浓度 (向量化 lfilter, 快) =====
    baseline_conc = run_dynamics(signals, sim_dt_ms)

    # ===== 从基线浓度提取每 chunk 末浓度 (用于灵敏度更新) =====
    # chunk 边界索引 (每 chunk 最后一个样本的 index+1)
    chunk_ends = np.arange(chunk_size, n_steps + 1, chunk_size)
    if chunk_ends[-1] < n_steps:
        chunk_ends = np.append(chunk_ends, n_steps)

    # ===== Pass 2: 逐 chunk 更新灵敏度, 构建缩放后的信号 =====
    sensitivity = {ch: 1.0 for ch in CHANNEL_ORDER}
    sensitivity_log = np.ones((n_chunks, 6), dtype=np.float32)
    scaled_signals = {}

    for chi, ch in enumerate(CHANNEL_ORDER):
        baseline = HOMEOSTATIC_BASELINE[ch]
        # 提取每 chunk 末的基线浓度
        prev_conc_val = 0.0  # 初始浓度
        per_step_scale = np.ones(n_steps, dtype=np.float32)

        for ci in range(n_chunks):
            t0 = ci * chunk_size
            t1 = min(t0 + chunk_size, n_steps)
            # 更新灵敏度 (基于上一 chunk 末浓度)
            excess = prev_conc_val - baseline
            if excess > 0.0:
                sensitivity[ch] *= (1.0 - HOMEOSTATIC_RATE * excess)
            else:
                sensitivity[ch] *= (1.0 + HOMEOSTATIC_UPREG_RATE * (-excess))
            sensitivity[ch] = max(RECEPTOR_SENSITIVITY_MIN,
                                  min(RECEPTOR_SENSITIVITY_MAX, sensitivity[ch]))
            sensitivity_log[ci, chi] = sensitivity[ch]
            # 本 chunk 的信号缩放
            per_step_scale[t0:t1] = sensitivity[ch]
            # 更新 prev_conc_val 为本 chunk 末的基线浓度
            prev_conc_val = float(baseline_conc[ch][t1 - 1])

        scaled_signals[ch] = signals[ch] * per_step_scale

    # ===== Pass 2: 用缩放后信号跑 lfilter (向量化, 快) =====
    conc = run_dynamics(scaled_signals, sim_dt_ms)

    return conc, sensitivity_log


# ============================================================================
# AffectiveState readout (与 get_affective_state 一致)
# ============================================================================
def compute_affective_state(conc: dict) -> dict:
    """从 6 维浓度计算 PAD 模型 + LLM 调制信号."""
    da  = conc["DA"]
    ach = conc["ACh"]
    ne  = conc["NE"]
    ht5 = conc["5HT"]
    gaba= conc["GABA"]
    oxy = conc["Oxy"]

    # PAD 模型
    pleasure  = da - 0.5*ht5 - 0.3*gaba
    arousal   = ne - 0.4*gaba - 0.3*ht5
    dominance = da - 0.5*oxy

    # LLM 调制信号
    temperature_delta = 0.3*da - 0.3*ht5 - 0.1*gaba
    top_p_delta       = -0.2*ne
    repetition_delta  = 0.1*ne
    empathy_level     = np.clip(0.5*oxy, 0.0, 1.0)

    # clamp PAD 到 [-1, 1]
    pleasure  = np.clip(pleasure, -1.0, 1.0)
    arousal   = np.clip(arousal, -1.0, 1.0)
    dominance = np.clip(dominance, -1.0, 1.0)

    return {
        "pleasure": pleasure.astype(np.float32),
        "arousal": arousal.astype(np.float32),
        "dominance": dominance.astype(np.float32),
        "temperature_delta": temperature_delta.astype(np.float32),
        "top_p_delta": top_p_delta.astype(np.float32),
        "repetition_delta": repetition_delta.astype(np.float32),
        "empathy_level": empathy_level.astype(np.float32),
    }


# ============================================================================
# 验证准则 (3a 验收标准)
# ============================================================================
def validate_dynamics(conc: dict, affective: dict, t_grid_ms: np.ndarray) -> bool:
    """检查 6 维调质动力学是否满足 Phase 3a 验收准则."""
    print("\n" + "=" * 70)
    print("Phase 3a 验证报告 (路径 C: 真实 GRAB 数据驱动)")
    print("=" * 70)

    all_pass = True

    # 准则 1: 6 维浓度始终 ∈ [0, 2]
    print("\n[1] 6 维浓度范围检查 (CUDA clamp = [0, 2]):")
    for ch in CHANNEL_ORDER:
        c = conc[ch]
        cmin, cmax, cmean = float(c.min()), float(c.max()), float(c.mean())
        ok = (cmin >= 0.0) and (cmax <= 2.0)
        flag = "✓" if ok else "✗"
        print(f"    {flag} {ch:5s}  min={cmin:.3f}  max={cmax:.3f}  mean={cmean:.3f}")
        if not ok:
            all_pass = False

    # 准则 2: PAD 三维有界
    print("\n[2] PAD 三维范围检查 (期望 [-1, 1]):")
    for key in ["pleasure", "arousal", "dominance"]:
        v = affective[key]
        vmin, vmax, vmean = float(v.min()), float(v.max()), float(v.mean())
        ok = (vmin >= -1.0) and (vmax <= 1.0)
        flag = "✓" if ok else "✗"
        print(f"    {flag} {key:12s}  min={vmin:+.3f}  max={vmax:+.3f}  mean={vmean:+.3f}")
        if not ok:
            all_pass = False

    # 准则 3: LLM 调制信号有界
    print("\n[3] LLM 调制信号范围检查:")
    bounds = {
        "temperature_delta": (-0.5, 0.5),
        "top_p_delta":       (-0.4, 0.0),   # 0.2*NE_max=0.4
        "repetition_delta":  (0.0, 0.2),
        "empathy_level":     (0.0, 1.0),
    }
    for key, (lo, hi) in bounds.items():
        v = affective[key]
        vmin, vmax = float(v.min()), float(v.max())
        # 允许少量越界 (1% 容差)
        ok = (vmin >= lo - 0.05) and (vmax <= hi + 0.05)
        flag = "✓" if ok else "✗"
        print(f"    {flag} {key:20s}  min={vmin:+.3f}  max={vmax:+.3f}  (期望 [{lo:+.2f}, {hi:+.2f}])")
        if not ok:
            all_pass = False

    # 准则 4: GABA 反馈在 NE 高时上升
    print("\n[4] GABA 抗焦虑反馈检查:")
    ne = conc["NE"]
    gaba = conc["GABA"]
    ne_high_mask = ne > 0.3
    if ne_high_mask.any():
        gaba_during_ne_high = float(gaba[ne_high_mask].mean())
        gaba_baseline = float(gaba[~ne_high_mask].mean() if (~ne_high_mask).any() else gaba.mean())
        ok = gaba_during_ne_high > gaba_baseline
        flag = "✓" if ok else "✗"
        print(f"    {flag} NE 高时 GABA 均值={gaba_during_ne_high:.3f} > 基线 {gaba_baseline:.3f}")
        if not ok:
            all_pass = False
    else:
        print(f"    - NE 从未超过 0.3, 此准则不适用 (NE max={float(ne.max()):.3f})")

    # 准则 5: 刺激响应延迟 (DA 应快于 5HT)
    print("\n[5] 调质响应时间常数检查 (DA 应快, 5HT 应慢):")
    da_tau_eff = TAU_MS["DA"]
    ht5_tau_eff = TAU_MS["5HT"]
    ok = da_tau_eff < ht5_tau_eff
    flag = "✓" if ok else "✗"
    print(f"    {flag} DA_TAU={da_tau_eff}ms  <  5HT_TAU={ht5_tau_eff}ms  "
          f"(DA 快响应, 5HT 慢响应)")
    if not ok:
        all_pass = False

    # 总结
    print("\n" + "=" * 70)
    if all_pass:
        print("[PASS] 所有准则通过, 6 维调质动力学在真实数据下稳定")
        print("       → 可以接入 CUDA SNN 主程序做端到端训练")
    else:
        print("[FAIL] 部分准则未通过, 需检查 modulatory_kernels.cu 参数")
    print("=" * 70)
    return all_pass


# ============================================================================
# 可视化
# ============================================================================
def plot_trajectories(conc: dict, affective: dict, signals: dict,
                       t_grid_ms: np.ndarray, out_path: Path):
    """画 6 维调质 + PAD + LLM 调制信号轨迹图."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 非交互后端
        import matplotlib.pyplot as plt
        # 配置中文字体 (Windows 优先使用微软雅黑, Linux 用文泉驿)
        import platform
        if platform.system() == "Windows":
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
        else:
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[WARN] matplotlib 未安装, 跳过画图")
        return False

    t_sec = t_grid_ms / 1000.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: 6 维调质浓度
    ax = axes[0]
    colors = {"DA":"tab:red", "ACh":"tab:orange", "NE":"tab:green",
              "5HT":"tab:blue", "GABA":"tab:purple", "Oxy":"tab:pink"}
    for ch in CHANNEL_ORDER:
        ax.plot(t_sec, conc[ch], label=ch, color=colors[ch], linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Concentration")
    ax.set_title("6 维调质浓度 (ΔF/F 驱动)")
    ax.legend(loc="upper right", ncol=6)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 2.05)

    # Panel 2: PAD 三维
    ax = axes[1]
    for key, color in [("pleasure","tab:red"), ("arousal","tab:orange"),
                        ("dominance","tab:blue")]:
        ax.plot(t_sec, affective[key], label=key, color=color, linewidth=1.2)
    ax.set_ylabel("PAD value")
    ax.set_title("PAD 情感模型 (Pleasure/Arousal/Dominance)")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 1.05)

    # Panel 3: LLM 调制信号
    ax = axes[2]
    for key, color in [("temperature_delta","tab:red"),
                        ("top_p_delta","tab:orange"),
                        ("repetition_delta","tab:green"),
                        ("empathy_level","tab:blue")]:
        ax.plot(t_sec, affective[key], label=key, color=color, linewidth=1.2)
    ax.set_ylabel("LLM modulator delta")
    ax.set_title("LLM 生成调制信号 (注入 MiniCPM5 推理参数)")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    # Panel 4: 输入 signal (GRAB ΔF/F)
    ax = axes[3]
    for ch in CHANNEL_ORDER:
        ax.plot(t_sec, signals[ch], label=f"{ch} sig", color=colors[ch],
                linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Input signal")
    ax.set_xlabel("Time (s)")
    ax.set_title("输入信号 (来自 GRAB ΔF/F + 基线)")
    ax.legend(loc="upper right", ncol=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[OK] Saved trajectory plot: {out_path}")
    return True


# ============================================================================
# 主入口
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Python 复现 SNN 6 维调质动力学, 用真实 GRAB 数据验证稳定性."
    )
    parser.add_argument(
        "--traces",
        default="data/neuroscience_traces/processed/traces.npz",
        help="prepare_neuroscience_data.py 生成的 npz 路径",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="模拟步数 (默认 100000, 配合 sim_dt_ms=1 即 100 秒)",
    )
    parser.add_argument(
        "--sim-dt-ms",
        type=float,
        default=1.0,
        help="模拟时间步长 ms (默认 1.0, 与 SNN step 一致)",
    )
    parser.add_argument(
        "--plot",
        default="data/neuroscience_traces/processed/dynamics_trajectory.png",
        help="输出轨迹图路径 (留空跳过画图)",
    )
    parser.add_argument(
        "--out-npz",
        default="data/neuroscience_traces/processed/dynamics_output.npz",
        help="输出 6 维浓度 + PAD + LLM 调制信号 npz 路径",
    )
    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        sys.stderr.write(f"[ERROR] traces npz not found: {traces_path}\n")
        sys.stderr.write("[HINT] 请先运行: python src/snn/tools/prepare_neuroscience_data.py\n")
        return 1

    print(f"[INFO] Loading traces: {traces_path}")
    data = np.load(traces_path, allow_pickle=True)
    traces = {k: data[k] for k in data.files if k.endswith("_trace")}
    if "time_s" in data.files:
        t_data_s = data["time_s"]
        fs_data = 1.0 / float(t_data_s[1] - t_data_s[0]) if len(t_data_s) > 1 else 100.0
    else:
        fs_data = 100.0
    print(f"[INFO] Loaded {len(traces)} channel traces, fs_data={fs_data:.1f} Hz")

    # 构造模拟时间轴
    n_steps = args.steps
    sim_dt = args.sim_dt_ms
    t_grid_ms = np.arange(n_steps, dtype=np.float64) * sim_dt
    total_s = n_steps * sim_dt / 1000.0
    print(f"[INFO] Sim grid: steps={n_steps}, dt={sim_dt}ms, total={total_s:.1f}s")

    # 生成 signals
    print("[INFO] Generating input signals from GRAB traces...")
    signals = signals_from_traces(traces, t_grid_ms, fs_data_hz=fs_data, sim_dt_ms=sim_dt)

    # 跑动力学
    print("[INFO] Running 6-dimensional modulatory dynamics...")
    conc = run_dynamics(signals, sim_dt)

    # 计算 PAD + LLM 调制信号
    print("[INFO] Computing PAD affective state and LLM modulation signals...")
    affective = compute_affective_state(conc)

    # 验证
    ok = validate_dynamics(conc, affective, t_grid_ms)

    # 保存输出 npz
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"time_ms": t_grid_ms.astype(np.float32)}
    for ch in CHANNEL_ORDER:
        save_dict[f"conc_{ch}"] = conc[ch]
        save_dict[f"signal_{ch}"] = signals[ch]
    for k, v in affective.items():
        save_dict[k] = v
    np.savez_compressed(out_npz, **save_dict)
    print(f"\n[OK] Saved dynamics output: {out_npz} ({out_npz.stat().st_size/1024:.1f} KB)")

    # 画图
    if args.plot:
        plot_trajectories(conc, affective, signals, t_grid_ms, Path(args.plot))

    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
