#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3a-C1: 事件驱动调质注入验证脚本
=======================================
验证 events.jsonl 事件流在 6 维调质动力学中的稳定性。

验证内容:
  1. JSONL 格式合法性 (字段完整、类型正确)
  2. 事件类型识别 (与 event_types.h 枚举对齐)
  3. 强度范围 [-50, +50]
  4. step_target 非负且递增
  5. 6 维调质浓度动力学模拟 (与 modulatory_kernels.cu 对齐)
     - 浓度范围 [0, 2]
     - PAD 情感模型有界
     - 稳态补偿生效 (受体灵敏度 ∈ [0.3, 1.0])
     - 事件后浓度能回归基线 (恢复能力)

用法:
  python validate_event_driven.py --events data/events/events_5k.jsonl --steps 5000
  python validate_event_driven.py --events data/events/events_5k.jsonl --steps 5000 --verbose
"""

import argparse
import json
import os
import sys
import math

# =============================================================================
# 基因映射表 (与 gene_event_map.h GENE_MAP_BASE 对齐)
# 6 维: [DA, ACh, NE, 5HT, GABA, Oxy]
# =============================================================================
GENE_MAP_BASE = {
    "food_tasty":       [ 0.40,  0.10,  0.05, -0.05,  0.00,  0.02],
    "food_bland":       [ 0.05,  0.00,  0.00,  0.00,  0.00,  0.00],
    "threat_physical":  [-0.20,  0.30,  0.60,  0.40,  0.10, -0.05],
    "threat_social":    [-0.15,  0.20,  0.45,  0.35,  0.05, -0.10],
    "praise":           [ 0.25,  0.10,  0.15, -0.05,  0.00,  0.20],
    "criticism":        [-0.10,  0.05,  0.20,  0.25,  0.00, -0.15],
    "social_bond":      [ 0.10,  0.05, -0.05,  0.05,  0.05,  0.35],
    "social_loss":      [-0.15,  0.00,  0.10,  0.30,  0.00, -0.25],
    "achievement":      [ 0.50,  0.15,  0.20, -0.10,  0.00,  0.05],
    "novelty":          [ 0.15,  0.40,  0.10,  0.00,  0.00,  0.00],
}

# 时间常数 (ms, 与 config.h 对齐)
TAU = [200.0, 300.0, 150.0, 300.0, 250.0, 500.0]  # DA, ACh, NE, 5HT, GABA, Oxy

# 基线浓度 (与 config.h MODULATORY_BASE 对齐)
BASELINE = [0.10, 0.25, 0.05, 0.10, 0.15, 0.05]

# 稳态补偿参数 (与 config.h 对齐)
HOMEOSTATIC_BASELINE = [0.15, 0.25, 0.20, 0.20, 0.25, 0.15]  # DA, ACh, NE, 5HT, GABA, Oxy
HOMEOSTATIC_RATE = 0.002
HOMEOSTATIC_UPREG_RATE = 0.001
HOMEOSTATIC_CLAMP_MIN = 0.3
HOMEOSTATIC_CLAMP_MAX = 1.0

CHANNEL_NAMES = ["DA", "ACh", "NE", "5HT", "GABA", "Oxy"]


def apply_modifiers(base_delta, modifier_flags, intensity):
    """应用修饰符 + intensity 调制 (与 gene_event_map.h apply_modifiers 对齐)。"""
    result = list(base_delta)
    scale = max(0.05, 1.0 + intensity * 0.02)
    for i in range(6):
        result[i] *= scale
    if modifier_flags & 1:  # MOD_PUBLIC
        result[5] *= 1.5  # Oxy
        result[2] *= 1.2  # NE
    if modifier_flags & 2:  # MOD_AUTHORITY
        result[0] *= 1.3  # DA
        result[3] *= 1.2  # 5HT
    return result


def parse_modifiers(modifiers_dict):
    """将 modifiers 字典转为位域标志。"""
    flags = 0
    if modifiers_dict.get("publicity") == "public":
        flags |= 1
    if modifiers_dict.get("authority") == "authority":
        flags |= 2
    if modifiers_dict.get("temporal") == "sustained":
        flags |= 4
    return flags


def load_events(path):
    """加载 events.jsonl, 返回事件列表。"""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                evt = json.loads(line)
                events.append((line_num, evt))
            except json.JSONDecodeError as e:
                print(f"  [FAIL] line {line_num}: JSON 解析错误: {e}")
    return events


def validate_format(events, total_steps):
    """验证 JSONL 格式合法性, 返回 (pass_count, fail_count, parsed_events)。"""
    passed = 0
    failed = 0
    parsed = []
    prev_step = -1

    for line_num, evt in events:
        errors = []
        # 必填字段
        if "event_type" not in evt:
            errors.append("缺少 event_type 字段")
        else:
            et = evt["event_type"]
            if et not in GENE_MAP_BASE:
                errors.append(f"未知事件类型: {et}")

        if "step_target" not in evt and "time_s" not in evt:
            errors.append("缺少 step_target 或 time_s")
        else:
            step = evt.get("step_target")
            if step is None:
                step = int(evt.get("time_s", 0) * 100)
            if step < 0:
                errors.append(f"step_target 为负: {step}")
            if step >= total_steps:
                errors.append(f"step_target {step} >= total_steps {total_steps}")
            if step < prev_step:
                errors.append(f"step_target {step} 未按递增排序 (前一个: {prev_step})")
            prev_step = step

        intensity = evt.get("intensity", 0)
        if not isinstance(intensity, (int, float)) or intensity < -50 or intensity > 50:
            errors.append(f"intensity 超出 [-50, +50]: {intensity}")

        if errors:
            failed += 1
            for e in errors:
                print(f"  [FAIL] line {line_num}: {e}")
        else:
            passed += 1
            modifiers = evt.get("modifiers", {})
            parsed.append({
                "line_num": line_num,
                "step": step,
                "event_type": evt["event_type"],
                "modifier_flags": parse_modifiers(modifiers),
                "intensity": int(intensity),
                "description": evt.get("description", ""),
            })

    return passed, failed, parsed


def simulate_dynamics(parsed_events, total_steps, verbose=False):
    """
    模拟 6 维调质动力学 (与 modulatory_kernels.cu launch_modulatory 对齐)。
    每 100 步更新一次浓度 (launch_modulatory 调度间隔)。
    """
    # 浓度初始化为基线
    conc = list(BASELINE)
    # 受体灵敏度初始化为 1.0
    sensitivity = [1.0] * 6
    # 衰减系数 (每 100 步)
    decay = [math.exp(-100.0 / t) for t in TAU]

    # 按调度步 (每 100 步) 模拟
    sched_steps = total_steps // 100
    # 事件索引按 step 排序
    event_idx = 0
    sorted_events = sorted(parsed_events, key=lambda e: e["step"])

    # 记录峰值和谷值
    peak = list(conc)
    trough = list(conc)

    # 事件触发记录
    event_log = []

    for sched in range(sched_steps):
        current_step = sched * 100

        # 基线信号 (与 launch_modulatory 中的基线注入一致)
        signal = list(BASELINE)

        # 派发到期事件
        while event_idx < len(sorted_events) and sorted_events[event_idx]["step"] <= current_step:
            evt = sorted_events[event_idx]
            base_delta = GENE_MAP_BASE[evt["event_type"]]
            delta = apply_modifiers(base_delta, evt["modifier_flags"], evt["intensity"])
            # 叠加事件信号到基线信号
            for i in range(6):
                signal[i] = max(0.0, signal[i] + delta[i])
            event_log.append({
                "sched_step": sched,
                "step": evt["step"],
                "event_type": evt["event_type"],
                "intensity": evt["intensity"],
                "delta": [round(d, 4) for d in delta],
                "conc_before": [round(c, 4) for c in conc],
            })
            event_idx += 1

        # 应用受体灵敏度
        eff_signal = [signal[i] * sensitivity[i] for i in range(6)]

        # 浓度更新: conc = conc * decay + eff_signal
        for i in range(6):
            conc[i] = conc[i] * decay[i] + eff_signal[i]
            peak[i] = max(peak[i], conc[i])
            trough[i] = min(trough[i], conc[i])

        # 记录事件后浓度
        if event_log and event_log[-1]["sched_step"] == sched:
            event_log[-1]["conc_after"] = [round(c, 4) for c in conc]

        # 稳态补偿 (每 100 步, 与 launch_modulatory 同步)
        for i in range(6):
            excess = conc[i] - HOMEOSTATIC_BASELINE[i]
            if excess > 0:
                sensitivity[i] *= (1.0 - HOMEOSTATIC_RATE * excess)
            else:
                sensitivity[i] *= (1.0 + HOMEOSTATIC_UPREG_RATE * (-excess))
            sensitivity[i] = max(HOMEOSTATIC_CLAMP_MIN, min(HOMEOSTATIC_CLAMP_MAX, sensitivity[i]))

        if verbose and (sched % 10 == 0 or (event_log and event_log[-1]["sched_step"] == sched)):
            conc_str = " ".join(f"{CHANNEL_NAMES[i]}={conc[i]:.4f}" for i in range(6))
            sens_str = " ".join(f"{CHANNEL_NAMES[i]}={sensitivity[i]:.3f}" for i in range(6))
            print(f"    step={current_step:6d}  {conc_str}  sens=[{sens_str}]")

    return conc, peak, trough, sensitivity, event_log


def validate_dynamics(parsed_events, total_steps, verbose=False):
    """验证动力学稳定性, 返回 (all_pass, results)。"""
    print("\n[2] 6 维调质动力学模拟...")

    conc, peak, trough, sensitivity, event_log = simulate_dynamics(
        parsed_events, total_steps, verbose
    )

    checks = []

    # 检查 1: 浓度范围 [0, 2]
    for i in range(6):
        ok = 0.0 <= trough[i] and peak[i] <= 2.0
        checks.append(("concentration_range", CHANNEL_NAMES[i], ok,
                       f"peak={peak[i]:.4f} trough={trough[i]:.4f}"))

    # 检查 2: PAD 有界 (简化检查: DA/5HT/GABA 不极端)
    # P = +DA - 5HT_low - GABA_low; A = +NE - GABA - 5HT; D = +DA - Oxy
    pleasure = conc[0] - (0.2 - conc[3]) - (0.2 - conc[4])  # DA - 5HT_deficit - GABA_deficit
    arousal = conc[2] - conc[4] - conc[3]
    dominance = conc[0] - conc[5]
    for name, val in [("Pleasure", pleasure), ("Arousal", arousal), ("Dominance", dominance)]:
        ok = -2.0 <= val <= 2.0
        checks.append(("pad_bounded", name, ok, f"value={val:.4f}"))

    # 检查 3: 受体灵敏度 ∈ [0.3, 1.0]
    for i in range(6):
        ok = HOMEOSTATIC_CLAMP_MIN <= sensitivity[i] <= HOMEOSTATIC_CLAMP_MAX
        checks.append(("receptor_sensitivity", CHANNEL_NAMES[i], ok,
                       f"sensitivity={sensitivity[i]:.4f}"))

    # 检查 4: 事件覆盖率 (scenario 模式期望覆盖多种类型)
    event_types_covered = set(e["event_type"] for e in parsed_events)
    coverage_ratio = len(event_types_covered) / len(GENE_MAP_BASE)
    checks.append(("event_coverage", "all_types", coverage_ratio >= 0.5,
                   f"{len(event_types_covered)}/{len(GENE_MAP_BASE)} types covered"))

    # 检查 5: 最终浓度回归基线 (恢复能力)
    # 仅当最后事件后有足够恢复窗口 (≥10 个调度间隔 = 1000 步) 时才检查
    # 否则跳过 (短测试中最后事件离结束太近, 浓度来不及衰减)
    last_event_step = max(e["step"] for e in parsed_events) if parsed_events else 0
    recovery_window = total_steps - last_event_step
    has_recovery_window = recovery_window >= 1000

    for i in range(6):
        if has_recovery_window:
            # 慢调质 (5HT, Oxy) 允许 3x 基线, 快调质允许 2x
            threshold = 3.0 if TAU[i] >= 300 else 2.0
            ok = conc[i] <= BASELINE[i] * threshold
            checks.append(("recovery", CHANNEL_NAMES[i], ok,
                           f"final={conc[i]:.4f} baseline={BASELINE[i]:.4f} threshold={BASELINE[i]*threshold:.4f}"))
        else:
            # 无足够恢复窗口: 检查浓度是否已开始衰减 (final < peak * 0.9)
            ok = conc[i] < peak[i] * 0.95
            checks.append(("recovery_decaying", CHANNEL_NAMES[i], ok,
                           f"final={conc[i]:.4f} peak={peak[i]:.4f} (window={recovery_window}步, 跳过基线恢复检查)"))

    # 打印结果
    pass_count = sum(1 for _, _, ok, _ in checks if ok)
    fail_count = len(checks) - pass_count

    print(f"  最终浓度: {' '.join(f'{CHANNEL_NAMES[i]}={conc[i]:.4f}' for i in range(6))}")
    print(f"  峰值:     {' '.join(f'{CHANNEL_NAMES[i]}={peak[i]:.4f}' for i in range(6))}")
    print(f"  受体灵敏度: {' '.join(f'{CHANNEL_NAMES[i]}={sensitivity[i]:.4f}' for i in range(6))}")
    print(f"  PAD:      P={pleasure:.4f} A={arousal:.4f} D={dominance:.4f}")

    if fail_count > 0:
        print(f"\n  失败项 ({fail_count}):")
        for check_name, channel, ok, detail in checks:
            if not ok:
                print(f"    [FAIL] {check_name}/{channel}: {detail}")
    else:
        print(f"  全部 {len(checks)} 项检查通过")

    # 打印事件日志
    if verbose and event_log:
        print(f"\n  事件触发日志 ({len(event_log)} 个事件):")
        for log in event_log:
            delta_str = " ".join(f"{CHANNEL_NAMES[i]}={log['delta'][i]:+.4f}" for i in range(6))
            print(f"    step={log['step']:6d} {log['event_type']:16s} int={log['intensity']:+3d}  delta=[{delta_str}]")

    return fail_count == 0, checks


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3a-C1 事件驱动调质注入验证脚本"
    )
    parser.add_argument("--events", type=str, required=True,
                        help="events.jsonl 文件路径")
    parser.add_argument("--steps", type=int, default=5000,
                        help="总训练步数 (默认 5000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出 (每 1000 步打印浓度 + 事件日志)")
    args = parser.parse_args()

    if not os.path.exists(args.events):
        print(f"[ERROR] 事件文件不存在: {args.events}", file=sys.stderr)
        return 1

    print(f"=" * 60)
    print(f"Phase 3a-C1 事件驱动调质注入验证")
    print(f"=" * 60)
    print(f"  事件文件: {args.events}")
    print(f"  总步数:   {args.steps}")

    # 加载事件
    events = load_events(args.events)
    print(f"\n[1] JSONL 格式验证: 加载 {len(events)} 行")

    # 格式验证
    fmt_pass, fmt_fail, parsed = validate_format(events, args.steps)
    print(f"  格式验证: {fmt_pass} PASS, {fmt_fail} FAIL")

    if fmt_fail > 0:
        print(f"\n[RESULT] 格式验证失败, 跳过动力学模拟")
        return 1

    if not parsed:
        print(f"\n[RESULT] 无有效事件, 跳过动力学模拟")
        return 1

    # 动力学验证
    dyn_ok, checks = validate_dynamics(parsed, args.steps, args.verbose)

    # 总结
    print(f"\n{'=' * 60}")
    if dyn_ok:
        print(f"[RESULT] ALL PASS — 事件驱动调质注入验证通过")
        return 0
    else:
        fail_count = sum(1 for _, _, ok, _ in checks if not ok)
        print(f"[RESULT] {fail_count} FAIL — 事件驱动调质注入验证未通过")
        return 1


if __name__ == "__main__":
    sys.exit(main())
