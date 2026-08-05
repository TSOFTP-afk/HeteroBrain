"""模拟器模块：复刻 C++ mod_simulator.h CurriculumModSimulator + GENE_MAP 事件映射。

与 src/snn/mod_simulator.h 及 modulatory_kernels.cu 的确定性注入路径逐条对齐：
  - 灵敏度稳态更新   (HOMEOSTATIC_RATE / HOMEOSTATIC_UPREG_RATE)
  - 事件增量 clamp   (单事件 [-1,1], 累加 [-1.5,1.5])
  - 非线性交互       (DA-5HT 拮抗 / NE→GABA 抑制 / Oxy 放大 DA)
  - 衰减+注入+clamp  (conc × exp(-100/tau) + signal, clamp [0,2])
事件库从 kb/events.json 加载 (与 C++ gene_event_map.h 数值一致)。
"""
import json
import math
import os
from typing import Dict, List, Tuple

# 通道顺序 [DA, ACh, NE, 5HT, GABA, Oxy] (GENE_MAP 列顺序)
MOD_TAU = [100.0, 200.0, 150.0, 300.0, 120.0, 500.0]
HOMEOSTATIC_BASELINE = [0.15, 0.25, 0.20, 0.20, 0.25, 0.15]
HOMEOSTATIC_RATE = 0.002
HOMEOSTATIC_UPREG_RATE = 0.001
SENS_MIN, SENS_MAX = 0.3, 1.0

_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kb")


def load_event_kb(path: str = None) -> Dict:
    """加载事件原子库 (kb/events.json)"""
    p = path or os.path.join(_KB_DIR, "events.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def apply_modifiers(base: List[float], intensity: int,
                    publicity: str, authority: str, temporal: str) -> List[float]:
    """与 C++ apply_modifiers 一致: intensity 缩放 + 修饰符"""
    result = [v for v in base]
    scale = max(0.05, 1.0 + intensity * 0.02)
    result = [v * scale for v in result]
    if publicity == "public":
        result[5] *= 1.5   # Oxy
        result[2] *= 1.2   # NE
    if authority == "authority":
        result[0] *= 1.3   # DA
        result[3] *= 1.2   # 5HT
    return result


class EventKB:
    """事件知识库: 类型 → 调质增量 + 修饰符 (对齐 C++ gene_event_map.h)"""

    def __init__(self, data: Dict = None):
        self.data = data or load_event_kb()
        self.order: List[str] = self.data["gene_map_order"]
        self.events: Dict = self.data["events"]
        self._validate()

    def _validate(self):
        for name, e in self.events.items():
            assert len(e["delta"]) == 6, f"{name} delta 必须 6 维"
            assert e["publicity"] in ("public", "private")
            assert e["authority"] in ("authority", "peer")
            assert e["temporal"] in ("momentary", "sustained")

    def has(self, event_type: str) -> bool:
        return event_type in self.events

    def event_delta(self, event_type: str, intensity: int) -> List[float]:
        """单事件 → 6 维调质增量 (GENE_MAP 列顺序)"""
        e = self.events[event_type]
        return apply_modifiers(e["delta"], intensity,
                               e["publicity"], e["authority"], e["temporal"])


class ConcentrationSimulator:
    """复刻 C++ mod_simulator.h CurriculumModSimulator (每 100 步 advance_block)"""

    def __init__(self, event_kb: EventKB):
        self.kb = event_kb
        self.conc = [0.0] * 6
        self.sensitivity = [1.0] * 6

    def reset(self):
        self.conc = [0.0] * 6
        self.sensitivity = [1.0] * 6

    def advance_block(self, events: List[Tuple[str, int]], base_signal: List[float]):
        """推进一个 100 步块 (与 launch_modulatory 每 100 步调用节奏一致)"""
        # 1. 灵敏度稳态更新
        for ch in range(6):
            excess = self.conc[ch] - HOMEOSTATIC_BASELINE[ch]
            if excess > 0.0:
                self.sensitivity[ch] *= (1.0 - HOMEOSTATIC_RATE * excess)
            else:
                self.sensitivity[ch] *= (1.0 + HOMEOSTATIC_UPREG_RATE * (-excess))
            self.sensitivity[ch] = max(SENS_MIN, min(SENS_MAX, self.sensitivity[ch]))
        # 2. 事件增量 (单事件 clamp [-1,1], 累加 clamp [-1.5,1.5])
        eff_event = [0.0] * 6
        for (etype, intensity) in events:
            delta = self.kb.event_delta(etype, intensity)
            for ch in range(6):
                eff_event[ch] += max(-1.0, min(1.0, delta[ch]))
        for ch in range(6):
            eff_event[ch] = max(-1.5, min(1.5, eff_event[ch]))
        # 3. 非线性交互 (复刻 modulatory_kernels.cu L515-545)
        if any(abs(v) > 1e-6 for v in eff_event):
            da, ach, ne, ht5, gaba, oxy = eff_event
            if da > 0.0 and ht5 > 0.0:   # DA-5HT 拮抗 (同号时)
                ant = 0.2 * min(da, ht5)
                da -= ant; ht5 -= ant
            if ne > 0.0 and gaba > 0.0:  # NE 抑制 GABA
                gaba = max(0.0, gaba - 0.3 * ne * gaba)
            if oxy > 0.0 and da > 0.0:   # Oxy 放大 DA 奖赏
                da *= (1.0 + 0.5 * oxy)
            eff_event = [da, ach, ne, ht5, gaba, oxy]
        # 4. 注入 (灵敏度 + 衰减 + clamp)
        for ch in range(6):
            signal = (base_signal[ch] + eff_event[ch]) * self.sensitivity[ch]
            self.conc[ch] = self.conc[ch] * math.exp(-100.0 / MOD_TAU[ch]) + signal
            self.conc[ch] = max(0.0, min(2.0, self.conc[ch]))


def target_pad_from_conc(conc: List[float]) -> List[float]:
    """统一 PAD 公式 (与 C++ pad_from_concentration 一致):
    P = DA - 0.5*5HT - 0.3*GABA;  A = NE - 0.4*GABA - 0.3*5HT;  D = DA - 0.5*Oxy"""
    da, ach, ne, ht5, gaba, oxy = conc
    p = da - 0.5 * ht5 - 0.3 * gaba
    a = ne - 0.4 * gaba - 0.3 * ht5
    d = da - 0.5 * oxy
    return [round(max(-1.0, min(1.0, v)), 3) for v in (p, a, d)]
