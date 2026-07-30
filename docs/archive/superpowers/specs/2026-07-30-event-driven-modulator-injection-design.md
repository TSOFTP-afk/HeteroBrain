# Phase 3a-C1 设计：事件驱动调质注入（端到端最小闭环）

> **状态**：设计已确认，待写实施计划
> **创建**：2026-07-30
> **方向文档**：[docs/snn-emotion-and-workspace-direction.md §3.4](../../snn-emotion-and-workspace-direction.md#L148)
> **范围**：Phase 3a-C1（3 阶段垂直切片的第 1 阶段）

---

## 1. 背景与动机

### 1.1 问题诊断

5K 步合成输入测试显示 DA=0.736、ACh=0.932、Pleasure=+0.482 —— 表面稳定，但这些数值的成因是 spike 统计恰好落入某个区间，**不是因为"吃到好吃的"或"解出题了"**。当前 `launch_modulatory` 的所有信号都来自 SNN 内部状态（spike/ TD error/ 解码误差），导致情绪是"网络自己跟自己玩"的产物，没有语义锚点。

### 1.2 核心洞察

情绪是事件在人身上的反馈，不是数据流的副产品。事件是离散脉冲，SNN 是连续系统 —— 事件触发调质浓度阶跃，然后按 tau 衰减，这正是现有 `conc[t+1] = conc[t]*decay + signal[t]` 方程的天然用法。

### 1.3 三阶段路径（本设计仅覆盖 C1）

| 阶段 | 内容 | 依赖 |
|------|------|------|
| **C1（本期）** | 端到端最小闭环：事件流→基因映射→set_event_signal→SNN→PAD | 无 |
| C2 | 三维度情境调制（调质状态×事件历史×时间节律）+ LLM 识别完整版 | C1 |
| C3 | 记忆交互闭环（海马检索+工作空间评估）| C1 + Phase 3b |

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 3a-C1 数据通路                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  events.jsonl ──→ EventScheduler ──→ set_event_signal[6]   │
│       ↑              │                    │                  │
│       │              │                    ↓                  │
│  离线生成         每 100 步        launch_modulatory         │
│  (本期)          检查时间戳        读取+清零+注入             │
│                                      │                       │
│                                      ↓                       │
│                            SNN 6 维调质浓度演化              │
│                                      │                       │
│                                      ↓                       │
│                         AffectiveState (PAD + LLM 调制)      │
│                                      │                       │
│                                      ↓                       │
│                    CSV 日志 (情绪轨迹可视化)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**设计要点**：
- C1 先用**离线生成的事件流**（events.jsonl），LLM 识别层作为 C2 任务（C1 留可插拔接口）
- 事件注入与现有 `set_empathy_signal` 模式同构（host 端设置→launch_modulatory 读取后清零）
- 事件注入走调质通道，不触碰输入电流（`d_input_current`），与字节/BPE 输入完全正交

---

## 3. 双层事件类型枚举

### 3.1 10 个主类型（按进化意义分类）

| 主类型 | 进化来源 | 默认效价 | 生物学依据 |
|--------|---------|---------|-----------|
| `food_tasty` | 高糖/高脂食物 | 正 | DA 释放（伏隔核），进化奖赏 |
| `food_bland` | 低营养食物 | 中 | DA 微弱释放 |
| `threat_physical` | 捕食者/受伤 | 负 | 5HT↑+NE↑（杏仁核），应激 |
| `threat_social` | 社交地位威胁 | 负 | 5HT↑+NE↑，社交应激 |
| `praise` | 社交接纳信号 | 正 | DA↑+Oxy↑，归属感 |
| `criticism` | 社交排斥信号 | 负 | 5HT↑+DA↓，社交疼痛 |
| `social_bond` | 亲密关系强化 | 正 | Oxy↑为主，依恋 |
| `social_loss` | 亲密关系丧失 | 负 | 5HT↑+Oxy↓，哀伤 |
| `achievement` | 打猎/任务成功 | 正 | DA↑（强烈），目标达成 |
| `novelty` | 新环境探索 | 中 | ACh↑+DA↑，惊奇 |

### 3.2 4 个修饰符维度

| 修饰符 | 取值 | 影响 |
|--------|------|------|
| `social_publicity` | private(0) / public(1) | public 放大 Oxy×1.5 + NE×1.2 |
| `authority_gap` | peer(0) / authority(1) | authority 放大 DA×1.3 + 5HT×1.2 |
| `temporal_extent` | momentary(0) / sustained(1) | sustained 拉长 duration×3 |
| `intensity` | -50..+50 | 连续强度调节（复用现有系统） |

**intensity 调制公式**：`final_delta = base_delta * max(0.05, 1.0 + intensity * 0.02)`

---

## 4. 基因硬编码映射表

### 4.1 GENE_MAP_BASE

intensity=0、修饰符全默认时的基准 6 维调质增量：

| 主类型 | DA | ACh | NE | 5HT | GABA | Oxy | duration_s |
|--------|-----|-----|-----|-----|------|-----|------------|
| food_tasty | +0.40 | +0.10 | +0.05 | -0.05 | 0.00 | +0.02 | 0.5 |
| food_bland | +0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.2 |
| threat_physical | -0.20 | +0.30 | +0.60 | +0.40 | +0.10 | -0.05 | 2.0 |
| threat_social | -0.15 | +0.20 | +0.45 | +0.35 | +0.05 | -0.10 | 3.0 |
| praise | +0.25 | +0.10 | +0.15 | -0.05 | 0.00 | +0.20 | 1.0 |
| criticism | -0.10 | +0.05 | +0.20 | +0.25 | 0.00 | -0.15 | 2.0 |
| social_bond | +0.10 | +0.05 | -0.05 | +0.05 | +0.05 | +0.35 | 5.0 |
| social_loss | -0.15 | 0.00 | +0.10 | +0.30 | 0.00 | -0.25 | 8.0 |
| achievement | +0.50 | +0.15 | +0.20 | -0.10 | 0.00 | +0.05 | 1.5 |
| novelty | +0.15 | +0.40 | +0.10 | 0.00 | 0.00 | 0.00 | 2.0 |

**注**：负值表示该调质浓度降低（如 food_tasty 的 5HT=-0.05 表示满足感略降 5HT）。实际注入时 `delta = max(0, base + signal)` 保证浓度不出现负值（与 CUDA clamp 一致）。

### 4.2 修饰符调制规则

```cpp
GeneMapEntry apply_modifiers(GeneMapEntry base, int modifier_flags, int intensity) {
    GeneMapEntry result = base;
    // intensity 调制
    float scale = std::max(0.05f, 1.0f + intensity * 0.02f);
    result.da_delta  *= scale;
    result.ach_delta *= scale;
    result.ne_delta  *= scale;
    result.ht5_delta *= scale;
    result.gaba_delta*= scale;
    result.oxy_delta *= scale;
    // publicity=public
    if (modifier_flags & MOD_PUBLIC) {
        result.oxy_delta *= 1.5f;
        result.ne_delta  *= 1.2f;
    }
    // authority=authority
    if (modifier_flags & MOD_AUTHORITY) {
        result.da_delta  *= 1.3f;
        result.ht5_delta *= 1.2f;
    }
    // temporal=sustained
    if (modifier_flags & MOD_SUSTAINED) {
        result.duration_s *= 3.0f;
    }
    return result;
}
```

### 4.3 生物学依据

- **food_tasty → DA↑**：伏隔核 DA 释放是进化奖赏的核心（Schultz 1997）
- **threat_physical → 5HT↑+NE↑**：杏仁核激活 5HT/NE 应激通路（LeDoux 2000）
- **praise → Oxy↑**：社交接纳触发催产素释放（Kosfeld 2005）
- **social_loss → Oxy↓**：哀伤伴随催产素系统下调（Panksepp 1998）
- **achievement → DA↑(强)**：目标达成是 DA 强烈释放的典型场景（Wolfram 2016）

---

## 5. set_event_signal API

### 5.1 接口设计

```cpp
// modulatory_kernels.cuh (修改)

// 设置外部事件驱动的 6 维调质增量 (host 端, 由事件调度器触发)
//   modulator_delta[6]: [DA, ACh, NE, 5HT, GABA, Oxy] 增量
//   duration_steps: 事件持续步数 (用于 plateau 型事件, 0=单次脉冲)
//   内部缓存, 由 launch_modulatory 读取后清零 (单次触发模型)
//   与 set_empathy_signal 兼容: set_event_signal 优先, empathy 作为 Oxy 通道 fallback
//   C1 行为: 仅支持 pulse 型 (duration_steps=0), plateau 型留 C2
void set_event_signal(const float modulator_delta[6], int duration_steps);
```

### 5.2 实现逻辑

```cpp
// modulatory_kernels.cu (修改)

// Phase 3a-C1: 6 维事件信号缓存 (扩展自 h_empathy_signal 模式)
static float h_event_signal[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
static int   h_event_duration_steps = 0;  // 剩余持续步数

void set_event_signal(const float modulator_delta[6], int duration_steps) {
    for (int i = 0; i < 6; ++i) {
        h_event_signal[i] = clamp(modulator_delta[i], -1.0f, 1.0f);
    }
    h_event_duration_steps = max(0, duration_steps);
}

// 在 launch_modulatory 内部读取 (替代当前 empathy 读取逻辑):
//   优先级: 显式参数 > h_event_signal > h_empathy_signal
float eff_event[6] = {0};
bool has_event = false;
for (int i = 0; i < 6; ++i) {
    if (fabsf(h_event_signal[i]) > 1e-6f) {
        eff_event[i] = h_event_signal[i];
        has_event = true;
    }
}
if (has_event) {
    // 叠加事件信号到各通道
    da_signal       += eff_event[0];
    ach_signal      += eff_event[1];
    ne_signal       += eff_event[2];
    ht5_signal      += eff_event[3];
    gaba_signal     += eff_event[4];
    oxytocin_signal += eff_event[5];
    // 单次触发: 清零缓存 (除非 duration > 0, 保留 for plateau)
    if (h_event_duration_steps <= 0) {
        for (int i = 0; i < 6; ++i) h_event_signal[i] = 0.0f;
    } else {
        h_event_duration_steps -= 100;  // 每 100 步递减
    }
} else {
    // fallback 到 empathy 信号 (保留 Phase 3a 兼容)
    float eff_empathy = (empathy_signal > 0.0f) ? empathy_signal : h_empathy_signal;
    oxytocin_signal = OXYTOCIN_BASE + OXYTOCIN_GAIN * eff_empathy;
    h_empathy_signal = 0.0f;
}
```

### 5.3 与稳态补偿的交互

事件信号叠加后，仍会经过受体灵敏度 `h_receptor_sensitivity[6]` 衰减（line 412-414）。这是期望行为：防止事件驱动导致病理滑移。但需验证稳态补偿不会完全阻断事件响应（灵敏度不归零，保底 30%）。

---

## 6. 事件调度器

### 6.1 EventScheduler 类

```cpp
// event_scheduler.h (新增)
#pragma once
#include <string>
#include <vector>

namespace stage2e {

enum EventType {
    EVT_FOOD_TASTY = 0,
    EVT_FOOD_BLAND,
    EVT_THREAT_PHYSICAL,
    EVT_THREAT_SOCIAL,
    EVT_PRAISE,
    EVT_CRITICISM,
    EVT_SOCIAL_BOND,
    EVT_SOCIAL_LOSS,
    EVT_ACHIEVEMENT,
    EVT_NOVELTY,
    EVT_COUNT
};

enum EventModifier {
    MOD_PRIVATE   = 0,
    MOD_PUBLIC    = 1 << 0,
    MOD_PEER      = 0,
    MOD_AUTHORITY = 1 << 1,
    MOD_MOMENTARY = 0,
    MOD_SUSTAINED = 1 << 2,
};

struct ScheduledEvent {
    int step_target;        // 触发 step
    int event_type;         // EventType 枚举
    int modifier_flags;     // EventModifier 位域
    int intensity;          // -50..+50
    float duration_s;       // 持续时间 (秒, 0=用 GENE_MAP 默认)
    std::string description; // 人类可读描述
};

class EventScheduler {
public:
    bool load_jsonl(const std::string& path);
    // 每 100 步调用: 派发所有 step_target <= current_step 的事件
    // 对每个事件: 查 GENE_MAP → apply_modifiers → set_event_signal
    void dispatch_pending(int current_step);
    size_t total_events() const { return events_.size(); }
    size_t dispatched_count() const { return next_event_idx_; }
private:
    std::vector<ScheduledEvent> events_;
    size_t next_event_idx_ = 0;
};

} // namespace stage2e
```

### 6.2 dispatch_pending 逻辑

```cpp
void EventScheduler::dispatch_pending(int current_step) {
    while (next_event_idx_ < events_.size() &&
           events_[next_event_idx_].step_target <= current_step) {
        const auto& evt = events_[next_event_idx_];
        // 1. 查 GENE_MAP_BASE
        GeneMapEntry entry = GENE_MAP_BASE[evt.event_type];
        // 2. 应用修饰符
        entry = apply_modifiers(entry, evt.modifier_flags, evt.intensity);
        // 3. 覆盖 duration (如果事件指定了)
        if (evt.duration_s > 0.0f) entry.duration_s = evt.duration_s;
        // 4. 转换为 6 维增量
        float delta[6] = {entry.da_delta, entry.ach_delta, entry.ne_delta,
                          entry.ht5_delta, entry.gaba_delta, entry.oxy_delta};
        // 5. 转换 duration_s 为步数 (假设 1 step = 1 ms)
        int duration_steps = (int)(entry.duration_s * 1000.0f);
        // 6. 注入
        set_event_signal(delta, duration_steps);
        next_event_idx_++;
    }
}
```

---

## 7. events.jsonl 格式

### 7.1 单事件结构

```json
{
  "event_id": 1,
  "step_target": 1000,
  "time_s": 10.0,
  "event_type": "food_tasty",
  "modifiers": {
    "publicity": "private",
    "authority": "peer",
    "temporal": "momentary"
  },
  "intensity": 30,
  "duration_s": 0.5,
  "description": "吃到一块巧克力"
}
```

### 7.2 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `event_id` | int | 是 | 事件唯一 ID |
| `step_target` | int | 是 | 触发 step（与 time_s 二选一） |
| `time_s` | float | 否 | 触发时间秒（若提供则自动算 step_target = time_s × 100，假设 1 step = 1 ms；事件调度每 100 步一次，时间分辨率 100ms） |
| `event_type` | string | 是 | 主类型枚举名 |
| `modifiers` | object | 否 | 修饰符，缺省全为默认(private/peer/momentary) |
| `intensity` | int | 否 | -50..+50，缺省 0 |
| `duration_s` | float | 否 | 持续时间，缺省用 GENE_MAP 默认 |
| `description` | string | 否 | 人类可读描述 |

### 7.3 示例事件序列（5K 步测试集）

```jsonl
{"event_id":1,"step_target":500,"event_type":"food_tasty","intensity":30,"description":"吃到巧克力"}
{"event_id":2,"step_target":1000,"event_type":"praise","modifiers":{"publicity":"public","authority":"authority"},"intensity":20,"description":"公开被上司表扬"}
{"event_id":3,"step_target":1500,"event_type":"threat_physical","intensity":-30,"description":"差点被车撞"}
{"event_id":4,"step_target":2000,"event_type":"achievement","intensity":40,"description":"完成项目"}
{"event_id":5,"step_target":2500,"event_type":"social_loss","intensity":20,"description":"朋友搬走"}
{"event_id":6,"step_target":3000,"event_type":"social_bond","modifiers":{"temporal":"sustained"},"intensity":10,"description":"与家人长时间相处"}
{"event_id":7,"step_target":3500,"event_type":"criticism","modifiers":{"publicity":"public"},"intensity":15,"description":"公开被批评"}
{"event_id":8,"step_target":4000,"event_type":"novelty","intensity":25,"description":"探索新环境"}
{"event_id":9,"step_target":4500,"event_type":"food_bland","intensity":-10,"description":"吃无味食物"}
```

---

## 8. 集成点

### 8.1 RunConfig 新增参数

```cpp
// run_config.h (修改)
struct RunConfig {
    // ... 现有字段 ...
    bool event_stream_enabled = false;
    std::string event_stream_path;  // --event-stream PATH
};
```

### 8.2 main.cpp 集成

```cpp
// main.cpp (修改)
// 启动时加载事件流
EventScheduler event_scheduler;
if (config.event_stream_enabled) {
    if (!event_scheduler.load_jsonl(config.event_stream_path)) {
        std::cerr << "[ERROR] failed to load event stream: " << config.event_stream_path << "\n";
        return 1;
    }
    std::cout << "[INFO] loaded " << event_scheduler.total_events() << " events\n";
}

// 主循环内 (line 801-922), 每 100 步派发事件
for (int step = start_step; step < total_steps; ++step) {
    // ... 现有逻辑 ...
    if (step % 100 == 0 && config.event_stream_enabled) {
        event_scheduler.dispatch_pending(step);
    }
    scheduler.step(step);
    // ... 现有逻辑 ...
}
```

### 8.3 与睡眠态的交互

事件注入应检查 `!is_sleeping_`（与 modulatory 一致）。睡眠态期间事件累积，待唤醒后批量注入。具体实现：`dispatch_pending` 在睡眠态时不调用，事件保留在队列中。

### 8.4 与现有输入模式的兼容矩阵

| 输入模式 | 事件流 | 行为 |
|---------|--------|------|
| `--synthetic-input` | 无 | 现状，纯烟雾测试 |
| `--synthetic-input` + `--event-stream` | 有 | 合成字节流 + 真实事件调质（推荐用于 C1 验证） |
| `--input-mode bpe` | 无 | 现状，BPE 训练 |
| `--input-mode bpe` + `--event-stream` | 有 | BPE 训练 + 事件调质叠加（生产模式） |

---

## 9. 文件改动清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/snn/event_types.h` | 新增 | EventType 枚举 + EventModifier 枚举 |
| `src/snn/gene_event_map.h` | 新增 | GeneMapEntry 结构 + GENE_MAP_BASE[10] + apply_modifiers() |
| `src/snn/event_scheduler.h` | 新增 | ScheduledEvent + EventScheduler 类声明 |
| `src/snn/event_scheduler.cpp` | 新增 | load_jsonl + dispatch_pending 实现 |
| `src/snn/modulatory_kernels.cuh` | 修改 | set_event_signal 声明 |
| `src/snn/modulatory_kernels.cu` | 修改 | h_event_signal[6] + set_event_signal + launch_modulatory 读取逻辑 |
| `src/snn/run_config.h` | 修改 | event_stream_enabled + event_stream_path 字段 |
| `src/snn/run_config.cpp` | 修改 | --event-stream 参数解析 |
| `src/snn/main.cpp` | 修改 | 加载事件流 + 调度器集成 |
| `src/snn/CMakeLists.txt` | 修改 | 添加 event_scheduler.cpp 到 SNN_TRAIN_SRCS |
| `src/snn/tools/generate_event_dataset.py` | 新增 | 生成 events.jsonl |
| `src/snn/tools/validate_event_driven.py` | 新增 | 5K 步验证脚本 |

---

## 10. 验证准则

5K 步测试需通过以下 5 项准则：

1. **事件响应可观测**：事件注入后 6 维调质浓度有 >5% 基线偏移
2. **PAD 方向正确**：正性事件（food_tasty/praise/achievement）Pleasure↑；负性事件（threat/criticism/social_loss）Pleasure↓
3. **衰减动力学**：事件间浓度按 tau 衰减，无事件时回落到基线 ±10%
4. **稳态补偿不阻断**：受体灵敏度在事件驱动响应后不归零（>0.3）
5. **LLM 调制信号波动**：temperature_delta 随 DA 变化（相关系数 >0.3）

---

## 11. C2/C3 大纲（后续阶段）

### C2: 三维度情境调制 + LLM 识别完整版
- 调质状态调制：DA 已高时同类事件增量衰减（边际效用）
- 事件历史调制：近期同类事件密度高时增量衰减（习惯化）
- 时间节律调制：昼夜/疲劳因子影响基线
- LLM 事件识别：MiniCPM5-1B 作为事件识别器，文本→EventType+修饰符+intensity

### C3: 记忆交互闭环（与 Phase 3b 合并）
- 事件→海马检索 top-k 类似记忆（PCA 签名相似度）
- 检索结果→工作空间评估事件意义
- 评估结果→调质强度调制
- 事件→写入海马带 emotion_tag

---

## 12. 风险与诚实评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 基因映射表数值不准 | 中 | 情绪反应方向错 | 用神经科学文献校准，5K 步验证方向性 |
| 事件信号被稳态补偿过度抑制 | 低 | 事件无响应 | 验证准则 4 检测，必要时调整 HOMEOSTATIC_RATE |
| duration_steps 实现复杂 | 中 | plateau 事件不工作 | C1 先只支持 pulse 型（duration=0），plateau 留 C2 |
| LLM 识别准确率不足 | C2 风险 | 事件类型错 | C1 用离线标注，C2 再上 LLM |
| 与睡眠态交互未充分测试 | 低 | 事件丢失 | C1 测试不涉及睡眠态，留后续 |

**诚实评估**：C1 是"事件→调质"通路的工程验证，不解决"理解"问题（事件识别是 C2 任务）。但 C1 能证明架构通路正确，为 C2/C3 打基础。基因映射表是人工设计的"出厂设置"，不代表真实人脑的精确映射，但方向性应正确。

---

## 附录 A: 神经科学依据汇总

| 事件类型 | 关键调质 | 文献依据 |
|---------|---------|---------|
| food_tasty → DA↑ | DA | Schultz 1997, 伏隔核 DA 释放 |
| threat_physical → 5HT↑+NE↑ | 5HT, NE | LeDoux 2000, 杏仁核应激通路 |
| praise → Oxy↑ | Oxy | Kosfeld 2005, 社交接纳催产素 |
| social_loss → Oxy↓ | Oxy | Panksepp 1998, 哀伤与催产素 |
| achievement → DA↑(强) | DA | Wolfram 2016, 目标达成 DA |
| novelty → ACh↑ | ACh | Hasselmo 2006, ACh 与注意/新奇 |

## 附录 B: 与方向文档 §3.4 的对应

本设计实现 §3.4 "实现路径" 的第 1-4 步：
1. ✅ 定义 EventType 枚举（10 主类型 × 4 修饰符）
2. ✅ 在 launch_modulatory 加 inject_event 接口（set_event_signal）
3. ✅ 定义基因硬编码映射表（GENE_MAP_BASE）
4. ✅ 主循环加事件调度器（EventScheduler）

第 5 步（扩展 generate_synthetic_grab_traces.py）改为独立 generate_event_dataset.py。
第 6 步（验证准则）见第 10 节。
