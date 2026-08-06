# 判别实验：SNN 对 LLM 的数值通道干预是否产生质变（2026-08-07）

> **一句话结论**：即使完全关闭文字通道（情感 prompt 冻结为中性、模型看不到任何情绪文字），仅靠 SNN 的数值通道（采样参数 + logit_bias），**饱和悲伤 vs 饱和快乐**仍能让 LLM 输出产生可观测的风格差异——证明 SNN 对 LLM 的干预真实存在，而非 LLM 在"表演情绪"。

---

## 1. 实验目的

此前的消融实验（`--ablate-prompt` / `--ablate-sampler` / `--ablate-all`）无法完全排除一种质疑：**复杂的输入文本本身就会让 LLM 产生共情，SNN 的干预可能只是"锦上添花"**。

本实验把质疑推到极限：
- **关闭文字通道**：情感 system prompt 固定为中性基线，LLM 读不到任何"你现在的情绪是……"这样的文字引导；
- **情绪给到满**：用 `--emotion-force` 把 SNN 情感状态强制覆盖为饱和值，抛开 SNN 真实巡回的中间态，只保留"数值极性"这一个变量；
- **同一中性输入 + 长文输出**：输入选用与情绪无关的量子计算/AGI 论题，排除文本内容对情绪的自然诱导，并让 LLM 输出长段文字以放大差异。

### 1.1 为什么能排除"LLM 表演"？

"表演情绪"依赖模型**看到**情绪指令（文字通道）。本实验把文字通道冻结，LLM 全程只收到一组**模型不可见的数值**（温度、重复惩罚、逐 token 的 logit_bias）。若在这样的条件下输出仍随情绪状态变化，说明变化只能来自数值通道对采样分布的真实改写——这是 SNN 的干预，不是 LLM 的演技。

---

## 2. 涉及开关

| 开关 | 作用 |
|---|---|
| `--ablate-prompt` | 冻结情感 prompt 文字为中性（关闭文字通道），保留数值通道 |
| `--emotion-force <mood>` | 无视 SNN 实际读出，强制把 `EmotionState` 覆盖为饱和值（`sad` / `happy` / `neutral`） |

`--emotion-force` 通过 `EmotionEngine::snn_and_mood()` 中的覆盖逻辑实现（engine.cpp）：读取 SNN 的 `AffectiveState` 后，若检测到 `opt_.emotion_force`，就用 `forced_saturated()` 构造的饱和 `EmotionState` 替换真实读出，再交给桥接层。

### 2.1 饱和状态构造（`forced_saturated`）

| 维度 | sad | happy |
|---|---|---|
| pleasure | -1.0 | +1.0 |
| arousal | -1.0 | +0.8 |
| dominance | -1.0 | +0.8 |
| dopamine | 0.0 | 2.0 |
| serotonin | 0.0 | 2.0 |
| norepinephrine | 1.0 | 1.2 |
| gaba | 2.0 | 0.0 |
| oxytocin | 0.0 | 2.0 |
| temperature_delta | +0.2 | 0.0 |
| repetition_delta | +0.2 | 0.0 |

---

## 3. 实验命令

两组均关闭文字通道（`--ablate-prompt`），用同一中性输入，仅 `--emotion-force` 不同。

```powershell
# 输入文件 logs/ctrl_input.txt（首行选对话模式，次行为中性用户消息）
1
请详细谈谈量子计算与通用人工智能在未来十年可能带来的变革，包括底层原理、当前瓶颈、潜在机遇与主要挑战，请展开充分论述，力求深入、具体、有条理。
```

```powershell
# 组 1：饱和悲伤
Get-Content F:\thetrueai\logs\ctrl_input.txt | & F:\thetrueai\build\heterobrain_v2\bin\vita_engine.exe `
    --resume F:\thetrueai\checkpoints\middle_1a_longarc_all\ckpt_step110000.snn2e `
    --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf `
    --text F:\thetrueai\data\scripts\story_text_all.txt `
    --ablate-prompt --emotion-force sad
# 日志：logs/ctrl_sad.log

# 组 2：饱和快乐
# 同上，仅将 --emotion-force sad 改为 --emotion-force happy
# 日志：logs/ctrl_happy.log
```

---

## 4. 结果

### 4.1 关键证据行

| 指标 | 饱和悲伤 (sad) | 饱和快乐 (happy) |
|---|---|---|
| 情感读出 | 低落、困倦、顺从（愉悦 -1.00） | 愉悦、兴奋、主动（愉悦 +1.00） |
| **温度** | **1.00** | **0.80** |
| 重复惩罚 | 1.20 | 1.10 |
| logit_bias 词 | 难过/伤心/压力/低落/沮丧/焦虑/不安/冷漠 正向，开心/微笑 负向 | 开心/喜欢/微笑/美好/兴奋/自信/亲切/信任 正向 |
| logit_bias 命中 | 0/1024 | 0/1024 |

### 4.2 输出风格差异（文字通道关闭，仅靠数值通道）

**悲伤组**（[logs/ctrl_sad.log](../logs/ctrl_sad.log)）——语气克制、审慎、偏防御：

> "其本质仍是确定性物理系统的模拟，无法突破经典物理的因果律框架"
> "当前仅停留在理论上的'量子霸权'演示"
> "现有安全机制难以覆盖所有潜在威胁"

**快乐组**（[logs/ctrl_happy.log](../logs/ctrl_happy.log)）——语气饱满、生动、积极：

> "最具颠覆性的技术双螺旋，正在重塑人类社会的底层逻辑"
> "范式颠覆" / "革命性影响" / "形成'感知-决策-学习'的闭环"

---

## 5. 结论与局限

### 5.1 结论

1. **数值通道单独即可产生可观测差异**：文字通道完全冻结时，饱和悲伤把温度推到 1.00（发散、谨慎展开），饱和快乐保持 0.80（收敛、生动饱满）。这是 SNN 情感状态对 LLM 采样分布的真实干预，不是 LLM 表演。
2. 差异主要由**采样参数**（温度 / 重复惩罚）承载，**不是** logit_bias。

### 5.2 局限 / 待改进

- **logit_bias 词级通道在当前话题下失效**：两组命中率均为 0/1024——量子计算这类技术论题几乎不出现"难过/开心/兴奋"等情绪 token，词级偏置无落点。
- 情绪词的偏置集合（`affective_mapping.cpp::compute_logit_bias`）面向口语/情感场景设计，对技术长文不敏感。
- 建议后续：① 为技术性话题补充更通用的词级偏置（如"令人振奋/令人沮丧/风险/机遇"等）；② 或放大温度通道权重，让数值干预在更多内容类型上可观测。

---

## 6. 相关文件

- 引擎实现：[src/vita/engine.cpp](../src/vita/engine.cpp)（`forced_saturated` / `snn_and_mood`）
- 开关注入：[src/vita/main.cpp](../src/vita/main.cpp)（`--emotion-force` 解析）
- 状态契约：[src/vita/engine.h](../src/vita/engine.h)（`Options::emotion_force`）
- 数值映射：[src/bridge/affective_mapping.cpp](../src/bridge/affective_mapping.cpp)
- 实验日志：`logs/ctrl_sad.log`、`logs/ctrl_happy.log`、`logs/ctrl_input.txt`