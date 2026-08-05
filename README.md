# VITA（维塔）— 人工情感核心 / VITA — Artificial Affective Core

> **SNN × LLM 异构认知架构**：SNN 作为**人工情感核心**（情感动力学 · 记忆检索 · 意图决策 · 调度器官），LLM 作为**语言皮层**（语言理解与表达）。
> **Heterogeneous cognitive architecture**: the SNN acts as the **artificial affective core** (affective dynamics · memory retrieval · intent decision · scheduling), while the LLM acts as the **language cortex** (language understanding and expression).
>
> 边缘可部署 · 中文原生 · 在线持续学习 · 无云端依赖 / Edge-deployable · Native Chinese · Online continual learning · No cloud dependency

---

## 项目定位 / Positioning

VITA 不是又一个 Transformer 大模型，也不是纯 SNN 研究项目。它是一台把"感受"与"表达"分开的**情感机器**：情感必须发生在 SNN 内部（神经元发放的动力学），语言表达交给 LLM。让 SNN 做它不可替代的事——**感受**（事件→情感动力学）、**记忆**（工作记忆/海马编码）、**决策**（意图 readout）与**调度**（调制 LLM 生成）。

VITA is not just another Transformer LLM, nor a pure SNN research project. It is an **affective machine** that separates *feeling* from *expression*: affect must emerge inside the SNN (spike-rate dynamics), while language expression is delegated to the LLM. The SNN does what it alone can — **feeling** (events → affective dynamics), **memory** (working memory / hippocampal encoding), **decision** (intent readout), and **scheduling** (modulating LLM generation).

| 子系统 / Subsystem | 职责 / Role | 实现 / Implementation |
|---|---|---|
| **SNN 情感核心** / Affective core | 情感动力学（6 维调质）、记忆（WM/海马）、意图、调度 / Affective dynamics (6-channel modulation), memory (WM/hippocampus), intent, scheduling | 自研 CUDA SNN：联合皮层 50K + 前额叶 5K + 运动皮层（60K 神经元 / 10.7M 突触 / 31 种生物机制） |
| **LLM 语言皮层** / Language cortex | 语言理解与生成、世界知识 / Language understanding, generation, world knowledge | llama.cpp + Qwen3-4B（INT4 GGUF，2.5GB）；MiniCPM5-1B 已弃用 |
| **Bridge 桥接层** / Bridge | 情感→调制信号（文字通道 + logit_bias + 采样参数）、对话→SNN 输入 / Affect → modulation signals; dialogue → SNN input | affective_mapping / emotion_bridge / snn_feedback |

**当前核心工程事实**（2026-08-06）：readout 根因已实锤——事件调制对联合皮层发放传导 <2.3%，情感目前只存在于外部浓度模拟器、尚未进入网络内部。修复方向是**事件直通联合皮层注入通道**（事件→固定子区域激活，与文本流并行），详见 [docs/bug-inventory-2026-08-04.md](file:///f:/thetrueai/docs/bug-inventory-2026-08-04.md)。

**Current engineering fact** (2026-08-06): the readout root cause is confirmed — event modulation propagates <2.3% into association-cortex firing; affect currently lives only in the external concentration simulator and never enters the network. The fix is a **direct event→association-cortex injection channel** (events activate fixed sub-regions, parallel to the text stream).

---

## 系统架构 / Architecture

```mermaid
flowchart TB
    subgraph In[输入 Input]
        D[对话文本 Dialogue<br/>append_text_stream]
        E[世界事件 World events<br/>浓度注入 + 皮层注入计划]
    end

    subgraph SNN[SNN 情感核心 / Affective Core<br/>CUDA 60K neurons]
        A[联合皮层 Association cortex<br/>50K 神经元群体编码]
        M[6 维调质浓度 6-ch modulation<br/>DA/ACh/NE/5HT/GABA/Oxy]
        R[readout 线性头<br/>rate → 浓度/PAD 预测]
        W[(WM 工作记忆 50 槽)]
        H[(海马记忆 PCA 签名)]
    end

    subgraph Bridge[桥接层 Bridge]
        AM[affective_mapping<br/>PAD → 情感文字/logit_bias/采样]
        FB[snn_feedback<br/>事件/共情/奖励回流]
    end

    subgraph LLM[LLM 语言皮层 / Language cortex]
        L[llama.cpp + Qwen3-4B]
    end

    subgraph Out[输出 Output]
        O[中文响应 Chinese response]
    end

    D --> A
    E --> M
    A --> R --> M
    A --> W --> H
    M --> AM --> L
    AM --> O
    O --> FB --> E
    FB -.-> M
```

### 数据流 / Data Flow

1. **对话输入**：用户对话文本经 `append_text_stream` 追加进文本流 → 群体编码注入感觉神经元（SNN"看见"对话字节）。
   **Dialogue input**: user text is appended to the text stream → population-coded injection into sensory neurons (the SNN "sees" dialogue bytes).
2. **事件注入**：世界事件 → 6 维调质浓度（浓度模拟器同源公式）→ 决定 STDP 三因子与监督目标；事件直通皮层注入为下一步计划。
   **Event injection**: world events → 6-channel modulation concentrations (same formula as the concentration simulator) → drives STDP three-factor learning and supervision targets; direct cortical injection is the next planned step.
3. **情感读出**：`get_affective_state` 读 6 维浓度 → PAD 情感模型（Pleasure/Arousal/Dominance）→ LLM 调制信号。
   **Affective readout**: 6-channel concentrations → PAD model → LLM modulation signals.
4. **LLM 生成**：情感文字（system）+ logit_bias（逐 token 干预采样）+ 采样参数调制 → llama.cpp 生成响应。
   **LLM generation**: affective text (system prompt) + logit_bias (token-level sampling intervention) + sampler modulation → response.
5. **记忆**：WM/海马以 PCA 签名编码网络状态，host 侧解码字节指纹注入 system（记忆内容回显）。
   **Memory**: WM/hippocampus encode network state as PCA signatures; host-side decoding injects byte fingerprints into the system prompt.
6. **回流**：对话/反馈 → `emit_event`/`emit_empathy`/`emit_embodied_reward` → 浓度与 STDP。
   **Feedback loop**: dialogue/feedback → event/empathy/reward emission → concentrations and STDP.

---

## 当前状态 / Current Status（2026-08-06）

| 项 / Item | 状态 / Status | 说明 / Notes |
|---|---|---|
| 异构引擎闭环 / Engine loop | ✅ | resume → SNN 推进 → Affective 读出 → 情感调制 → llama.cpp 生成（Qwen3-4B，思考模式保留） |
| OpenAI 兼容 serve / Serve mode | ✅ | `GET /v1/models` + `POST /v1/chat/completions`，Bearer 鉴权，SNN 每请求推进，情感跨请求演化 |
| SNN 记忆接入 / SNN memory | ✅ 一期 | WM/海马 PCA 签名 + host 解码字节指纹注入 system |
| logit_bias 通道 / Logit-bias channel | ✅ | SNN 逐 token 干预 LLM 采样分布 |
| 80K 重训 / 80K retrain | ✅ | N3F + `--curriculum-continuous`，decode PPL 137→19.8 |
| **readout 根因实锤** / Root cause confirmed | ⚠️ | 事件→皮层传导 <2.3%，情感未进网络内部，mod MSE 0.3519（判据 0.0339） |
| 事件→皮层注入通道 / Event→cortex channel | 📋 计划 | 事件直通联合皮层固定子区域，rate 携带事件信息（下一步） |

**当前训练配置**：N3F 在线学习 + `--curriculum-continuous` + `--bptt-window-size 400` + `--input-inject-interval 1`，checkpoint：`checkpoints/middle_1a_longarc_all/ckpt_step80000.snn2e`。

**Current training config**: N3F online learning + `--curriculum-continuous` + `--bptt-window-size 400` + `--input-inject-interval 1`; checkpoint: `checkpoints/middle_1a_longarc_all/ckpt_step80000.snn2e`.

---

## 快速开始 / Quick Start

### 1. 环境要求 / Requirements

| 组件 / Component | 版本 / Version | 用途 / Purpose |
|---|---|---|
| CUDA Toolkit | 13.x | SNN 子系统 / SNN subsystem |
| CMake | ≥ 3.18 | 构建 / Build |
| Python | 3.10+ | 数据生成与评测 / Data generation & evaluation |
| llama.cpp | 最新 / latest | LLM 推理 / LLM inference |

硬件：NVIDIA GPU（compute capability ≥ 8.6，6GB+ 显存）；模型文件置于 `F:\hb_models\`（如 `Qwen3-4B-Q4_K_M.gguf`）。
Hardware: NVIDIA GPU (CC ≥ 8.6, 6GB+ VRAM); model files in `F:\hb_models\` (e.g. `Qwen3-4B-Q4_K_M.gguf`).

### 2. 构建 / Build

> Windows 需先设置 MSVC 环境变量（VS DevShell）；`cmd /c` 被安全策略禁止，用 PowerShell。

```powershell
# SNN 训练子系统（snn_train）
cmake --build build/snn --target snn_train

# 异构引擎（vita_engine，CUDA 源需 -Xcompiler=/utf-8）
# 构建脚本：scripts/hb_build_cli.ps1
```

### 3. 训练 / Training

```powershell
# 80K 重训同款配置（长线课程数据）
snn_train --curriculum-stage 1 --curriculum-continuous `
    --bptt-window-size 400 --input-inject-interval 1 --learning-rule n3f --steps 80000

# 评估（120 样本，窗口必须 400）
snn_train --eval --curriculum-stage 1 --bptt-window-size 400
```

### 4. 对话引擎 / Dialogue Engine

```powershell
vita_engine.exe --resume checkpoints/middle_1a_longarc_all/ckpt_step80000.snn2e `
    --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf `
    --mod-interval 10 --steps-per-turn 10 --memory-budget-mb 4096
```

### 5. OpenAI 兼容 serve / OpenAI-Compatible Serve

```powershell
vita_engine.exe --serve --port 8899 --api-key <key> --model-name thetrueai `
    --resume checkpoints/<ckpt>.snn2e --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf
```

客户端配置：API 主机 `http://127.0.0.1:8899/v1`，API Key 与模型名默认 `thetrueai`。SNN 每请求推进 10 步，情感状态跨请求持续演化。
Client: base URL `http://127.0.0.1:8899/v1`; default API key and model name `thetrueai`. The SNN advances 10 steps per request; affect evolves across requests.

---

## 目录结构 / Directory Layout

```
vita/
├── src/
│   ├── snn/                # SNN 子系统（C++/CUDA）
│   │   ├── scheduler.cu    # 生物机制调度器（31 种机制）
│   │   ├── modulatory_kernels.cu    # 6 维调质 + AffectiveState readout
│   │   ├── mod_simulator.h          # 课程浓度模拟器（监督目标）
│   │   ├── input_encoding.cu        # 文本流群体编码注入
│   │   ├── bptt_curriculum.cu       # readout 监督头（调质/PAD/工具）
│   │   ├── wm_kernels.cu / hippocampal_kernels.cu  # 记忆
│   │   ├── event_types.h / gene_event_map.h        # 事件→调质映射
│   │   └── tools/         # 课程数据生成/长线剧本工具（Python）
│   ├── bridge/            # 桥接层：affective_mapping / emotion_bridge / snn_feedback
│   ├── vita/              # 引擎：engine / http_server / mini_json
│   └── llm/               # llama_backend（llama.cpp 封装）
├── data/
│   ├── events/            # 课程事件样本（curriculum_all.jsonl 204 段）
│   └── scripts/           # 长线叙事文本（story_text_all.txt）
├── curriculum_generator/  # 场景链生成器（Python）
├── docs/                  # 设计/训练计划/bug 清单
├── legacy/                # 上一代 SNN 研究代码（只读归档）
└── checkpoints/           # 训练检查点（git-ignore）
```

---

## 路线图 / Roadmap

- [ ] **事件→联合皮层注入通道**（当前计划）：事件直通联合皮层固定子区域，readout 有可学信号，SNN 内部涌现情感编码
  **Event→association-cortex injection channel** (current plan): direct event input to fixed cortical sub-regions so the readout has learnable signal and affect emerges inside the network.
- [ ] **浓度→发放即时调制**：调质浓度直接改变神经元兴奋性（神经调质生理角色），情感存在于网络内部
  **Concentration→firing modulation**: neuromodulators directly alter neuron excitability — affect lives inside the network.
- [ ] **双输入接口**：`inject_world`（LLM 理解转化器→事件→皮层）+ `inject_dialogue`（对话→神经签名）
  **Dual input interfaces**: `inject_world` (LLM comprehension translator → events → cortex) + `inject_dialogue` (dialogue → neural signatures).
- [ ] **意图决策闭环**：恢复工具 readout 训练，SNN 决策 → 沙盒执行 → 事件反馈 → 因果学习
  **Intent-decision loop**: restore tool-readout training; SNN decides → sandbox executes → events feed back → causal learning.
- [ ] **评估口径对齐**：判据从"MSE 稳态拟合"改为"事件→情感响应方向/时程/恢复"
  **Evaluation alignment**: criterion shifts from "steady-state MSE fitting" to "event → affective response direction/time-course/recovery".

---

## 关键设计决策 / Key Design Decisions

### 为什么 LLM 不微调？/ Why not fine-tune the LLM?
个性化与持续学习全部交给 SNN（在线 STDP/三因子），LLM 保持冻结、跨用户复用。情感调制走 prompt（文字）+ logit_bias（采样）双通道，不改权重。
All personalization and continual learning live in the SNN (online STDP/three-factor); the LLM stays frozen and reusable across users. Affect modulates via prompt text + logit_bias (sampling), never touching weights.

### 为什么 SNN 是核心而非附庸？/ Why is the SNN the core, not an accessory?
情感不能由"浓度模拟器"外部算好后塞进 prompt——那样会退化成普通角色扮演 LLM。VITA 的方向是让**事件直接塑造 SNN 发放**，情感作为网络内部自发的动力学状态涌现，LLM 只负责"读懂并表达"。当前 readout 根因（事件未进网络）正是这条路上的第一个硬骨头。
Affect must not be computed externally by a "concentration simulator" and stuffed into the prompt — that degenerates into ordinary role-play. VITA's direction is to let **events directly shape SNN firing**, so affect emerges as intrinsic network dynamics, with the LLM only "reading and expressing". The confirmed readout root cause (events never enter the network) is exactly the first hard problem on this path.

### 为什么保留 legacy/？/ Why keep legacy/?
`legacy/stage2e/` 的 BPTT trainer、PCA 签名、丘脑门控等模块经过 10K-100K 步训练验证，是可复用参考。
`legacy/stage2e/` (BPTT trainer, PCA signatures, thalamic gating) was validated over 10K–100K training steps and remains a reusable reference.

---

## 许可 / License

Apache License 2.0 — 见 [LICENSE](./LICENSE)。`legacy/` 旧代码继承自上一代项目（原 CC BY 4.0），新代码采用 Apache 2.0。
Apache License 2.0 — see [LICENSE](./LICENSE). `legacy/` code inherits from the previous project (originally CC BY 4.0); new code is Apache 2.0.

## 致谢 / Acknowledgements

- **Qwen3-4B**：阿里巴巴通义团队，Qwen 开源系列 / Alibaba Qwen team, open Qwen series
- **MiniCPM5-1B**：面壁智能 + 清华 + OpenBMB（历史默认模型，已弃用）/ ModelBest + Tsinghua + OpenBMB (former default, deprecated)
- **llama.cpp**：Georgi Gerganov，C++ LLM 推理事实标准 / the de-facto C++ LLM inference standard
- **LCCC-base 语料**：清华大学 + 三星 / Tsinghua University + Samsung
- **CUDA Toolkit**：NVIDIA
