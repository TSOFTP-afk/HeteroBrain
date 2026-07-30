# Phase 3: SNN 认知调度核心（情感核心 + 认知工作空间 + 工具编排）

> **方向更新**（2026-07-30）：原"双系统协作 + 6 分类头逻辑处理器"方案经评估存在核心问题——
> SNN 的 6 个分类头本质是 6 个共享 backbone 的线性分类器，2 层 MLP 即可替代，
> 现有 31 种生物机制中 29 种未被复用，SNN 存在性理由薄弱。
>
> **现调整为三层递进架构**（详见 [docs/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/snn-emotion-and-workspace-direction.md)）：
> - **Layer 1 情感核心**：6 维调质向量（DA/5HT/NE/ACh/GABA/催产素），维持跨轮次情感状态 + 神经调制 LLM
>   - 注：当前调质信号全来自 SNN 内部状态，情绪无语义锚点；Phase 3b 起增加外部事件驱动注入（方向文档 §3.4）
> - **Layer 2 认知工作空间**：256 槽黑板 + 读写头（替代原 50 槽 WM），支持多步推理
> - **Layer 3 工具编排**：6 工具 + 状态驱动调用信号 + DA reward RL 训练
>
> **本文档保留作为历史参考**——下文第 2-12 章是原"双系统协作"方案的详细设计，
> 其中 RAG 设计（第 4 章）、协调器框架（第 5 章）、模块清单（第 6 章）仍可作为参考，
> 但 SNN 部分的 6 分类头设计、内部默读机制、50K 标注数据策略**已废弃**，
> 以方向文档为准。

---

## 当前实施状态（2026-07-30 更新）

> 本节跟踪 Phase 3 三层架构的实际落地进度，替代下方第 7 章旧实施步骤。

### Phase 3a — 情感核心（进行中）

| 子项 | 状态 | 说明 |
|---|---|---|
| 6 维调质系统扩充（GABA/催产素） | ✅ 完成 | modulatory_kernels.cu 扩展到 6 维，含浓度动力学 + 衰减常数。注：当前信号源全为内部状态，事件驱动注入接口留待 Phase 3b（§3.4） |
| AffectiveState readout | ✅ 完成 | PAD 情感模型映射 + LLM 调制信号生成（temperature/top_p/repetition/empathy）。注：当前 readout 仅反映内部状态情绪，无语义锚点；事件驱动改造见方向文档 §3.4 |
| synapse_kernels 6 维 M_ij 门控 | ✅ 完成 | STDP kernel 加入 GABA 抑制门控 + 催产素增强社交可塑性 |
| 编译 + 250 步合成输入验证 | ✅ 通过 | criterion_modulatory_range=1, criterion_no_crash=1 |
| 稳态补偿（受体下调/上调） | ✅ 完成 | config.h + modulatory_kernels.cu + Python 验证脚本，1239/1239 子数据集通过 |
| 真实文字训练验证（BPE 数据 + 情感轨迹） | ⬜ 待做 | 需跑 prepare_bpe_data.py + 100K 步训练 + 接入 get_affective_state()。注：训练数据集需扩展为连续感官流 + 离散事件标注（§3.4 实现路径第5点） |
| LLM 调制接口接入（AffectiveState → LLM 推理） | ⬜ 待做 | 需实现 src/bridge/snn_llm_bridge.cpp。注：调制信号在事件驱动注入落地前仅有内部源，落地后需合并外部事件信号（§3.4） |

### Phase 3b-3g — 待启动

详见 [docs/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/snn-emotion-and-workspace-direction.md) §8.3 实施路径。

> **Phase 3b 特别说明**：事件驱动调质注入（§3.4）作为 Phase 3b 子任务，与认知黑板一并实现。
> 事件→工作空间写入→调质评估→SNN 演化→工具调用形成闭环，避免重复改动 launch_modulatory 接口。

---

> **历史架构声明**（2026-07-27，已废弃）：原 T2H 蒸馏方案让 SNN 承担语言生成（不擅长），
> 改为双系统协作：LLM 负责知识检索与语言生成，SNN 负责判断与路由决策。
> 该方案在 2026-07-30 被三层架构方案取代，原因见上方"方向更新"。

---

## 1. 架构调整动机

### 1.1 原方案问题

| 问题 | 说明 |
|---|---|
| SNN 生成质量天花板低 | 256 类字节 decoder 表达能力不足，中文生成可读性难达 4/5 |
| 蒸馏成本高 | 30K 步联合蒸馏 + 40GB LLM logits 缓存，工程复杂 |
| SNN 优势未发挥 | SNN 擅长稀疏决策，却被强行用于高维生成 |
| 知识不可控 | 蒸馏后知识冻结在突触权重中，无法热更新 |

### 1.2 新架构定位

> ⚠️ 下表中 SNN 的"逻辑处理器 + 决策路由"定位已废弃，调整为"前额叶认知调度器"（三层架构）。
> 详见方向文档 [docs/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/snn-emotion-and-workspace-direction.md)。

| 子系统 | 角色 | 类比大脑区域 |
|---|---|---|
| **LLM（MiniCPM5-1B）** | 知识库 + 语言生成器 | 新皮层（语言区）+ 海马体（语义记忆） |
| **RAG 向量库** | 外部事实知识 + 用户长程记忆 | 海马体（情景记忆索引） |
| **SNN（60K 神经元）** | ~~逻辑处理器 + 决策路由~~ → **前额叶认知调度器**（情感核心 + 认知工作空间 + 工具编排） | 前额叶（执行控制）+ 边缘系统（情感） |

### 1.3 设计原则

1. **LLM 保持完整**：不蒸馏、不脱离，始终作为知识库与主生成器
2. **SNN 做判断不做生成**：句子/轮次级决策，输出路由信号与评分
3. **SNN 保留内部默读**：decoder 不移除，但仅用于内部辅助决策，不直接输出
4. **知识库可热更新**：RAG 向量库独立于 LLM 权重，支持增量插入

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 3 双系统协作架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  用户输入                                                           │
│     │                                                               │
│     ↓                                                               │
│  ┌──────────────────────────────────────────┐                       │
│  │  阶段 1: SNN 前置判断（逻辑处理器）        │                       │
│  │  - 意图分类（对话/问答/闲聊/任务/情绪）    │                       │
│  │  - 情绪识别（中性/高兴/不满/困惑/愤怒）    │                       │
│  │  - 上下文相关性评分（0-1）                 │                       │
│  │  - 内部默读生成候选（不输出，辅助判断）    │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │ 决策信号                                   │
│                       ↓                                           │
│  ┌──────────────────────────────────────────┐                       │
│  │  阶段 2: LLM + RAG 生成（知识库 + 生成器） │                       │
│  │  - RAG 检索（FAISS 向量库 Top-K）         │                       │
│  │  - Prompt 构建（含 SNN 决策上下文）        │                       │
│  │  - MiniCPM5-1B 生成回复                  │                       │
│  │  - 流式输出到 CMD                         │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │ 生成的回复                                 │
│                       ↓                                           │
│  ┌──────────────────────────────────────────┐                       │
│  │  阶段 3: SNN 后置评估（质量判断）          │                       │
│  │  - 回复质量评分（流畅/相关/完整）          │                       │
│  │  - 重生成决策（binary）                   │                       │
│  │  - 记忆存档决策（binary）                 │                       │
│  │  - 与内部默读候选对比（一致性分数）        │                       │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                           │
│                       ↓                                           │
│                    输出给用户                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. SNN 逻辑处理器设计（⚠️ 已废弃 — 见方向文档 Layer 1/2/3）

> **废弃说明**（2026-07-30）：本章描述的"6 分类头 + 内部默读 + 50K 标注数据"方案已被取代。
> 原因：
> 1. 6 分类头本质是线性分类器，2 层 MLP 即可替代，60K 神经元严重过度
> 2. 现有 31 种生物机制中 29 种未被复用（仅用 PCA readout 一条线）
> 3. 内部默读机制自相矛盾（用生成质量差的 decoder 评估 LLM 质量）
> 4. 三阶段流水线 + 重生成循环延迟 > 4 秒
> 5. 50K 标注数据中"质量评分"和"重生成/记忆决策"标签难构造
>
> **取代方案**：见 [docs/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/snn-emotion-and-workspace-direction.md)
> - Layer 1 情感核心（替代 6 分类头）：6 维调质向量 + LLM 神经调制
> - Layer 2 认知工作空间（替代 50 槽 WM）：256 槽黑板 + 读写头
> - Layer 3 工具编排（替代三阶段流水线）：状态驱动工具调用 + RL 训练
>
> 本章内容保留作为历史参考，**不再作为实施依据**。

---

### 3.1 判断任务清单（句子/轮次级）

| 任务 | 类型 | 类别数 | 输出 | 用途 |
|---|---|---|---|---|
| 意图分类 | 多类 | 8 | intent_id | 决定 RAG 检索策略 + prompt 模板 |
| 情绪识别 | 多类 | 5 | emotion_id | 调整回复语气 |
| 上下文相关性 | 回归 | 1 | relevance_score [0,1] | 决定是否需要检索长程记忆 |
| 回复质量 | 回归 | 1 | quality_score [0,1] | 决定是否重生成 |
| 重生成决策 | 二类 | 2 | should_regenerate | 触发 LLM 重新生成 |
| 记忆存档决策 | 二类 | 2 | should_memorize | 写入 RAG 长程记忆 |

### 3.2 多任务分类头架构

复用 Phase 2 的 SNN 骨架（60K 神经元 + 10.7M 突触），**改造 decoder 为多任务头**：

```
SNN 前向 (T=50 步)
   ↓
spike_history[50][60K]
   ↓
PCA 签名 [1024]  ──────────→  内部默读 decoder [256 类]（辅助）
   ↓
多任务分类头：
   ├── intent_head     [1024 → 8]     意图分类
   ├── emotion_head    [1024 → 5]     情绪识别
   ├── relevance_head  [1024 → 1]     相关性回归
   ├── quality_head    [1024 → 1]     质量回归
   ├── regenerate_head [1024 → 2]     重生成决策
   └── memorize_head   [1024 → 2]     记忆存档决策
```

**改造点**：
- 保留 `decode_kernels.cu` 的 256 类 decoder 作为"内部默读"（不输出，辅助一致性判断）
- 新增 `multitask_head.cu`：6 个轻量线性头，共享 PCA 签名特征
- 训练方式：**监督学习**（标注数据驱动），不再是 BPTT 自回归

### 3.3 内部默读机制

SNN 的 256 类 decoder 保留但**不直接输出**：

```cpp
// 内部默读：生成候选字节序列（仅用于辅助决策）
void internal_monologue(const SpikeHistory& spikes,
                        std::vector<uint8_t>& candidate_bytes,
                        float& confidence);

// 用途：
// 1. 与 LLM 回复对比，计算一致性分数（SNN 是否"认同"LLM 说的）
// 2. 一致性低时，提高重生成概率
// 3. 作为 SNN 视角的质量评估输入
```

### 3.4 训练数据策略

SNN 不再需要 LLM 蒸馏，改用**标注数据监督训练**：

| 任务 | 数据来源 | 标注方式 |
|---|---|---|
| 意图分类 | LCCC + 自建 | 8 类规则匹配 + 人工校验 |
| 情绪识别 | LCCC + 公开情绪数据集 | 现成中文情绪数据集 |
| 相关性/质量 | LCCC 对话对 | 启发式打分 + 人工抽样 |
| 重生成/记忆 | LLM 生成 + 人工标注 | 对比 good/bad 回复对 |

**数据量目标**：~50K 标注样本（可从 LCCC 829MB 中采样 + 标注）

---

## 4. LLM + RAG 知识库设计

### 4.1 RAG 向量库

**技术选型**：FAISS（Facebook AI Similarity Search）

```
configs/rag.yaml:
  embedding_model: bge-small-zh-v1.5   # 中文 embedding, 512 维
  vector_db: faiss.IndexFlatIP          # 内积检索（cosine 归一化后）
  top_k: 5                              # 检索 Top-5 相关片段
  chunk_size: 256                       # 文档分块大小（字符）
  chunk_overlap: 64                     # 重叠
```

**知识来源**：
| 类别 | 内容 | 更新方式 |
|---|---|---|
| 领域知识 | WikiText-2 + 自建文档 | 离线批量索引 |
| 用户长程记忆 | 对话历史片段 | 在线增量插入（SNN 决策触发） |
| 事实知识 | CEval 子集 + 自建 QA | 离线批量索引 |

### 4.2 LLM 生成器

**职责**：基于 SNN 决策 + RAG 检索结果，生成最终回复

```cpp
class LLMGenerator {
public:
    // 构建增强 prompt
    std::string build_prompt(const std::string& user_input,
                              const SNNDecision& decision,
                              const std::vector<RAGResult>& retrieved);

    // 流式生成（CMD 输出）
    void generate_stream(const std::string& prompt,
                         std::function<void(const std::string&)> callback);

private:
    llama_model* model_;
    llama_context* ctx_;
    std::string chat_template_;  // Jinja 模板（Phase 1 已提取）
};
```

**Prompt 模板**（受 SNN 决策调制）：
```
<system>
你是一个中文对话助手。当前用户意图：{intent}，情绪：{emotion}。
请根据检索到的上下文和用户情绪调整回复语气。
</system>

<retrieved_context>
{rag_top_k_results}
</retrieved_context>

<history>
{dialog_history}
</history>

<user>{user_input}</user>
<assistant>
```

---

## 5. 协调器设计（三阶段流水线）

### 5.1 Coordinator 接口

```cpp
// src/heterobrain/coordinator.h
class Coordinator {
public:
    // 单轮对话处理
    std::string process_turn(const std::string& user_input);

private:
    SNNLogicProcessor snn_;      // 逻辑处理器
    RAGRetriever rag_;           // 知识检索
    LLMGenerator llm_;           // 语言生成

    // 阶段 1: SNN 前置判断
    SNNDecision pre_judge(const std::string& user_input);

    // 阶段 2: LLM + RAG 生成
    std::string generate(const std::string& user_input,
                         const SNNDecision& decision);

    // 阶段 3: SNN 后置评估
    SNNEvaluation post_evaluate(const std::string& user_input,
                                const std::string& reply);

    // 重生成循环（最多 2 次）
    std::string regenerate_loop(const std::string& user_input,
                                const SNNDecision& decision);
};
```

### 5.2 决策信号流

```
用户输入 "我今天好累啊"
   ↓
[阶段 1: SNN 前置判断]
   intent = EMOTIONAL_VENT      # 情绪宣泄
   emotion = SAD                # 悲伤
   relevance = 0.3              # 不需长程记忆检索
   internal_monologue = "辛苦了"  # SNN 默读候选
   ↓
[阶段 2: LLM + RAG 生成]
   RAG 检索（intent=情绪宣泄，跳过事实检索）
   Prompt 注入 emotion=SAD → 调整语气为共情
   LLM 生成："听起来今天很辛苦，要不要聊聊发生了什么？"
   ↓
[阶段 3: SNN 后置评估]
   quality = 0.85               # 质量良好
   monologue_consistency = 0.7  # 与默读候选较一致
   should_regenerate = false    # 不需重生成
   should_memorize = true       # 值得记住（用户情绪状态）
   ↓
输出："听起来今天很辛苦，要不要聊聊发生了什么？"
   ↓
[记忆存档]
   写入 RAG：{user: "今天好累", emotion: SAD, time: ...}
```

---

## 6. 关键模块清单

### 6.1 新增源码

| 文件 | 职责 |
|---|---|
| `src/snn/multitask_head.cu/.cuh` | 6 任务分类头（共享 PCA 特征） |
| `src/snn/logic_processor.cu/.cuh` | SNN 逻辑处理器封装（前置+后置） |
| `src/snn/internal_monologue.cu/.cuh` | 内部默读机制（保留 decoder） |
| `src/llm/rag/rag_retriever.cpp/.h` | FAISS 检索封装 |
| `src/llm/rag/embedding_client.cpp/.h` | Embedding 模型调用（bge-small-zh） |
| `src/llm/generator.cpp/.h` | LLM 生成器（llama.cpp 流式） |
| `src/llm/prompt_builder.cpp/.h` | Prompt 构建（含 SNN 决策上下文） |
| `src/heterobrain/coordinator.cpp/.h` | 三阶段协调器 |
| `src/heterobrain/main.cpp` | CMD 交互入口 |
| `src/bridge/snn_llm_bridge.cpp` | SNN ↔ LLM 通信（替换桩） |

### 6.2 新增脚本

| 脚本 | 职责 |
|---|---|
| `scripts/build_rag_index.py` | 构建 FAISS 索引（WikiText + 领域文档） |
| `scripts/prepare_snn_labels.py` | 标注数据准备（意图/情绪/质量） |
| `scripts/train_snn_multitask.bat` | SNN 多任务分类训练 |
| `scripts/build_heterobrain.bat` | 主程序构建脚本 |
| `scripts/run_heterobrain.bat` | CMD 交互启动脚本 |

### 6.3 新增配置

| 文件 | 内容 |
|---|---|
| `configs/rag.yaml` | RAG 参数（embedding 模型、top_k、chunk_size） |
| `configs/snn_multitask.yaml` | SNN 多任务训练超参（任务权重、学习率） |
| `configs/heterobrain.yaml` | 主配置（模型路径、协调器参数） |

### 6.4 新增数据

| 路径 | 内容 |
|---|---|
| `data/snn_labels/` | 标注数据（意图/情绪/质量） |
| `data/rag_index/` | FAISS 索引文件 |
| `data/rag_docs/` | 知识文档（WikiText、领域文档） |

---

## 7. 实施步骤

### 阶段 A: LLM 桥接落地（依赖: 无）

- [ ] **A1**: 实现 `src/bridge/snn_llm_bridge.cpp`（tokenize/embed/logits，llama.cpp C API）
- [ ] **A2**: 实现 `src/llm/generator.cpp`（流式生成，复用 Phase 1 Jinja 模板）
- [ ] **A3**: 实现 `src/llm/prompt_builder.cpp`（含 SNN 决策槽位）
- [ ] **A4**: 编写 `scripts/test_llm_gen.bat`，验证 LLM 流式生成

### 阶段 B: RAG 知识库（依赖: 无, 可与 A 并行）

- [ ] **B1**: 选型并安装 FAISS（Python binding 先行，C++ 后续）
- [ ] **B2**: 实现 `src/llm/rag/embedding_client.cpp`（bge-small-zh 调用）
- [ ] **B3**: 实现 `src/llm/rag/rag_retriever.cpp`（FAISS Top-K 检索）
- [ ] **B4**: 编写 `scripts/build_rag_index.py`，在 WikiText 样本上构建索引
- [ ] **B5**: 验证检索质量（Top-5 命中率 ≥ 60%）

### 阶段 C: SNN 多任务改造（依赖: 无, 可与 A/B 并行）

- [ ] **C1**: 实现 `src/snn/multitask_head.cu`（6 任务分类头）
- [ ] **C2**: 实现 `src/snn/internal_monologue.cu`（内部默读，复用 decoder）
- [ ] **C3**: 实现 `src/snn/logic_processor.cu`（前置判断 + 后置评估封装）
- [ ] **C4**: 编写 `scripts/prepare_snn_labels.py`（标注数据准备）
- [ ] **C5**: SNN 多任务训练（50K 标注样本，验证分类准确率）

### 阶段 D: 协调器集成（依赖: A, B, C）

- [ ] **D1**: 实现 `src/heterobrain/coordinator.cpp`（三阶段流水线）
- [ ] **D2**: 实现 `src/heterobrain/main.cpp`（CMD 交互入口）
- [ ] **D3**: 决策信号传递调试（SNN → LLM → SNN）
- [ ] **D4**: 重生成循环实现（最多 2 次）

### 阶段 E: 端到端验证（依赖: D）

- [ ] **E1**: 单轮对话冒烟测试（CMD 交互）
- [ ] **E2**: 多轮对话一致性测试（10 轮主题保持）
- [ ] **E3**: SNN 决策准确性测试（意图分类准确率 ≥ 80%）
- [ ] **E4**: RAG 检索增强效果测试（对比纯 LLM vs LLM+RAG）
- [ ] **E5**: 重生成机制测试（低质量回复触发重生成）

---

## 8. 验收标准

| 准则 | 指标 | 通过条件 |
|---|---|---|
| LLM 桥接 | llama.cpp 流式生成 | 中文回复 token/s ≥ 100 |
| RAG 检索 | Top-5 命中率 | ≥ 60%（人工标注子集） |
| SNN 意图分类 | 准确率 | ≥ 80%（8 类） |
| SNN 情绪识别 | 准确率 | ≥ 75%（5 类） |
| SNN 质量评估 | 与人工评分相关性 | Pearson ≥ 0.6 |
| 多轮一致性 | 10 轮主题保持率 | ≥ 70% |
| 重生成有效性 | 低质量回复拦截率 | ≥ 50% |
| 端到端延迟 | 单轮响应时间 | < 3 秒（含 RAG + LLM） |
| CMD 交互 | 可用性 | 连续 10 轮无崩溃 |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| SNN 分类头训练数据不足 | 准确率低 | 用 LLM 辅助标注 + 主动学习 |
| RAG 检索延迟高 | 端到端超 3 秒 | FAISS GPU + 缓存热点查询 |
| SNN 决策错误导致 LLM 误路由 | 回复跑题 | 置信度阈值 + 低置信时跳过 SNN 直接走 LLM |
| 内部默读增加 SNN 计算开销 | 延迟增加 | 默读仅在质量评估阶段触发，前置判断阶段跳过 |
| FAISS C++ 集成复杂 | 构建困难 | 先用 Python RAG 跑通 PoC，再迁移 C++ |
| 多任务分类头互相干扰 | 某些任务退化 | 任务权重调度 + GradNorm 自适应 |

---

## 10. 关键决策点

| 决策 | 选项 | 推荐 | 理由 |
|---|---|---|---|
| Embedding 模型 | bge-small-zh / bge-base-zh / LLM 自带 | **bge-small-zh** | 512 维，速度快，中文效果好 |
| 向量库 | FAISS / Chroma / Milvus | **FAISS** | C++ 原生，无外部依赖，单机足够 |
| SNN 训练方式 | BPTT 自回归 / 监督分类 / 联合 | **监督分类** | 任务明确，数据驱动，收敛快 |
| 重生成上限 | 1 次 / 2 次 / 3 次 | **2 次** | 平衡质量与延迟 |
| 默读触发条件 | 总是 / 仅低置信 / 仅评估阶段 | **仅评估阶段** | 减少前置阶段开销 |
| RAG 更新方式 | 全量重建 / 增量插入 | **增量插入** | 用户记忆需实时写入 |

---

## 11. 依赖与前置

| 依赖 | 状态 | 说明 |
|---|---|---|
| Phase 1 LLM 子系统 | ✅ | MiniCPM5-1B + llama.dll + Jinja 模板 |
| Phase 2 SNN 子系统 | ✅ | 60K 神经元 + PCA + decoder（改造成多任务头） |
| FAISS 库 | ⬜ 待安装 | `pip install faiss-cpu`（PoC）/ 源码编译（C++） |
| bge-small-zh 模型 | ⬜ 待下载 | HuggingFace, ~95MB |
| 标注数据 | ⬜ 待准备 | 从 LCCC 采样 + 规则标注 + 人工校验 |

---

## 12. 与原蒸馏方案的对比

| 维度 | 原 T2H 蒸馏方案 | 新双系统协作方案 |
|---|---|---|
| SNN 角色 | 学生（学生成） | 逻辑处理器（做判断） |
| LLM 角色 | 教师（蒸馏后脱离） | 知识库 + 主生成器（始终保留） |
| 知识库 | SNN 突触权重（冻结） | LLM 参数 + RAG 向量库（可热更新） |
| SNN 训练 | BPTT 联合蒸馏（30K 步） | 监督多任务分类（50K 样本） |
| 生成质量 | 依赖 SNN decoder（天花板低） | 依赖 LLM（质量高） |
| 工程复杂度 | 高（蒸馏 + 缓存 + 投影矩阵） | 中（RAG + 多任务头 + 协调器） |
| 可解释性 | 低（黑盒生成） | 高（SNN 决策可观测） |
| 在线学习 | SNN 突触 STDP | SNN 分类头微调 + RAG 增量 |

---

## 13. 后续衔接（Phase 4+ 预览）

Phase 3 完成后，双系统协作框架就位。后续阶段：

- **Phase 4: 在线持续学习** — SNN 分类头在线微调 + RAG 用户记忆增量 + 遗忘/巩固机制
- **Phase 5: 评测与优化** — CEval/CMMLU 评测 + 延迟/内存 profile + INT4 量化
- **Phase 6: 桌面应用** — CMD 升级为 GUI（PySide6 + PyInstaller 打包 EXE）
- **Phase 7: 发布** — 闭源分发，安装包（Inno Setup）

Phase 3 的 RAG 知识库 + SNN 决策路由为 Phase 4 的持续学习闭环奠定基础：
用户反馈 → SNN 评估 → RAG 记忆更新 → 下轮对话利用新知识。
