# HeteroBrain — 异构中文对话 AI 引擎

> **SNN × LLM 异构架构**：SNN 作为前额叶认知调度器（情感核心 + 认知工作空间 + 工具编排），
> LLM 作为语言生成与知识库。让 SNN 做它不可替代的事——维持跨轮次时序状态、STDP 在线学习、多时间尺度并行。
> 边缘可部署 · 中文原生 · 在线持续学习 · 无云端依赖

> **Phase 3 方向**（2026-07-30 更新）：SNN 定位从"6 分类头逻辑处理器"重构为"前额叶认知调度器"三层架构。
> 详见 [docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)。

---

## 项目定位

HeteroBrain 不是又一个 Transformer 大模型，也不是纯 SNN 研究项目。它是一个**面向边缘部署的异构中文对话引擎**，把两种互补的智能载体焊在一起：

| 模块 | 职责 | 实现 |
|---|---|---|
| **LLM 子系统** | 中文指令遵循、文本生成、知识检索（RAG） | MiniCPM5-1B（INT4 GGUF, 0.5GB）+ llama.cpp + FAISS + bge-small-zh |
| **SNN 子系统** | **前额叶认知调度器**：情感核心（6 维调质 + 事件驱动注入，[§3.4](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md#L148)）+ 认知工作空间（256 槽黑板）+ 工具编排（6 工具 + RL） | 自研 CUDA SNN（60K 神经元 / 10.7M 突触 / 31 种生物机制） |
| **Bridge 转换层** | spike signature ↔ LLM embedding 双向桥接、神经调制信号注入 | PCA 签名 + bge-small-zh embedding + 读写头 |

**设计原则**：
- 不与万亿参数模型卷规模，攻击 Transformer 在**长序列、流式推理、边缘部署、持续学习**上的弱项
- LLM 子系统**不微调**（零成本落地），SNN 子系统承载所有个性化与在线学习
- 中文 SOTA + INT4 0.5GB = 手机/浏览器/工控机均可运行
- 老老实实做工程，不再追"涌现"

---

## 与上一代项目的脱胎换骨

本仓库的前身是 `pure-snn-language`（私有归档）——一个纯 SNN/STDP 的科学探查项目。在 100K 步 LCCC 真实中文文本训练后，结论很清楚：

> **纯 STDP 能学到突触级、网络级结构，但学不到语义级结构。** 100K 步仅完成"柱间分化"（js_mean=0.65，达理论上限 94%），距离字节级语义映射还差 6-7 个数量级。

HeteroBrain 接受这个边界，把 SNN 降级为异构架构中的**记忆与筛选子系统**，让 Transformer LLM 承担它擅长的语言生成任务。研究方向 → 工程方向：

| 维度 | 上一代（研究） | 本代（工程） |
|---|---|---|
| 目标 | 验证纯 SNN 能否涌现语义 | 交付可对话的边缘 AI 引擎 |
| 主算法 | STDP / BPTT 代理梯度 | LLM 推理 + SNN 检索 + 投影层 |
| 评价 | 卡方检验 / silhouette / js_mean | BLEU / 困惑度 / 端到端对话评测 |
| 落地 | 实验报告 | 可执行二进制 / Docker 镜像 |
| 代码 | stage0/1/2/2e 散乱 | 模块化 src/{snn,llm,bridge} + legacy/ |

旧 SNN 代码完整保留在 [`legacy/`](./legacy/) 目录，作为 BPTT trainer、PCA 签名、丘脑门控等可复用模块的参考实现。

---

## 系统架构

```mermaid
flowchart TB
    subgraph Input
        U[用户中文输入]
    end

    subgraph Bridge[转换层 Bridge]
        T[BPE Tokenizer]
        E[Embedding 提取<br/>bge-small-zh 512维]
        P[PCA 签名投影<br/>50维]
    end

    subgraph SNN[SNN 认知调度核心 CUDA 60K神经元]
        L1[Layer 1 情感核心<br/>6维调质 DA/5HT/NE/ACh/GABA/催产素<br/>PAD情感模型 + LLM调制信号<br/>事件→基因映射→调质 §3.4]
        L2[Layer 2 认知黑板<br/>256槽 BlackboardSlot<br/>读写头 FACT/CONCEPT/GOAL/...]
        L3[Layer 3 工具编排<br/>6工具 + 状态驱动调用信号<br/>DA reward RL训练]
        R[(海马索引<br/>50K模式 PCA签名)]
    end

    subgraph LLM[LLM 子系统]
        G[MiniCPM5-1B INT4 GGUF]
        K[(知识库 RAG<br/>FAISS)]
    end

    subgraph Output
        O[中文响应]
    end

    U --> T --> E
    E --> P --> L2
    L1 -->|temperature/empathy 调制| G
    L2 -->|黑板内容导出 prompt| G
    L3 -->|工具调用信号| G
    L1 <--> L2 <--> L3
    L2 --> R
    G <--> K
    G --> O
    O -.->|用户反馈事件 离散<br/>→事件→调质映射 §3.4| L1
```

### 数据流

1. **输入**：用户中文文本 → BPE 分词 → embedding → 写入 SNN 认知黑板
2. **SNN 状态演化**：跨轮次维持 6 维调质状态（DA/5HT/NE/ACh/GABA/催产素），演化情感轨迹。调质信号由外部事件 + 内部认知状态共同驱动（§3.4）
3. **工具调用决策**：SNN 从内部状态 readout 工具调用信号（连续注意力，非 argmax），决定是否调用工具及调用哪个。注：内部状态含事件驱动注入产生的情绪背景（§3.4），非仅 spike 统计
4. **LLM 生成**：MiniCPM5-1B 接收 (用户输入 + 黑板内容 + RAG 知识 + SNN 神经调制参数) → 生成响应
5. **结果写回黑板**：工具结果 / LLM 输出 / 用户反馈写入黑板槽位，带情感印记 + 时间戳
6. **DA reward 闭环**：用户反馈 → DA 价值函数 → TD error 驱动 STDP，强化工具调用策略。注：当前 DA 仅来自内部 TD error；事件驱动注入后 DA 将叠加外部事件奖赏（§3.4）
7. **离线巩固**：睡眠重放期把黑板 HYPOTHESIS/GOAL 固化到海马长期记忆 + 突触权重

---

## 目录结构

```
HeteroBrain/
├── README.md                         # 本文件
├── LICENSE                           # Apache 2.0（工程友好）
├── .gitignore
├── CMakeLists.txt                    # 顶层 CMake
├── pyproject.toml                    # Python 桥接（暂留空骨架）
│
├── src/
│   ├── snn/                          # SNN 子系统（C++/CUDA）
│   │   ├── bptt_trainer.cu/.cuh      # BPTT 代理梯度训练器
│   │   ├── modulatory_kernels.cu/.cuh # 6 维调质系统 + AffectiveState readout
│   │   ├── synapse_kernels.cu/.cuh   # STDP + 6 维调质门控 M_ij
│   │   ├── pca_kernels.cu/.cuh       # PCA 签名提取（50 维）
│   │   ├── hippocampal_kernels.cu/.cuh # 海马索引（50K 模式）
│   │   ├── wm_kernels.cu/.cuh        # 工作记忆（50 槽，Phase 3b 替换为 256 槽黑板）
│   │   ├── scheduler.cu/.cuh         # 生物机制调度器（31 种机制）
│   │   └── decoder.cu/.cuh           # 256 类字节解码器
│   │
│   ├── llm/                          # LLM 子系统（C++ + llama.cpp）
│   │   ├── llama_runner.cpp/.h       # llama.cpp 推理封装
│   │   ├── tokenizer_bridge.cpp/.h   # BPE 分词器
│   │   └── prompt_builder.cpp/.h     # 上下文构造（SNN+RAG+用户）
│   │
│   ├── bridge/                       # 转换层
│   │   ├── spike_embedding.cpp/.h    # spike ↔ embedding 投影
│   │   ├── truth_filter.cpp/.h       # token 真实性筛选
│   │   └── pca_projection.cpp/.h     # [2048,1024] 投影矩阵
│   │
│   └── heterobrain/                  # 顶层引擎
│       ├── engine.cpp/.h             # 异构引擎主循环
│       ├── config.cpp/.h             # 配置加载
│       └── main.cpp                  # CLI 入口
│
├── legacy/                           # 上一代 SNN 研究代码（只读归档）
│   ├── cuda/ host/ include/          # Stage 0
│   ├── stage1/                       # BPTT 实验
│   ├── stage2/                       # 柱拓扑 + STDP
│   ├── stage2e/                      # 多机制 v4（BPTT trainer 来源）
│   └── stage3_spark/                 # DGX Spark 训练脚本
│
├── models/                           # 模型权重（git-ignore 大文件）
│   └── README.md                     # 下载说明
│
├── data/                             # 训练 / 测试数据
│   └── README.md
│
├── configs/                          # YAML 配置
│   ├── default.yaml
│   ├── edge_int4.yaml                # 边缘部署档
│   └── server_fp16.yaml              # 服务器档
│
├── scripts/                          # 工具脚本
│   ├── download_models.py            # 拉取 MiniCPM5-1B GGUF
│   ├── prepare_bpe_data.py           # 移植自 legacy
│   └── eval_perplexity.py            # 中文 PPL 评测
│
├── tests/                            # 测试
│   ├── test_bridge.py
│   ├── test_snn_retrieval.py
│   └── test_e2e_dialogue.py
│
└── docs/                             # 文档
    ├── developmental-training-master-spec.md  # 训练范式权威契约
    ├── roadmap.md                    # 路线图
    └── archive/                      # 历史方案 + 迁移文档（已归档）
        ├── phase3-t2h-distillation-plan.md
        ├── snn-emotion-and-workspace-direction.md
        ├── superpowers/               # 设计草稿 + 实施计划
        └── migration/                 # 迁移文档
```

---

## 快速开始

### 1. 环境要求

| 组件 | 版本 | 用途 |
|---|---|---|
| CUDA Toolkit | 13.x | SNN 子系统 |
| CMake | ≥ 3.18 | 构建 |
| Python | 3.10+ | 桥接 / 评测 |
| llama.cpp | 最新 | LLM 推理 |

硬件：NVIDIA GPU（compute capability ≥ 8.6，6GB+ 显存），或纯 CPU（仅 LLM 跑 INT4）。

### 2. 下载模型权重

```powershell
python scripts/download_models.py --model minicpm5-1b-int4
# 输出: models/minicpm5-1b-q4_k_m.gguf (~0.5GB)
```

### 3. 构建

> 当前可构建目标是 SNN 训练子系统 `snn_train`（独立可执行，不依赖 llama.cpp）。
> 顶层 `heterobrain_engine`（LLM + 桥接 + 引擎）待 Phase 3b-3f 实现后构建。

```powershell
# Windows (x64 VS DevShell) — 构建 SNN 训练子系统
cmake -S src/snn -B build/snn -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build/snn snn_train

# 顶层构建（当前仅生成 snn_subsystem / llm_subsystem / bridge_subsystem 空库,
# heterobrain_engine 待实现）
mkdir build; cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
# ninja heterobrain_engine   # 待实现 (Phase 3b-3f)

# Linux
# ./scripts/build.sh
```

### 4. 运行最小对话

> `heterobrain_engine` 尚未实现（Phase 3b-3f），以下命令为待实现后的运行示例。

```powershell
# 待实现 (Phase 3b-3f): 当前无此可执行文件
.\build\heterobrain_engine `
    --llm models/minicpm5-1b-q4_k_m.gguf `
    --config configs/default.yaml `
    --interactive
```

进入交互模式后直接输入中文即可对话。SNN 子系统首次启动需加载 60K 神经元权重（约 5 秒）。

---

## 路线图

### Phase 0 — 工程骨架（当前）

- [x] 创建 HeteroBrain 仓库与目录结构
- [x] 旧 SNN 代码迁移到 `legacy/`
- [ ] 顶层 CMakeLists / .gitignore / LICENSE
- [ ] 从 `legacy/stage2e` 抽出 BPTT trainer / PCA 签名到 `src/snn/`

### Phase 1 — LLM 子系统打通（MVP 对话）

- [ ] llama.cpp 集成，跑通 MiniCPM5-1B INT4 推理
- [ ] BPE 分词器封装
- [ ] Prompt 构造与上下文管理
- [ ] **里程碑**：可单轮中文对话（无 SNN）

### Phase 2 — SNN 子系统移植

- [ ] 从 `legacy/stage2e` 移植 BPTT trainer + PCA 签名
- [ ] 实现长程记忆库（基于 PCA 签名的 Top-K 检索）
- [ ] 在线 STDP 微调（不重训 LLM）
- [ ] **里程碑**：SNN 能从对话历史中检索相关片段

### Phase 3 — SNN 认知调度核心（情感核心 + 认知工作空间 + 工具编排）

> 详见 [docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)

- [~] **3a 情感核心**（进行中）：6 维调质向量 ✅ + AffectiveState readout ✅ + synapse 6 维 M_ij 门控 ✅ + 250 步验证 ✅ + 稳态补偿 ✅；待做：真实文字训练验证 + LLM 调制接口接入 + **事件驱动调质注入接口**（当前情绪无语义锚点，§3.4）
- [ ] **3b 认知黑板**：256 槽 BlackboardSlot + 读写头（替代原 50 槽 WM）+ **事件驱动调质注入**（launch_modulatory 加 inject_event + 基因硬编码映射表，与黑板一并实现，§3.4）
- [ ] **3c 黑板-LLM 桥接**：embedding 双向 + 导出 prompt
- [ ] **3d 工具编排**：6 工具集 + 状态驱动调用信号 + 黑板联动
- [ ] **3e 工具调用训练**：模仿学习冷启动 + RL 微调（复用 DA 价值函数 + PSW 突触）
- [ ] **3f 黑板-海马溢出**：短期→长期固化 + 情感印记
- [ ] **3g 端到端验证**：情感轨迹可视化 + 多步推理 + 工具调用 demo
- [ ] **里程碑**：SNN 在跨轮次情感维持 + 多步推理 + 工具调度上展现 LLM+RAG 做不到的能力

### Phase 4 — 评测与优化

- [ ] 中文对话评测（CEval / CMMLU 子集 + 人工评测）
- [ ] 困惑度对比（纯 LLM vs HeteroBrain）
- [ ] 延迟 / 内存 / 功耗 profile
- [ ] INT4 量化 + 边缘部署验证
- [ ] **里程碑**：边缘设备（手机/Jetson）可运行

### Phase 5 — 持续学习闭环

- [ ] 用户反馈写入 SNN 的 STDP 闭环
- [ ] 长程记忆库自动维护（遗忘 / 巩固）
- [ ] 多用户隔离
- [ ] **里程碑**：与同一用户多次对话后能记住早期话题

---

## 关键设计决策

### 为什么不微调 LLM？

- MiniCPM5-1B 已是 AA-Index 小模型第一，开箱即用
- 微调需要 4×A100 级别显卡，违背边缘部署定位
- SNN 在 BPTT 收敛阶段（grad_norm≈100）信号不稳定，强行微调会污染 LLM 权重
- 个性化与持续学习全部交给 SNN，LLM 保持冻结便于跨用户复用

### 为什么用 llama.cpp 而不是 transformers？

- C++ 原生，可嵌入现有 CUDA 工程，避免 Python ↔ C++ IPC 开销
- INT4 GGUF 量化成熟，0.5GB 权重可塞进手机
- 跨平台（Windows/Linux/macOS/Android）
- 推理速度在 CPU 上比 transformers 快 3-5×

### 为什么 SNN 不做语义生成？SNN 真正的不可替代角色是什么？

上一代项目 100K 步训练证明：纯 SNN 在合理规模下学不到字节级语义映射。HeteroBrain 接受这个边界。

**但"SNN 做什么"经历三次定位迭代**：
1. ~~长程模糊记忆 + token 真实性筛选~~（旧 README 描述，已废弃——筛选用弱评估器评强生成器，逻辑自相矛盾）
2. ~~6 分类头逻辑处理器~~（phase3 旧方案，已废弃——分类头本质是线性分类器，2 层 MLP 即可替代，SNN 严重过度）
3. **前额叶认知调度器**（当前方向，2026-07-30）：三层递进架构
   - **Layer 1 情感核心**：6 维调质向量，维持跨轮次情感状态，神经调制 LLM 生成参数
   - **Layer 2 认知工作空间**：256 槽黑板 + 读写头，让 SNN 从"只能口算"升级到"会打草稿"
   - **Layer 3 工具编排**：6 工具 + 状态驱动调用信号，让 SNN 从"被迫当计算器"升级到"调度外部能力"

这个定位立得住的依据：现有 SNN 代码已实现 31 种生物机制（4/6 调质就绪、海马索引、工作记忆、PSW STDP、丘脑门控、睡眠重放等），新方向能复用其中 20+ 种，而旧方案仅复用 PCA readout 一条线。详见 [docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)。

### 为什么保留 legacy/ 旧代码？

`legacy/stage2e/` 中的 BPTT trainer（CSR 稀疏格式 + 代理梯度 + 梯度裁剪）、PCA 签名提取、丘脑门控等模块经过 10K-100K 步训练验证，可直接复用到新工程。其他 stage0/1/2 代码作为研究档案保留。

---

## 性能指标（目标）

| 指标 | 纯 MiniCPM5-1B | HeteroBrain 目标 | 当前状态 |
|---|---|---|---|
| 中文 BLEU | 基准 | +5-10% | 未评测 |
| 多轮对话一致性 | 易丢上下文 | 显著改善 | 未评测 |
| 边缘推理延迟 (CPU) | 300ms | 350ms | 未测 |
| 内存占用 | 1.2GB | 1.8GB（含 SNN） | 未测 |
| 持续学习 | 不支持 | 支持 | 未实现 |

---

## 致谢

- **MiniCPM5-1B**：面壁智能 + 清华 + OpenBMB，AA-Index 小模型第一
- **llama.cpp**：Georgi Gerganov，C++ LLM 推理事实标准
- **LCCC-base 语料**：清华大学 + 三星，2020 年发布
- **CUDA Toolkit**：NVIDIA
- **上一代 pure-snn-language 项目**：本仓库 SNN 子系统的前身

---

## 许可

Apache License 2.0 — 见 [LICENSE](./LICENSE)。

`legacy/` 目录下的旧代码继承自上一代 `pure-snn-language` 项目，原 CC BY 4.0 许可；新代码采用 Apache 2.0 以利于工程化集成。
