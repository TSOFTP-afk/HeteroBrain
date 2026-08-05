# vita 路线图

> 从 SNN 研究项目转型为异构中文对话 AI 引擎的工程化路线。

## 已完成

### Phase 0 — 工程骨架 ✅

- [x] 创建 vita GitHub 仓库
- [x] 旧 SNN 代码迁移到 `legacy/`（126 文件, 1.19MB）
- [x] 顶层 `CMakeLists.txt` / `.gitignore` / `LICENSE` (Apache 2.0)
- [x] 工程化目录骨架 `src/{snn,llm,bridge,vita}/`
- [x] 配置文件 `configs/default.yaml`
- [x] 全新 README，告别 SNN/STDP 研究阶段

### Phase 1 — LLM 子系统打通 (MVP 对话) ✅

**目标**：跑通 MiniCPM5-1B INT4 推理，可单轮中文对话（无 SNN）。

**实际实现**：采用官方 `llama-cli` 工具（Release 构建）替代自定义 `llama_runner.cpp`，工程化更稳定、维护成本更低。后续如需嵌入式集成再回头实现 `src/llm/llama_runner.cpp`。

- [x] 引入 llama.cpp 源码（`F:\hb_llama\` junction，非子模块形式）
- [x] 编译 `llama-cli.exe`（Release + CUDA sm_86，`F:\thetrueai\build\bin\`，1.08 GB）
- [x] 启用 `LLAMA_BUILD_SERVER=ON` 以生成 cli 目标（支持 `--chat-template-file`）
- [x] 编写 `scripts/download_models.py`：拉取 MiniCPM5-1B GGUF（656 MB，Q4_K_M）
- [x] 从 GGUF 提取 Jinja chat 模板（9062 字节，`F:\hb_models\minicpm5-chat.jinja`）
- [x] 解决 GGUF 元数据 `general.architecture="llama"` 错误 → 用 `--chat-template-file` 显式指定
- [x] 编写 `scripts/test_minicpm5_zh.bat`：中文推理测试脚本
- [x] **里程碑达成**（2026-07-27）：`llama-cli + Jinja 模板` 单轮中文对话正常
  - 测试输入：`你好，请用中文简短介绍一下你自己（30字以内）。`
  - 模型输出：`我是MiniCPM系列模型，由面壁智能开发。`（含思考链）
  - 性能：Prompt 1150.5 t/s | Generation 248.2 t/s（RTX 3060，-ngl 99）
  - 详见 [logs/zh_inference2.log](file:///f:/thetrueai/logs/zh_inference2.log)

**待办（可推迟到 Phase 3 联调时再做）**：

- [ ] 实现 `src/llm/llama_runner.cpp/.h`：用 llama.cpp C API 嵌入式调用（替代 shell 调用 llama-cli）
- [ ] 实现 `src/llm/tokenizer_bridge.cpp/.h`：BPE 编码/解码
- [ ] 实现 `src/llm/prompt_builder.cpp/.h`：system + history + user 拼接
- [ ] 实现 `src/vita/main.cpp`：CLI 入口 + 交互循环

## 待启动

### Phase 2 — SNN 训练子系统移植 ✅

**目标**：将 `legacy/stage2e/` 的 SNN 训练子系统整体移植到新路径 `src/snn/`，作为 Phase 3 认知调度核心的前置依赖。

**实际实现**（2026-07-27 完成）：

- [x] 从 `legacy/stage2e` 移植 30+ 文件到 `src/snn/`（BPTT trainer + 全部 kernel + scheduler + decoder）
- [x] 创建 `src/snn/CMakeLists.txt`，独立 target `snn_train` + `snn_decoder`，CUDA sm_86
- [x] 创建 `src/bridge/snn_llm_bridge.h` 桥接桩（header-only，Phase 3 替换为 llama.cpp 调用）
- [x] 编写 `scripts/build_snn.bat` 构建脚本（vcvarsall + cmake + ninja）
- [x] 顶层 `CMakeLists.txt` 加入 `add_subdirectory(src/snn)`
- [x] **10K 步性能基线达成**：
  - perplexity = 9.86（达成 < 10 目标，参考 legacy 7.32）
  - accuracy = 66.66%（远超 legacy 39.62%）
  - P3-D 结构重建跳过 9 次（BPTT 模式）
  - 训练时长 ~70 分钟（笔记本 RTX 3060）
- [x] **Checkpoint 验证**：v3 格式，`--resume` 从 step 4000 恢复，loss 完全匹配（误差 0%）
- [x] Spec 文档：[.trae/specs/port-snn-training-subsystem/](file:///f:/thetrueai/.trae/specs/port-snn-training-subsystem/spec.md)

**注**：原计划的 `memory_index.cu` / `online_stdp.cu` 检索接口推迟到 Phase 3 时实现，因为 SNN 的检索能力需要先有 LLM embedding 对接才能定义 Top-K 语义。

### Phase 3 — SNN 认知调度核心（情感核心 + 认知工作空间 + 工具编排）

**目标**：SNN 从"6 分类头逻辑处理器"重构为"前额叶认知调度器"，复用现有 31 种生物机制中 20+ 种。
**方向文档（已归档）**：[docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)
**训练范式权威契约**：[docs/developmental-training-master-spec.md](file:///f:/thetrueai/docs/developmental-training-master-spec.md)

- [~] **3a 情感核心**（进行中）：6 维调质向量（DA/5HT/NE/ACh/GABA/催产素）✅ + AffectiveState readout ✅ + synapse 6 维 M_ij 门控 ✅ + 250 步合成输入验证 ✅ + 稳态补偿 ✅；待做：真实文字训练验证 + LLM 调制接口接入 + **事件驱动调质注入接口**（当前情绪无语义锚点，§3.4）
- [ ] **3b 认知黑板**：256 槽 BlackboardSlot + 读写头（替代原 50 槽 WM）+ 类型标签（FACT/CONCEPT/RELATION/GOAL/HYPOTHESIS/SCRATCH/ANCHOR）+ **事件驱动调质注入**（launch_modulatory 加 inject_event + 基因硬编码映射表，与黑板一并实现避免重复改接口，§3.4）
- [ ] **3c 黑板-LLM 桥接**：embedding 双向（bge-small-zh 512 维）+ 导出 prompt
- [ ] **3d 工具编排核心**：6 工具集（Transformer 生成器/计算器/草稿记录器/长程检索器/知识库查询/时钟）+ 状态驱动调用信号 + 黑板联动
- [ ] **3e 工具调用训练**：模仿学习冷启动 + RL 微调（复用现有 DA 价值函数 + PSW 贝叶斯突触做 reward 闭环）
  - 注：DA 信号当前来自内部 TD error，事件驱动注入后 DA 将叠加外部事件奖赏（如 hunt_success→DA↑，§3.4）
- [ ] **3f 黑板-海马溢出**：短期→长期固化 + 情感印记（HippoIndex 加 emotion_tag/user_id/real_timestamp 字段）
- [ ] **3g 端到端验证**：情感轨迹可视化 + 多步推理 demo + 工具调用流
- [ ] **里程碑**：SNN 在跨轮次情感维持 + 多步推理 + 工具调度上展现出 LLM+RAG 做不到的能力

### Phase 4 — 评测与优化

- [ ] 中文对话评测（CEval / CMMLU 子集 + 人工评测）
- [ ] 困惑度对比（纯 LLM vs vita）
- [ ] 延迟 / 内存 / 功耗 profile
- [ ] INT4 量化 + 边缘部署验证（手机 / Jetson / 浏览器）
- [ ] **里程碑**：边缘设备可运行

### Phase 5 — 持续学习闭环

- [ ] 用户反馈写入 SNN 的 STDP 闭环
- [ ] 长程记忆库自动维护（遗忘 / 巩固）
- [ ] 多用户隔离
- [ ] **里程碑**：与同一用户多次对话后能记住早期话题

## 阶段验收标准

| Phase | 验收指标 | 通过条件 |
|---|---|---|
| 1 | 中文单轮对话 | 人工评测可读性 ≥ 4/5 |
| 2 | SNN 检索准确率 | Top-5 命中率 ≥ 60% |
| 3 | 多轮对话一致性 | 10 轮对话后主题保持率 +20% |
| 4 | 边缘推理延迟 | CPU < 500ms / token |
| 5 | 持续学习 | 24h 后记得 ≥ 80% 早期话题 |

## 风险与缓解

| 风险 | 缓解措施 |
|---|---|
| llama.cpp 集成复杂度高 | 先用 Python transformers 跑通 PoC，再迁移到 C++ |
| SNN BPTT 收敛慢 | 接受 grad_norm≈100，靠 STDP 在线学习补充 |
| 投影矩阵训练数据不足 | 用 LCCC + WikiText-2 联合训练 |
| 边缘部署内存超限 | SNN 用 FP16, LLM 用 INT4, 总目标 < 2GB |
