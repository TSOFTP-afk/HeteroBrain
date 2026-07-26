# HeteroBrain 路线图

> 从 SNN 研究项目转型为异构中文对话 AI 引擎的工程化路线。

## 已完成

### Phase 0 — 工程骨架 ✅

- [x] 创建 HeteroBrain GitHub 仓库
- [x] 旧 SNN 代码迁移到 `legacy/`（126 文件, 1.19MB）
- [x] 顶层 `CMakeLists.txt` / `.gitignore` / `LICENSE` (Apache 2.0)
- [x] 工程化目录骨架 `src/{snn,llm,bridge,heterobrain}/`
- [x] 配置文件 `configs/default.yaml`
- [x] 全新 README，告别 SNN/STDP 研究阶段

## 进行中

### Phase 1 — LLM 子系统打通 (MVP 对话)

**目标**：跑通 MiniCPM5-1B INT4 推理，可单轮中文对话（无 SNN）。

- [ ] 引入 llama.cpp 作为 `third_party/llama.cpp` 子模块
- [ ] 实现 `src/llm/llama_runner.cpp/.h`：GGUF 加载 + 单批推理 + 流式输出
- [ ] 实现 `src/llm/tokenizer_bridge.cpp/.h`：BPE 编码/解码
- [ ] 实现 `src/llm/prompt_builder.cpp/.h`：system + history + user 拼接
- [ ] 实现 `src/heterobrain/main.cpp`：CLI 入口 + 交互循环
- [ ] 编写 `scripts/download_models.py`：拉取 MiniCPM5-1B GGUF
- [ ] **里程碑**：`./heterobrain_engine --interactive` 可单轮中文对话

## 待启动

### Phase 2 — SNN 子系统移植

**目标**：从对话历史中检索相关片段。

- [ ] 从 `legacy/stage2e` 移植 BPTT trainer（35KB CUDA 实现）
- [ ] 移植 PCA 签名提取（`legacy/stage2e/pca_kernels.cu`）
- [ ] 实现 `src/snn/memory_index.cu`：基于 PCA 签名的 Top-K 检索
- [ ] 实现 `src/snn/online_stdp.cu`：用户反馈触发的局部 STDP 更新
- [ ] **里程碑**：SNN 能从 100 轮对话历史中检索 Top-5 相关片段

### Phase 3 — Bridge 转换层

**目标**：SNN 检索结果影响 LLM 生成质量。

- [ ] PyTorch 离线训练 `[2048, 1024]` spike → embedding 投影矩阵
- [ ] 实现 `src/bridge/spike_embedding.cpp`：spike → embedding
- [ ] 实现 `src/bridge/truth_filter.cpp`：token 真实性筛选
- [ ] 实现 `src/bridge/pca_projection.cpp`：LLM embedding ↔ SNN PCA
- [ ] 三子系统联调
- [ ] **里程碑**：SNN 检索注入后，多轮对话一致性显著提升

### Phase 4 — 评测与优化

- [ ] 中文对话评测（CEval / CMMLU 子集 + 人工评测）
- [ ] 困惑度对比（纯 LLM vs HeteroBrain）
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
