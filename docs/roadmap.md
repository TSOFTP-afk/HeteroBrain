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
- [ ] 实现 `src/heterobrain/main.cpp`：CLI 入口 + 交互循环

## 待启动

### Phase 2 — SNN 训练子系统移植 ✅

**目标**：将 `legacy/stage2e/` 的 SNN 训练子系统整体移植到新路径 `src/snn/`，作为 Phase 3 T2H 蒸馏的前置依赖。

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

**注**：原计划的 `memory_index.cu` / `online_stdp.cu` 检索接口推迟到 Phase 3 T2H 蒸馏时实现，因为 SNN 的检索能力需要先有 LLM embedding 对接才能定义 Top-K 语义。

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
