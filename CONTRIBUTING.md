# 贡献指南 / Contributing Guide

欢迎参与 VITA（人工情感核心）的开发。本指南说明如何提交 Issue 与 Pull Request、遵循什么代码风格，以及如何让贡献流程对双方都高效友好。

- 打开 Issue 或 PR 前，请先阅读本指南与 [行为准则 CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)。
- 项目概览见 [README.md](./README.md)，架构与训练细节见 [docs/](./docs/)。

---

## 目录

- [开发环境](#开发环境)
- [代码结构](#代码结构)
- [提交 Issue](#提交-issue)
- [提交 Pull Request](#提交-pull-request)
- [代码风格](#代码风格)
- [测试要求](#测试要求)
- [文档要求](#文档要求)
- [提交信息规范](#提交信息规范)
- [行为准则](#行为准则)

---

## 开发环境

- Windows 10/11 + PowerShell（`cmd /c` 被安全策略禁止）
- MSVC 2022 + CUDA Toolkit 13.x + CMake ≥ 3.18
- NVIDIA GPU（CC ≥ 8.6）用于 SNN 训练与引擎验证
- Python 3.10+（课程数据生成/评测工具，可选）

构建方式见 [docs/QUICKSTART.md](./docs/QUICKSTART.md) 第 4 节。构建踩坑记录在文末"常见问题"。

## 代码结构

```
src/
├── snn/       # SNN 子系统（C++/CUDA）：scheduler、modulatory、bptt、记忆、生物拟真模块
├── bridge/    # 桥接层：affective_mapping、emotion_bridge、snn_feedback
├── vita/      # 引擎：engine、http_server、mini_json
└── llm/       # llama_backend（llama.cpp 封装）
data/          # 课程事件与叙事文本（JSONL/TXT）
curriculum_generator/  # 场景链生成器（Python）
legacy/        # 上一代 SNN 代码（只读归档，不修改）
docs/          # 设计/训练计划/bug 清单
```

**修改前请先读相关文档**：新增生物模块先看 [docs/bio-plausible-modules-spec.md](./docs/bio-plausible-modules-spec.md)；改动训练/评估流程先看 [docs/middle-school-training-plan.md](./docs/middle-school-training-plan.md)；`legacy/` 只读，改动请走新建路径。

## 提交 Issue

> 模板会自动填充，见 [.github/ISSUE_TEMPLATE/](./.github/ISSUE_TEMPLATE/)。

**提 Bug**（[bug_report 模板](./.github/ISSUE_TEMPLATE/bug_report.md)）请包含：

1. **环境**：OS / GPU / CUDA 版本 / 构建配置（MSVC 版本、Debug/Release）
2. **复现步骤**：完整命令（`snn_train ...` 或 `vita_engine ...`）+ 使用的 checkpoint 路径
3. **实际结果 vs 期望结果**：日志关键行（如 `[RateStat]`、`mod MSE`、报错行）
4. **影响范围**：是训练、评估、引擎还是 serve 模式

**提功能请求**（[feature_request 模板](./.github/ISSUE_TEMPLATE/feature_request.md)）请说明：动机（解决什么问题）、建议方案、替代方案、可验收标准。

**提问**：先在 [docs/](./docs/) 与 [README.md](./README.md) 中搜索，再开 Issue。

## 提交 Pull Request

1. **Fork** 本仓库，在 `main` 上开新分支：`git checkout -b feat/xxx`（命名见下）。
2. **小步提交**：每个逻辑变更一个 commit，提交信息见[规范](#提交信息规范)。
3. **本地验证**：按[测试要求](#测试要求)跑通相关检查，并自测你改动的路径。
4. **开 PR**（[PR 模板](./.github/PULL_REQUEST_TEMPLATE.md)）：描述变更、关联 Issue（`Closes #N`）、测试结果。
5. **等待评审**：维护者会逐条回复；请按反馈修改并推送（force-push 到自己的分支即可，**不要 force-push main**）。

**分支命名建议**：`feat/<功能>`、`fix/<bug>`、`docs/<文档主题>`、`refactor/<主题>`、`chore/<杂项>`。

**PR 检查清单**：

- [ ] 变更是否与项目定位一致（SNN 作为情感核心，不微调 LLM、w_tool=0 不恢复工具训练）
- [ ] 是否更新了受影响的文档（README / docs 相关 spec / API.md）
- [ ] 是否补充/更新了测试（见下）
- [ ] 是否新增了外部依赖（如新增 GGUF 模型、Python 库）并在 PR 中说明
- [ ] 提交信息是否符合规范

## 代码风格

**C++ / CUDA**（`src/snn`、`src/bridge`、`src/vita`、`src/llm`）：

- 命名：类 `PascalCase`，函数/变量 `snake_case`，常量 `SCREAMING_SNAKE_CASE`，成员尾缀 `_`（如 `weights_freeze_`）
- 头文件防护 `#ifndef SNN_STAGE2E_XXX_H`
- 注释用**中文**，关键机制处说明"为什么"（本仓踩坑多，注释是防回退的第一道防线）
- CUDA kernel 与 host 端接口分离：`.cuh` 声明 + `.cu` 实现；新模块先看 `insula_kernels.cuh/.cu` 与 `vta_kernels.cuh/.cu` 的既有范式
- 跨命名空间可见性：宏/常量用前缀限定；`stage2e::` 命名空间的常量在 `main.cpp`（非命名空间）不可直接用
- `__host__ __device__` 辅助函数放 `.cuh` 内联

**Python**（`src/snn/tools/`、`curriculum_generator/`）：

- 遵循 PEP 8；脚本尽量无外部依赖（或写入 requirements）
- 数据生成脚本保持可复现（固定 seed）

**通用**：

- 不要引入不必要的抽象；三行相似代码好过一次过早封装
- 不提交 `checkpoints/`、`*.csv` 训练产物、模型文件（已在 .gitignore）
- 不提交密钥/本地绝对路径（`F:\hb_models\...` 仅出现在文档示例中）

## 测试要求

- **C++/CUDA 改动**：至少保证 `snn_train` 与 `vita_engine` 能编译通过；逻辑改动尽量附带最小冒烟验证（如 300 步冷启动 / 3 样本 eval）
- **新增模块**：参照 M1-M4 的交付记录（见 [docs/bio-plausible-modules-spec.md](./docs/bio-plausible-modules-spec.md)），记录踩坑与验证输出
- **引擎/bridge 改动**：跑通 `test_emotion_bridge`（`src/bridge/`），serve 改动至少 curl 一次三个端点
- 训练相关的变更必须给出**同参数基线对比**（例如"90K→110K，mod MSE 0.184→0.0383"），没有对比的数值结论会被打回

## 文档要求

- 改 README 中的"当前状态"表时，同步更新日期与真实数据
- 新增 CLI 参数必须同步到 `--help` 文案与相关文档（QUICKSTART / API / spec）
- API 变更必须更新 [docs/API.md](./docs/API.md)（端点、请求/响应、错误码、示例）

## 提交信息规范

采用 Conventional Commits（英文动词，主题可用中文描述）：

```
<type>(<scope>): <subject>

<optional body: 为什么这么改 / 验证结果>
```

- `feat`：新功能（如生物模块 M4 脑岛）
- `fix`：bug 修复
- `docs`：文档（README / docs / 模板）
- `refactor`：重构（行为不变）
- `test`：测试
- `chore`：构建/杂项

示例：

```
feat(insula): 脑岛内感受模块（5 维 × 200 LIF + 窗口累计读取）

M4 交付：hunger/temp/comfort/fatigue/pain 经 15 柱 one-hot 注入，
窗口发放率→NE/Oxy 调制；修复读取相位踩坑（偶数步读取为 0）。
冒烟验证：hunger=0.500 comfort=0.500 @step200。
```

PowerShell 提交（不支持 heredoc）：

```powershell
git commit -m "docs(contributing): 新增贡献指南与社区文档"
```

## 行为准则

所有参与者须遵守 [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)：保持尊重、包容、建设性；不接受人身攻击、骚扰或恶意行为。违反者将被移出社区。

---

感谢你的贡献。任何"让 SNN 真正感受"的前进，都值得被认真对待。
