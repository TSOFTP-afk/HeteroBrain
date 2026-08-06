---
name: Bug 报告 / Bug report
about: 报告一个可以复现的缺陷，帮助改进 VITA
title: "[bug] 简要描述"
labels: bug
assignees: ''

---

<!-- 提交前请阅读 CONTRIBUTING.md 与 CODE_OF_CONDUCT.md -->

**描述 / Description**
清晰描述问题是什么。

**复现步骤 / Steps to Reproduce**
1. 使用的完整命令（`snn_train ...` 或 `vita_engine ...`，含所有参数）
2. checkpoint 路径（如有）
3. 触发操作

**实际结果 / Actual Behavior**
贴出关键日志行（如 `[RateStat]`、`mod MSE`、报错栈、exit code）。

**期望结果 / Expected Behavior**
你期望发生什么。

**环境 / Environment**
- OS：
- GPU：
- CUDA 版本：
- 构建配置（MSVC / Debug|Release / 是否用构建脚本）：
- 相关数据文件与 commit（`git log -1`）：

**影响范围 / Scope**
- [ ] 训练（snn_train）
- [ ] 评估（--curriculum-eval）
- [ ] 对话引擎（CMD）
- [ ] serve 模式（HTTP API）
- [ ] 数据生成（Python 工具）

**补充 / Additional Context**
截图、CSV、相关 issue 链接等。
