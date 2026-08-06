---
name: Pull Request
about: 提交代码变更
title: ""
labels: ""
assignees: ""
---

<!-- 提交前请阅读 CONTRIBUTING.md 与 CODE_OF_CONDUCT.md -->

**描述 / Description**
变更做了什么、为什么做。

**关联 Issue / Related Issues**
`Closes #N`（如有）。

**变更类型 / Type of Change**
- [ ] feat：新功能
- [ ] fix：bug 修复
- [ ] docs：文档
- [ ] refactor：重构（行为不变）
- [ ] test：测试
- [ ] chore：构建/杂项

**验证 / Verification**
- [ ] `snn_train` / `vita_engine` 编译通过
- [ ] 冒烟验证通过（如 300 步冷启动 / 3 样本 eval）
- [ ] 与基线对比数据（如 "mod MSE 0.184 → 0.0383"）
- [ ] 引擎/bridge 测试通过（如 `test_emotion_bridge`、serve 三端点 curl）
- [ ] 训练产物（checkpoints/*.snn2e、*.csv）未提交

**文档 / Docs**
- [ ] README 状态表已同步（含日期与真实数据）
- [ ] 新增 CLI 参数/API 变更已更新相应文档（QUICKSTART / API.md / spec）

**测试计划 / How to Test**
给出评审者可复现的命令（含完整参数）。

**补充 / Additional Context**
设计文档链接、踩坑记录、截图等。
