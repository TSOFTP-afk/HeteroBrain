# THE TRUE AI - Project Memory

> 本文件是项目自带记忆文档，随项目目录一起迁移。每个新会话开始时优先读取本文件，
> 以快速恢复项目上下文。本文件是对 TRAE IDE memory 系统的补充，确保跨路径/跨工具的连续性。
>
> **2026-07-31 文档归档**：历史方案文档已统一移至 `docs/archive/`。当前 `docs/` 只保留三份活文档：
> - [docs/developmental-training-master-spec.md](file:///f:/thetrueai/docs/developmental-training-master-spec.md) — 训练范式权威契约
> - [docs/enlightenment-design-spec.md](file:///f:/thetrueai/docs/enlightenment-design-spec.md) — 启蒙期设计契约（2026-08-01，情绪分层 × 沙盒架构 × 硬编码边界，修订 master-spec 启蒙期部分）
> - [docs/roadmap.md](file:///f:/thetrueai/docs/roadmap.md) — 项目阶段路线图
>
> 本文件保留**硬约束 + 阶段索引 + 工作流约定**，详细技术状态见上述三份活文档。

## 项目定位

**THE TRUE AI** 是 SNN+Transformer 混合架构的工程化方案，瞄准 Transformer 的弱项赛道：
长序列时序预测、边缘低功耗推理、在线连续学习、多模态时序融合。

**不是**：通用语言建模、代码生成、知识问答、数学推理、参数规模竞争。

## 硬约束（不可违反）

> **2026-07-30 方向更新**：SNN 定位从"token 真伪过滤器 + T2H 蒸馏"重构为"前额叶认知调度器"三层架构。
> 详见 [docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)。
> 下方约束 2-4 已更新，旧 T2H 蒸馏方案废弃。

1. **架构**：SNN+Transformer 混合，基底 LLM 用 MiniCPM5-1B（1B 参数，INT4 量化 0.5GB，中文 SOTA），面向边缘部署
2. **SNN 层**：60K 神经元，BPTT 架构，定位为"前额叶认知调度器"三层架构
   - Layer 1 情感核心：6 维调质向量 (DA/5HT/NE/ACh/GABA/催产素) + LLM 神经调制
     - 注：当前调质信号全来自 SNN 内部状态 (spike/ TD error/ 解码误差)，情绪无语义锚点
     - Phase 3b 起增加事件驱动注入（外部事件→基因硬编码映射→调质，见方向文档 §3.4）
   - Layer 2 认知工作空间：256 槽 BlackboardSlot + 读写头（替代原 50 槽 WM）
     - 事件驱动调质注入与 3b 一并实现，避免重复改动 launch_modulatory 接口（§3.4）
   - Layer 3 工具编排：6 工具 + 状态驱动调用信号 + DA reward RL 训练（复用 PSW 突触）
3. **转换层**：PCA 签名（50 维）↔ LLM embedding（bge-small-zh 512 维）双向桥接 + AffectiveState 调制信号注入（含内部认知信号 + 外部事件信号，§3.4 落地后）
4. **训练**：文字训练用 BPE token 流 + BPTT 代理梯度（next-token 预测，DA = 1 - 解码误差）；工具调用策略用 RL（DA reward + PSW 贝叶斯突触）
   - 注："DA = 1 - 解码误差"是纯内部信号；事件驱动注入后 DA 将叠加外部事件奖赏（如 hunt_success→DA↑），当前版本情绪无语义锚点（§3.4）
5. **算法**：BPTT 替代梯度（不是纯 STDP），跳过 P3-D 结构重构，PSW_ETA_ALPHA/BETA = 200.0
6. **突触**：STDP kernel 先算 delta_w 再更新 last_pre/post_spike；抑制突触用 [-W_MAX, 0] 钳位；6 维调质门控 M_ij = σ(Σ receptor·conc)
   - 注：conc 的来源将扩展为内部 signal + 外部事件注入（§3.4），动力学方程不变
7. **脑区**：6 脑区分区 (L4/L23/L5/L6/前额叶/运动)，每层内按 80/20 兴奋/抑制划分；抑制亚型 FS/LTS/SOM = 50:30:20

## 当前路径布局（迁移后）

| 用途 | 路径 | 说明 |
|------|------|------|
| 项目工作区 | `F:\thetrueai\` | 全英文，IDE 工作目录 |
| 项目源码 | `F:\thetrueai\src\` | SNN+Transformer 代码 |
| 模型 GGUF | `F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf` | 656MB，INT4 量化 |
| Jinja chat 模板 | `F:\hb_models\minicpm5-chat.jinja` | 9062 字节，从 GGUF 提取 |
| llama.cpp 源码 | `F:\hb_llama\` (junction) | 全英文 junction |
| 编译产物 | `F:\thetrueai\build\bin\` | llama-cli.exe + 9 个 DLL（1.08GB） |
| 编译脚本 | `F:\thetrueai\scripts\hb_*.ps1` + `build_llama_cli.bat` | 全 ASCII 内容 |
| TRAE memory | `c:\Users\26455\.trae-cn\memory\projects\-f-thetrueai\` | 迁移后由 TRAE 自动创建 |

## 阶段索引（详细见 [docs/roadmap.md](file:///f:/thetrueai/docs/roadmap.md)）

- [x] **Phase 0** 工程骨架 — 完成
- [x] **Phase 1** LLM 子系统打通 — 2026-07-27 完成（MiniCPM5-1B + Jinja 模板，248 t/s）
- [x] **Phase 2** SNN 训练子系统移植 — 2026-07-27 完成（60K 神经元 / 10.7M 突触，perplexity=9.86）
- [~] **Phase 3** SNN 认知调度核心（情感 + 工作空间 + 工具编排）— **进行中**
  - 训练范式权威契约：[docs/developmental-training-master-spec.md](file:///f:/thetrueai/docs/developmental-training-master-spec.md)
  - 方向推演记录（已归档）：[docs/archive/snn-emotion-and-workspace-direction.md](file:///f:/thetrueai/docs/archive/snn-emotion-and-workspace-direction.md)
  - 3a-A 6 维调质基础 ✅ / 3a-B 稳态补偿 ✅ / 3a-C1 事件驱动注入 ✅ / 3a-C2 事件叠加修复 ✅
  - 3a-D1 具身发育训练 ✅ / 3a-D2 启蒙期 ⬜ / 3a-D3 课程训练 ⬜（N3F 20K 对照完成，详见 [N3F 设计总览 §7 深层缺陷](file:///f:/thetrueai/docs/N3F-设计总览与实现说明.md#7-深层缺陷分析2026-08-01-代码审查发现)）/ 3a-D4 成年交付 ⬜ / 3a-D5 个性化 ⬜
  - 3b-3g 待启动（认知黑板 / 黑板-LLM 桥接 / 工具编排 / 工具 RL / 海马溢出 / 端到端验证）
- [ ] **Phase 4** 评测与优化
- [ ] **Phase 5** 持续学习闭环
- [ ] **Phase 6** 桌面应用（GUI + EXE 打包）

## 工作流约定

1. **所有 .ps1 脚本内容必须纯 ASCII**，中文用 `([char]0xXXXX)` 码点拼接
2. **编译/模型/源码路径全英文**：`F:\hb_*` 系列
3. **Write/Edit 工具操作工作区内文件**，路径可含 ASCII（迁移后已无中文）
4. **长命令写 .ps1**，避免 PowerShell 命令长度限制
5. **PowerShell 5.1 读 .ps1**：默认 UTF-16 LE BOM 或 ANSI（GBK），无 BOM 的 UTF-8 会被当 GBK；Write 工具写的是 UTF-8 无 BOM，所以 .ps1 内绝不能有非 ASCII 字符
6. **MiniCPM5-1B GGUF 已知问题（已解决）**：GGUF 内 `general.architecture="llama"`（应为 `minicpm5`），需用 `--chat-template-file F:\hb_models\minicpm5-chat.jinja --jinja` 显式指定 Jinja 模板
7. **模型推理测试约定**：输入文件需 UTF-8 无 BOM；中文 prompt 文件改用 Python 生成（`\u` 转义确保 ASCII 源码 → UTF-8 输出）

## 历史路径（已废弃，仅供追溯）

- 旧工作区：`f:\项目\THE TRUE AI\`（含中文，PS 5.1 编码问题，2026-07-26 迁出）
- 迁移原因：PowerShell 5.1 + 中文路径 = .ps1 内中文字面量被当 GBK 解析；编译/运行链路全程需 ASCII 路径
- 迁移详情见 [docs/archive/migration/migration-to-F-thetrueai.md](file:///f:/thetrueai/docs/archive/migration/migration-to-F-thetrueai.md)
- legacy 代码复用指南见 [docs/archive/migration/migration_from_legacy.md](file:///f:/thetrueai/docs/archive/migration/migration_from_legacy.md)

## 用户偏好（来自 TRAE user_profile）

- 沟通语言：中文
- UI 语言：中文
- 应用形态：GUI 界面 + 易打包 EXE
- 软件分发：偏好不开源
- 技术栈：Python（熟练）、C# mod、CUDA C++、桌面应用开发
- 领域：计算物理（大一本科）

## 联系点

- GitHub 仓库：私有（不开源），用 `gh` CLI 操作
- 旧 SNN 代码：`legacy/` 目录（已清理大文件，保留代码参考）
