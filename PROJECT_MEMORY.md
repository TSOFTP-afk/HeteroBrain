# THE TRUE AI - Project Memory

> 本文件是项目自带记忆文档，随项目目录一起迁移。每个新会话开始时优先读取本文件，
> 以快速恢复项目上下文。本文件是对 TRAE IDE memory 系统的补充，确保跨路径/跨工具的连续性。

## 项目定位

**THE TRUE AI** 是 SNN+Transformer 混合架构的工程化方案，瞄准 Transformer 的弱项赛道：
长序列时序预测、边缘低功耗推理、在线连续学习、多模态时序融合。

**不是**：通用语言建模、代码生成、知识问答、数学推理、参数规模竞争。

## 硬约束（不可违反）

1. **架构**：SNN+Transformer 混合，基底 LLM 用 MiniCPM5-1B（1B 参数，INT4 量化 0.5GB，中文 SOTA），面向边缘部署
2. **SNN 层**：60K 神经元，BPTT 架构，作为 token 真伪过滤器和在线连续学习组件
3. **转换层**：INT8-Spiking 编码（7 步位编码，64% 稀疏度），参考 SpikingBrain 2.0 的 T2H pipeline
4. **训练**：T2H 蒸馏（MiniCPM5-1B 作教师）+ 在线 STDP 微调
5. **算法**：BPTT 替代梯度（不是纯 STDP），跳过 P3-D 结构重构，PSW_ETA_ALPHA/BETA = 200.0
6. **突触**：STDP kernel 先算 delta_w 再更新 last_pre/post_spike；抑制突触用 [-W_MAX, 0] 钳位
7. **脑区**：每脑区按 80/20 兴奋/抑制划分 (SENSORY/ASSOCIATION/MOTOR)；运动区稳态目标 30Hz，其余 5Hz

## 当前路径布局（迁移后）

| 用途 | 路径 | 说明 |
|------|------|------|
| 项目工作区 | `F:\thetrueai\` | 全英文，IDE 工作目录 |
| 项目源码 | `F:\thetrueai\src\` | SNN+Transformer 代码 |
| 模型 GGUF | `F:\hb_models\MiniCPM5-1B-Q4_K_M.gguf` | 656MB，INT4 量化 |
| Jinja chat 模板 | `F:\hb_models\minicpm5-chat.jinja` | 9062 字节，从 GGUF 提取 |
| llama.cpp 源码 | `F:\hb_llama\` (junction) | 全英文 junction |
| **编译产物（新）** | `F:\thetrueai\build\bin\` | **llama-cli.exe + 9 个 DLL**（1.08GB） |
| 编译产物（旧） | `F:\hb_build\bin\` | 仅 llama-simple-chat.exe（已弃用） |
| 编译脚本 | `F:\thetrueai\scripts\hb_*.ps1` + `build_llama_cli.bat` | 全 ASCII 内容 |
| TRAE memory | `c:\Users\26455\.trae-cn\memory\projects\-F----thetrueai\` | 迁移后由 TRAE 自动创建 |

## 历史路径（已废弃，仅供追溯）

- 旧工作区：`f:\项目\THE TRUE AI\`（含中文，PS 5.1 编码问题，2026-07-26 迁出）
- 迁移原因：PowerShell 5.1 + 中文路径 = .ps1 内中文字面量被当 GBK 解析；编译/运行链路全程需 ASCII 路径

## 关键技术状态（截至 2026-07-26）

### llama.cpp 编译（Phase 1 已完成）
- 工具链：VS 2022 Build Tools + CMake 3.31.6 + Ninja + CUDA 13.3
- 目标 GPU：RTX 3060 (sm_86)
- CMake 选项：`-DGGML_CUDA=ON -DGGML_CUDA_ARCH=86 -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release`
- 编译产物（`F:\thetrueai\build\bin\`，约 1.08 GB）：
  - `llama-cli.exe`（10 KB，主入口）+ `llama-cli-impl.dll`（1.6 MB，CLI 逻辑）
  - `llama-server-impl.dll`（14 MB）、`llama-common.dll`（9.4 MB）、`llama.dll`（2.2 MB）
  - `ggml-cuda.dll`（30.9 MB）、`ggml-cpu.dll`、`ggml-base.dll`、`ggml.dll`、`mtmd.dll`
- 编译脚本：[scripts/build_llama_cli.bat](file:///f:/thetrueai/scripts/build_llama_cli.bat)（先调 vcvarsall，再 cmake + ninja llama-cli）

### MiniCPM5-1B GGUF 已知问题（已解决）
- GGUF 内 `general.architecture = "llama"`（应为 `minicpm5`），导致 `llama_model_chat_template()` 用 `llama.tokenizer.chat_template` 作 key 查找失败
- 后果：simple-chat 退回默认 chatml 模板 → MiniCPM5 输出严重退化（英文测试也输出 `Hello,` 后无限重复 `)`）
- 解决方案：用 `--chat-template-file F:\hb_models\minicpm5-chat.jinja --jinja` 显式指定 Jinja 模板
- Jinja 模板已从 GGUF offset 5113551 提取，9062 字节，写入 `F:\hb_models\minicpm5-chat.jinja`

### Phase 1 LLM 子系统打通里程碑（2026-07-27 完成）
- 测试脚本：[scripts/test_minicpm5_zh.bat](file:///f:/thetrueai/scripts/test_minicpm5_zh.bat)
- 测试日志：[logs/zh_inference2.log](file:///f:/thetrueai/logs/zh_inference2.log)
- 测试输入：`你好，请用中文简短介绍一下你自己（30字以内）。`
- 模型输出（含思考链）：
  - `[Start thinking]` 嗯，用户让我用中文介绍自己...（约 250 字思考过程）`[End thinking]`
  - `我是MiniCPM系列模型，由面壁智能开发。`
- 性能：Prompt 1150.5 t/s | Generation 248.2 t/s（RTX 3060，-ngl 99，全 GPU offload）
- 退出码：0（无错误）
- 结论：MiniCPM5-1B + Jinja 模板 + llama-cli 中文推理链路稳定，Phase 1 达成

### 模型推理测试约定
- 输入文件需 UTF-8 无 BOM（PS 5.1 写入用 `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)`）
- PowerShell 命令长度上限 32000 字符，复杂脚本必须写入 .ps1 文件执行
- .ps1 脚本内容**必须全 ASCII**（中文字面量会被 PS 5.1 当 GBK 误解析）；中文输出用 `([char]0xXXXX).ToString()` 拼接
- 中文 prompt 文件改用 Python 生成（`\u` 转义确保 ASCII 源码 → UTF-8 输出）

## 工作流约定

1. **所有 .ps1 脚本内容必须纯 ASCII**，中文用 `([char]0xXXXX)` 码点拼接
2. **编译/模型/源码路径全英文**：`F:\hb_*` 系列
3. **Write/Edit 工具操作工作区内文件**，路径可含 ASCII（迁移后已无中文）
4. **长命令写 .ps1**，避免 PowerShell 命令长度限制
5. **PowerShell 5.1 读 .ps1**：默认 UTF-16 LE BOM 或 ANSI（GBK），无 BOM 的 UTF-8 会被当 GBK；Write 工具写的是 UTF-8 无 BOM，所以 .ps1 内绝不能有非 ASCII 字符

## 路线图（简版）

- [x] Stage 0：环境搭建（CUDA 13.3、VS 2022、CMake、Ninja）
- [x] Stage 1：llama.cpp 编译（CUDA sm_86，simple-chat 已通过）
- [x] Stage 2：MiniCPM5-1B Q4_K_M 下载（656MB，已校验 GGUF magic header）
- [x] Stage 3：Jinja chat 模板从 GGUF 提取（9062 字节）
- [x] **Stage 4：编译 llama-cli**（Release 构建，`F:\thetrueai\build\bin\`，1.08 GB，含 CUDA 后端）
- [x] **Stage 5：用 llama-cli + Jinja 模板验证中文推理**（2026-07-27，输出正常，248 t/s）
- [x] **Phase 1 LLM 子系统打通里程碑达成**（Stage 0–5 全部完成）
- [x] **Stage 6：SNN 层集成**（Phase 2，2026-07-27 完成）
  - 移植路径：`legacy/stage2e/` → `src/snn/`（30+ 文件，60K 神经元 / 10.7M 突触）
  - 编译产物：`F:\thetrueai\build\snn\bin\snn_train.exe`（3MB，独立 target，不依赖 llama.cpp）
  - 构建脚本：[scripts/build_snn.bat](file:///f:/thetrueai/scripts/build_snn.bat)
  - 性能基线（10K 步字节模式 synthetic_input）：
    - perplexity = 9.86（达成 < 10 目标，参考 legacy 7.32）
    - accuracy = 66.66%（远超 legacy 39.62%）
    - P3-D 结构重建跳过 9 次（BPTT 模式，step 1000-9000）
    - 训练时长 ~70 分钟（笔记本 RTX 3060）
  - Checkpoint 验证：v3 格式，--resume 从 step 4000 恢复，loss 完全匹配（误差 0%）
  - LLM 桥接桩：[src/bridge/snn_llm_bridge.h](file:///f:/thetrueai/src/bridge/snn_llm_bridge.h)（header-only，Phase 3 替换为 llama.cpp 调用）
  - Spec 文档：[.trae/specs/port-snn-training-subsystem/](file:///f:/thetrueai/.trae/specs/port-snn-training-subsystem/spec.md)
- [ ] Stage 7：T2H 蒸馏管线（Phase 3，下一步）
- [ ] Stage 8：边缘部署验证

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
