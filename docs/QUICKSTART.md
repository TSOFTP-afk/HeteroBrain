# VITA 快速上手指南 / Quick Start Guide

> 从零开始构建 VITA（SNN × LLM 异构情感核心）并在本地跑通「训练 → 评估 → 对话引擎 → OpenAI 兼容 serve」全链路。
> 详细说明见 [README.md](../README.md)；API 契约见 [API.md](API.md)。

## 1. 环境要求 / Requirements

| 组件 | 版本 | 用途 |
|---|---|---|
| Windows | 10/11（x64） | 开发平台 |
| NVIDIA GPU | compute capability ≥ 8.6，显存 ≥ 6GB | SNN 训练 + LLM 推理（RTX 3060 验证通过） |
| CUDA Toolkit | 13.x | SNN 子系统（CUDA C++） |
| CMake | ≥ 3.18 | 构建（**不在 PATH**，用完整路径或 VS 环境） |
| MSVC | VS2022 + C++ 桌面开发工作负载 | 编译 |
| Python | 3.10+ | 课程数据生成与评测（可选） |
| llama.cpp | 最新（脚本自动克隆） | LLM 推理后端 |

> ⚠️ `cmd /c` 被本机安全策略禁止，所有命令请在 **PowerShell** 中执行。

## 2. 获取代码 / Get the Code

```powershell
git clone https://github.com/TSOFTP-afk/VITA.git
cd VITA
```

## 3. 准备模型与数据 / Models & Data

**LLM 模型**（GGUF，INT4）：

```powershell
# 手动下载后放到 F:\hb_models\（或 scripts/download_models.py 自动下载）
# 默认推荐：Qwen3-4B-Q4_K_M.gguf（2.50GB）
```

**课程数据**：仓库已内置合并后的长线课程数据，无需生成：

- `data/events/curriculum_all.jsonl`（204 段，正负比 2.38:1）
- `data/scripts/story_text_all.txt`（21936 字符）

如需重新生成课程数据（Python）：

```powershell
python src/snn/tools/generate_curriculum_data.py --stage middle_school
python src/snn/tools/merge_new_arcs.py        # 合并长线数据
```

## 4. 构建 / Build

### 4.1 SNN 训练子系统（snn_train）

```powershell
# 先进入 VS DevShell 设置 MSVC 环境变量（或从"开发人员 PowerShell"启动）
cmake --build build/snn --target snn_train
```

产物：`build/snn/bin/snn_train.exe`

### 4.2 异构引擎（vita_engine）

```powershell
# 构建脚本（含 llama.cpp 克隆与 -Xcompiler=/utf-8 等踩坑处理）
powershell -ExecutionPolicy Bypass -File scripts/hb_build_cli.ps1
```

产物：`build/root/bin/vita_engine.exe`（约 35MB）

## 5. 训练 / Training

```powershell
# 续跑训练（110K → 120K 示例：事件注入 + 具身 + 连续课程）
snn_train --resume checkpoints/middle_1a_longarc_all/ckpt_step110000.snn2e --steps 120000 `
    --curriculum data/events/curriculum_all.jsonl --curriculum-stage 1 --curriculum-continuous `
    --bptt-window-size 400 --learning-rule n3f --curriculum-lr 0.0100 `
    --embodied --embodied-scene hunger_feeding --input-mode byte `
    --text data/scripts/story_text_all.txt --seed 42
```

- `--steps` 必须大于 resume 步数
- 训练速度约 140ms/步（20K 步 ≈ 47 分钟）
- 检查点写入 `checkpoints/`（已被 git-ignore）

## 6. 评估 / Evaluation

> ⚠️ 两个铁律：**eval 必须 `--bptt-window-size 400`**（默认 50 步 < 事件 offset 100，事件永不注入，MSE 纯失真）；**训练带 `--curriculum-continuous` 则 eval 必须同带**（否则 gtr 口径不一致）。

```powershell
# 120 样本评估（约 1.8h）
snn_train --resume checkpoints/middle_1a_longarc_all/ckpt_step110000.snn2e --steps 111200 `
    --curriculum-eval --curriculum-eval-samples 120 --curriculum data/events/curriculum_all.jsonl `
    --curriculum-stage 1 --curriculum-continuous --bptt-window-size 400 --input-mode byte `
    --text data/scripts/story_text_all.txt --seed 42
```

**关注指标**（判据两级）：
1. **事件可辨性**：`[RateStat] event_subregion` ratio > 2（事件子区域激活 vs 全皮层）
2. **readout 可纠正性**：`mod MSE`（110K 实测 0.0383 为历史最优）

**情绪涌现诊断**（`--eval-emergent`，L1 事件扩散 / L2 readout 权重分布 / L3 模式效价区分度）：

```powershell
# 在上一条 eval 命令基础上加 --eval-emergent
```

## 7. 对话引擎 / Interactive Dialogue Engine

```powershell
vita_engine.exe --resume checkpoints/middle_1a_longarc_all/ckpt_step110000.snn2e `
    --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf `
    --mod-interval 10 --steps-per-turn 10 --memory-budget-mb 4096
```

流程：resume → 每轮 SNN 推进 10 步 → Affective 读出（DA/ACh/NE/5HT/GABA/Oxy + PAD）→ EmotionBridge 情感调制（文字 + logit_bias + 采样参数）→ llama.cpp 生成中文回复。每轮约 3-5 秒。

## 8. OpenAI 兼容 Serve / OpenAI-Compatible Serve

```powershell
vita_engine.exe --serve --port 8899 --api-key thetrueai --model-name thetrueai `
    --resume checkpoints/<ckpt>.snn2e --llm F:\hb_models\Qwen3-4B-Q4_K_M.gguf
```

- Base URL：`http://127.0.0.1:8899/v1`
- 端点：`GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/world`（详见 [API.md](API.md)）
- 鉴权：`Authorization: Bearer thetrueai`
- 任意 OpenAI 兼容客户端即可接入（如第三方 Chat UI 配置 API 主机/Key/模型名）

## 9. 常见问题 / Troubleshooting

| 问题 | 原因与解法 |
|---|---|
| 构建报 `config.h(440) undefined` | heterobrain/engine 目标编译 CUDA 源必须加 `-Xcompiler=/utf-8`（GBK 误解析中文注释） |
| `LNK1104 无法打开 bin\snn_train.exe` | eval/训练进程仍占用 exe → 先停掉进程再重编译 |
| checkpoint 拒绝覆盖（code=4） | 检查点文件已存在且不允许覆盖 → 重跑前删除旧 checkpoint 目录 |
| eval 结果全是 baseline（mod MSE ≈ 0.35） | 缺 `--bptt-window-size 400` 或事件未进网络；核对命令是否与训练同参数 |
| 中文回复乱码（.NET 客户端） | 请求/响应须 UTF-8；PowerShell 请求体用 `[Text.Encoding]::UTF8.GetBytes()` 传 byte[] |
| PowerShell 中文请求体服务端收坏字节 | `Invoke-RestMethod -Body <string>` 默认 Latin1 → 用 byte[] |
| `GGML_ASSERT(n_tokens_all <= n_batch)` | 多轮历史超 batch → `n_batch=2048` + 分段 prompt 解码 |
| 模型回复空（`<|im_end|>`） | 该模型（MiniCPM5-1B）已弃用 → 统一用 Qwen3-4B |

## 10. 下一步 / Next Steps

- 读 [docs/bio-plausible-modules-spec.md](bio-plausible-modules-spec.md)：生物拟真模块（杏仁核/HPA/脑岛/VTA-DA）设计与踩坑
- 读 [docs/middle-school-training-plan.md](middle-school-training-plan.md)：课程训练全貌与命令模板
- 参与贡献：见 [CONTRIBUTING.md](../CONTRIBUTING.md)
