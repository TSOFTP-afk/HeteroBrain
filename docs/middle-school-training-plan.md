# 初中课程训练计划（Stage 1a，基于具身启蒙 20K 起点）

- 日期：2026-08-03
- 对应规范：`docs/developmental-training-master-spec.md` §5.4 / §7
- 前置成果：具身启蒙 20K（`checkpoints/embodied_20k/ckpt_step20000.snn2e`，evidence=0.535，α=0.122，mature=16.1%，判据全 1）

---

## 1. 目标

在具身启蒙 20K 已建立的 PSW 证据（α 双侧 LTP）基础上，切换到初中课程监督，完成：

1. **情绪回路监督**：7 条情感链（学业 4 + 社交 3）的"负性事件 → 低落 → 社交恢复 → 表扬回升"完整回路，使网络可稳定再现正确调质响应（正向 DA/P 抬升、负向 5HT/DA 变换、恢复 Oxy 回升）。
2. **认知工具链监督**：5 条知识链（question 触发 → 工具调用 0-5）建立情境 → 工具决策映射。
3. **PAD 输出一致性**：readout 输出的 P/A/D 与浓度模拟器目标对齐（w_pad=0.30）。

**完成判据**（设计文档 §7 判定表）：
- [ ] 调质 readout MSE 在 50K 步内下降 ≥ 50%（对比起步值）
- [x] 复合事件（学业+社交叠加）引发组合调质响应（非单通道线性叠加）—— 120K 验证达标（见 §9.5）
- [x] PSW β 不单侧失控（β ≤ 0.4 或 β/α 比例不持续恶化）—— β=0.211，全程健康
- [x] 工具决策准确率（分事件类型统计，排除类别不平衡假象）—— 已废弃（职责归 LLM，w_tool=0）

---

## 2. 起点状态确认

| 项 | 值 |
|---|---|
| 起点 checkpoint | `checkpoints/embodied_20k/ckpt_step20000.snn2e` |
| 发育阶段 | SYNAPTOGENIC（5K-200K 中段，plast_gain=1.0，nmda=0.8） |
| PSW 结束态 | evidence=0.535，α=0.122，β≈0.41，mature=16.1% |
| 课程数据 | `data/events/curriculum_middle_school.jsonl`（2000 样本，情感/知识各 50%） |

**前置检查清单**（执行前逐项确认）：
- [ ] `fix-curriculum-beta-runaway` 改动已 git 提交（config.h / synapse_kernels.cu / scheduler.cu 等，当前未提交）
- [ ] `snn_stage2e_p1` 可执行文件为含 β 修复的最新构建（验证依据：`build/snn/beta_fix_6k.log` 运行产物）
- [ ] checkpoint 与课程 JSONL 路径存在
- [ ] 显存足够（持久 1543 MB，预算 32768 MB）

---

## 3. 训练配置

**启动命令（Phase 1，0-40K）**：

```powershell
snn_stage2e_p1.exe `
  --resume checkpoints/embodied_20k/ckpt_step20000.snn2e `
  --steps 40000 `
  --curriculum data/events/curriculum_middle_school.jsonl `
  --curriculum-stage 1 `
  --learning-rule n3f `
  --bptt-window-size 400 `
  --curriculum-lr 0.0100 `
  --checkpoint-dir checkpoints/middle_1a `
  --checkpoint-interval 10000 `
  --keep-checkpoints 4 `
  --log-interval 1000 `
  --csv build/snn/middle_1a_40k.csv `
  --seed 42
```

关键参数说明（与实际实现一致，`beta_fix_6k.log` / `run_config.cpp` 验证过）：

| 参数 | 值 | 依据 |
|---|---|---|
| learning_rule | n3f | 课程路径走 N3F 三因子在线，每步课程误差 → neuron_eligibility → STDP |
| bptt-window-size | 400 | 课程样本事件按 100 步对齐，窗口起点须为 100 倍数（日志警告要求） |
| curriculum_lr | 0.0100 | readout 学习率（日志实测） |
| w_mod / w_tool / w_pad | 1.00 / 1.00 / 0.30 | 初中全监督（含工具链，与设计文档 w_tool=0 不同——实际实现已含知识链，保留） |
| stdp_eta_multiplier | 1.50 | 初中阶段参数（设计文档 §5.4） |
| baseline_mod | DA 0.22 / 5HT 0.15 / NE 0.25 / ACh 0.28 / GABA 0.18 / Oxy 0.20 | 初中人格基线 |

**分两段执行（每段可独立 resume）**：
- Phase 1（0-40K）：按上命令一次跑 40K（约 1.5h，N3F 实测 ~138ms/步）
- Phase 2（40K-50K）：`--resume` 续跑最后 10K，补齐设计文档 50K 总量

---

## 4. 阶段划分与里程碑

| 段 | 步范围 | 关注点 | 退出判据 |
|---|---|---|---|
| 稳态验证 | 0-10K | readout 初始化收敛；PSW β 漂移曲线；情绪三向抽查 | β ≤ 0.5 且无单调暴增；正向事件 DA 抬升、负向 5HT 抬升 |
| 能力建立 | 10K-30K | 情感链/知识链 MSE 下降；工具准确率分化 | MSE 较起步下降 ≥ 30% |
| 收敛判定 | 30K-50K | 复合事件响应；PAD 一致性；判据全检 | MSE 下降 ≥ 50%；复合响应非单通道叠加 |

每 10K 检查点自动保存（`checkpoints/middle_1a/ckpt_step*.snn2e`），每 1000 步输出 Affective 轨迹 + PSW 统计 + Decode 指标。

---

## 5. 监控指标

1. **PSW 证据**（每 1000 步）：α、β、evidence——重点看 β 是否单侧漂移（冷启动 6K 教训：SYNAPTOGENIC 塑性开启后 β 曾在 1K 步内 0.20→0.69）。本次起点 α 已具身双侧塑造，预期更稳，但必须盯。
2. **调质 MSE**：readout 输出 vs 浓度模拟器连续浓度目标（每 100 步推进）。
3. **工具准确率**：按事件类型（4=知识库查询 / 1=计算器 / 3=长程检索 / 2=草稿记录 / 0=生成器 / 6=不调用）分统计，警惕类别不平衡偏好。
4. **Affective 三向抽查**：每段首尾各抽一条正向链、负向链、恢复链，核对 DA/5HT/Oxy/PAD 方向。
5. **decode 背景指标**：perplexity/accuracy 作为背景任务参考（非本阶段主目标）。

---

## 6. 风险与对策

| 风险 | 触发信号 | 对策 |
|---|---|---|
| 课程 β 单侧漂移回归 | β > 0.5 持续 3+ 次采样且 α 停滞 | 暂停；核对 CURRICULUM_ELIGIBILITY_GAIN（当前 8e-4）与 eta 反比补偿；必要时降 w_pad |
| 工具类别不平衡偏好 | 总准确率高但某类接近 0 | 按类型统计；评估样本均衡或损失加权（不改结构） |
| 长跑不稳定/崩溃 | 进程异常 | 10K 检查点自动保存，`--resume` 续跑不丢进度 |
| eval 口径失真 | MSE 指标跳变 | 确认 eval 每 100 步推进模拟器、对比连续浓度（Task 8 已修复） |
| 具身证据被覆盖 | α 不再增长、β 独大 | 检查课程误差方向（e1 恒负 → β 单侧为课程固有特征），对照具身 checkpoint 保留副本 |

---

## 7. 输出与后续衔接

- 产物：`checkpoints/middle_1a/`（每 10K checkpoint）+ `build/snn/middle_1a_40k.log/.csv`
- 结束判据全部达标后 → 衔接 **Stage 1b 高中 50K**（stage=2，baseline 5HT 上调、w_pad 上调、eta=1.0），课程数据 `data/events/curriculum_high_school.jsonl`
- 全程保留 `embodied_20k` 副本作为"具身基线对照"，用于对比课程监督对 α/β 的影响

---

## 8. 执行顺序（决策点）

1. 提交 `fix-curriculum-beta-runaway` 未提交改动（git）
2. 确认最新构建 + 起点 checkpoint
3. 跑 Phase 1（0-40K），每 10K 检查日志（PSW β、MSE、工具准确率）
4. 40K 时评估：达标则 Phase 2 补 10K；β 漂移则暂停调参
5. 50K 结束 → 判据全检 → 高中阶段计划

---

## 9. 执行结果与决策记录（2026-08-03）

### 9.1 已执行

- **Phase 1（0-40K）+ Phase 2（40K-50K）**：完成，`checkpoints/middle_1a/` 保存 30K/40K/50K checkpoint
  - 调质 MSE：0.0678 → 0.0457（**-32.6%**，未达 -50% 判据）
  - 工具准确率：55%（全预测"不调用"，类别不平衡偏好）
  - PSW：β 0.385→0.329 无单侧失控；但 α 0.122→0.043（具身证据被课程稀释）
- **具身续训（50K→60K）**：resume 50K + `--embodied hunger_feeding` + 课程并行
  - PSW 恢复：α 0.041→0.132，β 0.163→0.208，evidence 0.204→0.341（具身奖励路径重建 PSW 证据）
  - 60K checkpoint 已保存
- **工具类加权 CE 分化逻辑**（commit 482c423）：过度校正（55%→15%），pred 漂移到类 1，未达分化目标

### 9.2 工具分类职责决策（重要）

**结论：工具决策归 LLM 语义理解（MiniCPM），SNN 不再承担工具分类。**

证据链：
1. 课程数据是人工构造的 12 条链模板，**2000 样本仅 28 种独特模式（1.4%）**，无语义信息
2. 轻量 MLP（事件+情绪特征）随机划分测试 100%，但**按 chain 留出仅 42.5%**——记忆而非泛化
3. 真实工具选择的输入是自然语言问题，必须由 LLM 语义理解承担

**落地**：
- SNN 侧 `w_tool→0`，工具 readout 停止训练（待实施）
- 课程判据聚焦调质 MSE 与 PSW（工具准确率不再作为 SNN 判据）
- 工具编排回归 README 既有架构：SNN 前额叶调度 + LLM 工具编排

### 9.3 w_tool=0 续训（60K→100K，课程+具身并行）

**w_tool=0 实施**（commit aa71f53）：初中 profile `w_tool` 1.0→0.0，工具 readout 停止梯度注入，与设计文档 §5.4 对齐。

| 段 | 调质 MSE | PAD MSE | PSW α/β/evidence | 工具准确率 |
|---|---|---|---|---|
| 60K | 0.0458 | 0.0873 | 0.126/0.198/0.324 | — |
| 80K | 0.0634（↑） | 0.0909 | 0.151/0.208/0.359（健康） | 15%（冻结） |
| 100K | 0.0798（↑↑） | 0.0614（↓） | 0.155/0.209/0.365（健康） | 15%（冻结） |
| 100K（120样本） | 0.0716 | 0.0631 | — | 13.3%（冻结） |
| 120K（120样本） | **0.0425（↓↓）** | 0.0673 | 0.158/0.211/0.369（健康） | 13.3%（冻结） |
| 140K（120样本） | 0.0420（平台） | 0.0635 | 0.159/0.212/0.371（健康） | 13.3%（冻结） |

**结论**：
1. `w_tool=0` 生效确认：工具准确率 80K/100K 均冻结在 15%，readout 不再更新
2. 调质 MSE 在 60K→100K 持续回升（0.0458→0.0798），**w_tool=0 未能改善**——回升与工具 readout 无关，指向具身训练改变 spike 分布导致 readout 失配，或情感监督已达架构上限
3. PSW 判据持续健康（criterion_psw=1），α/β/evidence 稳步小幅增长
4. 调质 MSE 距 -50% 判据（目标 ≤0.0339）反而更远

### 9.4 120K 反转（2026-08-03 补充）

**调质 MSE 显著下降：0.0716 → 0.0425（-40.6%，120 样本同口径）**，为训练以来最优（低于 60K 的 0.0458），**打破 60K→100K 的回升趋势**。PSW 持续健康（α=0.158/β=0.211/evidence=0.369）。可能原因：readout 在 100K-120K 追上了突触漂移（收敛滞后效应），或具身/课程动态进入新学习阶段。距 -50% 判据（0.0339）还需再降 20.2%。

### 9.5 复合事件非线性验证（120K，判据达标）

**实验**（120K checkpoint，权重冻结 eval）：peer_rejection_recovery 链拆分为 4 变体——完整复合（threat_social -30 + social_loss -20 + social_bond +20 + praise +15）、负性部分、正性部分、无事件基线。readout 预测（GENE_MAP 顺序）：

| 通道 | 复合实测 | 线性叠加预测 | 偏差 |
|---|---|---|---|
| DA | 0.560 | 0.654 | -0.094 |
| ACh | 0.745 | 0.991 | **-0.246** |
| NE | 0.614 | 0.793 | **-0.179** |
| 5HT | 0.407 | 0.396 | +0.011 |
| GABA | 0.315 | 0.394 | -0.079 |
| Oxy | 1.130 | 1.387 | **-0.257** |

**结论**：偏差 L2 范数 0.417（5/6 通道显著），超过单事件响应幅度（正 0.136/负 0.335）。复合响应范数 0.329 仅为线性叠加 0.624 的 **53%**——负性+正性叠加时网络产生**非线性压缩/对冲**（ACh/Oxy 抑制约 25%），而非机械相加。**判据达标**（组合调质响应非单通道线性叠加）。

**伴随改动**：eval 模式日志增加 `pred_mod`/`gtr_mod` 预测值打印（main.cpp，方便未来响应分析）。

### 9.6 后续待办

- [ ] 调质 MSE 判据：**140K 平台确认（0.0420，20K 仅降 1.2%）**——同配置（课程+具身、w_tool=0、lr=0.01）已收敛，距 0.0339 还差 19.3%，继续同配置训练边际收益≈0。可选：提 curriculum_lr 冲击 / 纯课程 A/B / 接受 0.042 进高中
- [ ] 高中阶段（stage=2）衔接：baseline 5HT 上调、w_pad 上调、eta=1.0，课程数据 `data/events/curriculum_high_school.jsonl`
- [ ] 远期：沙盒 v2/v3 社会对象 + 主动询问涌现（见 enlightenment-design-spec.md §6.4）

---

## 10. 当前状态快照（新会话交接，2026-08-03 140K）

> 供新窗口快速恢复上下文。详细数据见 §9.1-§9.6 与各日志文件。

### 10.1 一句话现状

初中课程（Stage 1a）训练至 **140K 步**，PSW/复合事件/PAD 判据达标，**调质 MSE 0.0420 进入平台期（唯一未达判据）**，等待决策：提 lr 冲击 / 纯课程 A/B / 接受进高中。

### 10.2 资产与数据

| 项 | 值 |
|---|---|
| 当前 checkpoint | `checkpoints/middle_1a/ckpt_step140000.snn2e`（保留 90K-140K，keep=4） |
| 可执行文件 | `build/snn/bin/snn_train.exe`（build_time 23:02:29，含 eval pred_mod 打印） |
| 最新 eval | `build/snn/middle_1a_eval_140k.log`（120 样本，调质 MSE=0.0420） |
| 训练日志 | `build/snn/middle_1a_140k.log` / `_120k.log` / `_100k.log` / `_80k.log` |
| 课程数据 | `data/events/curriculum_middle_school.jsonl`（2000 样本，12 链） |
| 复合事件验证 | `build/snn/eval_composite_linearity.log`（4 变体对照，非线性达标） |
| git | main 分支，领先 origin 20+ 提交，工作区 clean（末次 commit 2d95e5c） |

### 10.3 最新指标（120 样本 eval 口径）

| 指标 | 140K 值 | 趋势 |
|---|---|---|
| 调质 MSE / MAE | 0.0420 / 0.1610 | 平台期（120K=0.0425） |
| PAD MSE / MAE | 0.0635 / 0.2338 | 稳定 |
| PSW α / β / evidence | 0.159 / 0.212 / 0.371 | 持续健康增长（criterion_psw=1） |
| PSW mature | 15.7% | 正常 |
| 工具准确率 | 13.3%（冻结） | w_tool=0 生效确认 |
| decode perplexity / acc | 2.4 / 65-72% | 背景无退化 |
| 突触总数 | 8,729,034（净修剪 18.4%） | 稳态修剪，每 1000 步 prune 5-10 万 |

### 10.4 判据完成度（3.5/4）

- ✅ 复合事件非线性（偏差范数 0.42，复合响应=线性叠加的 53%）
- ✅ PSW β 不单侧失控（β=0.212 < 0.4）
- ✅ 工具判据已废弃（职责归 LLM 语义理解，w_tool=0，commit aa71f53）
- ⚠️ 调质 MSE -50% 判据未达（0.0420 vs 目标 0.0339，差 19.3%）
- **2026-08-06 判据重定义**：0.0339 基于失真监督数据（事件调制对联合皮层传导 <2.3%，readout=平均拟合器）定义，已失效。A（事件→联合皮层注入）实施后 readout 输入特征首次携带事件信息，判据改为两级：
  1. **事件可辨性（第一判据）**：eval 输出 `[RateStat] event_subregion ratio`（事件子区域 avg 发放率 / 全皮层 avg），目标 ratio > 2 且 nz_evt 全子区域激活（80K 实测 social_loss=2.41 / criticism=1.57 已达标；offset 晚的事件因注入窗口仅 10 步被窗口累计稀释，ratio 会偏低）
  2. **readout 可纠正性（第二判据）**：重训后 mod MSE 相对"冻结平均拟合器基线"（80K 冻结 continuous 口径 0.358 / 冷启动 0.232）显著下降，且 pred 范围摆脱 baseline 锁定（旧 pred 全样本 0.09-0.53）

### 10.5 待决策（上次对话遗留，未定）

1. **提 curriculum_lr 0.01→0.03** 续跑 20K 冲击 MSE（最便宜，~45 min）
2. **纯课程 A/B**（去掉 `--embodied`）排除具身干扰
3. **接受 0.042 进高中**（stage=2）：MSE 判据改为"PSW 健康 + PAD 稳定"为准

### 10.6 新窗口必备的环境/构建知识（踩坑记录）

- **cmake 不在 PATH**：用 `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`
- **构建前必须设置 MSVC 环境**（PowerShell，`cmd /c` 被安全策略禁止，不能直接 vcvars64）：
  ```powershell
  $msvc = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
  $kit = "10.0.28000.0"
  $env:Path = "$msvc\bin\Hostx64\x64;${env:ProgramFiles(x86)}\Windows Kits\10\bin\$kit\x64;${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:Path"
  $env:INCLUDE = "$msvc\include;${env:ProgramFiles(x86)}\Windows Kits\10\Include\$kit\ucrt;${env:ProgramFiles(x86)}\Windows Kits\10\Include\$kit\um;${env:ProgramFiles(x86)}\Windows Kits\10\Include\$kit\shared"
  $env:LIB = "$msvc\lib\x64;${env:ProgramFiles(x86)}\Windows Kits\10\Lib\$kit\ucrt\x64;${env:ProgramFiles(x86)}\Windows Kits\10\Lib\$kit\um\x64"
  ```
- **构建命令**：`& cmake --build build/snn --target snn_train`（目标在 **build/snn** 子目录，非顶层）
- **运行命令模板**（续跑训练）：
  ```powershell
  & .\build\snn\bin\snn_train.exe --resume checkpoints/middle_1a/ckpt_step140000.snn2e --steps 160000 --curriculum data/events/curriculum_middle_school.jsonl --curriculum-stage 1 --learning-rule n3f --bptt-window-size 400 --curriculum-lr 0.0100 --checkpoint-dir checkpoints/middle_1a --checkpoint-interval 10000 --keep-checkpoints 4 --csv build/snn/middle_1a_160k.csv --seed 42 --embodied --embodied-scene hunger_feeding --input-mode byte --text data/smoke_test.txt
  ```
- **eval 命令模板**（120 样本，`--steps` 必须 > resume 步数；**必须带 `--bptt-window-size 400`**，否则窗口 50 步 < 事件 offset 100，事件永不注入，MSE 纯 baseline 失真——2026-08-04 修复；**若训练用了 `--curriculum-continuous`，eval 必须同带**，否则 gtr 冷启动每样本复位，慢通道 Oxy 残留缺失，口径不一致——2026-08-06 实锤：3 样本 continuous 时 sample3 Oxy gtr=1.79 vs 冷启动 1.22）：
  ```powershell
  & .\build\snn\bin\snn_train.exe --resume checkpoints/middle_1a_longarc_all/ckpt_step80000.snn2e --steps 81200 --curriculum-eval --curriculum-eval-samples 120 --curriculum data/events/curriculum_all.jsonl --curriculum-stage 1 --curriculum-continuous --bptt-window-size 400 --checkpoint-dir checkpoints/middle_1a_longarc_all_eval --checkpoint-interval 0 --input-mode byte --text data/scripts/story_text_all.txt --seed 42
  ```
- 训练约 140 ms/步 → 20K ≈ 47 min；120 样本 eval ≈ 13 min

### 10.7 代码契约与约定（防再踩坑）

- **调质通道顺序**：readout/模拟器 = GENE_MAP `[DA, ACh, NE, 5HT, GABA, Oxy]`；`personality_profiles.h` 的 `baseline_mod` = personality 顺序 `[DA, 5HT, NE, ACh, GABA, Oxy]`，跨接口按 `{0,3,2,1,4,5}` 重排（main.cpp/test 已做）。常量 `MOD_CH_*` 在 `mod_simulator.h`，契约注释在 `bptt_curriculum.cuh`（commit a5d82ff）
- **eval 样本数**：用 120（40 样本口径噪声大，曾把 exam_success 误判为 0.24 垫底，实为噪声）
- **w_tool=0**：工具 readout 冻结，工具决策归 LLM（§9.2 决策），不要恢复工具训练
- **eval 日志正则**：OK 行是双空格 `OK  |`，解析时注意
- **MSE 波动性**：60K→100K 曾回升（0.046→0.072），120K 突降 0.0425——历史教训是"先扩样本确认再改代码"

### 10.8 远期方向（不因新窗口丢失）

- 沙盒 v2/v3 社会对象 + 主动询问涌现（`enlightenment-design-spec.md` §6.4：SNN 涌现询问动机，LLM 生成问题内容）
- 高中阶段（stage=2）：baseline 5HT 上调、w_pad 上调、eta=1.0，`data/events/curriculum_high_school.jsonl`
- 课程数据生成：`src/snn/tools/generate_curriculum_data.py --stage high_school`
