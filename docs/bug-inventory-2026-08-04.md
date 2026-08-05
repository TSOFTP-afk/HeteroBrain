# SNN 课程训练全量审计 — Bug 清单（2026-08-04）

> **创建**：2026-08-04
> **触发**：发现"课程浓度模拟器跨样本不重置"P0 bug（140K 调质 MSE 0.0420 平台期为失真口径），修复后全量清查 src/snn 同类问题（跨样本/跨窗口状态残留 + 训练/eval 口径不一致）。
> **状态**：追踪文档。已修复项标注 ✅；待修项按优先级推进。
> **行号基准**：审计时点（2026-08-04 上午）；后续代码改动会使行号偏移，定位以变量名/函数名为准。

---

## 0. 审计范围

- 方法：2 路并行代码级审计，覆盖 src/snn 24 个源文件 + tools 数据生成器
- 重点关注：样本/窗口/step 边界状态重置、训练 vs eval 口径一致性、窗口与事件 offset 对齐

---

## 1. 已修复 ✅

### 1.1 P0：课程浓度模拟器跨样本不重置（原 bug，本次会话修复）

- **问题**：`CurriculumModSimulator curriculum_mod_sim_`（scheduler 成员）在训练/评估中跨样本持续运行，从不调用 `reset()`。慢通道 Oxy tau=500 > 窗口 400 步，残留累积 → 监督目标漂移到稳态不动点。**调质 MSE 0.0420 是"稳态拟合"假象，非"事件→调质响应"误差**。且 60K→120K 的 MSE 波动（0.046→0.0798→0.0425）大部分为该漂移伪影。
- **修复**：
  - `scheduler.cuh` 新增 `reset_curriculum_target()` 声明
  - `scheduler.cu` 实现（`curriculum_mod_sim_.reset()` + 刷新 `curriculum_target_mod_curr_/pad_curr_`）
  - eval 路径（main.cpp 样本循环开头）与训练路径（窗口边界）调用
- **验证**：修复后 20 样本 eval——gtr_mod 随样本变化（旧 eval 恒为不动点）、`jsonl_ref == MSE`（目标 = baseline + 本样本事件响应）、真实 MSE 0.0589（旧口径 0.0420 虚低 40%）。**判据 0.0339 在新口径下不达标**。
- **诊断结论**：readout 是"平均状态拟合器"——正性高 DA 事件（目标 1.2-1.3）严重低估（预测 ~0.5）、负性事件（目标 0.30）严重高估。140K 训练监督目标失真导致 readout 学偏，需在正确监督下重新训练验证。

### 1.2 P0-1：训练窗口边界 reset 位置错误（修复引入的回归，已修）

- **问题**：`reset_curriculum_target()` 初版插入在窗口末 readout 更新块**之前** → readout 更新时 `curriculum_target_mod_curr_` 为全零 → **每个窗口把 readout 朝"输出零"拉一步**（N3F 训练路径 readout 的唯一更新入口被污染）。
- **修复**：reset 移到 readout 更新块**之后**、`launch_curriculum_accum_clear` **之前**。窗口末 readout 更新仍用旧窗口末模拟浓度目标。
- **影响面**：仅 N3F 训练续跑路径（eval 冻结 readout 不受影响；140K 训练完成于修复前，未受影响）。

### 1.3 配套：eval 窗口防护 + 文档命令模板

- eval 代码增加防护：事件最大 offset ≥ 窗口时告警（防再以默认 window=50 评估——旧 eval 事件从未注入网络/模拟器）。
- `docs/middle-school-training-plan.md` §10.6 eval 命令模板补 `--bptt-window-size 400`。
- 已知限制（非 bug）：offset=400 边界事件（10.7% 样本）在窗口 400 下不注入，与训练 140K 行为一致（公平评估）；完整注入需 window=500。

### 1.4 B3：eval 统一冻结突触权重（本次会话修复）

- **问题**：`--curriculum-eval` 只冻结 BPTT + readout 更新；STDP/PSW 证据/CaMKII/scaling/结构可塑性/证据衰减在 eval 期间仍持续修改突触权重 → 评估样本间网络漂移，eval 非严格冻结（n_eval×win≥1000 时衰减/重建必触发）。
- **修复**：
  - `scheduler.cuh` 新增 `weights_freeze_` 标志 + `set_weights_freeze()`/`weights_freeze()`
  - `scheduler.cu` step() 内四处权重写入门控 `!weights_freeze_`：STDP（L557）、CaMKII（%10 块）、scaling（%100 块）、结构可塑性（%1000 块）
  - `main.cpp` curriculum-eval 分支 `scheduler.set_weights_freeze(true)`
  - 注意：modulatory / 抑制性网络 / STP / 共激活采样等网络动力学保留（非权重），保证事件注入与网络状态评估正常
- **验证**：3 样本 eval 正常完成；训练路径不受影响（freeze 默认 false）。
- **残余（B7，P2）**：`--eval-mode`（字节预测评估）路径仍未设 freeze——decode 冻结但 BPTT 照常反传、STDP 照常运行，属同一根因的下半段。

### 1.5 B1：抑制电流每步清零时序（本次会话修复）

- **问题**：`d_inhibitory_current` 每步清零（`p1_noop_clear_float`），但 `launch_inhibitory_network` 每 10 步才全量覆写 → 竞争/门控信号每 10 步仅生效 1 步，9/10 步抑制电流为 0；P3 稀疏竞争（判据 [17][18]）实际几乎失效（判据只统计 inhib_updates 计数故"通过"）。
- **修复**：移除每步清零调用（`p1_noop_clear_float` kernel 一并删除，无其他使用方）；`p3_inhibitory_competition_kernel` 无条件写全部条目，值在两次 %10 更新间持续生效；步 0 前由 network_init 零初始化。`blocks` 变量声明移至唯一使用处（统计 kernel）。
- **验证**：500 步 N3F 续训正常完成，无崩溃/异常。

### 1.6 B4：resume 后 PCA CPU 镜像未恢复（本次会话修复）

- **问题**：checkpoint 只存 GPU `d_pca_W`（不存 `d_pca_mean_`）；CPU 镜像 `h_pca_W_`（构造随机）/`h_mean_fr_`（零）不随 resume 恢复 → 下一 `PCA_SYNC_INTERVAL` 周期用随机/零镜像覆盖 GPU 基矩阵，PCA 流形破坏（签名/海马/WM 检索）。附带缺口：`d_pca_mean_` 也从未持久化。
- **修复**（scheduler_checkpoint.cu）：
  - `make_sections` 末尾新增 `pca_mean` 节（`self->d_pca_mean_`）——放节表末尾保证旧 checkpoint 前缀兼容（缺失即零，与 h_mean_fr_ 构造初值一致）
  - `load_checkpoint` 在 restore_state 后 D2H 回填 `h_pca_W_ ← d_pca_W`（前 N_ASSOCIATION×K）、`h_mean_fr_ ← d_pca_mean_`
- **验证**：旧 checkpoint（无 pca_mean 节）加载无警告 + 镜像回填；新 checkpoint 保存（含 pca_mean 节）并回读成功（full layout match，无"未找到 section"）。

### 1.7 B6/B7/P1-6/B11/P1-2 + B2/B5 范式切换（2026-08-04 第二批次修复）

- **B6**（main.cpp curriculum-eval 分支）：补 `scheduler.decode_update_weights = false` — 不传 `--eval-mode` 时 decode 每 3 步（INPUT_INJECT_INTERVAL=3）持续学习，评估被 decode 学习污染（§8.3 严重性升级）。
- **B7**（main.cpp eval_mode 分支）：补 `set_bptt_freeze(true)` + `set_weights_freeze(true)` — 原只冻结 decode，BPTT 照常反传、STDP 照常运行；两个评估入口（eval_mode / curriculum_eval）冻结策略统一为"评估全冻结"。
- **P1-6**（main.cpp resume 分支）：resume + 课程模式先 `launch_curriculum_accum_clear` + `reset_curriculum_target()` — 首窗口不再混入 checkpoint 时点的中途累计 spike（readout 特征污染），目标不再全零直到首个 advance。
- **P1-4 → 范式切换**：按 spec §7.2 删除课程 eligibility 注入到突触（B5 根源）——教学信号不再写共享 `d_neuron_eligibility` 缓冲；跨窗口残留问题（P1-4）随注入删除自然消失。同步删除 `launch_curriculum_eligibility_update`（含 P1-4 临时加的 decay 变体）与 `CURRICULUM_ELIGIBILITY_GAIN` 常量。
- **B2**（scheduler.cu bptt_step 课程分支）：跳过 `backward_curriculum + update` — BPTT 梯度直接写 weight 会被 STDP `w=W_MAX·α/(α+β)` 重算覆盖（突触级训练名存实亡）；BPTT 课程模式只训 readout 头，突触塑形回归 STDP。
- **B5**（scheduler.cu decode 路径）：decode eligibility 门控由 `!n3f_mode()` 扩为 `!n3f_mode() && !curriculum_mode_` — BPTT 课程模式下 decode 字节预测信号也不得写共享 eligibility 缓冲干扰课程 STDP 证据。
- **B11**（main.cpp 具身初始化）：补 `stage2e::motor_rng_seed(config.seed)` — 不同 `--seed` 沙盒动作序列可复现（原固定兜底种子）。
- **P1-2**（event_scheduler.cpp load_jsonl）：加载后统计非 100 倍数 step_target 并告警（launch_modulatory 每 100 步读取，违约事件被静默丢弃；实测动态/因果类事件流文件 82% 违约）。
- **残余**：P1-6 的 `curriculum_sample_idx` 未持久化（resume 后样本序号重开，已文档化）；范式切换后纯课程 N3F 模式突触学习只剩纯 STDP（readout 监督保留），需在正确监督下重训验证。

---

## 2. P1 级待修（确认 bug）

| 编号 | 位置 | 问题 | 影响 | 建议修复 | 状态 |
|---|---|---|---|---|---|
| **B3** | main.cpp L898 + scheduler.step() | eval（`--curriculum-eval`）只冻结 BPTT + readout 更新；**STDP/PSW 证据/CaMKII/scaling/结构可塑性/证据衰减在 eval 期间持续修改突触权重** | 评估样本间网络漂移，eval 结果非严格冻结；n_eval×win≥1000 时衰减/重建必触发 | eval 路径设统一全局 freeze 标志，step() 内跳过 stdp/scaling/structural/camkii 权重写入 | ✅ 已修（§1.4） |
| **B1** | scheduler.cu L566-588 | `d_inhibitory_current` **每步**清零，但 `launch_inhibitory_network` 每 10 步才写 → 竞争/门控信号每 10 步仅生效 1 步，9/10 步抑制电流为 0 | P3 稀疏竞争机制（判据 [17][18]）实际几乎失效（计数递增故判据"通过"） | 清零移入 %10 块（写前清），或移除每步清零 | ✅ 已修（§1.5） |
| **B2** | synapse_kernels + bptt_trainer | BPTT `bptt_sgd_update_kernel` 直接写 `weight`；STDP `stdp_dual_trace_kernel` 在 post 发放时按 `w = W_MAX·α/(α+β)` 重算覆盖；证据衰减每 1000 步再全量重算 → **BPTT 突触级训练几乎无效**（梯度无法经 PSW 留存） | BPTT 课程模式下突触权重训练名存实亡（仅 readout 头生效） | **架构级决策**：BPTT 梯度折算进 α/β，或 BPTT 模式关闭 STDP 权重写入（仅保留 trace） | ✅ 已修（§1.7 范式切换：BPTT 课程模式只训 readout） |
| **B4** | scheduler_checkpoint.cu | resume 后 PCA CPU 镜像 `h_pca_W_/h_mean_fr_` 未从 checkpoint 恢复 → 下一同步周期用随机镜像覆盖 GPU 基矩阵 | resume 后 PCA 流形被破坏（影响签名/海马/WM 检索） | resume 时同步 `d_pca_W→h_pca_W_`、`d_pca_mean→h_mean_fr_` | ✅ 已修（§1.6） |
| **B5** | decode_kernels.cu L120-131 + scheduler.cu L1577 | `d_neuron_eligibility` 三路共享（decode/课程/具身）。仅 N3F 模式禁用 decode eligibility；**BPTT 课程模式下 decode 字节预测信号持续写共享缓冲** | BPTT 课程模式下 STDP 证据被字节预测信号主导（§7.6 已知问题仅 N3F 规避） | eligibility 独立缓冲，或 BPTT 课程模式也禁用 decode eligibility 写入 | ✅ 已修（§1.7 范式切换：课程 eligibility 注入删除 + decode 门控扩到课程模式） |
| **P1-4** | scheduler.cu n3f_online_step | N3F 课程 eligibility 只在事件注入步启动 kernel（λ 衰减仅注入时应用一次），无事件步**不衰减** → 旧窗口教学信号冻结值残留进新窗口前 ~100 步（gain=8e-4 量级较小） | 跨窗口教学信号污染（纯课程 N3F 模式；具身模式每步衰减可缓解） | 衰减与注入解耦：每步 `elig = λ·elig`，仅事件步叠加教学项；或窗口边界清零 | ✅ 已修（§1.7 范式切换：课程 eligibility 注入整体删除，残留问题消失） |
| **P1-6** | main.cpp resume + scheduler_checkpoint.cu L274 | resume 首窗口：`d_curriculum_accum_spikes` 保留 checkpoint 时点中途值（注释称"延续部分窗口状态"但样本序号已从 0 重开，语义破缺）；`curriculum_target_mod_curr_` 为全零直到首个 advance | resume 后首窗口 readout 特征混入旧样本发放 + 目标全零 | resume 时若课程模式则先 `launch_curriculum_accum_clear`；持久化 `curriculum_sample_idx` 或重对齐样本 | ✅ 已修（§1.7 accum_clear + 目标复位；sample_idx 持久化留残余） |

---

## 3. P2 级待修

| 编号 | 位置 | 问题 | 建议 |
|---|---|---|---|
| B6 | main.cpp L735 + run_config | `--curriculum-eval` 不隐式开启 `--eval-mode` → decode 权重在评估期间继续更新（不传 `--eval-mode` 时） | ✅ 已修（§1.7 curriculum_eval 自动置 decode 冻结） |
| B7 | scheduler.cu L1718-1756 + main.cpp | `bptt_freeze_` 只在课程分支生效；非课程字节预测分支无 freeze 检查；`--eval-mode` 路径从未设置 freeze → eval_mode 下 BPTT 每窗口照常反传更新 | ✅ 已修（§1.7 eval_mode 设置 freeze；字节分支保持原状，课程/评估两入口已统一） |
| B8 | main.cpp 训练 vs eval | 课程训练注入 BPE token/字节 + 事件；课程 eval 只注入事件 → bpe 模式输入口径不一致 | eval 也注入同口径输入（或文档化约束） |
| B10 | embodied_body.h + scheduler_checkpoint | `BodyState` 是 host 端对象且不入 checkpoint，resume 后身体从场景初值重新演化 | 加入 checkpoint section，或文档化"世界状态随 resume 重置" |
| B11 | embodied_motor.cpp L19-30 | `motor_rng_seed(config.seed)` 从未被 main.cpp 调用 → 固定兜底种子，不同 `--seed` 沙盒序列相同 | ✅ 已修（§1.7 main.cpp 具身初始化接入 config.seed） |
| B12 | hippocampal_kernels.cu L403-457 | 睡眠重放注入只写 `d_replay_injection`，**从未并入 `d_input_current`** → 重放对网络零效果（仅消耗海马 importance） | scheduler.step() 中把 d_replay_injection 并入输入 |
| P1-2 | event_scheduler.cpp L151-154 + main.cpp L1189 | 非 100 倍数步的事件在下一个 step 的 `reset_event_signal()` 被提前清零，`launch_modulatory`（step%100==0）永远读不到 → 事件静默丢弃（当前数据契约内不触发） | ✅ 已修（§1.7 加载时告警；课程数据契约内安全，事件流违约不再静默） |
| P2-10 | modulatory_kernels.cu L569-574 | plateau 事件 duration 机制死代码：`reset_event_signal()` 每步清零信号但不清理 duration → duration 分支永不执行（当前所有调用方 duration 恒传 0） | 若启用 plateau 需调整 reset 策略 |
| P2-9 | scheduler.cu L1818 | `advance_curriculum_target` 同 rel 事件收集上限 8 个，网络侧全部注入 → 目标/网络不一致（数据契约 ≤4 事件内安全） | 上限校验或动态扩容 |

---

## 4. P3 级

| 编号 | 位置 | 问题 |
|---|---|---|
| B13 | main.cpp 窗口切换 | 窗口边界步（rel==win）的 spike 被累计到新窗口（每窗口多 1 帧噪声，1/win 量级） |
| B14 | scheduler.cu eval | eval 期间 `neuron_byte_counts`/海马索引/WM/PCA 持续更新（`--curriculum-eval` 不存 checkpoint 故不污染；`--eval-mode` 路径会污染） |

---

## 5. 已排查并确认干净的区域

- 窗口簿记（window_start/rel/sample_idx/curriculum_cur_sample）切换顺序
- 累计 spike 缓冲（每窗口起点 accum_clear；allocator 分配即清零）
- BPTT 窗口缓冲（V/S history 全量覆写、dL_dW 更新后清零、via_W 每反向步清零）
- BPTT 窗口对齐（窗口末触发时 set_curriculum_mode 与 accum 仍属窗口 k，无 off-by-one）
- 事件注入时序（C2 修复完整：reset → 逐事件累加 → 同 rel%100 步 advance 与网络注入读同一批事件）
- spike_flags / input_current / nmda_current / motor_input_current 每步清零或覆写
- CSR 重建对齐数组 remap（存活搬运 + 新槽 α/β=MIN + 尾部清零，无 NaN/残留）
- 发育阶段切换 / E0 消融（纯函数，无迁移状态）
- curriculum_loader（load_jsonl 每次 samples_.clear()，无共享状态）
- 课程 kernel 误差缓冲（launch_curriculum_error 全量覆写）

---

## 6. 修复优先级建议（2026-08-04 更新）

1. ✅ 已完成：**P0/P0-1**（前次会话）、**B3/B1/B4**（本次会话，见 §1.4-1.6）
2. **B7**（eval_mode 冻结，B3 残余下半段）→ **P1-6**（resume accum 残留）
3. **P1-4**（eligibility 衰减）→ **B5**（eligibility 隔离）
4. **B2**（BPTT/PSW 双写）——**架构级，需用户决策后再动**：BPTT 梯度折算进 α/β vs BPTT 模式关闭 STDP 权重写入。注：设计 spec §7.2 已定方向（突触层回归 STDP+沙盒反馈、BPTT 只训 readout 头），此决策的最终答案是落实范式切换而非修时序
5. P2 按需推进；P3 低优先

---

## 7. 关联决策记录

- **判据重定义**：调质 MSE 判据（旧 0.0339，基于失真口径）需在修复后重定基线；当前真实 MSE（20 样本）0.0589。
- **readout 学偏**：140K readout 未学会事件响应（平均状态拟合器）。待决策：修复后续跑训练验证可纠正性 vs 重置 readout 冷启动重训。
- **B2 架构决策**：BPTT 与 PSW 的突触级训练并存方式，未定。

---

## 8. 全量清查修订记录（2026-08-04 复核，与代码/数据实测对齐）

> 对照设计文档（enlightenment-design-spec.md §7.2/§9.5、middle-school-training-plan.md）与数据文件实测后，对清单内若干条目的描述修正如下；其余条目（P0/B1-B5/P1-4/P1-6/P2-9/P2-10/B14 等）经复核仍如原文所述。

1. **P1-2 数据契约声明失实**：原文称"当前数据契约内不触发"。实测 `data/events/` 非课程文件（dynamic/causal/parallel2 类）**1091/1330（82%）事件为非 100 倍数 offset**（全文件 5899 事件中占 18.5%）；走 `--event-stream-path` 会被静默丢弃。仅课程文件（offset 100/200/300/400）安全。→ 修正为"课程数据契约内安全；事件流数据违约无校验，建议生成器对齐或加载时告警"。
2. **B13 描述方向反了**：实测边界步（rel==win）本就是新窗口的 rel=0（窗口切换在 step() 之前），N3F 窗口恰好 win 帧，**无"多 1 帧"**；真正偏差是 BPTT 模式 readout 触发在 rel=win-1、累计仅 win-1 帧（末帧触发后累计，未使用），为 1/win 欠计数。→ 修正描述；当前 n3f 训练路径无此问题，P3 维持低优先。
3. **B6 严重性升级**：`INPUT_INJECT_INTERVAL=3` → `--curriculum-eval` 期间 decode 权重每 3 步更新一次（warmup 后持续学习），比"评估期间继续更新"更严重。修复方向同 B3（curriculum_eval 自动置 decode 冻结）。
4. **B7 根因归纳**：两个评估入口冻结策略互为镜像且都不完整——`--eval-mode` 冻结 decode 但 BPTT/STDP 照常；`--curriculum-eval` 冻结 BPTT（本次已加权重冻结）但 decode 照常。建议统一为单一"评估冻结"入口。
5. **B3 的 scaling 一项已过时**：`launch_synaptic_scaling` 为 PSW 空壳（显式跳过，防破坏 α/β 一致性）——eval 期间实际活跃的权重写入是 STDP/证据衰减/结构重建/CaMKII（间接），已随 §1.4 一并冻结。
6. **B2 需求判定**：按设计 spec §7.2（2026-08-03）"不再注入 neuron_eligibility 到突触、突触塑形回归 STDP+沙盒反馈"，P1-4/B5/B2 的最终修复方向是**落实范式切换（删除课程 eligibility 注入、BPTT 只训 readout）**，而非修补旧机制的时序。
7. **B4 附带缺口已修**：`d_pca_mean_` 原本也未持久化，本次随 B4 一并保存（§1.6）。
