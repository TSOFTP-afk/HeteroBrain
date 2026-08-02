# N3F 三因子在线学习 — 设计总览与实现说明

> 整理时间: 2026-08-01
> 适用范围: Stage2e 课程训练（Phase 3a-D3）的突触学习算法之一，与 BPTT 并列
> 关联文件: [N3F训练进度_2026-08-01.md](N3F训练进度_2026-08-01.md)（实验进度与对照结果）、[run_config.h](../src/snn/run_config.h)、[scheduler.cu](../src/snn/scheduler.cu)、[bptt_curriculum.cu](../src/snn/bptt_curriculum.cu)、[synapse_kernels.cu](../src/snn/synapse_kernels.cu)

---

## 1. 一句话定义

**N3F（Neuromodulator-Gated 3-Factor，调质门控三因子在线学习）** 是课程训练模式下的一套**局部、在线**的突触学习规则：
每步用课程监督（调质预测 MSE + 工具调用 CE）产生教学信号，注入到**神经元级 eligibility**，
再由 STDP kernel 以 `突触级 e1 trace × 神经元级教学信号` 的乘积调制突触证据，实现三因子闭环。
**无窗口重放、无历史缓冲、无全局反传**，每一步都是 O(N) 局部更新。

---

## 2. 动机与科学意义

### 2.1 为什么需要 N3F（相对 BPTT）

| 维度 | BPTT | N3F |
|---|---|---|
| 学习规则 | 全局误差反传（代理梯度） | 局部三因子（pre · post · 调质） |
| 时序信用分配 | 窗口重放（V/S history + 反向循环） | 突触级/神经元级 eligibility trace |
| 生物学合理性 | 低（大脑无全局误差反传通道） | 高（对应多巴胺/乙酰胆碱等调质门控突触可塑性） |
| 内存 | 需要 V/S 历史缓冲 + 梯度缓冲 | 不需要（无窗口缓冲） |
| 结构可塑性 | 跳过结构重建（V/S history 索引依赖） | **完整保留**（共激活→新突触 + 弱突触修剪） |
| 每步成本 | 窗口末一次性前向+反向+更新 | 每步小规模在线更新 |

N3F 的意义在于：它是本项目"全量神经元模拟 + 发育性训练"路线上**唯一符合生物学习机制**的监督路径，
让"任务学习"与"结构发育"在同一套局部规则下协同进行，而不是像 BPTT 那样用全局梯度去塑形一个生物网络。

### 2.2 三因子的生物对应

1. **pre 因子**：突触级快 trace `e1`（10 步累积的 STDP 历史，signed：+LTP / −LTD），对应突触前活动痕迹；
2. **post 因子**：突触后神经元发放（STDP 时序配对），对应突触后活动；
3. **调质因子**：神经元级 `neuron_eligibility`（教学信号注入的 credit，signed：+强化 / −削弱），
   对应多巴胺等调质系统对"该神经元当前行为是否值得强化"的评价。

三者的乘积 `e1[i] · neuron_eligibility[post]` 自然给出正确的 LTP/LTD 方向——这是神经科学中
"三因子规则 / eligibility trace × 奖励调制"的标准形式，也正是"调质门控"（Neuromodulator-Gated）名称的由来。

### 2.3 与现有解码路径的关系

N3F 的神经元级 eligibility 注入机制与 Stage2e 原有**在线线性解码器**（字节预测，
[decode_kernels.cu](../src/snn/decode_kernels.cu) 的 `decode_eligibility_update_kernel`）**同构**：
误差经 readout 权重转置反传到神经元 → `neuron_elig[i] = λ·old − g·backprop_signal`。
N3F 只是把误差源从"字节预测"换成"课程监督"（调质 + 工具双任务），量级与增益重新标定。

---

## 3. 机制原理（每步闭环流程）

```
课程样本事件 → launch_modulatory 调质注入
      │
      ▼
scheduler.step()  （网络动力学 + STDP 突触更新，读取 neuron_eligibility 调制证据）
      │
      ▼
n3f_online_step(step)  ──────────────────────────────────────────────┐
  1. readout 前向（当前帧 spike）→ 调质 logits[6] + 工具 logits[7]   │
  2. 课程误差 → mod_error[6] (MSE) + tool_error[7] (CE) + 总 loss   │
  3. 教学信号注入神经元级 eligibility：                              │
       backprop_signal[i] = Σ_m w_mod·W_mod[i,m]·err_mod[m]         │
                          + Σ_t w_tool·W_tool[i,t]·err_tool[t]      │
       neuron_elig[i] = λ·neuron_elig[i] − g·backprop_signal[i]     │
       （λ = exp(−1/NEURON_ELIB_TAU)，g = 0.1 课程增益，clamp ±1）    │
      └─────────────────────────────────────────────────────────────┘
      │（下次 scheduler.step() 内生效）
      ▼
STDP kernel: evidence = ELIGIBILITY_EVIDENCE_GAIN · e1[i] · neuron_eligibility[post] · M_ij · plasticity_factor
      ▼
权重/结构演化（含结构可塑性：共激活建突触 + 弱突触修剪）
```

- **readout 权重更新时机**：不在每步，而在**窗口边界**（每 `bptt_window_size` 步，默认 400），
  用窗口累计 spike 帧 + 当前 readout 误差做一次 SGD 更新（与 BPTT 的 readout 更新等价，仅触发点不同）。
- **每步固定教学目标**：一个课程样本对应一个窗口，窗口内每一步的教学目标都是该样本的 `target_mod` / `target_tool`。

### 3.1 关键常量

| 常量 | 值 | 含义 | 出处 |
|---|---|---|---|
| `NEURON_ELIB_TAU` | 20.0 ms | 神经元级 eligibility 衰减时间常数（与 e1 一致保持时序耦合） | [config.h](../src/snn/config.h#L328) |
| `ELIGIBILITY_EVIDENCE_GAIN` | 0.1 | 突触级 trace 替代瞬时 delta_w 的增益 | [config.h](../src/snn/config.h#L325) |
| `DECODE_ELIGIBILITY_GAIN` | 0.01 | 解码误差反传增益（N3F 课程用更大的 0.1，误差量级差异 ~256 倍） | [config.h](../src/snn/config.h#L326) |
| `CURRICULUM_N_TOOL` | 7 | 工具类数（0-5 工具索引，6 = 不调用） | [bptt_curriculum.cuh](../src/snn/bptt_curriculum.cuh#L22) |

---

## 4. 代码实现导览

| 模块 | 文件 | 关键函数 | 职责 |
|---|---|---|---|
| 配置入口 | [run_config.h](../src/snn/run_config.h#L48-L52) | `--learning-rule n3f` | 选择学习算法；N3F 时**不初始化 BPTT trainer**（[main.cpp](../src/snn/main.cpp#L752-L754)） |
| 训练调度 | [main.cpp](../src/snn/main.cpp#L1004-L1013) | 窗口边界分支 | N3F 模式窗口末用累计帧更新 readout 权重 |
| 训练调度 | [main.cpp](../src/snn/main.cpp#L1106-L1109) | `n3f_online_step(step)` | 每步在线教学信号注入 |
| 调度器 | [scheduler.cu](../src/snn/scheduler.cu#L1743-L1785) | `BioMechanismScheduler::n3f_online_step` | 1) readout 前向 2) 误差 3) eligibility 注入 |
| 课程 Kernel | [bptt_curriculum.cu](../src/snn/bptt_curriculum.cu#L414-L472) | `curriculum_readout_forward_frame_kernel` (Kernel 6) | 当前帧 bool spike → logits |
| 课程 Kernel | [bptt_curriculum.cu](../src/snn/bptt_curriculum.cu#L484-L532) | `curriculum_eligibility_update_kernel` (Kernel 7) | 课程误差 → 神经元级 eligibility |
| 突触更新 | [synapse_kernels.cu](../src/snn/synapse_kernels.cu#L262-L273) | STDP 证据计算 | `e1 × neuron_eligibility[post]` 替代瞬时 delta_w |
| 状态恢复 | [scheduler_checkpoint.cu](../src/snn/scheduler_checkpoint.cu#L399-L401) | checkpoint 加载 | N3F 模式不重建 BPTT trainer（在线学习无窗口状态可恢复） |
| 学习规则标记 | [scheduler.cuh](../src/snn/scheduler.cuh#L256-L260) | `set_learning_rule` / `n3f_mode()` | 运行时切换算法 |

### 4.1 核心 Kernel 语义

**Kernel 6 — readout 前向（当前帧）**：`logits[m] = Σ_i W_mod[i·6+m]·spike[i]`，spike ∈ {0,1}。
6 个 block 算调质头、7 个 block 算工具头，每个 block 内做全局归约。

**Kernel 7 — 课程误差 → eligibility**：
```
backprop_signal[i] = w_mod·Σ_m W_mod[i,m]·err_mod[m] + w_tool·Σ_t W_tool[i,t]·err_tool[t]
neuron_elig[i] = clamp(λ·neuron_elig[i] − g·backprop_signal[i], −1, 1)
```
符号约定与 decode 路径一致：backprop = 责任，credit = −blame，`neuron_elig > 0` = 应强化、`< 0` = 应削弱。

**STDP 证据调制**：`effective_signal = ELIGIBILITY_EVIDENCE_GAIN · e1[i] · neuron_eligibility[post]`，
乘积符号天然给出 LTP/LTD 方向；`e1` 为 10 步累积的突触级 STDP trace，提供时序信用分配。

---

## 5. 训练与实验结果

### 5.1 训练命令（100K → 120K）

```
snn_train.exe --curriculum data/events/curriculum_middle_school.jsonl \
  --curriculum-stage 1 --resume checkpoints/curriculum_middle/ckpt_step100000.snn2e \
  --learning-rule n3f --bptt-window-size 400 --curriculum-lr 0.01 \
  --input-mode byte --text data/smoke_test.txt \
  --checkpoint-interval 50000 --checkpoint-dir checkpoints/curriculum_middle \
  --steps 120000 --csv run_curriculum_n3f_20k.csv
```

### 5.2 进度

| 项目 | 状态 |
|---|---|
| 进度 | 100000 → 120000（20000 步，30.9 分钟） |
| Checkpoint | `checkpoints/curriculum_middle/ckpt_step120000.snn2e`（1.45GB） |
| 速度 | 优化后 0.0605 s/步（16.5 步/秒），无 CSV 0.044 s/步 |
| 指标 | perplexity 3.07、decode_acc 55.3%、累计脉冲 74.4M |

### 5.3 已修复的工程问题

1. **CSV 采样拖慢训练**：旧代码每步全量拷回 137MB + CPU 遍历 32M 元素（1 s/步）。
   新增 [csv_stats.cu](../src/snn/csv_stats.cu) GPU 单次遍历归约，每步仅回读 ~24 字节，提速 ~2 倍。
2. **结构重建 GPU hang**：`csr_rebuild_kernel` 只重排 `d_synapses`/`d_row_ptr`，未同步重排 12 个 10.7M 对齐数组
   （col_idx / eligibility / weights_cache 等），导致重建后 STDP/NMDA 越界 hang。
   新增 `AlignedSynapseArrays` + `remap_aligned_arrays_kernel` + 尾部清零修复；**N3F 完整保留结构重建**。

### 5.4 BPTT vs N3F 对照评估（20 样本、权重冻结、checkpoint 前向统计）

| 指标 | BPTT (ckpt_step100000) | N3F (ckpt_step120000) |
|---|---|---|
| 工具调用准确率 | 12/20 = 60.0% | 12/20 = 60.0% |
| 调质预测 MSE | **0.1291** | 0.2529 |
| 调质预测 MAE | **0.2708** | 0.4156 |
| 工具 softmax（末样本） | 类6=1.107, 类1=0.034 | 类6=2.037, 类1=0.959 |

**结论要点**：
1. 工具准确率持平（60%），但两者都存在**默认类（类 6）偏置**——12 个 target=6 全对、8 个非 6 全错，该指标未真正区分能力；
2. BPTT 调质拟合更优（MSE 0.13 vs 0.25）：N3F 的 readout 更新路径（窗口边界触发）收敛偏弱；
3. N3F 工具类区分信号更强（类 1 softmax 0.959 vs 0.034），被类 6 偏置掩盖——若消除偏置 N3F 有潜力；
4. N3F 独有能力完整保留：结构重建 + 每步在线学习。

---

## 6. 已知不足与改进方向

- [ ] **readout 收敛偏弱**：readout 权重只在窗口边界（400 步）更新一次，与每步教学信号的节奏不匹配；
      候选方案：每步更新 / 窗口末改用当前帧而非累计帧 / 增大 curriculum_lr。
- [ ] **工具 readout 默认类偏置**：类 6 得分系统性偏高，掩盖真实区分能力；候选：目标均衡采样 / 偏置项初始化。
- [ ] **eligibility 增益是固定启发式（0.1）**：随训练阶段（发育阶段、误差量级）自适应调度可能更稳。
- [ ] **无从头（from-scratch）验证**：当前 20K 结果继承自 BPTT 100K 地基，不能证明 N3F 的在线学习容量。
      需要一次同课程、同窗口、同步数的从头对照实验（N3F 0→20K vs BPTT 0→20K 曲线对比，0.044 s/步成本极低）。

---

## 7. 深层缺陷分析（2026-08-01 代码审查发现）

> 本节记录 §6 已知不足之外、经代码审查发现的架构与实现缺陷，
> 按严重程度分级（P0=阻塞 / P1=严重 / P2=中等 / P3=轻微）。

### 7.0 修复状态汇总（2026-08-01 已实施）

| 缺陷 | 级别 | 状态 | 修复方式 |
|------|------|------|---------|
| §7.1 教学目标与事件注入时序失配 | P0 | ✅ 已修复 | 删除课程 eligibility 注入（spec §7.2） |
| §7.2 窗口末 readout 用单步误差 | P0 | ✅ 已修复 | 窗口末累计帧前向 + 累计误差再更新 |
| §7.3 评估样本数不足 | P1 | ◐ 部分修复 | 默认样本数 20→100；轨迹评估待做 |
| §7.4 累计帧 vs 当前帧不一致 | P1 | ✅ 已消解 | readout 更新统一用累计帧（与 BPTT 一致）；课程已不注入突触，教学信号矛盾消除 |
| §7.5 学习信号滞后一步 | P2 | ✅ 已修复 | `n3f_embodied_step` 在 scheduler.step() 之前注入 |
| §7.6 与解码路径竞争缓冲 | P2 | ⬜ 待做 | 需独立 eligibility 缓冲（设计决策，延后） |
| §7.7 无 readout warmup | P2 | ✅ 已修复 | 新增 `--curriculum-readout-warmup` |
| §7.8 readout 更新无裁剪 | P3 | ✅ 已修复 | `CURRICULUM_READOUT_WEIGHT_CLIP` 裁剪 |
| §7.9 静态全局变量不可重入 | P3 | ✅ 已修复 | 移入 PersistentBuffers（target/loss 缓冲） |
| §7.10 复用 BPTT 参数语义混乱 | P3 | ✅ 已修复 | run_config.h 注释文档化参数复用关系 |

---

### 7.1 P0：教学目标与事件注入时序失配

**现象**：一个课程窗口内，每一步的教学目标 `target_mod` 都是同一个值（窗口末的期望调质状态），
但事件是分步注入的（不同步 offset 触发不同事件）。

```
步 0:   无事件 → 目标调质=[0.55, 0.30, ...]  ← 虚假误差：网络还没看到事件
步 100: 事件注入 → 目标调质=[0.55, 0.30, ...]  ← 正确
步 200: 无事件 → 目标调质=[0.55, 0.30, ...]  ← 正确
```

**后果**：窗口前半段的"预测误差"是虚假的——网络在**没有看到事件**时就被要求预测**事件后的调质状态**。
这些虚假误差通过 `curriculum_eligibility_update_kernel` 注入到 `neuron_eligibility` 中，
累积到 STDP 证据调制，**污染了突触权重更新方向**。

**涉及代码**：
- 教学目标设置：[main.cpp#L1020-L1026](file:///f:/thetrueai/src/snn/main.cpp#L1020-L1026)
- 每步事件注入：[main.cpp#L1043-L1058](file:///f:/thetrueai/src/snn/main.cpp#L1043-L1058)
- eligibility 注入（Kernel 7）：[bptt_curriculum.cu#L484-L514](file:///f:/thetrueai/src/snn/bptt_curriculum.cu#L484-L514)

**修复方向**：教学目标应随事件注入步进式变化（事件前=基线，事件后=目标），或窗口前半段降低调质损失权重。

### 7.2 P0：窗口末 readout 更新使用单步误差而非累计误差

**现象**：N3F 模式窗口末的 readout 权重更新（[main.cpp#L1006-L1013](file:///f:/thetrueai/src/snn/main.cpp#L1006-L1013)）
使用 `d_curriculum_error` 和 `d_curriculum_tool_error`，但这两个缓冲中存放的是**上一步**（即窗口最后一步）
的 `n3f_online_step` 写入的误差，而非窗口 400 步的累计误差。

**涉及代码**：主循环 N3F 分支：[main.cpp#L1004-L1013](file:///f:/thetrueai/src/snn/main.cpp#L1004-L1013)
```cuda
if (config.learning_rule == "n3f") {
    launch_curriculum_readout_update(..., config.curriculum_lr, ...);
    // 注意：此时 d_curriculum_error 是上一步（rel=399）的 n3f 误差，不是窗口累计
}
```

**后果**：readout 更新仅基于窗口最后一步的误差，样本内前 399 步的信息被丢弃，收敛速度下降。

**修复方向**：窗口末做一次累计误差计算（而非使用最后一步的瞬时误差），或改为每步更新 readout。

### 7.3 P1：评估样本数不足 + 无轨迹评估

**现象**：评估（[main.cpp#L859-L939](file:///f:/thetrueai/src/snn/main.cpp#L859-L939)）仅用 20 个样本、
仅在窗口末做一次前向判断。

**细节**：
1. 20 个样本中 12 个 target=6，统计意义不足（8 个非 6 样本全错 = 60% 准确率，与随机无显著差异）
2. 只取窗口末累计帧做一次前向，丢失调质轨迹的时间信息
3. 工具调用用 argmax 判断，不看 softmax 置信度

**后果**：60% 的准确率可能只是"窗口末发放率恰好与默认类 6 相关"，无法判断真正学会了什么。

**修复方向**：评估样本 ≥ 200；窗口内多点采样评估调质轨迹；输出 softmax 置信度分布。

### 7.4 P1：Readout 更新用累计帧 vs 教学信号用当前帧

**现象**：N3F 的 readout 权重更新（[bptt_curriculum.cu#L230-L252](file:///f:/thetrueai/src/snn/bptt_curriculum.cu#L230-L252)）
使用窗口累计平均发放率 `rate[i]` 作为输入特征：

```cuda
// Kernel 4: readout 权重更新
mod_row[m] -= lr * w_mod * mod_error[m] * spike_rates[i];  // spike_rates = 窗口累计率
```

而每步教学信号（`curriculum_readout_forward_frame_kernel`，[bptt_curriculum.cu#L414](file:///f:/thetrueai/src/snn/bptt_curriculum.cu#L414)）
使用当前帧 bool spike 作为输入特征：

```cuda
// Kernel 6: 当前帧前向
logits[m] = Σ_i W_mod[i*6+m] · spike[i]  // spike ∈ {0,1}
```

**后果**：readout 权重的优化目标和教学信号的目标不一致——一个优化"累计发放率预测"，一个优化"当前帧预测"。
两者可能不在同一最优点。

**修复方向**：N3F 的 readout 更新改用当前帧模式（与教学信号一致），或两者统一为一种特征。

### 7.5 P2：学习信号滞后一步

**现象**：主循环中 `scheduler.step()` 和 `n3f_online_step()` 的执行顺序：

```
scheduler.step(step)         → 内部 STDP kernel 读取 neuron_eligibility（第 N-1 步注入的）
n3f_online_step(step)        → 注入 neuron_eligibility（第 N 步的，但 STDP 已经跑完）
```

**涉及代码**：[main.cpp#L1101-L1108](file:///f:/thetrueai/src/snn/main.cpp#L1101-L1108)
```cuda
scheduler.step(step);                              // 第 N 步动力学 + STDP
if (config.learning_rule == "n3f") {
    scheduler.n3f_online_step(step);               // 第 N 步教学信号注入 → 第 N+1 步才生效
}
```

**后果**：教学信号注入 → 下一次 `scheduler.step()` 的 STDP 才生效，**滞后一步**。
在快速动力学（AdEx 步长 1ms）场景下影响不大，但若每步 spike 模式变化剧烈，会累积相位误差。

**修复方向**：将 eligibility 注入移到 `scheduler.step()` 内部、在 STDP kernel 之前执行；
或接受滞后（当前设计），但需文档化该边界。

### 7.6 P2：N3F 与解码路径竞争同一 neuron_eligibility 缓冲

**现象**：N3F 和字节解码器共享 `d_neuron_eligibility` 全局缓冲
（[scheduler.cu#L1528-L1532](file:///f:/thetrueai/src/snn/scheduler.cu#L1528-L1532)）：

```cuda
// N3F 课程模式: 禁用 decode eligibility
if (!n3f_mode()) {
    launch_decode_eligibility_update(buf);
}
```

**后果**：如果未来需要同时启用课程训练和字节预测（混合监督），两者会互相覆盖 `neuron_eligibility`。
当前设计是二选一，不是混合。

**修复方向**：让课程和字节解码使用独立的 eligibility 缓冲，或合并为一个统一的教学信号。

### 7.7 P2：无 readout warmup 阶段

**现象**：课程训练启动时，readout 权重是小随机初始化（`init_scale=0.01`），
没有预热阶段让 readout 先收敛到合理范围（[main.cpp#L774-L776](file:///f:/thetrueai/src/snn/main.cpp#L774-L776)）：

```cuda
stage2e::launch_curriculum_readout_init(allocator.buffers(), 0.01f, config.seed + 0xA11CEu);
```

**后果**：训练初期 readout 权重接近零，前几个窗口的误差信号主要由随机权重产生，
这些随机误差通过 `curriculum_eligibility_update_kernel` 注入到 `neuron_eligibility`，
**污染了突触权重更新**。BPTT 模式有 warmup（1000 步），N3F 没有。

**修复方向**：N3F 启动后先跑若干个纯前向窗口（冻结教学信号注入），让 readout 先收敛。

### 7.8 P3：readout 更新无权重裁剪

**现象**：`curriculum_readout_update_kernel`（[bptt_curriculum.cu#L230-L252](file:///f:/thetrueai/src/snn/bptt_curriculum.cu#L230-L252)）
没有权重裁剪：

```cuda
mod_row[m] -= lr * w_mod * mod_error[m] * spike_rates[i];
// 无 clip 保护
```

**后果**：若某些神经元 `spike_rates[i]` 系统性偏高，对应 readout 权重会发散。
虽然 `lr=0.01` 较小，但长时间训练后仍可能数值不稳定。

**修复方向**：添加权重裁剪（例如 `clip(W_mod, -10, 10)`）。

### 7.9 P3：静态全局变量导致不可重入

**现象**：`launch_curriculum_error`（[bptt_curriculum.cu#L293-L300](file:///f:/thetrueai/src/snn/bptt_curriculum.cu#L293-L300)）
使用静态懒分配：

```cuda
static float* s_d_target = nullptr;
static float* s_d_loss = nullptr;
```

**后果**：多流或多 GPU 场景下静态变量共享冲突。当前项目单 GPU 无影响，但代码可移植性受限。

**修复方向**：改为 PersistentBuffers 成员变量或函数参数传入。

### 7.10 P3：N3F 复用 BPTT 配置参数，语义混乱

**现象**：`run_config.h` 中 N3F 复用 BPTT 的参数，但语义不同：

| 参数 | BPTT 语义 | N3F 使用 | 是否匹配 |
|------|-----------|----------|---------|
| `bptt_window_size` | 窗口重放长度 | readout 更新间隔 | 部分匹配（窗口概念一致） |
| `bptt_lr` | BPTT 学习率 | N3F 不使用（N3F 不初始化 BPTT trainer） | ❌ 不匹配 |
| `curriculum_lr` | readout 学习率 | readout 学习率 | ✅ 匹配 |

**后果**：配置文件语义不清晰，`bptt_lr` 在 N3F 模式下被忽略易引起困惑。

**修复方向**：添加 N3F 专用参数（如 `n3f_readout_update_interval`），或明确文档化参数复用关系。

---

## 8. 定位与后续路线

1. **短期**：修复 readout 更新节奏 + 类偏置 + 增益调度 → 跑从头 N3F 对照，判断规则本身的收敛能力；
2. **中期**：若从头收敛成立，N3F 可作为"发育一致性"主线——结构与权重在同一局部规则下协同演化，
   避免"BPTT 全局梯度塑形 → 中途换局部规则"的信用分配冲突；
3. **长期**：N3F 与具身训练（[embodied](../src/snn/embodied_env.h)）天然契合——调质（DA 奖励）本身就是
   三因子规则的第三因子，N3F 是"调质门控突触可塑性"在任务监督下最自然的落地形态。
