# 从 legacy 复用代码指南

> HeteroBrain 的 SNN 子系统从 `legacy/stage2e/` 移植而来。本文档列出可复用模块与移植注意事项。

## 可直接复用的模块

### 1. BPTT Trainer（核心）

**来源**: `legacy/stage2e/bptt_trainer.cu` (35KB) + `bptt_trainer.cuh` (12KB)

**功能**: BPTT 代理梯度训练器，CSR 稀疏格式，含梯度裁剪与数值稳定。

**移植目标**: `src/snn/bptt_trainer.cu/.cuh`

**移植要点**:
- 保留 CSR 稀疏突触格式（10.7M 突触内存可控）
- 保留双精度 `d_norm_sq_scratch`（防 10M 突触平方和溢出 float）
- 保留 `bptt_clip_dL_dV_kernel`（防 50 步反向累积指数爆炸）
- 保留 V_prev 钳位到 [-5, 5]
- 删除 `namespace stage2e`，改为 `namespace heterobrain::snn`

### 2. PCA 签名提取

**来源**: `legacy/stage2e/pca_kernels.cu` (10KB) + `pca_kernels.cuh` (5KB)

**功能**: 从联合皮层神经元发放模式提取 PCA 签名（256 维），用于长程记忆索引。

**移植目标**: `src/snn/pca_signatures.cu/.cuh`

**移植要点**:
- PCA W 矩阵需保留初始化（小随机值，非零）
- 输入：前 N_ASSOCIATION_NEURONS_2E 个神经元的瞬时发放（bool → float）
- 输出：256 维 float 签名向量

### 3. 突触与神经元 Kernel

**来源**:
- `legacy/stage2e/synapse_kernels.cu` (37KB) — NMDA/AMPA/GABA + STDP 双trace + 树突区室化
- `legacy/stage2e/neuron_kernels.cu` (17KB) — AdEx 神经元 + 适应性 + 阈值动态

**功能**: SNN 前向计算核心。

**移植目标**: `src/snn/synapse_kernels.cu/.cuh` + `src/snn/neuron_kernels.cu/.cuh`

**移植要点**:
- 保留树突区室化（前馈连接专用 Ca²⁺ 动力学，防 L5/L6 chi² 停滞）
- 保留抑制性突触 [-W_MAX, 0] 钳位（防权重归零）
- 保留 STDP 时序：先算 Δw 再更新 last_pre/post_spike

### 4. 网络初始化

**来源**: `legacy/stage2e/network_init.cu` (39KB)

**功能**: 50 柱拓扑生成 + 1/√K 平衡态权重缩放 + PSW 初始化。

**移植目标**: `src/snn/network_init.cu/.cuh`

**移植要点**:
- 50 非重叠柱偏好（每柱偏好 256/50≈5 个字节）
- 80% 兴奋 / 20% 抑制（FS/LTS/SOM 三亚型）
- 独立前额叶（5000 神经元 / 50 组）

## 不复用的模块

| 模块 | 原因 |
|---|---|
| `legacy/stage2e/scheduler.cu` (80KB) | 异构架构不再需要多时间尺度流水线调度 |
| `legacy/stage2e/thalamic_gate.cu` | 丘脑门控是研究阶段的输入控制，异构架构由 Bridge 接管 |
| `legacy/stage2e/hippocampal_kernels.cu` | 海马重放未实现，Phase 5 再考虑 |
| `legacy/stage2e/wm_kernels.cu` | 工作记忆槽位由 LLM 上下文管理替代 |
| `legacy/stage2e/modulatory_kernels.cu` | 神经调质在异构架构中无对应概念 |
| `legacy/stage0/1/2/` | 早期实验代码，已被 stage2e 完全替代 |

## 关键约束（从 project_memory 继承）

移植时必须遵守以下硬约束，否则会重现已知 bug：

1. **STDP 时序**: delta_w 必须在更新 last_pre_spike/last_post_spike 之前计算
2. **抑制性突触钳位**: [-W_MAX, 0]，不是 [0, 1]
3. **树突区室化**: 前馈连接用独立 Ca²⁺ 动力学（τ=10ms, max=0.12 < rebound 阈值 0.15）
4. **BPTT + P3-D 互斥**: BPTT 模式下必须跳过 P3-D 结构重建（防 CSR 索引错位 GPU 死锁）
5. **PSW_ETA_ALPHA/BETA**: BPTT 模式下需提升到 200.0（加速 PSW 成熟）
6. **Checkpoint 完整性**: 必须保存完整 d_synapses_（32MB，含 STDP 状态），不是 d_weights_

## 移植步骤建议

1. 先复制 `bptt_trainer.cu/.cuh` 到 `src/snn/`，改 namespace，编译通过
2. 复制 `pca_kernels.cu/.cuh`，写一个独立测试：随机输入 → 提取签名 → 验证维度
3. 复制 `network_init.cu`，初始化 60K 神经元网络，保存 checkpoint
4. 复制 `synapse_kernels.cu` + `neuron_kernels.cu`，跑 1K 步前向，验证 spike/step ∈ [50, 200]
5. 实现 `memory_index.cu`：基于 PCA 签名的余弦相似度 Top-K
6. 实现 `online_stdp.cu`：用户反馈触发的局部 STDP 更新（不改全局拓扑）
