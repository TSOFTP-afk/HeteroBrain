# HeteroBrain ISA 动态协作架构

## 1. 目标与边界

本设计将 Indecisive Swarm Algorithm（ISA）的**动态单源拓扑**应用于
HeteroBrain 的模块/Agent 协作层，而不是 SNN 神经元、突触或 token 采样层。

严格 ISA 部分只有四项：

1. 每个目标 Agent 在一轮中只接收一个主要来源；
2. 来源是全局控制者，或一个同行 Agent；
3. 每次状态更新前重新采样有向拓扑；
4. 拓扑更新与状态更新一一对应，即 `T = tau_d / tau_n = 1`。

语义匹配、信任权重、成本权重、强制正典检查和人工审批属于 HeteroBrain
工程扩展，不应被描述成原论文结论。

## 2. Agent 映射

| ISA | HeteroBrain |
|---|---|
| Controller | Story Director / 全局任务目标 |
| Agent | Planner、Writer、SNN Memory、Vector Memory、Critic 等 |
| State update | 一次规划、草稿、审稿或修订 |
| Temporal edge | 本轮 Agent 接受哪个来源的信息 |
| Herding target | 当前场景目标、章节目标或一般任务验收条件 |

路由粒度默认是一次任务/修订轮。小说模式可选择场景级或段落级，但禁止默认
使用 token 级随机切换，因为它会破坏 KV cache、文风和上下文连续性。

## 3. 单轮执行语义

每轮严格分成三个阶段：

1. **Snapshot**：冻结上一轮全部 Agent 状态；
2. **Resample**：每个目标 Agent 在 Controller 与一个 Peer 之间选择单一来源；
3. **Update**：所有 Agent 仅使用 Snapshot 和选中来源同步更新状态。

```text
round k snapshot
       |
       v
resample G(k): one incoming edge per target
       |
       v
synchronous agent updates
       |
       v
deterministic acceptance gates
       |
       v
round k+1
```

不可逆操作、正典写入、外部发布和事实声明不得仅依赖随机路由。它们必须经过
图外的确定性验收门。

## 4. 两阶段选择

对目标 Agent `i`：

```text
controller, probability alpha
one peer,  probability gamma = 1 - alpha
```

在选中 peer 分支后，再使用相关性、可靠性、新鲜度与成本对候选同行加权：

```text
score_j =
    w_rel   * relevance_j
  + w_trust * reliability_j
  + w_fresh * freshness_j
  - w_cost  * normalized_cost_j
```

第二阶段是工程扩展，但仍保持“每轮只采纳一个来源”的 ISA 约束。

## 5. 三类记忆

### 5.1 硬记忆

结构化正典、人物状态、时间线、权限和不可违反约束。硬记忆必须可审计、可编辑，
不能由 SNN 模糊召回替代。

### 5.2 向量记忆

负责语义召回和高精度 Top-K 候选生成，作为 SNN 增益实验的强基线。

### 5.3 SNN 联想记忆

负责时间共激活、熟悉度、新奇度、情绪/意象联想和在线个性化。SNN 输出是
`association_score` 或 `memory_consistency_score`，不是事实真值。

原计划中的 `truth_filter` 应调整为 `memory_consistency_scorer`。事实验证依靠
RAG 证据、工具调用或独立 Verifier。

## 6. 小说创作拓扑示例

```text
round 1: Writer        <- Story Director
round 2: Writer        <- SNN Memory
round 3: Critic        <- Writer
round 4: Writer        <- Critic
round 5: Canon Checker <- Writer
```

前四轮属于动态协作图。第五轮可以参与路由，但“是否写入正典”仍由确定性验收门
决定。

## 7. 对照实验

必须同时实现三类拓扑，才能把收益归因于 ISA：

| Baseline | 拓扑 |
|---|---|
| ASA | 每轮融合全部来源 |
| LFSA | 任务开始时固定单一来源/固定流水线 |
| ISA | 每轮选择一个来源并重新采样 |

通用指标：

- 任务成功率、失败率；
- 每任务模型调用数、token、延迟和能耗；
- 路由熵、Agent 覆盖率、循环/孤岛出现率；
- SNN 联想召回相对向量检索的增益。

小说指标：

- 人物/时间线/世界规则矛盾率；
- 伏笔回收率、长距回调成功率；
- 主题漂移和重复短语比例；
- 人工文笔、人物声线和情感评分。

## 8. 分阶段落地

1. 先打通 LLM Provider、结构化记忆和向量检索；
2. 定义稳定的 SNN 记忆记录与读取接口；
3. 完成固定 LFSA 流水线；
4. 接入本分支的 ISA topology core；
5. 增加同步 Orchestrator 和路由日志；
6. 加入确定性验收门；
7. 实现 ASA/LFSA/ISA 对照及消融实验；
8. 证明显著增益后再学习路由权重。

当前 `dynamic_router` 只实现步骤 4 的可复现拓扑采样核心，不宣称已经完成
HeteroBrain 端到端协作。
