# Phase 3a-D: 发育式认知训练系统设计

> **状态**: 设计稿
> **作者**: session 2026-07-30
> **依赖**: Phase 3a-C1 (事件驱动调质注入) 已完成
> **后续**: Phase 3b (认知工作空间) 的前置基础

---

## 1. 背景与动机

### 1.1 当前系统的根本局限

Phase 3a-C1 实现了"事件→调质"映射，但事件是**无根标签**：

```
food_tasty → DA↑0.40   # SNN 不知道"食物"是什么
praise → DA↑0.25        # SNN 不知道"表扬"的社会含义
```

SNN 只是对 6 个数字做条件反射，缺少：
- **感觉接地**: 没有感官通道，"食物"只是抽象标签
- **因果学习**: 没有学到"哭→喂养→满足"的因果结构
- **概念构建**: 没有 grounded 符号系统

### 1.2 用户洞察

> "不行没有事实意义，这些仅仅是事件产生的情绪，没有事实事件的支撑，实际上我觉得我们应该从0岁开始训练"

核心主张：**情绪必须有事实事件支撑**，应模拟人类从出生开始的发育过程，让 SNN 在真实的事件序列中成长，使调质响应扎根于语义因果。

### 1.3 设计目标

让 SNN 经历 0-18 岁的完整叙事流，通过**因果链时序**学习事件之间的因果结构，而非对单事件做反射。验证 SNN 能否：
1. 学到因果预测（哭泣预测喂养到来）
2. 形成阶段一致的情绪模式（婴儿期 vs 青少年期）
3. 实现概念接地（"食物"稳定触发 DA 响应）

---

## 2. 核心设计

### 2.1 因果链范式

传统方式：单事件脉冲
```
[事件标签] → [调质增量]        (无语义, 机械反射)
```

发育方式：因果链序列
```
[饥饿叙事] → [微负DA]           (因果链前段: 需求)
[哭泣叙事] → [NE↑]              (因果链前段: 行动)
[妈妈回应] → [Oxy↑微]           (因果链中段: 回应)
[喂养叙事] → [DA↑强]            (因果链后果: 满足)
[满足叙事] → [5HT↑]             (情绪收尾: 平静)
                                    ↓
                 SNN 学到: 哭泣预测喂养(因果), 非标签反射
```

**关键创新**：一个"场景"不再是单事件，而是 4-8 个微事件构成的因果链序列，SNN 经历整个序列而非孤立标签。

### 2.2 发育阶段划分（月龄粒度）

| 阶段 | 月龄 | 叙事特征 | 核心事件类型 | 学习目标 |
|------|------|----------|-------------|---------|
| 新生儿 | 0-1月 | 极简词(1-3字) | 饥饿/温暖/惊吓 | 基本因果(哭→回应) |
| 婴儿 | 2-12月 | 短语(2-5字) | 喂养/安抚/探索 | 情绪接地 |
| 幼儿 | 13-36月 | 短句(3-7字) | 学语/探索/社交 | 物体概念 |
| 学前 | 3-6岁 | 完整句 | 幼儿园/规则/想象 | 社交规则 |
| 学龄 | 7-12岁 | 短段落 | 学习/友谊/竞争 | 社会比较 |
| 青少年 | 13-18岁 | 复杂叙事 | 身份/目标/挫折 | 抽象思维 |

### 2.3 时间映射（标准模式）

- **1 月龄 = 2000 训练步**
- 0-36月：37 × 2000 = 74,000 步
- 3-18岁：16 年 × 12 月 × 2000 = 384,000 步（年份按12月计）
- **总步数 ≈ 458,000 步**（约 460K 步）
- 每场景约占 200-400 步，月均 5-10 个场景

---

## 3. 数据结构

### 3.1 发育场景结构

```json
{
  "scene_id": "infant_002_hunger",
  "age_months": 2,
  "developmental_stage": "infant",
  "scene_type": "hunger_feeding_cycle",
  "learning_goal": "hunger_cry_feeding_causality",
  "narrative_full": "宝宝饿了。哭着。妈妈来了。抱起。喂奶。饱了。舒服。",
  "causal_chain": ["hunger", "cry", "mother_response", "feeding", "satisfaction"],
  "event_segments": [
    {
      "text": "宝宝饿了",
      "event_type": "food_bland",
      "intensity": -20,
      "step_offset": 0,
      "modifiers": {"publicity": "private", "authority": "authority", "temporal": "momentary"}
    },
    {
      "text": "哭着",
      "event_type": "threat_physical",
      "intensity": -15,
      "step_offset": 40,
      "modifiers": {"publicity": "private", "authority": "peer", "temporal": "momentary"}
    },
    {
      "text": "妈妈来了",
      "event_type": "social_bond",
      "intensity": 10,
      "step_offset": 80,
      "modifiers": {"publicity": "private", "authority": "authority", "temporal": "momentary"}
    },
    {
      "text": "喂奶",
      "event_type": "food_tasty",
      "intensity": 30,
      "step_offset": 120,
      "modifiers": {"publicity": "private", "authority": "authority", "temporal": "momentary"}
    },
    {
      "text": "饱了",
      "event_type": "achievement",
      "intensity": 15,
      "step_offset": 160,
      "modifiers": {"publicity": "private", "authority": "peer", "temporal": "momentary"}
    },
    {
      "text": "舒服",
      "event_type": "social_bond",
      "intensity": 10,
      "step_offset": 200,
      "modifiers": {"publicity": "private", "authority": "peer", "temporal": "sustained"}
    }
  ],
  "scene_duration_steps": 250,
  "sensory_tags": ["warmth", "sweet", "satiety"]
}
```

### 3.2 发育事件流文件 (developmental_events.jsonl)

每行一个微事件（展平 event_segments），供 EventScheduler 加载：

```json
{"scene_id":"infant_002_hunger","age_months":2,"segment_idx":0,"step_target":2040,"event_type":"food_bland","intensity":-20,"description":"宝宝饿了","learning_goal":"hunger_cry_feeding_causality"}
{"scene_id":"infant_002_hunger","age_months":2,"segment_idx":1,"step_target":2080,"event_type":"threat_physical","intensity":-15,"description":"哭着","learning_goal":"hunger_cry_feeding_causality"}
...
```

### 3.3 叙事 BPE token 流 (narrative_tokens.bin)

所有场景的叙事文本按年龄顺序拼接，经 BPE 分词后输出为 int32 token 流：
```
[场景1文本tokens] [场景2文本tokens] ... [场景N文本tokens]
```

每个 token 对应一个训练步，场景间插入间隔 token（如 `<scene_break>`）。

---

## 4. 混合叙事生成架构

### 4.1 模板层（确定性骨架）

`developmental_templates.py` 定义 53 个时间点的场景骨架：

```python
DEVELOPMENTAL_SCENES = [
    # 新生儿期 (0-1月)
    {
        "age_months": 0,
        "stage": "neonatal",
        "scenes": [
            {
                "scene_type": "hunger_feeding",
                "causal_chain": ["hunger", "cry", "mother_response", "feeding", "satisfaction"],
                "event_types": ["food_bland", "threat_physical", "social_bond", "food_tasty", "achievement"],
                "intensities": [-20, -15, 10, 30, 15],
                "learning_goal": "basic_causality"
            },
            # ... 更多场景
        ]
    },
    # 婴儿期 (2-12月)
    # ...
]
```

### 4.2 LLM 填充层（自然语言叙事）

`narrative_generator.py` 调用 LLM 生成年龄适配叙事：

```python
def generate_narrative(scene_skeleton, age_months):
    prompt = f"""
你是儿童发展心理学专家。请为 {age_months} 月龄的婴儿生成一个简短叙事。

场景类型: {scene_skeleton['scene_type']}
因果链: {' → '.join(scene_skeleton['causal_chain'])}

要求:
1. 叙事长度: {get_narrative_length(age_months)} 个字
2. 语言复杂度: {get_language_complexity(age_months)}
3. 将因果链的每个环节都体现在叙事中
4. 输出格式: 用句号分隔的短句, 每句对应因果链一个环节

输出仅叙事文本, 不要解释。
"""
    return llm_call(prompt)
```

### 4.3 缓存机制

LLM 生成的叙事按 `scene_id` 缓存到 `data/developmental/narrative_cache/`，避免重复调用。

---

## 5. 系统组件

### 5.1 组件清单

| 组件 | 文件 | 职责 |
|------|------|------|
| 发育模板库 | `src/snn/tools/developmental_templates.py` | 53个时间点的场景骨架 |
| LLM叙事生成器 | `src/snn/tools/narrative_generator.py` | 模板骨架 → 自然语言叙事 |
| 发育数据集组装器 | `src/snn/tools/generate_developmental_dataset.py` | 整合模板+LLM → JSONL |
| 叙事BPE转换器 | `src/snn/tools/narrative_to_bpe.py` | 叙事文本 → BPE token流 |
| 发育验证器 | `src/snn/tools/validate_developmental.py` | 因果学习+阶段一致性验证 |

### 5.2 数据流

```
developmental_templates.py (场景骨架)
         ↓
narrative_generator.py (LLM填充叙事)
         ↓
generate_developmental_dataset.py
    ├──→ developmental_events.jsonl  (供 --event-stream)
    └──→ narrative_text.txt          (供 narrative_to_bpe.py)
                ↓
         narrative_to_bpe.py
                ↓
         narrative_tokens.bin         (供 --bpe-data)
                ↓
snn_train.exe --bpe-data narrative_tokens.bin
             --event-stream developmental_events.jsonl
             --steps 460000
                ↓
         validate_developmental.py (验证因果学习)
```

### 5.3 与现有系统的集成

| 现有组件 | 发育模式扩展 |
|---------|------------|
| `--event-stream` (单事件脉冲) | 直接复用, 微事件按 step_offset 派发 |
| `--bpe-data` (BPE token流) | 直接复用, 输入发育叙事 token 流 |
| `launch_modulatory` (事件信号读取) | 无需修改 |
| `hippocampal_kernels` (情节记忆) | 新角色: 存储发育场景记忆 (Phase 3b 深化) |
| `EventScheduler` (事件调度) | 无需修改, 已支持 step_target 时序派发 |

**关键**: 无需修改 CUDA 核心，仅需准备数据 + 验证脚本。

---

## 6. 发育场景设计（按阶段）

### 6.1 新生儿期 (0-1月) — 感觉运动接地

**核心场景** (~5个/月):
1. `hunger_feeding`: 饥饿→哭→喂养→满足 (因果链原型)
2. `warmth_safety`: 寒冷→哭→包裹→温暖 (温度调节)
3. `startle_recover`: 惊吓→哭→安抚→平静 (惊吓反射)
4. `sleep_wake`: 困倦→哭→哄睡→入睡 (睡眠周期)
5. `discomfort_change`: 不适→哭→换尿布→舒适 (身体护理)

**叙事示例** (2月龄):
```
宝宝饿了。哭着。妈妈来了。抱起。喂奶。饱了。舒服。
```

### 6.2 婴儿期 (2-12月) — 情绪接地

**新增场景类型**:
- `exploration`: 抓握→品尝→发现 (物体探索)
- `social_smile`: 看到妈妈→微笑→回应→愉悦 (社交微笑)
- `separation_anxiety`: 妈妈离开→哭→妈妈回来→安心 (依恋)
- `novelty_surprise`: 新玩具→惊奇→探索→满足 (好奇心)

**叙事示例** (8月龄):
```
宝宝看到球。伸手抓。拿到了。摇一摇。响了。好开心。
```

### 6.3 幼儿期 (13-36月) — 语言与概念

**新增场景类型**:
- `language_acquisition`: 指物→命名→重复→理解 (词汇习得)
- `autonomy`: 想自己做→尝试→成功/失败→骄傲/挫败 (自主性)
- `peer_interaction`: 遇到同伴→互动→分享/争抢→友谊/冲突
- `rule_learning`: 被禁止→哭闹→解释→接受 (规则内化)

### 6.4 学前期 (3-6岁) — 社交规则

**新增场景类型**:
- `kindergarten_entry`: 进入幼儿园→陌生→适应→交友
- `imaginative_play`: 想象游戏→角色扮演→合作→创造
- `emotional_regulation`: 生气→表达→引导→平静
- `empathy_development`: 看到别人哭→关心→安慰→共情

### 6.5 学龄期 (7-12岁) — 社会比较

**新增场景类型**:
- `academic_achievement`: 学习→考试→好成绩→骄傲
- `academic_failure`: 学习→考试→差成绩→挫败→重新努力
- `friendship_deep`: 共同活动→信任→深层友谊
- `competition`: 比赛→胜利/失败→风度/嫉妒
- `bullying_experience`: 被欺负→委屈→求助→解决

### 6.6 青少年期 (13-18岁) — 抽象思维

**新增场景类型**:
- `identity_formation`: 思考自我→探索→困惑→确立
- `romantic_interest`: 心动→接近→交往/被拒→喜悦/心碎
- `rebellion`: 规则束缚→反抗→冲突→和解
- `future_anxiety`: 思考未来→迷茫→规划→希望
- `moral_dilemma`: 面临选择→权衡→决定→反思

---

## 7. 验证准则

### 7.1 因果预测学习

**方法**: 在"哭泣"段结束时记录调质状态，检查是否能预测"喂养"段的调质响应方向。

```python
def validate_causal_prediction(events_log):
    # 对每个 hunger_feeding 场景:
    #   cry段结束时的调质状态 → 应能预测 feeding段的方向
    #   即: cry段NE↑ → feeding段DA↑(正相关)
    correlation = compute_cross_segment_correlation(events_log, "cry", "feeding")
    assert correlation > 0.3  # 正相关阈值
```

### 7.2 阶段一致性

**方法**: 对比婴儿期和青少年期的调质响应模式，应存在显著差异。

```python
def validate_stage_consistency(infant_log, teen_log):
    # 婴儿期: 基本需求事件(饥饿/温暖)占主导
    # 青少年期: 复杂社交事件(身份/挫败)占主导
    infant_entropy = compute_event_type_entropy(infant_log)
    teen_entropy = compute_event_type_entropy(teen_log)
    assert teen_entropy > infant_entropy  # 青少年情绪更复杂
```

### 7.3 概念接地

**方法**: 检查同类事件跨年龄的调质响应方向一致性。

```python
def validate_concept_grounding(all_events_log):
    # "食物"类事件(跨所有年龄)应稳定触发 DA↑
    food_events = filter_by_type(all_events_log, "food_tasty")
    da_responses = [e["da_delta"] for e in food_events]
    assert all(d > 0 for d in da_responses)  # 方向一致
    assert np.std(da_responses) < 0.2  # 响应稳定
```

---

## 8. 实施路线图

### Phase 3a-D1: 核心数据管线 (最小可行)
1. 编写新生儿期(0-1月)模板库
2. 实现 LLM 叙事生成器(带缓存)
3. 实现 developmental_events.jsonl 生成
4. 实现 narrative_to_bpe 转换
5. 端到端验证: 新生儿期 2K 步训练

### Phase 3a-D2: 完整发育时间线
1. 扩展模板库到 0-18 岁
2. 批量生成叙事(带缓存)
3. 完整 460K 步训练
4. 三项验证(因果预测/阶段一致/概念接地)

### Phase 3a-D3: 深化(可选)
1. 海马体情节记忆集成
2. 发育场景记忆回放
3. 与 Phase 3b 认知工作空间对接

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 叙事质量不稳定 | 数据噪声 | 模板约束 + 缓存 + 人工抽检 |
| BPE 词表覆盖不足 | 婴儿词汇 OOV | 预扩展词表或用字节回退 |
| 460K 步训练太久 | 验证延迟 | 分阶段验证(D1 先验证新生儿期) |
| 因果学习不显现 | 核心目标失败 | 增加场景重复率 + 调整时间映射 |
| 调质饱和/病理滑移 | 训练崩溃 | 稳态补偿机制已就绪(Phase 3a-B) |

---

## 10. 成功标准

1. **数据完整性**: 0-18岁发育数据集生成完成, 500+场景, 3000+微事件
2. **训练稳定性**: 460K步训练无崩溃, 调质浓度全程在 [0,2] 范围
3. **因果学习**: 哭泣段调质状态能预测喂养段方向 (相关系数 > 0.3)
4. **阶段差异**: 婴儿期与青少年期调质模式有显著差异 (熵差 > 0.5)
5. **概念接地**: 同类事件跨年龄调质响应方向一致 (100% 方向一致)

---

## 附录 A: 与现有文档的关系

- `docs/snn-emotion-and-workspace-direction.md` §3.4: 事件驱动调质注入 → 本文档深化为发育式
- `docs/superpowers/specs/2026-07-30-event-driven-modulator-injection-design.md`: C1 单事件 → D 发育因果链
- `PROJECT_MEMORY.md`: 需更新 Phase 3a-D 条目

## 附录 B: 术语表

- **因果链 (causal_chain)**: 一个场景内按时间顺序的微事件序列, 如 "饥饿→哭→喂养→满足"
- **微事件 (event_segment)**: 因果链中的一个环节, 对应一次调质注入
- **场景 (scene)**: 一个完整的因果链序列, 包含 4-8 个微事件
- **发育阶段 (developmental_stage)**: 按年龄划分的认知发育阶段
- **概念接地 (concept_grounding)**: 抽象符号(如"食物")与调质响应的稳定关联
