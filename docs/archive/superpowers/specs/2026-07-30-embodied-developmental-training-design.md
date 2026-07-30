# 具身发育式认知训练系统设计 (Embodied Developmental Training)

> **状态**: Phase 3a-D 设计文档
> **日期**: 2026-07-30
> **替代**: `2026-07-30-developmental-cognitive-training-design.md` (旧版事件链方案, 已废弃)
> **依赖**: Phase 3a-B (稳态补偿), Phase 3a-C1 (事件驱动调质注入)

---

## 1. 动机与问题诊断

### 1.1 旧方案的根本缺陷

旧版发育训练方案 (`2026-07-30-developmental-cognitive-training-design.md`) 本质是**巴甫洛夫反射弧**:

```
[预设事件标签] → [基因映射表查表] → [6维调质增量] → [SNN被动响应]
```

**四个缺失:**

| 缺失 | 后果 |
|------|------|
| **没有身体** | SNN 不知道"饿"是什么感觉, 只收到 `food_bland -25` 标签 |
| **没有感知** | 没有触/听/视信号, 只有抽象事件 ID |
| **没有行动** | SNN 不能影响事件发生, "哭→妈妈来"是预设脚本 |
| **没有因果** | 因果链是模板硬编码的, 不是 SNN 学到的预测结构 |

### 1.2 新方案核心思想

给 SNN 一个**虚拟婴儿身体 + 概率响应环境**, 让它通过**感知-行动闭环**经历真实因果链:

```
[身体状态 hunger↑] → [内感态信号] → SNN 活动演化
                                      ↓
                                  [哭输出]
                                      ↓
                              [环境概率响应]
                                      ↓
                              [妈妈来 → 喂养]
                                      ↓
                              [hunger↓ → DA↑]
                                      ↓
                              [闭环奖励反馈]
```

**关键转变:**
- 情绪 = 身体状态演化的**结果** (不再是事件标签)
- 事件 = SNN 行动导致的**环境反馈** (不再是天上掉的)
- 因果链 = SNN 通过闭环**学到的预测结构** (不再是预设脚本)

---

## 2. 系统架构

### 2.1 五层闭环架构

```
┌──────────────────────────────────────────────────────────┐
│              Layer 1: BodyState (虚拟婴儿身体)              │
│  hunger∈[0,1]  temp∈[0,1]  comfort∈[0,1]                 │
│  fatigue∈[0,1] arousal∈[0,1]                              │
│  每环境步演化: hunger+=Δh, temp→ambient, comfort=f(diaper)│
└───────┬────────────────────────────────┬─────────────────┘
        │ 内感态信号(15柱)                │ 身体状态更新
        ▼                                │
┌──────────────────────────────────────────────────────────┐
│         Layer 2: SensoryInput (50柱多模态感知)             │
│  触觉10│听觉10│视觉10│嗅觉5│内感态15                       │
│  launch_multi_sensory_inject → d_input_current (L4层)     │
└───────┬──────────────────────────────────────────────────┘
        │ spike注入
        ▼
┌──────────────────────────────────────────────────────────┐
│              SNN 主网络 (60K神经元, 现有, 不改)             │
│  L4→L2/3→L5→L6  +  6维调质 + 稳态补偿 + STDP              │
│  W_pred世界模型  +  w_value价值函数  +  decode行为模仿      │
└───────┬──────────────────────────────┬───────────────────┘
        │ L5→运动皮层突触               │ V(s), W_pred误差
        ▼                              │
┌──────────────────────────────────────────────────────────┐
│         Layer 3: MotorOutput (5K运动皮层, 50组→5动作)      │
│  哭(组0-9) 手(10-19) 脚(20-29) 吸吮(30-39) 注视(40-49)   │
│  组内投票 → 组间softmax → action_prob[5] → 采样动作        │
└───────┬──────────────────────────────────────────────────┘
        │ 动作输出
        ▼
┌──────────────────────────────────────────────────────────┐
│         Layer 4: Environment (概率响应环境, C++实时)        │
│  妈妈响应: P(来)=σ(a·cry+b·arousal-c·mom_fatigue)         │
│  喂养量: amount=f(来, suck_strength) → hunger-=amount     │
│  温度: temp += k·(ambient-temp) + 包裹修正                │
│  舒适: comfort=f(diaper_dirty, holding, position)         │
│  DA reward = Δcomfort - Δhunger·w + Δarousal·w2           │
└───────┬──────────────────────────────────────────────────┘
        │ 状态更新 + 感知变化
        ▼
    (回到 Layer 1, 闭环)
```

### 2.2 Layer 5: 学习目标 (四合一)

| 目标 | 实现方式 | 复用现有代码 |
|------|----------|-------------|
| **世界模型预测** | W_pred 预测 next subcolumn_fr, 预测误差驱动 STDP | ✓ `w_pred_update_kernel` |
| **价值学习** | DA reward → TD error → w_value 更新 | ✓ `launch_da_value_function` (扩展 reward 参数) |
| **行为模仿** | 教师信号: hunger>0.6 时 target=CRY, 误差驱动运动皮层权重 | 新增 `motor_teacher_kernel` |
| **好奇心探索** | W_pred 预测误差大 → ACh↑ → 探索驱动 | ✓ 预测误差已有, ACh 注入新增 |

### 2.3 时间尺度映射

| 层级 | 映射 | 理由 |
|------|------|------|
| **SNN步** | 1步 = 1ms (不变) | 保持现有神经元动力学 (τ_m=9.37ms) |
| **环境步** | 1环境步 = 100 SNN步 = 100ms | 与 `launch_modulatory` 同步, 身体状态每100ms更新 |
| **场景** | 1场景 = 3000环境步 = 300K SNN步 = 5分钟 | 一个完整因果链 (饥饿→哭→喂养→满足) 约5分钟 |
| **D1验证** | 5场景 × 10次 = 50场景 = 15M步 | 统计因果学习 (D1实际只跑5K步验证不崩) |

---

## 3. 详细设计

### 3.1 Layer 1: BodyState (虚拟婴儿身体)

#### 3.1.1 数据结构

```cpp
// embodied_body.h
struct BodyState {
    // 核心内感态 (5维, ∈[0,1])
    float hunger;       // 饥饿度: 每环境步+0.001, 喂养时-0.3*suck
    float temperature;  // 体感温度: 理想0.5, 向ambient收敛
    float comfort;      // 舒适度: f(diaper, holding, position)
    float fatigue;      // 疲劳度: 每步+0.0005, 睡眠归零
    float arousal;      // 唤醒度: f(hunger, comfort, fatigue)
    
    // 环境衍生状态
    float ambient_temp;     // 环境温度 [0,1]
    float diaper_dirty;     // 尿布脏度 [0,1], 累积, 换尿布归零
    bool  is_held;          // 是否被抱
    bool  is_fed;           // 是否在喂养
    float mom_fatigue;      // 妈妈疲劳度 [0,1], 影响响应概率
    
    // 衍生计算
    float discomfort() const;  // 不适感 = f(hunger, 1-comfort, fatigue)
    float distress() const;    // 痛苦感 = f(discomfort, arousal)
};
```

#### 3.1.2 演化方程 (每环境步 = 100 SNN步)

```cpp
void BodyState::step(float dt) {
    // dt = 1.0 (一个环境步)
    // 饥饿: 基础速率 + 唤醒加成
    hunger += 0.001f * (1.0f + arousal * 0.3f);
    if (is_fed) hunger -= 0.3f;  // 喂养时大幅下降 (外部设置is_fed)
    hunger = clamp(hunger, 0.0f, 1.0f);
    
    // 温度: 向ambient指数收敛
    temperature += 0.01f * (ambient_temp - temperature);
    
    // 尿布: 饥饿时排泄加快
    diaper_dirty += 0.0003f * (hunger > 0.5f ? 1.5f : 1.0f);
    diaper_dirty = clamp(diaper_dirty, 0.0f, 1.0f);
    
    // 舒适度: 受尿布/温度/抱影响
    comfort = clamp(
        0.7f
        - diaper_dirty * 0.5f
        - fabsf(temperature - 0.5f) * 0.6f
        + (is_held ? 0.2f : 0.0f),
        0.0f, 1.0f
    );
    
    // 疲劳: 持续累积
    fatigue += 0.0005f;
    fatigue = clamp(fatigue, 0.0f, 1.0f);
    
    // 唤醒: 综合驱动
    arousal = clamp(
        0.3f
        + hunger * 0.4f
        + (1.0f - comfort) * 0.3f
        - fatigue * 0.2f,
        0.0f, 1.0f
    );
    
    // 妈妈疲劳: 每步微增, 响应后增加
    mom_fatigue += 0.0002f;
    mom_fatigue = clamp(mom_fatigue, 0.0f, 1.0f);
}
```

#### 3.1.3 内感态信号编码 (15柱: 柱35-49)

每维3柱 (low/mid/high 阈值激活):

```cpp
void BodyState::encode_interoception(float out[15]) const {
    // hunger: 柱35(low<0.3) 柱36(mid 0.3-0.7) 柱37(high>0.7)
    out[0] = (hunger < 0.3f) ? 1.0f : 0.0f;
    out[1] = (hunger >= 0.3f && hunger < 0.7f) ? 1.0f : 0.0f;
    out[2] = (hunger >= 0.7f) ? 1.0f : 0.0f;
    // temperature: 柱38-40 (cold<0.4, ok 0.4-0.6, hot>0.6)
    out[3] = (temperature < 0.4f) ? 1.0f : 0.0f;
    out[4] = (temperature >= 0.4f && temperature < 0.6f) ? 1.0f : 0.0f;
    out[5] = (temperature >= 0.6f) ? 1.0f : 0.0f;
    // comfort: 柱41-43 (low<mid<high)
    out[6] = (comfort < 0.3f) ? 1.0f : 0.0f;
    out[7] = (comfort >= 0.3f && comfort < 0.7f) ? 1.0f : 0.0f;
    out[8] = (comfort >= 0.7f) ? 1.0f : 0.0f;
    // fatigue: 柱44-46
    out[9] = (fatigue < 0.3f) ? 1.0f : 0.0f;
    out[10] = (fatigue >= 0.3f && fatigue < 0.7f) ? 1.0f : 0.0f;
    out[11] = (fatigue >= 0.7f) ? 1.0f : 0.0f;
    // arousal: 柱47-49
    out[12] = (arousal < 0.3f) ? 1.0f : 0.0f;
    out[13] = (arousal >= 0.3f && arousal < 0.7f) ? 1.0f : 0.0f;
    out[14] = (arousal >= 0.7f) ? 1.0f : 0.0f;
}
```

### 3.2 Layer 2: SensoryInput (50柱多模态感知)

#### 3.2.1 柱分配

| 柱范围 | 模态 | 信号源 | 激活逻辑 |
|--------|------|--------|----------|
| 0-4 | 触觉-被抱 | is_held | is_held=true → 柱0-4全部激活 |
| 5-9 | 触觉-抚摸 | 抚摸强度 | 按强度激活柱5-9中的K个 |
| 10-14 | 听觉-妈妈声 | 妈妈在场说话 | mom_present && speaking → 柱10-14 |
| 15-19 | 听觉-噪音 | 环境噪音 | 噪音强度 → 柱15-19按强度 |
| 20-24 | 视觉-光 | 光照强度 | 光强 → 柱20-24按强度 |
| 25-29 | 视觉-人脸 | 妈妈脸出现 | mom_visible → 柱25-29 |
| 30-32 | 嗅觉-奶味 | 喂养中 | is_fed → 柱30-32 |
| 33-34 | 嗅觉-妈妈味 | 妈妈近 | mom_present → 柱33-34 |
| 35-49 | 内感态 | BodyState | 见 3.1.3 |

#### 3.2.2 感知信号结构

```cpp
// embodied_sensory.h
struct SensorySignals {
    // 触觉
    bool  is_held;          // 被抱
    float touch_intensity;  // 抚摸强度 [0,1]
    // 听觉
    bool  mom_speaking;     // 妈妈在说话
    float noise_intensity;  // 噪音强度 [0,1]
    // 视觉
    float light_level;      // 光照 [0,1]
    bool  mom_visible;      // 妈妈脸可见
    // 嗅觉
    bool  milk_smell;       // 奶味
    bool  mom_smell;        // 妈妈气味
    
    // 编码为50维柱信号
    void encode(float out[50]) const;
};
```

#### 3.2.3 注入 Kernel

```cpp
// multi_sensory_inject.cu
__global__ void multi_sensory_inject_kernel(
    const float* __restrict__ sensory_signals,  // [50] 柱信号强度
    float* __restrict__ input_current,           // d_input_current
    const ThalamicGateState* __restrict__ gate_states,
    int n_columns,
    int neurons_per_column,
    int l4_size);

// Host launcher
void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states);
```

**注入逻辑 (每柱):**
1. 信号强度 `s = sensory[col]` ∈ [0,1]
2. 丘脑门控 `gate = gate_states[col].gate_signal` ∈ [0,1]
3. 在柱内 L4 层 (200神经元) 用 xorshift32 哈希激活 K = `int(100 * s * gate)` 个神经元
4. 每个激活神经元: `atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN * s * gate)`

### 3.3 Layer 3: MotorOutput (50组→5动作)

#### 3.3.1 动作定义

```cpp
// embodied_motor.h
enum ActionId {
    ACT_CRY = 0,    // 哭: 触发妈妈响应概率
    ACT_HAND = 1,   // 手部动作: D1无直接效果, 消耗能量
    ACT_FOOT = 2,   // 脚部动作: D1无直接效果, 消耗能量
    ACT_SUCK = 3,   // 吸吮: 喂养时 hunger-=0.3*suck
    ACT_GAZE = 4,   // 注视: D1无直接效果
    ACT_COUNT = 5
};

struct MotorReadout {
    float action_raw[5];     // 原始发放率 (softmax前)
    float action_prob[5];    // softmax概率
    int   action_sampled;    // 采样的离散动作
    // 连续值 (用于环境效果)
    float cry_intensity;     // = action_prob[ACT_CRY]
    float suck_strength;     // = action_prob[ACT_SUCK]
    float limb_movement;     // = (action_prob[ACT_HAND] + action_prob[ACT_FOOT]) / 2
};
```

#### 3.3.2 读出流程 (每环境步)

```cpp
MotorReadout read_motor_output(const bool* d_motor_spike_flags) {
    // 1. 拷贝spike_flags到host
    bool h_spike_flags[5000];
    cudaMemcpy(h_spike_flags, d_motor_spike_flags, ...);
    
    // 2. 按组聚合 (50组, 每组100神经元)
    float group_rate[50];
    for (int g = 0; g < 50; ++g) {
        int count = 0;
        for (int i = g * 100; i < (g + 1) * 100; ++i) {
            if (h_spike_flags[i]) ++count;
        }
        group_rate[g] = (float)count / 100.0f;
    }
    
    // 3. 5个动作各对应10组
    MotorReadout m;
    for (int a = 0; a < 5; ++a) {
        float sum = 0;
        for (int g = a * 10; g < (a + 1) * 10; ++g) {
            sum += group_rate[g];
        }
        m.action_raw[a] = sum / 10.0f;
    }
    
    // 4. Softmax (τ=0.5)
    float max_val = *std::max_element(m.action_raw, m.action_raw + 5);
    float exp_sum = 0;
    for (int a = 0; a < 5; ++a) {
        m.action_prob[a] = expf((m.action_raw[a] - max_val) / 0.5f);
        exp_sum += m.action_prob[a];
    }
    for (int a = 0; a < 5; ++a) m.action_prob[a] /= exp_sum;
    
    // 5. 采样动作
    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int a = 0; a < 5; ++a) {
        cum += m.action_prob[a];
        if (r < cum) { m.action_sampled = a; break; }
    }
    
    // 6. 连续值
    m.cry_intensity = m.action_prob[ACT_CRY];
    m.suck_strength = m.action_prob[ACT_SUCK];
    m.limb_movement = (m.action_prob[ACT_HAND] + m.action_prob[ACT_FOOT]) * 0.5f;
    
    return m;
}
```

### 3.4 Layer 4: Environment (概率响应环境)

#### 3.4.1 妈妈响应模型

```cpp
// embodied_env.h
class EmbodiedEnvironment {
    BodyState body;
    int   mom_response_countdown;  // 妈妈响应剩余环境步 (0=不在路上)
    bool  mom_present;             // 妈妈是否在场
    float mom_response_prob;       // 当前步响应概率 (用于日志)
    
    void step_env(const MotorReadout& motor);
    float compute_reward(const BodyState& prev) const;
};
```

**妈妈响应概率 (每环境步计算):**

> **D1 教师辅助机制:** D1 阶段 SNN 运动皮层尚未训练, cry_intensity 可能持续为 0. 为保证闭环可验证, 妈妈响应以 **hunger 为主驱动** (基因硬编码保底: 婴儿生理需求→母亲响应). D2 阶段 SNN 学会有效哭后, 切换为 cry_intensity 为主驱动.

```cpp
float EmbodiedEnvironment::compute_mom_response_prob() {
    // D1: hunger为主驱动 (基因保底, 确保闭环可验证)
    // D2: 将切换为 cry_intensity 为主驱动
    // σ(2.0*hunger + 1.5*arousal - 1.0*mom_fatigue - 0.5)
    float x = 2.0f * body.hunger  // 生理需求驱动 (基因硬编码)
            + 1.5f * body.arousal
            - 1.0f * body.mom_fatigue
            - 0.5f;
    return 1.0f / (1.0f + expf(-x));
}
```

**响应流程:**
```cpp
void EmbodiedEnvironment::step_env(const MotorReadout& motor) {
    BodyState prev = body;
    
    // 1. 妈妈响应判定 (概率)
    if (!mom_present && mom_response_countdown <= 0) {
        float p = compute_mom_response_prob();
        mom_response_prob = p;
        // cry_intensity作为额外驱动
        float p_cry = p * (0.5f + motor.cry_intensity);
        if ((float)rand() / RAND_MAX < p_cry) {
            // 妈妈开始响应, 延迟5-20环境步 (0.5-2秒)
            mom_response_countdown = 5 + rand() % 16;
        }
    }
    
    // 2. 响应倒计时
    if (mom_response_countdown > 0) {
        mom_response_countdown--;
        if (mom_response_countdown == 0) {
            mom_present = true;
            body.is_held = true;
            body.mom_fatigue += 0.05f;  // 响应消耗妈妈精力
        }
    }
    
    // 3. 妈妈在场时的效果
    if (mom_present) {
        // 喂养: 如果饥饿>0.3则喂奶
        if (body.hunger > 0.3f) {
            body.is_fed = true;
            body.hunger -= 0.3f * motor.suck_strength;
        }
        // 换尿布: 如果尿布脏>0.7
        if (body.diaper_dirty > 0.7f) {
            body.diaper_dirty = 0.0f;
        }
        // 妈妈离开: 饥饿<0.2且尿布干净
        if (body.hunger < 0.2f && body.diaper_dirty < 0.3f) {
            mom_present = false;
            body.is_held = false;
            body.is_fed = false;
        }
    }
    
    // 4. 身体状态演化
    body.step(1.0f);
    
    // 5. 动作能量消耗
    body.fatigue += motor.limb_movement * 0.002f;
}
```

#### 3.4.2 DA奖励信号

```cpp
float EmbodiedEnvironment::compute_reward(const BodyState& prev) const {
    float reward = 0.0f;
    // 舒适度提升 = 正奖励
    reward += (body.comfort - prev.comfort) * 1.0f;
    // 饥饿下降 = 正奖励 (满足感)
    reward += (prev.hunger - body.hunger) * 0.5f;
    // 唤醒适度变化 = 小正奖励 (探索)
    reward += (body.arousal - prev.arousal) * 0.2f;
    // 疲劳 = 小负奖励
    reward -= body.fatigue * 0.1f;
    return reward;
}
```

### 3.5 Layer 5: 学习目标实现

#### 3.5.1 世界模型预测 (复用 W_pred)

**现有代码:** `w_pred_predict_kernel` 预测 `pred_fr = W_pred · fr_prev`, `w_pred_update_kernel` 用预测误差更新 W_pred.

**集成方式:** 无需修改, W_pred 自动学习预测下一个环境步的 subcolumn_fr. 当环境因果结构稳定时 (如哭→妈妈来), W_pred 会学到这个时序模式.

#### 3.5.2 价值学习 (扩展 DA reward)

**现有代码:** `launch_da_value_function` 计算 `δ = reward + γ·V(s') - V(s)`, 但 reward 来自外部参数.

**修改:** 在 `launch_modulatory` 中, 当 `embodied_mode=true` 时, 用环境计算的 reward 替代默认 reward:

```cpp
// modulatory_kernels.cu - launch_modulatory 内
float effective_reward = external_reward;
if (embodied_reward_active) {
    effective_reward = embodied_reward;  // 来自 Environment::compute_reward
}
float delta = effective_reward + GAMMA * h_v_sp - h_v_s;
```

#### 3.5.3 行为模仿 (新增 motor_teacher_kernel)

**教师信号规则 (D1阶段, 基于基因硬编码):**
- `hunger > 0.6` → target = ACT_CRY
- `is_fed && hunger > 0.3` → target = ACT_SUCK
- `fatigue > 0.7` → target = ACT_GAZE (闭眼休息)
- 其他 → 无教师信号 (让好奇心探索驱动)

```cpp
// motor_teacher.cu
__global__ void motor_teacher_kernel(
    float* __restrict__ motor_weights,  // L5→运动皮层突触权重
    const bool* __restrict__ spike_flags_l5,  // L5层spike
    int target_action,                      // 教师目标动作 (-1=无)
    float teacher_lr,                       // 学习率
    int n_motor_neurons,
    int motor_group_size);
```

**学习规则:** 如果 `target_action >= 0`, 增强目标动作对应组(10组)的L5→运动皮层突触权重, 抑制其他组:

```
if (target >= 0):
    for g in 0..49:
        group_action = g / 10
        sign = (group_action == target) ? +1 : -1
        for neuron in group:
            if spike_flags_l5[pre]:
                motor_weights[neuron] += sign * teacher_lr
```

#### 3.5.4 好奇心探索 (ACh 注入)

**机制:** 当 W_pred 预测误差大时 (世界模型"惊讶"), 注入 ACh 驱动探索.

> **注意:** 不能复用 `h_event_signal[1]` (ACh通道), 因为该缓存是单次触发模型 (launch_modulatory 读取后清零). 好奇心是持续信号, 需独立缓存.

```cpp
// modulatory_kernels.cu 新增
static float h_curiosity_ach = 0.0f;  // 好奇心驱动的ACh持续信号

void set_curiosity_ach(float pred_error) {
    // pred_error ∈ [0,1], 来自 W_pred 预测误差 (余弦距离)
    h_curiosity_ach = pred_error * 0.3f;  // ACh增量, clamp到[0, 0.3]
}

// 在 launch_modulatory 内, ACh信号计算:
float ach_signal = baseline_ach                 // 基线
                 + h_event_signal[1]            // 事件驱动 (单次脉冲)
                 + h_curiosity_ach;             // 好奇心驱动 (持续)
// h_curiosity_ach 不清零, 每次环境步由 set_curiosity_ach 更新
```

---

## 4. 与现有代码集成

### 4.1 命令行接口

新增参数:

```bash
snn_train.exe --embodied --steps 5000 --checkpoint-interval 0
```

| 参数 | 说明 |
|------|------|
| `--embodied` | 启用具身发育模式 (替代 --event-stream) |
| `--embodied-config PATH` | 身体参数配置JSON (可选, 默认新生儿参数) |
| `--embodied-scene ID` | 指定场景 (可选, 默认按月龄自动选择) |

### 4.2 main.cpp 训练循环修改

```cpp
// main.cpp 新增
#include "embodied_body.h"
#include "embodied_env.h"
#include "embodied_motor.h"
#include "multi_sensory_inject.cuh"

// 初始化
EmbodiedEnvironment env;
env.load_config(config.embodied_config_path);  // 可选
MotorReadout motor;
float sensory[50] = {0};
float embodied_reward = 0.0f;
bool embodied_active = config.embodied_mode;

// 训练循环 (时序: 环境步在scheduler.step之前执行, 动作读出的是上一环境步的运动皮层输出)
for (int step = 0; step < total_steps; ++step) {
    // === 现有: 事件调度 (兼容, 但具身模式下通常不用) ===
    if (event_stream_active) event_scheduler.dispatch_pending(step);
    
    // === 新增: 每100步环境闭环 (在scheduler.step之前) ===
    bool is_env_step = (step % 100 == 0 && step > 0);
    if (is_env_step && embodied_active) {
        // 1. 感知注入 (当前身体状态 → 50柱信号 → L4层)
        //    注意: 这会在scheduler.step()内被神经元读取
        env.compute_sensory_signals(sensory);
        stage2e::launch_multi_sensory_inject(sensory, allocator.d_input_current, 
                                              scheduler.d_gate_states_for_inject());
        
        // 2. 动作读出 (读的是"上一环境步"产生的运动皮层spike)
        //    因为当前环境步的spike还没被scheduler.step()产生
        motor = stage2e::read_motor_output(allocator.d_motor_spike_flags);
        
        // 3. 环境响应 + 身体演化 (动作 → 环境反馈 → 身体状态更新)
        BodyState prev_body = env.get_body_state();
        env.step_env(motor);
        
        // 4. DA奖励计算 (身体状态改善程度 → reward)
        embodied_reward = env.compute_reward(prev_body);
        stage2e::set_embodied_reward(embodied_reward);  // 传给launch_modulatory
        
        // 5. 行为模仿教师信号 (基因硬编码: hunger>0.6→教CRY)
        int target_action = env.get_teacher_signal();
        if (target_action >= 0) {
            stage2e::launch_motor_teacher(target_action, teacher_lr);
        }
    }
    
    // === 现有: 主步进 (神经元更新+突触更新+调质更新) ===
    scheduler.step(step);
    
    // === CSV日志 (扩展) ===
    if (step % 100 == 0 && embodied_active) {
        log_embodied_state(step, env, motor, embodied_reward);
    }
}
```

### 4.3 文件清单

#### 新增文件

| 文件 | 职责 |
|------|------|
| `src/snn/embodied_body.h` | BodyState结构 + 演化方程 + 内感态编码 |
| `src/snn/embodied_env.h` | EmbodiedEnvironment类 (妈妈响应+动作效果+奖励) |
| `src/snn/embodied_env.cpp` | 环境模型实现 |
| `src/snn/embodied_motor.h` | MotorReadout结构 + 动作枚举 |
| `src/snn/embodied_motor.cpp` | 运动皮层读出 + 动作采样实现 |
| `src/snn/multi_sensory_inject.cu` | 多模态感知注入kernel |
| `src/snn/multi_sensory_inject.cuh` | 注入kernel头文件 |
| `src/snn/motor_teacher.cu` | 行为模仿教师kernel |
| `src/snn/motor_teacher.cuh` | 教师kernel头文件 |
| `src/snn/tools/generate_embodied_config.py` | 生成身体参数配置JSON |
| `src/snn/tools/validate_embodied.py` | 具身训练验证脚本 |
| `src/snn/test_embodied.cpp` | 具身环境单元测试 |

#### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/snn/main.cpp` | 新增 `--embodied` 模式 + 环境闭环调用 + 日志扩展 |
| `src/snn/run_config.h` | 新增 `embodied_mode`, `embodied_config_path` 字段 |
| `src/snn/run_config.cpp` | 新增命令行解析 |
| `src/snn/modulatory_kernels.cu` | 新增 `set_embodied_reward()` + `set_curiosity_ach()` + `launch_modulatory` 读取 reward/curiosity |
| `src/snn/modulatory_kernels.cuh` | 新增 `set_embodied_reward()` + `set_curiosity_ach()` 声明 |
| `src/snn/CMakeLists.txt` (或build脚本) | 新增源文件 |

---

## 5. D1 阶段范围 (新生儿期最小可行验证)

### 5.1 D1 目标

**验证标准: 跑 5K 步不崩.** 不期望学到复杂因果.

具体验证项:
1. 程序不崩溃 (CUDA 无错误, 无段错误)
2. 6维调质浓度维持在 [0, 2] 范围
3. 运动皮层有输出 (action_prob 非均匀)
4. 环境闭环运行 (身体状态在演化, 妈妈响应在触发)
5. DA reward 有变化 (不是恒定值)
6. CSV 日志完整输出

### 5.2 D1 场景定义

5个新生儿核心场景, 每场景3000环境步 (300K SNN步):

| 场景 | 初始状态 | 因果链 |
|------|----------|--------|
| hunger_feeding | hunger=0.8 | 饥饿→哭→妈妈来→喂养→满足 |
| warmth_safety | temp=0.2 (冷) | 寒冷→哭→妈妈来→包裹→温暖 |
| startle_recover | noise=1.0 (突发噪音) | 惊吓→哭→妈妈来→安抚→平静 |
| sleep_wake | fatigue=0.9 | 困倦→闹→妈妈来→哄睡→入睡 |
| discomfort_change | diaper=0.9 | 不适→哭→妈妈来→换尿布→舒适 |

**D1 训练流程 (5K步):**
- 5K步 = 50环境步 = 5秒模拟时间
- 只能跑完一个场景的前50环境步 (约1/60个完整场景)
- 主要验证系统稳定性, 不是学习效果

### 5.3 D1 不包含

- 婴儿期(2-12月)及以后的场景
- LLM叙事增强 (留待 D2)
- 睡眠重放与发育记忆固化
- 多场景切换 (D1只跑单场景前50步)
- BPTT 代理梯度 (具身模式下禁用)
- 长期训练的因果学习验证

---

## 6. 后续路线图

| 阶段 | 内容 | 依赖 |
|------|------|------|
| **D1** | 新生儿期最小闭环验证 (5K步不崩) | 本文档 |
| **D2** | 5场景×10次重复=15M步, 验证统计因果学习 | D1通过 |
| **D3** | LLM叙事增强: 接入MiniCPM5生成自然语言叙事 | D2通过 |
| **D4** | 婴儿期(2-12月)场景扩展: 物体抓取/爬行/咿呀学语 | D2通过 |
| **D5** | 幼儿期(1-3岁)场景扩展: 走路/说话/社交 | D4通过 |
| **D6** | 学龄期+青少年期: 复杂社交/成就/失败 | D5通过 |

---

## 7. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 环境模型位置 | C++实时模拟 | 真正感知-行动闭环, SNN动作影响环境 |
| 感知编码方式 | 50柱分配多模态 | 复用现有L4注入架构, 改动最小 |
| 动作解码方式 | 运动皮层50组→5动作 | 复用现有d_motor_spike_flags, 组映射自然 |
| 学习目标 | 四合一 (世界模型+价值+模仿+好奇心) | 覆盖认知发育的核心学习模式 |
| 时间尺度 | 环境步=100 SNN步 | 与launch_modulatory同步, 调质动力学自然耦合 |
| D1验证标准 | 5K步不崩 | 用户明确要求, 不期望学习效果 |
| 妈妈响应模型 | 概率响应 (σ函数) | 接近真实, 能学到"哭不一定立即来"的概率因果 |
| 旧方案处理 | 废弃 `2026-07-30-developmental-cognitive-training-design.md` | 反射弧方案缺乏事实支撑 |
