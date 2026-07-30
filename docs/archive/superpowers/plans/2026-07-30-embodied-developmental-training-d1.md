# Phase 3a-D1: 具身发育式认知训练 — 新生儿期最小可行验证 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现具身发育训练系统(虚拟婴儿身体+概率响应环境+感知-行动闭环),端到端验证 5K 步不崩溃。

**Architecture:** BodyState(5维内感态) → 50柱多模态感知注入L4 → SNN主网络 → 5K运动皮层读出5动作 → 概率响应环境 → DA奖励闭环。C++实时环境模拟,每100 SNN步执行一次环境步。

**Tech Stack:** C++17 (身体/环境/动作读出), CUDA 17 (感知注入kernel/教师kernel), CMake/ninja (构建)

**Spec:** `docs/superpowers/specs/2026-07-30-embodied-developmental-training-design.md`

---

## 文件结构

| 文件 | 职责 | 创建/修改 |
|------|------|----------|
| `src/snn/embodied_body.h` | BodyState结构 + 演化方程 + 内感态编码 (header-only) | 创建 |
| `src/snn/embodied_motor.h` | MotorReadout结构 + ActionId枚举 (header-only) | 创建 |
| `src/snn/embodied_motor.cpp` | 运动皮层读出 + softmax + 采样实现 | 创建 |
| `src/snn/embodied_env.h` | EmbodiedEnvironment类声明 | 创建 |
| `src/snn/embodied_env.cpp` | 环境模型实现 (妈妈响应+动作效果+奖励) | 创建 |
| `src/snn/multi_sensory_inject.cuh` | 多模态感知注入kernel声明 | 创建 |
| `src/snn/multi_sensory_inject.cu` | 多模态感知注入kernel实现 | 创建 |
| `src/snn/motor_teacher.cuh` | 行为模仿教师kernel声明 | 创建 |
| `src/snn/motor_teacher.cu` | 行为模仿教师kernel实现 | 创建 |
| `src/snn/test_embodied.cpp` | 具身环境单元测试 (纯host, 无CUDA) | 创建 |
| `src/snn/run_config.h` | 新增 embodied_mode 字段 | 修改 |
| `src/snn/run_config.cpp` | 新增 --embodied 命令行解析 | 修改 |
| `src/snn/modulatory_kernels.cu` | 新增 set_embodied_reward + set_curiosity_ach | 修改 |
| `src/snn/modulatory_kernels.cuh` | 新增声明 | 修改 |
| `src/snn/main.cpp` | 新增 --embodied 模式 + 环境闭环调用 | 修改 |
| `src/snn/CMakeLists.txt` | 新增源文件 + test_embodied target | 修改 |

**关键设计决策:**
- BodyState 和 MotorReadout 的结构体定义放在 header-only 文件中, 供多个 .cpp/.cu 包含
- 纯 host 逻辑 (BodyState演化/环境模型/动作读出) 与 CUDA kernel 分离, 便于单元测试
- test_embodied.cpp 不链接 CUDA, 用 stub 替代 CUDA 函数

---

### Task 1: BodyState 虚拟婴儿身体

**Files:**
- Create: `src/snn/embodied_body.h`

- [ ] **Step 1: 编写 BodyState header**

```cpp
// src/snn/embodied_body.h
#ifndef SNN_EMBODIED_BODY_H
#define SNN_EMBODIED_BODY_H

#include <cmath>
#include <algorithm>

namespace stage2e {

// =============================================================================
// Phase 3a-D1: 虚拟婴儿身体状态
// 5维内感态 (hunger/temperature/comfort/fatigue/arousal) + 环境衍生状态
// 每环境步 (100 SNN步) 演化一次
// =============================================================================
struct BodyState {
    // 核心内感态 (5维, ∈[0,1])
    float hunger;       // 饥饿度: 每环境步+0.001, 喂养时-0.3
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
    
    // 初始化为新生儿默认状态
    void init_default() {
        hunger = 0.3f;
        temperature = 0.5f;
        comfort = 0.7f;
        fatigue = 0.0f;
        arousal = 0.3f;
        ambient_temp = 0.5f;
        diaper_dirty = 0.0f;
        is_held = false;
        is_fed = false;
        mom_fatigue = 0.0f;
    }
    
    // 初始化为指定场景
    void init_scene(const char* scene_id) {
        init_default();
        std::string s(scene_id);
        if (s == "hunger_feeding") {
            hunger = 0.8f;
        } else if (s == "warmth_safety") {
            temperature = 0.2f;
            ambient_temp = 0.2f;
        } else if (s == "startle_recover") {
            arousal = 0.9f;
        } else if (s == "sleep_wake") {
            fatigue = 0.9f;
        } else if (s == "discomfort_change") {
            diaper_dirty = 0.9f;
            comfort = 0.2f;
        }
    }
    
    // 演化方程 (每环境步调用一次)
    void step(float dt) {
        // dt = 1.0 (一个环境步)
        // 饥饿: 基础速率 + 唤醒加成
        hunger += 0.001f * (1.0f + arousal * 0.3f);
        if (is_fed) hunger -= 0.3f;
        hunger = std::max(0.0f, std::min(1.0f, hunger));
        
        // 温度: 向ambient指数收敛
        temperature += 0.01f * (ambient_temp - temperature);
        
        // 尿布: 饥饿时排泄加快
        diaper_dirty += 0.0003f * (hunger > 0.5f ? 1.5f : 1.0f);
        diaper_dirty = std::max(0.0f, std::min(1.0f, diaper_dirty));
        
        // 舒适度: 受尿布/温度/抱影响
        comfort = std::max(0.0f, std::min(1.0f,
            0.7f
            - diaper_dirty * 0.5f
            - std::fabs(temperature - 0.5f) * 0.6f
            + (is_held ? 0.2f : 0.0f)
        ));
        
        // 疲劳: 持续累积
        fatigue += 0.0005f;
        fatigue = std::max(0.0f, std::min(1.0f, fatigue));
        
        // 唤醒: 综合驱动
        arousal = std::max(0.0f, std::min(1.0f,
            0.3f
            + hunger * 0.4f
            + (1.0f - comfort) * 0.3f
            - fatigue * 0.2f
        ));
        
        // 妈妈疲劳: 每步微增
        mom_fatigue += 0.0002f;
        mom_fatigue = std::max(0.0f, std::min(1.0f, mom_fatigue));
    }
    
    // 内感态信号编码 (15柱: 柱35-49, 每维3柱 low/mid/high)
    void encode_interoception(float out[15]) const {
        // hunger: 柱35-37
        out[0] = (hunger < 0.3f) ? 1.0f : 0.0f;
        out[1] = (hunger >= 0.3f && hunger < 0.7f) ? 1.0f : 0.0f;
        out[2] = (hunger >= 0.7f) ? 1.0f : 0.0f;
        // temperature: 柱38-40
        out[3] = (temperature < 0.4f) ? 1.0f : 0.0f;
        out[4] = (temperature >= 0.4f && temperature < 0.6f) ? 1.0f : 0.0f;
        out[5] = (temperature >= 0.6f) ? 1.0f : 0.0f;
        // comfort: 柱41-43
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
    
    // 衍生计算
    float discomfort() const {
        return hunger * 0.4f + (1.0f - comfort) * 0.4f + fatigue * 0.2f;
    }
    
    float distress() const {
        return discomfort() * (0.5f + arousal * 0.5f);
    }
};

} // namespace stage2e

#endif // SNN_EMBODIED_BODY_H
```

- [ ] **Step 2: 验证 header 可编译**

Run: `cd f:\thetrueai && cl /std:c++17 /EHsc /I src\snn /c src\snn\embodied_body.h /Fo:NUL 2>&1 | findstr /V "warning"`
Expected: 无 error 输出 (可能有 C4514 未引用内联函数警告, 忽略)

- [ ] **Step 3: 提交**

```bash
git add src/snn/embodied_body.h
git commit -m "feat(snn): add BodyState virtual infant body (Phase 3a-D1 Task 1)"
```

---

### Task 2: MotorReadout 动作读出

**Files:**
- Create: `src/snn/embodied_motor.h`
- Create: `src/snn/embodied_motor.cpp`

- [ ] **Step 1: 编写 MotorReadout header**

```cpp
// src/snn/embodied_motor.h
#ifndef SNN_EMBODIED_MOTOR_H
#define SNN_EMBODIED_MOTOR_H

#include <cuda_runtime.h>

namespace stage2e {

// 5个动作 (50组运动皮层, 每动作10组)
enum ActionId {
    ACT_CRY  = 0,  // 哭: 触发妈妈响应概率
    ACT_HAND = 1,  // 手部动作: 消耗能量
    ACT_FOOT = 2,  // 脚部动作: 消耗能量
    ACT_SUCK = 3,  // 吸吮: 喂养时 hunger-=0.3*suck
    ACT_GAZE = 4,  // 注视: 无直接效果
    ACT_COUNT = 5
};

struct MotorReadout {
    float action_raw[5];     // 原始发放率 (softmax前)
    float action_prob[5];    // softmax概率
    int   action_sampled;    // 采样的离散动作
    float cry_intensity;     // = action_prob[ACT_CRY]
    float suck_strength;     // = action_prob[ACT_SUCK]
    float limb_movement;     // = (action_prob[ACT_HAND] + action_prob[ACT_FOOT]) / 2
};

// 从 GPU d_motor_spike_flags 读出动作 (5K神经元, 50组×100)
// 调用后 motor 中填充读出结果
MotorReadout read_motor_output(const bool* d_motor_spike_flags);

// 纯 host 版本: 从 host spike_flags 读出 (供单元测试使用)
MotorReadout read_motor_output_host(const bool* h_spike_flags, int n_neurons,
                                     int n_groups, int group_size);

} // namespace stage2e

#endif // SNN_EMBODIED_MOTOR_H
```

- [ ] **Step 2: 编写 MotorReadout 实现**

```cpp
// src/snn/embodied_motor.cpp
#include "embodied_motor.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace stage2e {

// 通用读出逻辑 (host端)
static MotorReadout compute_readout(const bool* spike_flags, int n_neurons,
                                     int n_groups, int group_size) {
    MotorReadout m = {};
    
    // 1. 按组聚合发放率
    float group_rate[50];
    for (int g = 0; g < n_groups && g < 50; ++g) {
        int count = 0;
        int start = g * group_size;
        int end = std::min(start + group_size, n_neurons);
        for (int i = start; i < end; ++i) {
            if (spike_flags[i]) ++count;
        }
        group_rate[g] = (float)count / (float)group_size;
    }
    
    // 2. 5个动作各对应10组
    for (int a = 0; a < 5; ++a) {
        float sum = 0;
        for (int g = a * 10; g < (a + 1) * 10; ++g) {
            sum += group_rate[g];
        }
        m.action_raw[a] = sum / 10.0f;
    }
    
    // 3. Softmax (τ=0.5)
    float max_val = *std::max_element(m.action_raw, m.action_raw + 5);
    float exp_sum = 0;
    for (int a = 0; a < 5; ++a) {
        m.action_prob[a] = expf((m.action_raw[a] - max_val) / 0.5f);
        exp_sum += m.action_prob[a];
    }
    for (int a = 0; a < 5; ++a) m.action_prob[a] /= exp_sum;
    
    // 4. 采样动作 (简单随机, 用 rand())
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    m.action_sampled = 4;  // 默认最后一个
    for (int a = 0; a < 5; ++a) {
        cum += m.action_prob[a];
        if (r < cum) { m.action_sampled = a; break; }
    }
    
    // 5. 连续值
    m.cry_intensity = m.action_prob[ACT_CRY];
    m.suck_strength = m.action_prob[ACT_SUCK];
    m.limb_movement = (m.action_prob[ACT_HAND] + m.action_prob[ACT_FOOT]) * 0.5f;
    
    return m;
}

MotorReadout read_motor_output_host(const bool* h_spike_flags, int n_neurons,
                                     int n_groups, int group_size) {
    return compute_readout(h_spike_flags, n_neurons, n_groups, group_size);
}

MotorReadout read_motor_output(const bool* d_motor_spike_flags) {
    // 拷贝到 host
    bool h_spike_flags[5000];
    cudaMemcpy(h_spike_flags, d_motor_spike_flags, 5000 * sizeof(bool),
               cudaMemcpyDeviceToHost);
    // 50组 × 100神经元 = 5000
    return compute_readout(h_spike_flags, 5000, 50, 100);
}

} // namespace stage2e
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/embodied_motor.h src/snn/embodied_motor.cpp
git commit -m "feat(snn): add MotorReadout action decoder (Phase 3a-D1 Task 2)"
```

---

### Task 3: EmbodiedEnvironment 概率响应环境

**Files:**
- Create: `src/snn/embodied_env.h`
- Create: `src/snn/embodied_env.cpp`

- [ ] **Step 1: 编写环境模型 header**

```cpp
// src/snn/embodied_env.h
#ifndef SNN_EMBODIED_ENV_H
#define SNN_EMBODIED_ENV_H

#include "embodied_body.h"
#include "embodied_motor.h"
#include <string>

namespace stage2e {

// =============================================================================
// Phase 3a-D1: 概率响应环境
// 妈妈响应模型: P(来)=σ(2.0*hunger + 1.5*arousal - 1.0*mom_fatigue - 0.5)
// 响应延迟: 5-20环境步 (0.5-2秒)
// 动作效果: cry→触发响应, suck→喂养, hand/foot→消耗能量
// DA奖励: Δcomfort + Δhunger*0.5 + Δarousal*0.2 - fatigue*0.1
// =============================================================================
class EmbodiedEnvironment {
public:
    BodyState body;
    int   mom_response_countdown;  // 妈妈响应剩余环境步 (0=不在路上)
    bool  mom_present;             // 妈妈是否在场
    float mom_response_prob;       // 当前步响应概率 (日志用)
    float last_reward;             // 上次计算的reward (日志用)
    float noise_intensity;         // 环境噪音强度 [0,1]
    float light_level;             // 光照强度 [0,1]
    bool  mom_speaking;            // 妈妈在说话
    bool  mom_visible;             // 妈妈脸可见
    
    // 初始化场景
    void init(const std::string& scene_id = "hunger_feeding");
    
    // 环境步进 (每100 SNN步调用)
    void step_env(const MotorReadout& motor);
    
    // 计算DA奖励 (需要传入step_env之前的body状态)
    float compute_reward(const BodyState& prev) const;
    
    // 获取教师信号 (-1=无教师)
    int get_teacher_signal() const;
    
    // 计算感知信号 (50柱)
    void compute_sensory_signals(float out[50]) const;
    
    // 获取当前body状态 (供compute_reward之前保存)
    BodyState get_body_state() const { return body; }
    
private:
    float compute_mom_response_prob() const;
    void apply_mom_effects(const MotorReadout& motor);
};

} // namespace stage2e

#endif // SNN_EMBODIED_ENV_H
```

- [ ] **Step 2: 编写环境模型实现**

```cpp
// src/snn/embodied_env.cpp
#include "embodied_env.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace stage2e {

void EmbodiedEnvironment::init(const std::string& scene_id) {
    body.init_scene(scene_id.c_str());
    mom_response_countdown = 0;
    mom_present = false;
    mom_response_prob = 0.0f;
    last_reward = 0.0f;
    noise_intensity = 0.0f;
    light_level = 0.5f;
    mom_speaking = false;
    mom_visible = false;
    
    // 场景特定初始化
    if (scene_id == "startle_recover") {
        noise_intensity = 1.0f;
    }
}

float EmbodiedEnvironment::compute_mom_response_prob() const {
    // D1: hunger为主驱动 (基因保底)
    float x = 2.0f * body.hunger
            + 1.5f * body.arousal
            - 1.0f * body.mom_fatigue
            - 0.5f;
    return 1.0f / (1.0f + expf(-x));
}

void EmbodiedEnvironment::apply_mom_effects(const MotorReadout& motor) {
    if (!mom_present) return;
    
    // 喂养: 如果饥饿>0.3则喂奶
    if (body.hunger > 0.3f) {
        body.is_fed = true;
        body.hunger -= 0.3f * motor.suck_strength;
        body.hunger = std::max(0.0f, body.hunger);
    } else {
        body.is_fed = false;
    }
    
    // 换尿布: 如果尿布脏>0.7
    if (body.diaper_dirty > 0.7f) {
        body.diaper_dirty = 0.0f;
    }
    
    // 妈妈在场时设置感知
    mom_speaking = true;
    mom_visible = true;
    
    // 妈妈离开: 饥饿<0.2且尿布干净
    if (body.hunger < 0.2f && body.diaper_dirty < 0.3f) {
        mom_present = false;
        body.is_held = false;
        body.is_fed = false;
        mom_speaking = false;
        mom_visible = false;
    }
}

void EmbodiedEnvironment::step_env(const MotorReadout& motor) {
    // 1. 妈妈响应判定 (概率)
    if (!mom_present && mom_response_countdown <= 0) {
        float p = compute_mom_response_prob();
        mom_response_prob = p;
        float p_cry = p * (0.5f + motor.cry_intensity);
        if ((float)rand() / (float)RAND_MAX < p_cry) {
            mom_response_countdown = 5 + rand() % 16;
        }
    }
    
    // 2. 响应倒计时
    if (mom_response_countdown > 0) {
        mom_response_countdown--;
        if (mom_response_countdown == 0) {
            mom_present = true;
            body.is_held = true;
            body.mom_fatigue += 0.05f;
        }
    }
    
    // 3. 妈妈在场时的效果
    apply_mom_effects(motor);
    
    // 4. 身体状态演化
    body.step(1.0f);
    
    // 5. 动作能量消耗
    body.fatigue += motor.limb_movement * 0.002f;
    body.fatigue = std::min(1.0f, body.fatigue);
    
    // 6. 噪音衰减 (startle场景)
    if (noise_intensity > 0) {
        noise_intensity *= 0.95f;
        if (noise_intensity < 0.01f) noise_intensity = 0.0f;
    }
}

float EmbodiedEnvironment::compute_reward(const BodyState& prev) const {
    float reward = 0.0f;
    reward += (body.comfort - prev.comfort) * 1.0f;
    reward += (prev.hunger - body.hunger) * 0.5f;
    reward += (body.arousal - prev.arousal) * 0.2f;
    reward -= body.fatigue * 0.1f;
    return reward;
}

int EmbodiedEnvironment::get_teacher_signal() const {
    // 基因硬编码教师信号
    if (body.hunger > 0.6f) return ACT_CRY;
    if (body.is_fed && body.hunger > 0.3f) return ACT_SUCK;
    if (body.fatigue > 0.7f) return ACT_GAZE;
    return -1;  // 无教师信号
}

void EmbodiedEnvironment::compute_sensory_signals(float out[50]) const {
    std::memset(out, 0, 50 * sizeof(float));
    
    // 触觉-被抱 (柱0-4)
    if (body.is_held) {
        for (int i = 0; i < 5; ++i) out[i] = 1.0f;
    }
    // 触觉-抚摸 (柱5-9): 被抱时有抚摸
    if (body.is_held) {
        for (int i = 5; i < 10; ++i) out[i] = 0.7f;
    }
    
    // 听觉-妈妈声 (柱10-14)
    if (mom_present && mom_speaking) {
        for (int i = 10; i < 15; ++i) out[i] = 0.8f;
    }
    // 听觉-噪音 (柱15-19)
    if (noise_intensity > 0) {
        for (int i = 15; i < 20; ++i) out[i] = noise_intensity;
    }
    
    // 视觉-光 (柱20-24)
    for (int i = 20; i < 25; ++i) out[i] = light_level;
    // 视觉-人脸 (柱25-29)
    if (mom_visible) {
        for (int i = 25; i < 30; ++i) out[i] = 0.9f;
    }
    
    // 嗅觉-奶味 (柱30-32)
    if (body.is_fed) {
        for (int i = 30; i < 33; ++i) out[i] = 0.8f;
    }
    // 嗅觉-妈妈味 (柱33-34)
    if (mom_present) {
        out[33] = 0.7f;
        out[34] = 0.7f;
    }
    
    // 内感态 (柱35-49)
    float interoception[15];
    body.encode_interoception(interoception);
    for (int i = 0; i < 15; ++i) out[35 + i] = interoception[i];
}

} // namespace stage2e
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/embodied_env.h src/snn/embodied_env.cpp
git commit -m "feat(snn): add EmbodiedEnvironment probabilistic response env (Phase 3a-D1 Task 3)"
```

---

### Task 4: 具身环境单元测试

**Files:**
- Create: `src/snn/test_embodied.cpp`

- [ ] **Step 1: 编写单元测试**

```cpp
// src/snn/test_embodied.cpp
// 具身环境单元测试 (纯host, 无CUDA依赖)
// 测试 BodyState演化 + MotorReadout读出 + Environment响应 + 奖励计算
#include "embodied_body.h"
#include "embodied_motor.h"
#include "embodied_env.h"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace stage2e;

static int g_test_pass = 0;
static int g_test_fail = 0;

#define TEST(cond, msg) do { \
    if (cond) { g_test_pass++; } \
    else { g_test_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) TEST(fabsf((a)-(b)) < (eps), msg)

// 测试 1: BodyState 默认初始化
void test_body_init_default() {
    BodyState b;
    b.init_default();
    ASSERT_NEAR(b.hunger, 0.3f, 1e-5f, "default hunger=0.3");
    ASSERT_NEAR(b.temperature, 0.5f, 1e-5f, "default temp=0.5");
    ASSERT_NEAR(b.comfort, 0.7f, 1e-5f, "default comfort=0.7");
    ASSERT_NEAR(b.fatigue, 0.0f, 1e-5f, "default fatigue=0");
    ASSERT_NEAR(b.arousal, 0.3f, 1e-5f, "default arousal=0.3");
}

// 测试 2: BodyState 场景初始化
void test_body_init_scene() {
    BodyState b;
    b.init_scene("hunger_feeding");
    ASSERT_NEAR(b.hunger, 0.8f, 1e-5f, "hunger_feeding hunger=0.8");
    
    b.init_scene("warmth_safety");
    ASSERT_NEAR(b.temperature, 0.2f, 1e-5f, "warmth_safety temp=0.2");
    
    b.init_scene("discomfort_change");
    ASSERT_NEAR(b.diaper_dirty, 0.9f, 1e-5f, "discomfort_change diaper=0.9");
}

// 测试 3: BodyState 演化 — 饥饿增加
void test_body_step_hunger() {
    BodyState b;
    b.init_default();
    b.is_fed = false;
    float h0 = b.hunger;
    b.step(1.0f);
    TEST(b.hunger > h0, "hunger increases after step");
    TEST(b.hunger <= 1.0f, "hunger clamped to 1.0");
}

// 测试 4: BodyState 演化 — 喂养降低饥饿
void test_body_step_feed() {
    BodyState b;
    b.init_default();
    b.hunger = 0.8f;
    b.is_fed = true;
    float h0 = b.hunger;
    b.step(1.0f);
    TEST(b.hunger < h0, "hunger decreases when fed");
    // hunger += 0.001*(1+0.3*0.3) - 0.3 ≈ -0.299
    ASSERT_NEAR(b.hunger, 0.8f + 0.001f*(1.0f+0.3f*0.3f) - 0.3f, 0.01f, "feed amount");
}

// 测试 5: BodyState 内感态编码
void test_body_encode_interoception() {
    BodyState b;
    b.init_default();
    float out[15];
    b.encode_interoception(out);
    // hunger=0.3 → mid 激活
    TEST(out[1] > 0.5f, "hunger=0.3 activates mid");
    TEST(out[0] < 0.5f, "hunger=0.3 does not activate low");
    TEST(out[2] < 0.5f, "hunger=0.3 does not activate high");
    
    b.hunger = 0.8f;
    b.encode_interoception(out);
    TEST(out[2] > 0.5f, "hunger=0.8 activates high");
}

// 测试 6: MotorReadout host 读出
void test_motor_readout_host() {
    // 模拟5000个神经元, 全部不发放
    bool flags[5000];
    std::memset(flags, 0, sizeof(flags));
    MotorReadout m = read_motor_output_host(flags, 5000, 50, 100);
    // 全不发放 → softmax均匀
    ASSERT_NEAR(m.action_prob[0], 0.2f, 0.01f, "no spike → uniform prob");
    
    // 让组0-99 (ACT_CRY) 全部发放
    std::memset(flags, 0, sizeof(flags));
    for (int i = 0; i < 1000; ++i) flags[i] = true;
    m = read_motor_output_host(flags, 5000, 50, 100);
    TEST(m.action_prob[ACT_CRY] > 0.5f, "cry groups spike → cry prob high");
    TEST(m.cry_intensity > 0.5f, "cry_intensity > 0.5");
}

// 测试 7: Environment 妈妈响应概率
void test_env_mom_response_prob() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    // hunger=0.8, arousal≈0.3+0.8*0.4=0.62, mom_fatigue=0
    float p = env.compute_mom_response_prob();
    TEST(p > 0.8f, "high hunger → high mom response prob");
    
    env.body.hunger = 0.1f;
    env.body.arousal = 0.2f;
    p = env.compute_mom_response_prob();
    TEST(p < 0.3f, "low hunger → low mom response prob");
}

// 测试 8: Environment 教师信号
void test_env_teacher_signal() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    // hunger=0.8 > 0.6 → ACT_CRY
    TEST(env.get_teacher_signal() == ACT_CRY, "hunger>0.6 → CRY");
    
    env.body.hunger = 0.4f;
    env.body.is_fed = true;
    TEST(env.get_teacher_signal() == ACT_SUCK, "fed+hungry → SUCK");
    
    env.body.hunger = 0.1f;
    env.body.is_fed = false;
    env.body.fatigue = 0.8f;
    TEST(env.get_teacher_signal() == ACT_GAZE, "fatigued → GAZE");
    
    env.body.fatigue = 0.1f;
    TEST(env.get_teacher_signal() == -1, "no teacher when comfortable");
}

// 测试 9: Environment 感知信号编码
void test_env_sensory_signals() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    float sig[50];
    env.compute_sensory_signals(sig);
    
    // 妈妈不在 → 触觉柱0-4为0
    TEST(sig[0] < 0.5f, "not held → touch=0");
    // hunger=0.8 → 内感态柱37 (hunger high) 激活
    TEST(sig[37] > 0.5f, "hunger=0.8 → interoception high");
    
    // 模拟妈妈在场
    env.mom_present = true;
    env.body.is_held = true;
    env.mom_speaking = true;
    env.mom_visible = true;
    env.compute_sensory_signals(sig);
    TEST(sig[0] > 0.5f, "held → touch=1");
    TEST(sig[10] > 0.5f, "mom speaking → auditory");
    TEST(sig[25] > 0.5f, "mom visible → face");
}

// 测试 10: Environment 奖励计算
void test_env_reward() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    BodyState prev = env.body;
    
    // 模拟饥饿下降 (喂养)
    env.body.hunger = 0.5f;  // 从0.8降到0.5
    float r = env.compute_reward(prev);
    TEST(r > 0, "hunger decrease → positive reward");
    
    // 模拟饥饿上升
    env.body.hunger = 0.9f;
    r = env.compute_reward(prev);
    TEST(r < 0, "hunger increase → negative reward");
}

// 测试 11: Environment step_env 闭环
void test_env_step闭环() {
    EmbodiedEnvironment env;
    env.init("hunger_feeding");
    
    MotorReadout motor = {};
    motor.cry_intensity = 0.9f;  // 大声哭
    motor.suck_strength = 0.5f;
    motor.limb_movement = 0.1f;
    
    // 跑50个环境步, 验证不崩 + 妈妈最终会来
    bool mom_came = false;
    for (int i = 0; i < 50; ++i) {
        BodyState prev = env.body;
        env.step_env(motor);
        env.last_reward = env.compute_reward(prev);
        if (env.mom_present) mom_came = true;
        TEST(env.body.hunger >= 0.0f && env.body.hunger <= 1.0f, "hunger in range");
        TEST(env.body.comfort >= 0.0f && env.body.comfort <= 1.0f, "comfort in range");
    }
    // hunger=0.8 + cry=0.9 → 妈妈应该会在50步内来
    TEST(mom_came, "mom comes within 50 steps when hungry+crying");
}

int main() {
    srand(42);  // 固定种子, 可复现
    
    test_body_init_default();
    test_body_init_scene();
    test_body_step_hunger();
    test_body_step_feed();
    test_body_encode_interoception();
    test_motor_readout_host();
    test_env_mom_response_prob();
    test_env_teacher_signal();
    test_env_sensory_signals();
    test_env_reward();
    test_env_step闭环();
    
    printf("\n=== Embodied Environment Tests ===\n");
    printf("PASS: %d\n", g_test_pass);
    printf("FAIL: %d\n", g_test_fail);
    printf("Result: %s\n", g_test_fail == 0 ? "ALL PASS" : "HAS FAILURES");
    return g_test_fail == 0 ? 0 : 1;
}
```

- [ ] **Step 2: 验证测试可编译运行 (不链接CUDA)**

Run (在VS开发环境下):
```powershell
cd f:\thetrueai\build
cl /std:c++17 /EHsc /utf-8 /I ..\src\snn ..\src\snn\test_embodied.cpp ..\src\snn\embodied_env.cpp ..\src\snn\embodied_motor.cpp /Fe:test_embodied.exe
.\test_embodied.exe
```
Expected: `PASS: 11` + `FAIL: 0` + `Result: ALL PASS`

- [ ] **Step 3: 提交**

```bash
git add src/snn/test_embodied.cpp
git commit -m "test(snn): add embodied environment unit tests (Phase 3a-D1 Task 4)"
```

---

### Task 5: 多模态感知注入 CUDA Kernel

**Files:**
- Create: `src/snn/multi_sensory_inject.cuh`
- Create: `src/snn/multi_sensory_inject.cu`

- [ ] **Step 1: 编写 kernel header**

```cpp
// src/snn/multi_sensory_inject.cuh
#ifndef SNN_MULTI_SENSORY_INJECT_CUH
#define SNN_MULTI_SENSORY_INJECT_CUH

#include "thalamic_gate.cuh"
#include "config.h"

namespace stage2e {

// 多模态感知注入: 50柱信号 → L4层 input_current
// 每柱按信号强度×丘脑门控激活K个L4神经元
void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states);

} // namespace stage2e

#endif // SNN_MULTI_SENSORY_INJECT_CUH
```

- [ ] **Step 2: 编写 kernel 实现**

```cpp
// src/snn/multi_sensory_inject.cu
#include "multi_sensory_inject.cuh"
#include <cuda_runtime.h>

namespace stage2e {

// 每 thread 处理一个柱 (50柱)
__global__ void multi_sensory_inject_kernel(
    const float* __restrict__ sensory_signals,  // [50]
    float* __restrict__ input_current,           // d_input_current
    const ThalamicGateState* __restrict__ gate_states,
    int n_columns,
    int neurons_per_column,
    int l4_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_columns) return;
    
    float s = sensory_signals[col];
    if (s < 0.01f) return;  // 信号太弱, 跳过
    
    float gate = gate_states[col].gate_signal;
    float effective = s * gate;
    if (effective < 0.01f) return;
    
    // 在柱内 L4 层激活 K 个神经元
    int K = (int)(100.0f * effective);  // 最多100个
    if (K < 1) K = 1;
    
    int sensory_base = col * neurons_per_column;  // L4在柱首
    
    // xorshift32 哈希选择K个神经元 (与 input_encoding.cu 一致)
    unsigned int state = (unsigned int)(col * 2654435761u + 12345u);
    for (int k = 0; k < K; ++k) {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        int neuron_offset = (int)(state % (unsigned int)l4_size);
        int neuron_idx = sensory_base + neuron_offset;
        atomicAdd(&input_current[neuron_idx], POP_CODING_GAIN * effective);
    }
}

void launch_multi_sensory_inject(
    const float sensory[50],
    float* d_input_current,
    const ThalamicGateState* d_gate_states)
{
    // 50柱, 1 block × 50 threads (或 1 block × 32 threads, 2 blocks)
    float* d_sensory = nullptr;
    cudaMalloc(&d_sensory, 50 * sizeof(float));
    cudaMemcpy(d_sensory, sensory, 50 * sizeof(float), cudaMemcpyHostToDevice);
    
    multi_sensory_inject_kernel<<<1, 50>>>(
        d_sensory,
        d_input_current,
        d_gate_states,
        N_COLUMNS_2E,        // 50
        NEURONS_PER_COLUMN_2E,  // 1000
        COL_L4_SIZE_2E       // 200
    );
    
    cudaFree(d_sensory);
}

} // namespace stage2e
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/multi_sensory_inject.cuh src/snn/multi_sensory_inject.cu
git commit -m "feat(snn): add multi-sensory injection CUDA kernel (Phase 3a-D1 Task 5)"
```

---

### Task 6: 行为模仿教师 CUDA Kernel

**Files:**
- Create: `src/snn/motor_teacher.cuh`
- Create: `src/snn/motor_teacher.cu`

- [ ] **Step 1: 编写教师 kernel header**

```cpp
// src/snn/motor_teacher.cuh
#ifndef SNN_MOTOR_TEACHER_CUH
#define SNN_MOTOR_TEACHER_CUH

namespace stage2e {

// 行为模仿教师: 增强目标动作对应组的L5→运动皮层突触权重
// target_action=-1 时无操作
void launch_motor_teacher(
    int target_action,
    float teacher_lr);

} // namespace stage2e

#endif // SNN_MOTOR_TEACHER_CUH
```

- [ ] **Step 2: 编写教师 kernel 实现**

```cpp
// src/snn/motor_teacher.cu
#include "motor_teacher.cuh"
#include "config.h"
#include <cuda_runtime.h>

namespace stage2e {

// 教师 kernel: 每 thread 处理一个运动皮层神经元 (5K)
// 增强目标动作组的权重, 抑制其他组
__global__ void motor_teacher_kernel(
    float* __restrict__ motor_synapse_weights,  // L5→Motor突触权重
    const bool* __restrict__ l5_spike_flags,     // L5层spike (上一步)
    int target_action,
    float teacher_lr,
    int n_motor_neurons,
    int motor_group_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_motor_neurons) return;
    if (target_action < 0) return;
    
    int group = i / motor_group_size;  // 0..49
    int group_action = group / 10;     // 0..4
    float sign = (group_action == target_action) ? 1.0f : -0.5f;
    
    // 简化: 直接调整权重 (实际应通过L5 spike eligibility)
    // D1阶段用简化版本: 只要L5有spike就调整
    if (l5_spike_flags && l5_spike_flags[i % 50000]) {  // 简化映射
        motor_synapse_weights[i] += sign * teacher_lr;
        // clamp
        if (motor_synapse_weights[i] > 1.0f) motor_synapse_weights[i] = 1.0f;
        if (motor_synapse_weights[i] < -1.0f) motor_synapse_weights[i] = -1.0f;
    }
}

void launch_motor_teacher(
    int target_action,
    float teacher_lr)
{
    // D1阶段: 简化版, 不实际修改突触权重 (避免需要访问scheduler内部状态)
    // 只记录教师信号, 让现有STDP机制自然学习
    // 真正的motor_teacher_kernel留待D2阶段实现完整版
    
    // D1: 打印教师信号 (日志用)
    if (target_action >= 0) {
        // 静态变量记录最近教师信号, 供main.cpp日志读取
        // (实际实现见 main.cpp 的日志部分)
    }
}

} // namespace stage2e
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/motor_teacher.cuh src/snn/motor_teacher.cu
git commit -m "feat(snn): add motor teacher kernel stub (Phase 3a-D1 Task 6)"
```

---

### Task 7: DA奖励 + 好奇心ACh 注入接口

**Files:**
- Modify: `src/snn/modulatory_kernels.cuh`
- Modify: `src/snn/modulatory_kernels.cu`

- [ ] **Step 1: 在 modulatory_kernels.cuh 添加声明**

在 `set_event_signal` 声明之后添加:

```cpp
// Phase 3a-D1: 具身训练 reward + curiosity 接口
void set_embodied_reward(float reward);
void set_curiosity_ach(float pred_error);
float get_last_embodied_reward();
float get_last_curiosity_ach();
```

- [ ] **Step 2: 在 modulatory_kernels.cu 添加实现**

在 `set_event_signal` 实现之后 (约 line 290 后) 添加:

```cpp
// Phase 3a-D1: 具身训练 reward + curiosity 缓存
static float h_embodied_reward = 0.0f;
static float h_curiosity_ach = 0.0f;

void set_embodied_reward(float reward) {
    // clamp reward 到合理范围
    if (reward < -1.0f) reward = -1.0f;
    if (reward > 1.0f) reward = 1.0f;
    h_embodied_reward = reward;
}

void set_curiosity_ach(float pred_error) {
    // pred_error ∈ [0,1], ACh增量 clamp到[0, 0.3]
    if (pred_error < 0.0f) pred_error = 0.0f;
    if (pred_error > 1.0f) pred_error = 1.0f;
    h_curiosity_ach = pred_error * 0.3f;
}

float get_last_embodied_reward() { return h_embodied_reward; }
float get_last_curiosity_ach() { return h_curiosity_ach; }
```

- [ ] **Step 3: 在 launch_modulatory 中集成 reward + curiosity**

找到 `launch_modulatory` 函数中计算 `delta` (TD error) 的位置, 修改为使用 embodied reward:

```cpp
// 原代码 (约 line 460 附近):
// float delta = reward + GAMMA * h_v_sp - h_v_s;

// 修改为:
float effective_reward = reward;
// Phase 3a-D1: 具身模式覆盖 reward
if (fabsf(h_embodied_reward) > 1e-6f) {
    effective_reward = h_embodied_reward;
}
float delta = effective_reward + GAMMA * h_v_sp - h_v_s;
```

在 ACh 信号计算位置 (约 line 420 附近), 添加 curiosity:

```cpp
// 原代码:
// float ach_signal = ...;

// 修改为:
float ach_signal = /* 原有计算 */;
// Phase 3a-D1: 好奇心驱动的ACh
ach_signal += h_curiosity_ach;
```

- [ ] **Step 4: 提交**

```bash
git add src/snn/modulatory_kernels.cuh src/snn/modulatory_kernels.cu
git commit -m "feat(snn): add embodied reward + curiosity ACh injection (Phase 3a-D1 Task 7)"
```

---

### Task 8: RunConfig 新增 --embodied 参数

**Files:**
- Modify: `src/snn/run_config.h`
- Modify: `src/snn/run_config.cpp`

- [ ] **Step 1: 在 run_config.h 添加字段**

在 `event_stream_path` 字段之后添加:

```cpp
// Phase 3a-D1: 具身发育训练模式
bool embodied_mode = false;
std::string embodied_scene = "hunger_feeding";  // 默认场景
```

- [ ] **Step 2: 在 run_config.cpp 添加命令行解析**

在 `--event-stream` 解析之后添加:

```cpp
} else if (arg == "--embodied") {
    // Phase 3a-D1: 具身发育训练模式
    config->embodied_mode = true;
} else if (arg == "--embodied-scene") {
    value = require_value(&i, "--embodied-scene");
    if (!value) return false;
    config->embodied_scene = value;
    config->embodied_mode = true;  // 指定场景自动启用
}
```

在 `print_usage` 函数中添加帮助文本:

```cpp
printf("  --embodied                 Enable embodied developmental mode (Phase 3a-D1)\n");
printf("  --embodied-scene ID        Specify scene: hunger_feeding|warmth_safety|startle_recover|sleep_wake|discomfort_change\n");
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/run_config.h src/snn/run_config.cpp
git commit -m "feat(snn): add --embodied CLI flag (Phase 3a-D1 Task 8)"
```

---

### Task 9: main.cpp 集成环境闭环

**Files:**
- Modify: `src/snn/main.cpp`

- [ ] **Step 1: 添加 include 和初始化**

在 main.cpp 顶部 include 区域添加:

```cpp
#include "embodied_body.h"
#include "embodied_env.h"
#include "embodied_motor.h"
#include "multi_sensory_inject.cuh"
#include "motor_teacher.cuh"
```

在 main() 函数中, 事件调度器初始化之后添加:

```cpp
// === Phase 3a-D1: 具身发育环境初始化 ===
stage2e::EmbodiedEnvironment embodied_env;
bool embodied_active = config.embodied_mode;
if (embodied_active) {
    embodied_env.init(config.embodied_scene);
    printf("[P1] 具身发育模式已启用: 场景=%s\n", config.embodied_scene.c_str());
    printf("[P1]   初始状态: hunger=%.2f temp=%.2f comfort=%.2f fatigue=%.2f arousal=%.2f\n",
           embodied_env.body.hunger, embodied_env.body.temperature,
           embodied_env.body.comfort, embodied_env.body.fatigue, embodied_env.body.arousal);
}
stage2e::MotorReadout motor_readout = {};
float sensory_signals[50] = {0};
float embodied_reward = 0.0f;
```

- [ ] **Step 2: 在训练循环中添加环境闭环**

找到训练循环 `for (int step = start_step; step < total_steps; ++step)` 内部, 在 `event_scheduler.dispatch_pending(step)` 之后, `scheduler.step(step)` 之前添加:

```cpp
// === Phase 3a-D1: 每100步环境闭环 ===
bool is_env_step = (step > 0 && step % 100 == 0);
if (is_env_step && embodied_active) {
    // 1. 感知注入 (当前身体状态 → 50柱信号 → L4层)
    embodied_env.compute_sensory_signals(sensory_signals);
    stage2e::launch_multi_sensory_inject(
        sensory_signals,
        allocator.d_input_current,
        scheduler.d_gate_states_for_inject());
    
    // 2. 动作读出 (上一环境步的运动皮层输出)
    motor_readout = stage2e::read_motor_output(allocator.d_motor_spike_flags);
    
    // 3. 环境响应 + 身体演化
    stage2e::BodyState prev_body = embodied_env.get_body_state();
    embodied_env.step_env(motor_readout);
    
    // 4. DA奖励计算
    embodied_reward = embodied_env.compute_reward(prev_body);
    stage2e::set_embodied_reward(embodied_reward);
    
    // 5. 行为模仿教师信号
    int target_action = embodied_env.get_teacher_signal();
    stage2e::launch_motor_teacher(target_action, 0.01f);
    
    // 6. 日志 (每100步)
    printf("[EMBODIED step=%d] hunger=%.2f comfort=%.2f arousal=%.2f cry=%.2f suck=%.2f reward=%.4f mom=%s\n",
           step, embodied_env.body.hunger, embodied_env.body.comfort,
           embodied_env.body.arousal, motor_readout.cry_intensity,
           motor_readout.suck_strength, embodied_reward,
           embodied_env.mom_present ? "YES" : "no");
}
```

- [ ] **Step 3: 在 CSV 日志中添加具身字段**

找到 CSV 采样区域 (约 `if (step % CSV_SAMPLE_INTERVAL == 0)` 附近), 在现有 CSV 行末尾添加具身字段:

```cpp
// Phase 3a-D1: 具身状态追加到CSV
if (embodied_active) {
    fprintf(csv_f, ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d",
            embodied_env.body.hunger, embodied_env.body.temperature,
            embodied_env.body.comfort, embodied_env.body.fatigue,
            embodied_env.body.arousal, motor_readout.cry_intensity,
            embodied_reward, embodied_env.mom_present ? 1 : 0);
}
fprintf(csv_f, "\n");
```

- [ ] **Step 4: 提交**

```bash
git add src/snn/main.cpp
git commit -m "feat(snn): integrate embodied environment loop in main.cpp (Phase 3a-D1 Task 9)"
```

---

### Task 10: CMakeLists.txt 更新

**Files:**
- Modify: `src/snn/CMakeLists.txt`

- [ ] **Step 1: 添加新源文件到 SNN_TRAIN_SRCS**

在 `event_scheduler.cpp` 之后添加:

```cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_env.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_motor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_sensory_inject.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/motor_teacher.cu
```

- [ ] **Step 2: 添加 test_embodied target**

在 `test_event_scheduler` target 之后添加:

```cmake
# =============================================================================
# Phase 3a-D1: Embodied Environment 单元测试 (无 CUDA 依赖)
# =============================================================================
add_executable(test_embodied
    ${CMAKE_CURRENT_SOURCE_DIR}/test_embodied.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_env.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embodied_motor.cpp
)
target_include_directories(test_embodied PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
    target_compile_options(test_embodied PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4305>
    )
endif()
set_target_properties(test_embodied PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
```

- [ ] **Step 3: 提交**

```bash
git add src/snn/CMakeLists.txt
git commit -m "build(snn): add embodied sources and test target to CMake (Phase 3a-D1 Task 10)"
```

---

### Task 11: 编译 + 单元测试 + 5K步集成测试

**Files:** 无 (验证步骤)

- [ ] **Step 1: 重新生成 CMake + 编译 test_embodied**

Run (在 VS 开发环境下):
```powershell
cd f:\thetrueai\build
cmake .. -G Ninja
ninja test_embodied
.\bin\test_embodied.exe
```
Expected: `PASS: 11` + `FAIL: 0` + `Result: ALL PASS`

- [ ] **Step 2: 编译 snn_train (含具身模块)**

Run:
```powershell
ninja snn_train
```
Expected: 编译成功, 无 error

- [ ] **Step 3: 运行 5K 步具身训练测试**

Run:
```powershell
.\bin\snn_train.exe --embodied --embodied-scene hunger_feeding --steps 5000 --synthetic-input --input-mode byte --no-bptt --checkpoint-interval 0 --csv run_embodied_5k.csv 2>&1 | Tee-Object run_embodied_5k.log
```
Expected:
1. 程序正常退出 (exit code 0)
2. 日志中出现 `[P1] 具身发育模式已启用`
3. 每100步出现 `[EMBODIED step=XXX]` 日志行
4. hunger 从 0.8 开始变化 (不恒定)
5. mom 在某些步显示 YES
6. 6维调质浓度均在 [0, 2] 范围
7. 24个准则中大部分 PASS (PSW/CV 可能因短步数失败, 与非具身模式一致)

- [ ] **Step 4: 验证 CSV 日志包含具身字段**

Run:
```powershell
Get-Content run_embodied_5k.csv -Head 3
```
Expected: CSV 行末尾包含 8 个具身字段 (hunger,temp,comfort,fatigue,arousal,cry,reward,mom_present)

- [ ] **Step 5: 最终提交**

```bash
git add -A
git commit -m "test(snn): verify embodied 5K step run passes (Phase 3a-D1 complete)"
```

---

## 验证清单 (D1 完成标准)

| # | 验证项 | 通过条件 |
|---|--------|----------|
| 1 | test_embodied 单元测试 | 11/11 PASS |
| 2 | snn_train 编译 | 无 error |
| 3 | 5K步不崩溃 | exit code 0 |
| 4 | 6维调质 [0,2] | criterion_modulatory_range PASS |
| 5 | 运动皮层有输出 | action_prob 非均匀 (cry_intensity 有变化) |
| 6 | 环境闭环运行 | hunger 在变化, mom_present 出现 YES |
| 7 | DA reward 有变化 | reward 不恒定 |
| 8 | CSV 日志完整 | 含8个具身字段 |
