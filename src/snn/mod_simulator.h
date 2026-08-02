#ifndef SNN_STAGE2E_MOD_SIMULATOR_H
#define SNN_STAGE2E_MOD_SIMULATOR_H

// =============================================================================
// Stage 2e 课程浓度模拟器 (host 端, header-only)
// =============================================================================
// 背景: 课程训练监督目标 target_mod 与网络内部调质浓度不同源
//   (目标 = 纯累加注入, 内部 = 衰减 + 基线 + 事件 + 交互 + 灵敏度)
// 本模拟器复刻 modulatory_kernels.cu 课程模式(确定性)注入路径,
//   生成"期望浓度"作为训练监督目标 (每 100 步推进), 与 GPU 内部动力学对齐.
//
// 与 modulatory_kernels.cu 的对应关系:
//   - 灵敏度稳态更新: L428-442 (HOMEOSTATIC_RATE / HOMEOSTATIC_UPREG_RATE)
//   - 事件增量 clamp: L309-326 (set_event_signal, 单事件 [-1,1], 累加 [-1.5,1.5])
//   - 非线性交互:     L515-545 (DA-5HT 拮抗 / NE→GABA 抑制 / Oxy 放大 DA)
//   - 灵敏度应用:     L562-569 (signal × receptor_sensitivity)
//   - 衰减+注入+clamp: L48-75  (conc × exp(-100/tau) + signal, clamp [0,2])
//
// 线程安全: 单实例单线程使用 (scheduler 与 main.cpp 各持一个, 互不共享)
// =============================================================================

#include <cmath>
#include <algorithm>
#include "event_types.h"
#include "gene_event_map.h"
#include "config.h"

namespace stage2e {

// 通道索引 (GENE_MAP 列顺序 [DA, ACh, NE, 5HT, GABA, Oxy])
enum { MOD_CH_DA = 0, MOD_CH_ACH, MOD_CH_NE, MOD_CH_5HT, MOD_CH_GABA, MOD_CH_OXY, MOD_CH_COUNT = 6 };

// 通道衰减常数 (ms): 与 config.h DA_TAU=100/ACH_TAU=200/NE_TAU=150/HT5_TAU=300/GABA_TAU=120/OXYTOCIN_TAU=500 一致
static const float MOD_TAU_ARR[MOD_CH_COUNT] = {
    DA_TAU, ACH_TAU, NE_TAU, HT5_TAU, GABA_TAU, OXYTOCIN_TAU,
};

// 稳态基线 (索引同通道顺序): 与 config.h HOMEOSTATIC_BASELINE_* 一致
//   (modulatory_kernels.cu L282-289 的文件内数组为 HOMEOSTATIC_BASELINE,
//    此处改名避免混淆, 值完全一致)
static const float HOMEOSTATIC_BASELINE_ARR[MOD_CH_COUNT] = {
    HOMEOSTATIC_BASELINE_DA,
    HOMEOSTATIC_BASELINE_ACH,
    HOMEOSTATIC_BASELINE_NE,
    HOMEOSTATIC_BASELINE_HT5,
    HOMEOSTATIC_BASELINE_GABA,
    HOMEOSTATIC_BASELINE_OXY,
};

// 事件默认修饰符: event_default_modifier_flags(EventType) 由 gene_event_map.h 提供
//   (与 Python generate_curriculum_data.py EVENT_DEFAULT_MOD 一致), 此处直接使用.

// 课程浓度模拟器: 复刻 modulatory_kernels.cu 课程模式(确定性)注入路径
//   conc[ch] = clamp(conc[ch]*decay[ch] + signal[ch], 0, 2)
//   signal[ch] = (base_signal[ch] + eff_event[ch]) * sensitivity[ch]
//   sensitivity 稳态更新与事件非线性交互与 modulatory_kernels.cu 完全一致
// 线程安全: 单实例单线程使用 (scheduler 与 main.cpp 各持一个)
class CurriculumModSimulator {
public:
    CurriculumModSimulator() { reset(); }

    // 复位到冷启动状态 (conc=0, sensitivity=1)
    void reset() {
        for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
            conc_[ch] = 0.0f;
            sensitivity_[ch] = 1.0f;
        }
    }

    // 每 100 步调用一次 (与 launch_modulatory 节奏一致):
    //   base_signal[6]: 该阶段基线, GENE_MAP 列顺序 [DA,ACh,NE,5HT,GABA,Oxy]
    //   event_types/intensities: 本 100 步块内到期的事件 (可 0 个)
    //   内部: 1) 灵敏度稳态更新 2) 事件增量(修饰符+交互) 3) 衰减+注入
    void advance_block(const int* event_types, const int* intensities, int n_events,
                       const float base_signal[6]) {
        // ---- 1. 灵敏度稳态更新 (复刻 modulatory_kernels.cu L428-442) ----
        // current_means 取当前 conc_ 各通道均值: 因所有神经元浓度相同, 均值 = conc_
        for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
            float excess = conc_[ch] - HOMEOSTATIC_BASELINE_ARR[ch];
            if (excess > 0.0f) {
                // 下调: 持续超阈 → 受体脱敏
                sensitivity_[ch] *= (1.0f - HOMEOSTATIC_RATE * excess);
            } else {
                // 上调: 低于基线 → 受体缓慢恢复
                sensitivity_[ch] *= (1.0f + HOMEOSTATIC_UPREG_RATE * (-excess));
            }
            // clamp 到 [MIN, MAX]
            if (sensitivity_[ch] < RECEPTOR_SENSITIVITY_MIN)
                sensitivity_[ch] = RECEPTOR_SENSITIVITY_MIN;
            if (sensitivity_[ch] > RECEPTOR_SENSITIVITY_MAX)
                sensitivity_[ch] = RECEPTOR_SENSITIVITY_MAX;
        }

        // ---- 2. 事件增量 (复刻 set_event_signal L309-326 + 交互 L505-545) ----
        float eff_event[MOD_CH_COUNT] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < n_events; ++i) {
            GeneMapEntry e = GENE_MAP_BASE[event_types[i]];
            e = apply_modifiers(e, event_default_modifier_flags((EventType)event_types[i]),
                                intensities[i]);
            float v[MOD_CH_COUNT] = {
                e.da_delta, e.ach_delta, e.ne_delta, e.ht5_delta, e.gaba_delta, e.oxy_delta
            };
            for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
                // 单事件 delta 先 clamp 到 [-1, 1]
                if (v[ch] < -1.0f) v[ch] = -1.0f;
                if (v[ch] >  1.0f) v[ch] =  1.0f;
                // 累加 (多事件叠加)
                eff_event[ch] += v[ch];
            }
        }
        // 叠加后整体 clamp 到 [-1.5, 1.5] (set_event_signal L317-319)
        for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
            if (eff_event[ch] < -1.5f) eff_event[ch] = -1.5f;
            if (eff_event[ch] >  1.5f) eff_event[ch] =  1.5f;
        }

        // 非线性交互 (复刻 L515-545, 仅在任一通道非零时)
        //   索引: [0]=DA, [1]=ACh, [2]=NE, [3]=5HT, [4]=GABA, [5]=Oxy
        bool has_event = false;
        for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
            if (std::fabs(eff_event[ch]) > 1e-6f) { has_event = true; break; }
        }
        if (has_event) {
            float da   = eff_event[MOD_CH_DA];
            float ach  = eff_event[MOD_CH_ACH];
            float ne   = eff_event[MOD_CH_NE];
            float ht5  = eff_event[MOD_CH_5HT];
            float gaba = eff_event[MOD_CH_GABA];
            float oxy  = eff_event[MOD_CH_OXY];

            // 规则 1: DA-5HT 拮抗 (仅在两者同号时生效, 避免反向增强)
            if (da > 0.0f && ht5 > 0.0f) {
                float antagonism = 0.2f * std::fmin(da, ht5);
                da -= antagonism;
                ht5 -= antagonism;
            }
            // 规则 2: NE 抑制 GABA (NE↑ 时 GABA 释放减弱)
            if (ne > 0.0f && gaba > 0.0f) {
                float inhibition = 0.3f * ne * gaba;
                gaba -= inhibition;
                if (gaba < 0.0f) gaba = 0.0f;
            }
            // 规则 3: Oxy 放大 DA 奖赏 (仅在 DA 正向时生效)
            //   注: 交互后 DA 可能超过 1.5 上限, 这是 modulatory_kernels.cu 的
            //       既有行为 (交互在 clamp 之后), 保持完全一致, 不"修正"
            if (oxy > 0.0f && da > 0.0f) {
                da *= (1.0f + 0.5f * oxy);
            }

            // 写回 eff_event
            eff_event[MOD_CH_DA]   = da;
            eff_event[MOD_CH_ACH]  = ach;
            eff_event[MOD_CH_NE]   = ne;
            eff_event[MOD_CH_5HT]  = ht5;
            eff_event[MOD_CH_GABA] = gaba;
            eff_event[MOD_CH_OXY]  = oxy;
        }

        // ---- 3. 衰减 + 注入 + clamp (复刻 kernel L48-75 + 灵敏度 L562-569) ----
        for (int ch = 0; ch < MOD_CH_COUNT; ++ch) {
            float signal = (base_signal[ch] + eff_event[ch]) * sensitivity_[ch];
            conc_[ch] = conc_[ch] * std::exp(-100.0f / MOD_TAU_ARR[ch]) + signal;
            if (conc_[ch] < 0.0f) conc_[ch] = 0.0f;
            if (conc_[ch] > 2.0f) conc_[ch] = 2.0f;
        }
    }

    // 当前浓度 (GENE_MAP 列顺序)
    const float* conc() const { return conc_; }
    // 当前受体灵敏度
    const float* sensitivity() const { return sensitivity_; }

private:
    float conc_[MOD_CH_COUNT];
    float sensitivity_[MOD_CH_COUNT];
};

// 统一 PAD 映射: 与 get_affective_state (modulatory_kernels.cu L741-749) 公式完全一致
//   P = DA - 0.5*5HT - 0.3*GABA;  A = NE - 0.4*GABA - 0.3*5HT;  D = DA - 0.5*Oxy;  clamp [-1,1]
//   conc 顺序: [DA, ACh, NE, 5HT, GABA, Oxy]
inline void pad_from_concentration(const float conc[6], float pad[3]) {
    pad[0] = conc[MOD_CH_DA] - 0.5f * conc[MOD_CH_5HT] - 0.3f * conc[MOD_CH_GABA];
    pad[1] = conc[MOD_CH_NE] - 0.4f * conc[MOD_CH_GABA] - 0.3f * conc[MOD_CH_5HT];
    pad[2] = conc[MOD_CH_DA] - 0.5f * conc[MOD_CH_OXY];
    for (int i = 0; i < 3; ++i) {
        if (pad[i] >  1.0f) pad[i] =  1.0f;
        if (pad[i] < -1.0f) pad[i] = -1.0f;
    }
}

} // namespace stage2e

#endif // SNN_STAGE2E_MOD_SIMULATOR_H
