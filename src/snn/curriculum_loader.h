#ifndef SNN_STAGE2E_CURRICULUM_LOADER_H
#define SNN_STAGE2E_CURRICULUM_LOADER_H

#include <string>
#include <vector>
#include <cstddef>
#include "event_types.h"
#include "gene_event_map.h"
#include "personality_profiles.h"

namespace stage2e {

// 课程样本中的单个事件 (复用 GENE_MAP 事件类型)
struct CurriculumEvent {
    int   step_offset;       // 相对样本起点的步偏移
    int   event_type;        // EventType 枚举
    int   intensity;         // -50..+50
    std::string description;
};

// 一个课程样本: 事件序列 + 目标调质轨迹 + 目标 PAD + 目标工具调用
// 对应 spec §5.2 CurriculumSample
struct CurriculumSample {
    int    sample_id;
    float  target_modulators[6];   // 期望调质响应 (DA/5HT/NE/ACh/GABA/Oxy)
    float  target_pad[3];          // 期望 PAD 情感状态 [Pleasure, Arousal, Dominance]
    int    target_tool_call;       // 0-5 工具索引, -1=初中/启蒙不训工具
    std::vector<CurriculumEvent> events;
};

// 课程数据加载器: 加载 curriculum JSONL
// 每行一个课程样本, 格式:
// {
//   "sample_id": 1,
//   "events": [{"step_offset":0,"event_type":"exam_success","intensity":30}, ...],
//   "target_modulators": [0.55, 0.30, 0.30, 0.40, 0.25, 0.30],
//   "target_pad": [0.5, 0.4, 0.5],
//   "target_tool": -1
// }
class CurriculumLoader {
public:
    // 加载课程 JSONL 文件, 成功返回 true
    bool load_jsonl(const std::string& path, CurriculumStage stage);

    size_t total_samples() const { return samples_.size(); }
    bool   empty() const { return samples_.empty(); }
    CurriculumStage stage() const { return stage_; }

    // 测试用: 直接访问样本列表
    const std::vector<CurriculumSample>& samples() const { return samples_; }

    // 取第 idx 个样本 (越界返回 nullptr)
    const CurriculumSample* sample(size_t idx) const {
        if (idx >= samples_.size()) return nullptr;
        return &samples_[idx];
    }

private:
    std::vector<CurriculumSample> samples_;
    CurriculumStage stage_ = STAGE_ENLIGHTENMENT;
};

} // namespace stage2e

#endif // SNN_STAGE2E_CURRICULUM_LOADER_H
