#ifndef SNN_STAGE2E_EVENT_SCHEDULER_H
#define SNN_STAGE2E_EVENT_SCHEDULER_H

#include <string>
#include <vector>
#include <cstddef>
#include "event_types.h"
#include "gene_event_map.h"

namespace stage2e {

// 一次被调度的事件 (从 events.jsonl 解析得到)
struct ScheduledEvent {
    int         step_target;      // 触发 step
    int         event_type;       // EventType 枚举
    int         modifier_flags;   // EventModifier 位域
    int         intensity;        // -50..+50
    float       duration_s;       // 持续时间 (秒, 0=用 GENE_MAP 默认)
    std::string description;      // 人类可读描述
};

// 事件调度器: 加载 JSONL 事件流, 每 100 步派发到期事件
// 派发逻辑: 查 GENE_MAP_BASE → apply_modifiers → set_event_signal
class EventScheduler {
public:
    // 加载 events.jsonl 文件, 成功返回 true
    bool load_jsonl(const std::string& path);

    // 每 100 步调用: 派发所有 step_target <= current_step 的事件
    // 对每个事件: 查 GENE_MAP → apply_modifiers → set_event_signal
    void dispatch_pending(int current_step);

    size_t total_events() const { return events_.size(); }
    size_t dispatched_count() const { return next_event_idx_; }
    bool   empty() const { return events_.empty(); }

    // 测试用: 直接访问事件列表 (用于单元测试)
    const std::vector<ScheduledEvent>& events() const { return events_; }

private:
    std::vector<ScheduledEvent> events_;
    size_t next_event_idx_ = 0;
};

} // namespace stage2e

#endif // SNN_STAGE2E_EVENT_SCHEDULER_H
