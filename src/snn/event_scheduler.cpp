#include "event_scheduler.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// 前向声明 set_event_signal, 避免引入 modulatory_kernels.cuh 的 CUDA 依赖
// (实现位于 modulatory_kernels.cu, 由 snn_train target 链接)
namespace stage2e {
void set_event_signal(const float modulator_delta[6], int duration_steps);
// Phase 3a-C2: 每步开始前清零事件信号缓存, 允许同 step 多事件叠加
//   旧实现没有此函数, 同一 step 多个事件调用 set_event_signal 时会相互覆盖
//   新实现: dispatch_pending 处理某 step 的事件之前先 reset, 让各事件累加
void reset_event_signal();
}

namespace stage2e {

// =============================================================================
// 轻量 JSON 字段提取 (仅支持扁平 key:value, 不支持嵌套对象解析)
// 对于 modifiers 对象, 用独立函数处理
// =============================================================================

// 从 JSON 行中提取 "key":"value" 或 "key":number 的值字符串
static std::string extract_field(const std::string& line, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = line.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    // 跳过 : 和空格
    while (pos < line.size() && (line[pos] == ':' || line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return "";
    // 字符串值
    if (line[pos] == '"') {
        size_t start = pos + 1;
        size_t end = line.find('"', start);
        if (end == std::string::npos) return "";
        return line.substr(start, end - start);
    }
    // 数字/布尔值 (读到 , 或 } 结束)
    size_t start = pos;
    while (pos < line.size() && line[pos] != ',' && line[pos] != '}' && line[pos] != '\n') pos++;
    return line.substr(start, pos - start);
}

// 从 JSON 行中提取 modifiers 子字段 (如 "publicity":"public")
static std::string extract_modifier(const std::string& line, const std::string& key) {
    return extract_field(line, key);
}

static int parse_int_or(const std::string& s, int default_val) {
    if (s.empty()) return default_val;
    return atoi(s.c_str());
}

static float parse_float_or(const std::string& s, float default_val) {
    if (s.empty()) return default_val;
    return (float)atof(s.c_str());
}

bool EventScheduler::load_jsonl(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "[EventScheduler] cannot open: %s\n", path.c_str());
        return false;
    }
    events_.clear();
    next_event_idx_ = 0;

    std::string line;
    int line_num = 0;
    while (std::getline(fin, line)) {
        line_num++;
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') continue;
        // 跳过不含 event_type 的行
        if (line.find("\"event_type\"") == std::string::npos) continue;

        ScheduledEvent evt;
        evt.event_type = EVT_COUNT;
        evt.modifier_flags = 0;
        evt.intensity = 0;
        evt.duration_s = 0.0f;

        // 解析 step_target 或 time_s
        std::string step_str = extract_field(line, "step_target");
        if (!step_str.empty()) {
            evt.step_target = parse_int_or(step_str, 0);
        } else {
            std::string time_str = extract_field(line, "time_s");
            if (!time_str.empty()) {
                float t = parse_float_or(time_str, 0.0f);
                // 事件调度每 100 步一次, time_s 单位为秒
                // 设计文档示例: time_s=15.0 → step=1500 (即 1s = 100 调度步)
                evt.step_target = (int)(t * 100.0f);
            } else {
                fprintf(stderr, "[EventScheduler] line %d: missing step_target/time_s\n", line_num);
                continue;
            }
        }

        // 解析 event_type
        std::string type_str = extract_field(line, "event_type");
        evt.event_type = event_type_from_string(type_str.c_str());
        if (evt.event_type >= EVT_COUNT) {
            fprintf(stderr, "[EventScheduler] line %d: unknown event_type '%s'\n", line_num, type_str.c_str());
            continue;
        }

        // 解析 modifiers (可选)
        std::string pub = extract_modifier(line, "publicity");
        if (pub == "public") evt.modifier_flags |= MOD_PUBLIC;
        std::string auth = extract_modifier(line, "authority");
        if (auth == "authority") evt.modifier_flags |= MOD_AUTHORITY;
        std::string temp = extract_modifier(line, "temporal");
        if (temp == "sustained") evt.modifier_flags |= MOD_SUSTAINED;

        // 解析 intensity (可选, 默认 0)
        std::string int_str = extract_field(line, "intensity");
        evt.intensity = parse_int_or(int_str, 0);

        // 解析 duration_s (可选, 0=用 GENE_MAP 默认)
        std::string dur_str = extract_field(line, "duration_s");
        evt.duration_s = parse_float_or(dur_str, 0.0f);

        // 解析 description (可选)
        evt.description = extract_field(line, "description");

        events_.push_back(evt);
    }

    // 按 step_target 排序 (确保 dispatch 顺序正确)
    std::sort(events_.begin(), events_.end(),
              [](const ScheduledEvent& a, const ScheduledEvent& b) {
                  return a.step_target < b.step_target;
              });

    // P1-2 修复 (2026-08-04): 事件流数据契约告警 — launch_modulatory 每 100 步
    //   才读取事件信号, 非 100 倍数 offset 的事件在下一个 step 被
    //   reset_event_signal() 提前清零, 永远读不到 → 静默丢弃.
    //   课程数据 (offset 100/200/300/400) 安全; 事件流文件 (dynamic/causal/
    //   parallel2 类) 实测 82% 事件违约, 在此告警以便生成器对齐.
    {
        int aligned = 0;
        for (const auto& evt : events_)
            if (evt.step_target % 100 == 0) aligned++;
        if (aligned != static_cast<int>(events_.size())) {
            fprintf(stderr, "[EventScheduler] 警告: %d/%zu 事件 step_target 非 100 倍数"
                    " (launch_modulatory 每 100 步读取, 违约事件将被静默丢弃)\n",
                    static_cast<int>(events_.size()) - aligned, events_.size());
        }
    }

    fprintf(stdout, "[EventScheduler] loaded %zu events from %s\n", events_.size(), path.c_str());
    return true;
}

void EventScheduler::dispatch_pending(int current_step) {
    // Phase 3a-C2 修复: 检测 step 切换, 在处理新 step 的事件前清零信号缓存
    //   旧实现: 每个 set_event_signal 覆盖前一个, 同 step 多事件只剩最后一个
    //   新实现: 进入新 step 时先 reset, 后续多个事件累加到一起
    //   注意: launch_modulatory 每 100 步才读取一次, 所以这里只 reset 一次
    //         (在进入新 step 时 reset, 不是每事件 reset)
    if (current_step != last_dispatch_step_) {
        reset_event_signal();
        last_dispatch_step_ = current_step;
    }

    int event_count_this_step = 0;
    while (next_event_idx_ < events_.size() &&
           events_[next_event_idx_].step_target <= current_step) {
        const ScheduledEvent& evt = events_[next_event_idx_];

        // 1. 查 GENE_MAP_BASE
        GeneMapEntry entry = GENE_MAP_BASE[evt.event_type];

        // 2. 应用修饰符 + intensity
        entry = apply_modifiers(entry, evt.modifier_flags, evt.intensity);

        // 3. 覆盖 duration (如果事件指定了)
        if (evt.duration_s > 0.0f) entry.duration_s = evt.duration_s;

        // 4. 转换为 6 维增量
        float delta[6] = {
            entry.da_delta,
            entry.ach_delta,
            entry.ne_delta,
            entry.ht5_delta,
            entry.gaba_delta,
            entry.oxy_delta
        };

        // 5. C1 仅支持 pulse 型 (duration_steps=0), plateau 留 C2
        int duration_steps = 0;

        // 6. 注入到 modulatory 缓存 (累加模式, 同 step 多事件会叠加)
        set_event_signal(delta, duration_steps);
        event_count_this_step++;
        // Phase 3a-F (M1): 事件→杏仁核 LA 注入 (与 set_event_signal 并列,
        //   负性事件同时经 scheduler.amygdala_event_inject 累积皮质醇应激)
        if (on_event_inject) {
            on_event_inject(evt.event_type, static_cast<float>(evt.intensity));
        }

        if (!evt.description.empty()) {
            const char* tag = (event_count_this_step > 1) ? "[Event-Superposed]" : "[Event]";
            fprintf(stdout, "%s step=%d type=%d intensity=%d (simul#%d) desc=%s\n",
                    tag, evt.step_target, evt.event_type, evt.intensity,
                    event_count_this_step, evt.description.c_str());
        }

        next_event_idx_++;
    }
}

} // namespace stage2e
