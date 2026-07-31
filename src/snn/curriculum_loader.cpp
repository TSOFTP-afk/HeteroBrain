#include "curriculum_loader.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

namespace stage2e {

// =============================================================================
// 轻量 JSON 辅助 (课程格式: 每行一个样本, 含数组字段)
// =============================================================================

static int parse_int_or(const std::string& s, int default_val) {
    if (s.empty()) return default_val;
    return atoi(s.c_str());
}

// 提取 "key":value 的原始子串 (含数组/字符串/数字)
static std::string extract_raw(const std::string& line, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = line.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    while (pos < line.size() && (line[pos] == ':' || line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size()) return "";
    if (line[pos] == '[') {
        // 数组: 找到匹配的 ]
        size_t end = line.find(']', pos);
        if (end == std::string::npos) return "";
        return line.substr(pos, end - pos + 1);
    }
    if (line[pos] == '"') {
        size_t start = pos + 1;
        size_t end = line.find('"', start);
        if (end == std::string::npos) return "";
        return line.substr(start, end - start);
    }
    size_t start = pos;
    while (pos < line.size() && line[pos] != ',' && line[pos] != '}' && line[pos] != '\n') pos++;
    return line.substr(start, pos - start);
}

// 解析 "[a, b, c]" → 浮点数组
static bool parse_float_array(const std::string& s, float* out, int n) {
    if (s.empty() || s[0] != '[') return false;
    std::string inner = s.substr(1, s.find(']') - 1);
    std::stringstream ss(inner);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, ',')) {
        if (idx >= n) break;
        // trim
        size_t a = token.find_first_not_of(" \t");
        size_t b = token.find_last_not_of(" \t");
        if (a != std::string::npos) {
            out[idx] = (float)atof(token.substr(a, b - a + 1).c_str());
        } else {
            out[idx] = 0.0f;
        }
        idx++;
    }
    return idx >= 1;
}

// 解析 events 数组 "[{...},{...}]" → 事件列表
// 格式: [{"step_offset":0,"event_type":"exam_success","intensity":30}, ...]
static bool parse_events_array(const std::string& s,
                               std::vector<CurriculumEvent>& out,
                               int line_num) {
    if (s.empty() || s[0] != '[') return false;
    size_t pos = 1;
    while (pos < s.size()) {
        size_t brace_start = s.find('{', pos);
        if (brace_start == std::string::npos) break;
        size_t brace_end = s.find('}', brace_start);
        if (brace_end == std::string::npos) break;
        std::string obj = s.substr(brace_start, brace_end - brace_start + 1);

        CurriculumEvent evt;
        evt.step_offset = parse_int_or(extract_raw(obj, "step_offset"), 0);
        std::string type_str = extract_raw(obj, "event_type");
        evt.event_type = event_type_from_string(type_str.c_str());
        evt.intensity = parse_int_or(extract_raw(obj, "intensity"), 0);
        evt.description = extract_raw(obj, "description");
        if (evt.event_type >= EVT_COUNT) {
            fprintf(stderr, "[CurriculumLoader] line %d: unknown event_type '%s'\n",
                    line_num, type_str.c_str());
            pos = brace_end + 1;
            continue;
        }
        out.push_back(evt);
        pos = brace_end + 1;
    }
    return !out.empty();
}

bool CurriculumLoader::load_jsonl(const std::string& path, CurriculumStage stage) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "[CurriculumLoader] cannot open: %s\n", path.c_str());
        return false;
    }
    samples_.clear();
    stage_ = stage;

    std::string line;
    int line_num = 0;
    while (std::getline(fin, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue;
        if (line.find("\"events\"") == std::string::npos) continue;

        CurriculumSample sample;
        sample.sample_id = parse_int_or(extract_raw(line, "sample_id"), line_num);
        sample.target_tool_call = parse_int_or(extract_raw(line, "target_tool"), -1);

        // 目标调质轨迹 (6 维)
        std::string mod_str = extract_raw(line, "target_modulators");
        if (!parse_float_array(mod_str, sample.target_modulators, 6)) {
            fprintf(stderr, "[CurriculumLoader] line %d: missing/invalid target_modulators\n", line_num);
            continue;
        }

        // 目标 PAD (3 维, 可选)
        std::string pad_str = extract_raw(line, "target_pad");
        if (!parse_float_array(pad_str, sample.target_pad, 3)) {
            sample.target_pad[0] = 0.0f;
            sample.target_pad[1] = 0.0f;
            sample.target_pad[2] = 0.0f;
        }

        // 事件序列
        std::string evts_str = extract_raw(line, "events");
        if (!parse_events_array(evts_str, sample.events, line_num)) {
            fprintf(stderr, "[CurriculumLoader] line %d: missing/invalid events array\n", line_num);
            continue;
        }

        samples_.push_back(sample);
    }

    fprintf(stdout, "[CurriculumLoader] loaded %zu samples (stage=%s) from %s\n",
            samples_.size(), personality_profile(stage).name, path.c_str());
    return !samples_.empty();
}

} // namespace stage2e
