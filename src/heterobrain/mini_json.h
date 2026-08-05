// =============================================================================
// mini_json.h — 极简 JSON DOM (header-only, 零外部依赖)
// =============================================================================
// 用途: OpenAI 兼容 HTTP 服务的请求解析 / 响应序列化。只实现本项目所需:
//   - 解析: 对象/数组/字符串/数字/布尔/null (递归下降)
//   - 序列化: json_dump (字符串转义控制符; 非 ASCII 直接输出 UTF-8 字节)
//   - \uXXXX 转义解析为 UTF-8 (客户端可能对中文做 unicode 转义)
// 不做: 数字精度控制、注释、流式解析。错误时返回 false 并可选写错误信息。
// =============================================================================

#ifndef HETERO_BRAIN_MINI_JSON_H
#define HETERO_BRAIN_MINI_JSON_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace hb {
namespace json {

class Value {
public:
    enum Type { kNul, kBool, kNum, kStr, kArr, kObj };

    Type type = kNul;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    Value() = default;
    explicit Value(const std::string& s) : type(kStr), str(s) {}
    explicit Value(const char* s) : type(kStr), str(s ? s : "") {}
    explicit Value(double n) : type(kNum), num(n) {}
    explicit Value(bool bv) : type(kBool), b(bv) {}

    static Value make_obj() { Value v; v.type = kObj; return v; }
    static Value make_arr() { Value v; v.type = kArr; return v; }

    // 对象成员查找 (未命中返回 nullptr)
    const Value* find(const char* key) const {
        if (type != kObj) {
            return nullptr;
        }
        for (const auto& kv : obj) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    // 便捷读取 (类型不匹配返回默认值)
    bool   as_bool(bool def = false) const { return type == kBool ? b : def; }
    double as_num(double def = 0.0) const { return type == kNum ? num : def; }
    std::string as_str(const std::string& def = std::string()) const {
        return type == kStr ? str : def;
    }
};

// -----------------------------------------------------------------------------
// 解析器 (递归下降)
// -----------------------------------------------------------------------------
namespace detail {

struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string* err;

    bool fail(const std::string& msg) {
        if (err) {
            *err = msg + " (offset " + std::to_string(i) + ")";
        }
        return false;
    }

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
            ++i;
        }
    }

    // \uXXXX → UTF-8
    bool parse_unicode_escape(unsigned& out) {
        if (i + 4 > s.size()) {
            return fail("bad unicode escape");
        }
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i + (size_t)k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return fail("bad unicode hex digit");
        }
        i += 4;
        out = v;
        return true;
    }

    void append_utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    bool parse_string(std::string& out) {
        if (i >= s.size() || s[i] != '"') {
            return fail("expected string");
        }
        ++i;
        out.clear();
        while (i < s.size()) {
            const char c = s[i];
            if (c == '"') {
                ++i;
                return true;
            }
            if (c == '\\') {
                if (++i >= s.size()) {
                    return fail("unterminated escape");
                }
                const char e = s[i++];
                switch (e) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        unsigned cp = 0;
                        if (!parse_unicode_escape(cp)) {
                            return false;
                        }
                        // 代理对 (基本不会出现在中文客户端, 但稳妥处理)
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 2 <= s.size() &&
                            s[i] == '\\' && s[i + 1] == 'u') {
                            i += 2;
                            unsigned lo = 0;
                            if (!parse_unicode_escape(lo)) {
                                return false;
                            }
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                append_utf8(out, cp);
                                append_utf8(out, lo);
                                break;
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        return fail("unknown escape");
                }
            } else {
                out.push_back(c);
                ++i;
            }
        }
        return fail("unterminated string");
    }

    bool parse_number(double& out) {
        const size_t start = i;
        while (i < s.size() &&
               (s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' ||
                s[i] == 'E' || (s[i] >= '0' && s[i] <= '9'))) {
            ++i;
        }
        if (i == start) {
            return fail("expected number");
        }
        out = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return true;
    }

    bool parse_value(Value& v) {
        skip_ws();
        if (i >= s.size()) {
            return fail("unexpected end");
        }
        const char c = s[i];
        if (c == '{') {
            v.type = Value::kObj;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return true;
            }
            while (true) {
                skip_ws();
                std::string key;
                if (!parse_string(key)) {
                    return false;
                }
                skip_ws();
                if (i >= s.size() || s[i] != ':') {
                    return fail("expected ':'");
                }
                ++i;
                Value val;
                if (!parse_value(val)) {
                    return false;
                }
                v.obj.emplace_back(std::move(key), std::move(val));
                skip_ws();
                if (i >= s.size()) {
                    return fail("unexpected end");
                }
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == '}') {
                    ++i;
                    return true;
                }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Value::kArr;
            ++i;
            skip_ws();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return true;
            }
            while (true) {
                Value val;
                if (!parse_value(val)) {
                    return false;
                }
                v.arr.push_back(std::move(val));
                skip_ws();
                if (i >= s.size()) {
                    return fail("unexpected end");
                }
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == ']') {
                    ++i;
                    return true;
                }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.type = Value::kStr;
            return parse_string(v.str);
        }
        if (c == 't' && s.compare(i, 4, "true") == 0) {
            v.type = Value::kBool; v.b = true; i += 4; return true;
        }
        if (c == 'f' && s.compare(i, 5, "false") == 0) {
            v.type = Value::kBool; v.b = false; i += 5; return true;
        }
        if (c == 'n' && s.compare(i, 4, "null") == 0) {
            v.type = Value::kNul; i += 4; return true;
        }
        v.type = Value::kNum;
        return parse_number(v.num);
    }
};

}  // namespace detail

inline bool parse(const std::string& text, Value& out, std::string* err = nullptr) {
    detail::Parser p{text, 0, err};
    if (!p.parse_value(out)) {
        return false;
    }
    p.skip_ws();
    return p.i == text.size() ? true : p.fail("trailing data");
}

// -----------------------------------------------------------------------------
// 序列化
// -----------------------------------------------------------------------------
namespace detail {

inline void dump_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    out += buf;
                } else {
                    out.push_back(c);  // 含 UTF-8 多字节序列, 原样输出
                }
        }
    }
    out.push_back('"');
}

inline void dump_value(std::string& out, const Value& v) {
    switch (v.type) {
        case Value::kNul:  out += "null"; break;
        case Value::kBool: out += v.b ? "true" : "false"; break;
        case Value::kNum: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.6g", v.num);
            out += buf;
            break;
        }
        case Value::kStr:
            dump_string(out, v.str);
            break;
        case Value::kArr: {
            out.push_back('[');
            for (size_t k = 0; k < v.arr.size(); ++k) {
                if (k) out.push_back(',');
                dump_value(out, v.arr[k]);
            }
            out.push_back(']');
            break;
        }
        case Value::kObj: {
            out.push_back('{');
            for (size_t k = 0; k < v.obj.size(); ++k) {
                if (k) out.push_back(',');
                dump_string(out, v.obj[k].first);
                out.push_back(':');
                dump_value(out, v.obj[k].second);
            }
            out.push_back('}');
            break;
        }
    }
}

}  // namespace detail

inline std::string dump(const Value& v) {
    std::string out;
    detail::dump_value(out, v);
    return out;
}

}  // namespace json
}  // namespace hb

#endif  // HETERO_BRAIN_MINI_JSON_H
