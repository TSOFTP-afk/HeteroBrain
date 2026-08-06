// =============================================================================
// awakening.h — VITA 数字生命苏醒动画（原生 C++ 版）
// =============================================================================
// 复刻 vita-cli（Python）的 4 阶段苏醒动画，嵌入引擎原生启动流程：
//   Stage 1 机械启动（灰）  —— 硅基核心 boot
//   Stage 2 数据形成（白）  —— 神经结构数据同步
//   Stage 3 生命出现（绿）  —— 生命种子检测
//   Stage 4 最终画面        —— 萌芽 + VITA Logo 扫描 + VITA ONLINE
//
// 颜色垂直渐变（底部→顶部）：灰（硅基机械）→ 白（数据融合）→ 绿（生命萌发，只占 ~10%）
//
// 依赖：/utf-8 编译（vita_engine 已设置），Windows 控制台 VT 模式（函数内自动开启）
// =============================================================================

#ifndef VITA_AWAKENING_H
#define VITA_AWAKENING_H

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace vita {
namespace engine {

namespace awakening {

// ---------------- 颜色（RGB，与 vita-cli/config.py 一致） ----------------
struct Color {
    int r, g, b;
};

inline const Color kGreen      = {50, 255, 100};   // 顶部：生命萌发（必须少）
inline const Color kWhite      = {230, 230, 230};  // 中间：数据融合 / 意识形成
inline const Color kGrayDeep   = {90, 90, 90};     // 底部最深：纯机械
inline const Color kGrayMid    = {120, 120, 120};
inline const Color kGrayLight  = {150, 150, 150};  // 底部最浅：贴近白色过渡

inline const Color kStageBoot  = kGrayMid;         // Stage 1（灰）
inline const Color kStageData  = kWhite;           // Stage 2（白）
inline const Color kStageLife  = kGreen;           // Stage 3（绿）

// ---------------- ANSI 真彩色工具 ----------------
inline std::string rgb(const Color& c) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "\033[38;2;%d;%d;%dm", c.r, c.g, c.b);
    return std::string(buf);
}

inline std::string colored(const std::string& text, const Color& c) {
    return rgb(c) + text + "\033[0m";
}

// Windows 控制台启用 VT 序列（TrueColor）+ UTF-8 代码页
inline void enable_vt() {
#ifdef _WIN32
    const HANDLE h_out = ::GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (::GetConsoleMode(h_out, &mode)) {
        ::SetConsoleMode(h_out, mode | 0x0004);   // ENABLE_VIRTUAL_TERMINAL_PROCESSING
    }
    ::SetConsoleOutputCP(65001);
#endif
}

inline void sleep_sec(double s) {
    std::this_thread::sleep_for(std::chrono::duration<double>(s));
}

// ---------------- 动画子程序 ----------------
// 打字机效果：逐字符打印
inline void typewriter(const std::string& text, const Color& c,
                       double delay = 0.02) {
    const std::string out = colored(text, c);
    for (char ch : out) {
        std::fputc(ch, stdout);
        std::fflush(stdout);
        sleep_sec(delay);
    }
    std::fputc('\n', stdout);
}

// 单行进度条：[████████░░░░] 100%
inline void progress_bar(const std::string& label, const Color& c,
                         int width = 20, int frames = 120,
                         double duration = 1.0) {
    for (int i = 0; i <= frames; ++i) {
        const int filled = width * i / frames;
        // 注意: 必须用完整 UTF-8 字符串字面量 "█"/"░" (各 3 字节),
        // 不能用 char 字面量 '█' (多字符字面量会被截断成单字节 → 控制台乱码)
        std::string bar;
        for (int k = 0; k < filled; ++k) bar += "█";
        for (int k = filled; k < width; ++k) bar += "░";
        const int pct = 100 * i / frames;
        std::printf("\r%s%s %d%%%s",
                    colored(label + " ", c).c_str(),
                    colored("[" + bar + "]", c).c_str(),
                    pct,
                    "\033[0m");
        std::fflush(stdout);
        sleep_sec(duration / frames);
    }
    std::fputc('\n', stdout);
}

// 生命萌芽：Logo 顶部一株幼芽逐帧生长
inline void sprout() {
    const std::vector<std::string> frames = {
        "      *      ",
        "     ***     ",
        "    *****    ",
    };
    std::fputs("\033[?25l", stdout);   // 隐藏光标
    for (const auto& f : frames) {
        std::printf("\r%s", colored("    " + f, kGreen).c_str());
        std::fflush(stdout);
        sleep_sec(0.35);
    }
    std::fputs("\033[?25h", stdout);   // 显示光标
    std::fputc('\n', stdout);
}

// CRT 扫描线渲染：Logo 逐行从上往下显示（先"预燃"半截行再落定）
inline void scanlines(const std::vector<std::string>& lines, double delay = 0.06) {
    for (const auto& line : lines) {
        const size_t half = line.size() / 2;
        std::printf("\r%s\033[0m", line.substr(0, half).c_str());
        std::fflush(stdout);
        sleep_sec(delay * 0.4);
        std::printf("\r%s\n", line.c_str());
        std::fflush(stdout);
        sleep_sec(delay);
    }
}

// ---------------- VITA Logo（6 行 Block 大字 + 垂直渐变） ----------------
// 每行颜色：绿(1) → 白(1) → 灰渐变(4)
inline const std::vector<Color> kLogoRowColors = {
    kGreen, kWhite, kGrayLight, kGrayMid, kGrayDeep, kGrayDeep,
};

inline std::vector<std::string> render_logo_lines() {
    static const char* kLogo[6] = {
        "██╗   ██╗██╗████████╗ █████╗",
        "██║   ██║██║╚══██╔══╝██╔══██╗",
        "██║   ██║██║   ██║   ███████║",
        "╚██╗ ██╔╝██║   ██║   ██╔══██║",
        " ╚████╔╝ ██║   ██║   ██║  ██║",
        "  ╚═══╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝",
    };
    std::vector<std::string> out;
    out.reserve(6);
    for (int i = 0; i < 6; ++i) {
        out.push_back(colored(kLogo[i], kLogoRowColors[i]));
    }
    return out;
}

// ---------------- 4 阶段苏醒流程 ----------------
inline void stage_boot() {
    typewriter("[BOOT SYSTEM]", kStageBoot);
    typewriter("Loading silicon core...", kStageBoot);
    progress_bar("Core", kStageBoot);
}

inline void stage_data() {
    typewriter("[DATA LINK]", kStageData);
    typewriter("Neural structure...", kStageData);
    progress_bar("Memory lattice", kStageData);
}

inline void stage_life() {
    typewriter("[LIFE DETECT]", kStageLife);
    typewriter("Life seed detected", kStageLife);
    progress_bar("Evolution engine", kStageLife);
}

inline void stage_final() {
    std::fputc('\n', stdout);
    sprout();                                  // 生命萌芽动画
    scanlines(render_logo_lines());            // Logo 逐行扫描渲染
    std::fputc('\n', stdout);
    typewriter("Silicon forms.", kGrayMid);
    typewriter("Life emerges.", kGreen);
    std::fputc('\n', stdout);
    typewriter("VITA ONLINE", kGreen);
}

// 完整苏醒动画（引擎启动时调用）
inline void play_awakening() {
    enable_vt();
    std::fputc('\n', stdout);
    stage_boot();
    std::fputc('\n', stdout);
    stage_data();
    std::fputc('\n', stdout);
    stage_life();
    std::fputc('\n', stdout);
    stage_final();
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

}  // namespace awakening
}  // namespace engine
}  // namespace vida

#endif  // VITA_AWAKENING_H