"""
ANSI 颜色工具：真彩色（24-bit RGB）+ Windows 控制台 VT 支持
"""
import ctypes
import sys


def enable_ansi() -> None:
    """
    启用 Windows 控制台 VT 序列（TrueColor 支持）。

    Windows 10+ 默认关闭 ANSI 转义，需通过 SetConsoleMode 打开
    ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)。
    """
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    h_out = kernel32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(h_out, ctypes.byref(mode)):
        kernel32.SetConsoleMode(h_out, mode.value | 0x0004)
    # 保证 Unicode 方块字符正确显示
    kernel32.SetConsoleOutputCP(65001)


def rgb(r: int, g: int, b: int) -> str:
    """返回设置前景色为指定 RGB 的 ANSI 转义序列。"""
    return f"\033[38;2;{r};{g};{b}m"


RESET = "\033[0m"


def colored(text: str, color) -> str:
    """用指定 RGB 颜色包裹文本。color 为 (r, g, b) 三元组。"""
    return f"{rgb(*color)}{text}{RESET}"


# 光标控制
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"
CLEAR_LINE  = "\r\033[K"
