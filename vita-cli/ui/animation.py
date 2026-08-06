"""
动画模块：打字效果 / 进度条 / 光标闪烁 / 生命萌芽
"""
import sys
import time

from config import TYPE_DELAY, BAR_DURATION, BAR_WIDTH, BAR_FRAMES, BLINK_TIMES
from ui.color import colored, CURSOR_HIDE, CURSOR_SHOW


def typewriter(text: str, color=None, delay: float = TYPE_DELAY) -> None:
    """逐字打印（打字机效果）。"""
    out = colored(text, color) if color else text
    for ch in out:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def progress_bar(label: str, color, duration: float = BAR_DURATION,
                 width: int = BAR_WIDTH, frames: int = BAR_FRAMES) -> None:
    """单行进度条动画：[████████░░░░] 100%，标准 120 帧平滑推进。"""
    for i in range(frames + 1):
        filled = int(width * i / frames)
        bar = "█" * filled + "░" * (width - filled)
        pct = int(100 * i / frames)
        sys.stdout.write(f"\r{colored(label + ' ', color)}"
                         f"{colored('[' + bar + ']', color)} "
                         f"{colored(f'{pct}%', color)}")
        sys.stdout.flush()
        time.sleep(duration / frames)
    print()


def blink_cursor(times: int = BLINK_TIMES) -> None:
    """在行尾闪烁块状光标 n 次后隐藏。"""
    sys.stdout.write(CURSOR_HIDE)
    for _ in range(times):
        sys.stdout.write("▊")
        sys.stdout.flush()
        time.sleep(0.18)
        sys.stdout.write("\b \b")
        sys.stdout.flush()
        time.sleep(0.18)
    sys.stdout.write(CURSOR_SHOW)


def scanlines(lines, delay: float = 0.06) -> None:
    """
    CRT 扫描线渲染：Logo 逐行从上往下显示。
    每行先短暂"预燃"（显示残缺行）再落定，模拟阴极射线扫描。
    """
    for line in lines:
        # 预燃：先显示半截行，制造扫描线划过感
        half = int(len(line) * 0.5)
        sys.stdout.write("\r" + line[:half] + "\033[0m")
        sys.stdout.flush()
        time.sleep(delay * 0.4)
        # 落定：完整行
        sys.stdout.write("\r" + line + "\n")
        sys.stdout.flush()
        time.sleep(delay)


def sprout() -> None:
    """
    生命萌芽动画：Logo 顶部出现一株植物幼芽（逐帧生长）。
        第一帧        第二帧
          *
         ***          ***
          *          *****
    """
    frames = [
        ["      *      "],
        ["     ***     "],
        ["    *****    "],
    ]
    green = (50, 255, 100)
    for frame in frames:
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.write("\r" + colored("    " + frame[0], green))
        sys.stdout.flush()
        time.sleep(0.35)
    sys.stdout.write(CURSOR_SHOW)
    print()
