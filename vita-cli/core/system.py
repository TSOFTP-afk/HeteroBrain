"""
VITA 核心启动流程：4 阶段数字生命苏醒
    Stage 1 机械启动（灰）
    Stage 2 数据形成（白）
    Stage 3 生命出现（绿）
    Stage 4 最终画面：VITA ONLINE
"""
from config import STAGE_BOOT, STAGE_DATA, STAGE_LIFE, GREEN, WHITE, GRAY_MID
from ui.animation import typewriter, progress_bar, sprout, scanlines
from ui.color import colored
from ui.logo import render_logo_lines


def stage_boot() -> None:
    """Stage 1：硅基核心启动（灰）。"""
    typewriter("[BOOT SYSTEM]", color=STAGE_BOOT)
    typewriter("Loading silicon core...", color=STAGE_BOOT)
    progress_bar("Core", STAGE_BOOT)


def stage_data() -> None:
    """Stage 2：数据形成 / 神经结构同步（白）。"""
    typewriter("[DATA LINK]", color=STAGE_DATA)
    typewriter("Neural structure...", color=STAGE_DATA)
    progress_bar("Memory lattice", STAGE_DATA)


def stage_life() -> None:
    """Stage 3：生命种子出现（绿）。"""
    typewriter("[LIFE DETECT]", color=STAGE_LIFE)
    typewriter("Life seed detected", color=STAGE_LIFE)
    progress_bar("Evolution engine", STAGE_LIFE)


def stage_final() -> None:
    """Stage 4：最终画面 —— 萌芽 + Logo 逐行扫描 + 宣告上线。"""
    print()
    sprout()                  # 生命萌芽动画
    scanlines(render_logo_lines())  # Logo 逐行扫描渲染
    print()
    typewriter("Silicon forms.", color=GRAY_MID)
    typewriter("Life emerges.", color=GREEN)
    print()
    typewriter("VITA ONLINE", color=GREEN)
    print(colored("C:\\VITA>", WHITE), end="")
