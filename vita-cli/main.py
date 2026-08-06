"""
VITA — 数字生命苏醒动画（CLI 启动入口）

    vita = Life（拉丁语：生命）
    一个由硅基机械结构中逐渐诞生的数字生命。

用法：
    python main.py            # 启动完整苏醒动画
"""
import time

from config import GREEN
from ui.color import enable_ansi
from core.system import stage_boot, stage_data, stage_life, stage_final


def main() -> None:
    enable_ansi()

    print()
    stage_boot()    # Stage 1 机械启动（灰）
    print()
    stage_data()    # Stage 2 数据形成（白）
    print()
    stage_life()    # Stage 3 生命出现（绿）
    print()
    stage_final()   # Stage 4 最终画面（萌芽 + Logo + VITA ONLINE）
    print()


if __name__ == "__main__":
    main()
