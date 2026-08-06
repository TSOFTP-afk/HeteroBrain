"""
VITA Logo：ASCII Block 大字 + 垂直渐变
    ███ 绿（顶，~10%）生命萌发
    ▓▓▓ 白（中，~20%）数据融合
    ▒▒▒ 灰（底，~70%）硅基机械
"""
from config import GREEN, WHITE, GRAY_LIGHT, GRAY_MID, GRAY_DEEP
from ui.color import colored

# 6 行 Logo（Box Drawing + Block 字符）
LOGO = [
    "██╗   ██╗██╗████████╗ █████╗",
    "██║   ██║██║╚══██╔══╝██╔══██╗",
    "██║   ██║██║   ██║   ███████║",
    "╚██╗ ██╔╝██║   ██║   ██╔══██║",
    " ╚████╔╝ ██║   ██║   ██║  ██║",
    "  ╚═══╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝",
]

# 每行颜色：绿(1) → 白(1) → 灰渐变(4)
_ROW_COLORS = [
    GREEN,
    WHITE,
    GRAY_LIGHT,
    GRAY_MID,
    GRAY_DEEP,
    GRAY_DEEP,
]


def render_logo() -> str:
    """渲染完整 Logo（含垂直渐变），返回多行字符串。"""
    return "\n".join(render_logo_lines())


def render_logo_lines():
    """渲染 Logo 的每一行（已上色），供逐行扫描动画使用。"""
    return [colored(line, _ROW_COLORS[i]) for i, line in enumerate(LOGO)]
