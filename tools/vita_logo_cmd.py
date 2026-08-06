import ctypes
import sys

kernel32 = ctypes.windll.kernel32
h_out = kernel32.GetStdHandle(-11)

# 设置 UTF-8 输出，确保方块字符正确显示
kernel32.SetConsoleOutputCP(65001)
sys.stdout.reconfigure(encoding='utf-8')

# 前景色
C_GRAY  = 0x08  # 灰色：硅基机械
C_WHITE = 0x0F  # 白色：过渡
C_GREEN = 0x0A  # 亮绿：生命萌发
C_DIM   = 0x07  # 默认

def color(c):
    kernel32.SetConsoleTextAttribute(h_out, c)

# 9x7 大字幅像素字体，笔画更粗、比例更端正
LETTERS = {
    'V': [
        "100000001",
        "100000001",
        "100000001",
        "100000001",
        "010000010",
        "001000100",
        "000111000",
    ],
    'I': [
        "111111111",
        "000010000",
        "000010000",
        "000010000",
        "000010000",
        "000010000",
        "111111111",
    ],
    'T': [
        "111111111",
        "000010000",
        "000010000",
        "000010000",
        "000010000",
        "000010000",
        "000010000",
    ],
    'A': [
        "000010000",
        "000111000",
        "001000100",
        "001000100",
        "011111110",
        "001000100",
        "001000100",
    ],
}

# 从上到下：绿(小) → 白(过渡) → 灰(大)
ROW_COLORS = [C_GREEN, C_WHITE, C_WHITE, C_GRAY, C_GRAY, C_GRAY, C_GRAY]

# 双宽像素块，让 CMD 里的像素接近正方形
BLOCK = '██'
SPACE = '  '

def draw_vita():
    for row in range(7):
        color(ROW_COLORS[row])
        line_parts = []
        for idx, ch in enumerate("VITA"):
            bits = LETTERS[ch][row]
            line_parts.append(''.join(BLOCK if b == '1' else SPACE for b in bits))
            if idx < 3:
                line_parts.append(SPACE * 4)
        print(''.join(line_parts))
    color(C_DIM)

if __name__ == '__main__':
    # 类似 llama.cpp 启动画面的 loading 文字
    color(C_GRAY)
    print("Loading model...")
    print()
    draw_vita()
    color(C_GRAY)
    print("\nvita_engine ready")
    color(C_DIM)
