"""MiniCPM 本地推理客户端 (llama-cli 子进程, 零远程 API 费用)。

调用参数对齐 scripts/test_minicpm5_zh.bat (2026-07-27 验证通过:
  RTX 3060 -ngl 99, Generation 248 t/s):
  llama-cli -m <gguf> --chat-template-file <jinja> --jinja -f <prompt> -st
           -n <tokens> -ngl 99 -c 2048 --temp 0.7 --top-p 0.9 --top-k 40
           --repeat-penalty 1.1 --seed <seed> --no-warmup

职责 (用户定义): "简单的逻辑链条交给本地 MiniCPM" —
  给定学生生活场景, 让 MiniCPM 生成该情境的情感反应链 (事件序列),
  软件侧做 schema 校验 + 与 GENE_MAP 对齐, 不合格回退规则引擎。
"""
import json
import os
import re
import subprocess
import tempfile
from typing import List, Optional

# 默认路径 (可按环境覆盖)
DEFAULT_LLAMA_CLI = r"F:\thetrueai\build\bin\llama-cli.exe"
DEFAULT_MODEL = r"F:\thetrueai\models\MiniCPM5-1B-Q4_K_M.gguf"
DEFAULT_JINJA = r"F:\hb_models\minicpm5-chat.jinja"

VALID_EVENTS = ["novelty", "achievement", "praise", "social_bond", "question",
                "threat_social", "social_loss", "criticism",
                "food_tasty", "food_bland", "threat_physical"]

# 事件默认强度区间 (MiniCPM 只给事件类型序列时, 软件补确定性强度)
EVENT_DEFAULT_INTENSITY = {
    "novelty": 20, "achievement": 35, "praise": 25, "social_bond": 20,
    "question": 25, "threat_social": -25, "social_loss": -20, "criticism": -20,
    "food_tasty": 25, "food_bland": -5, "threat_physical": -30,
}


class MiniCPMClient:
    def __init__(self, cli: str = None, model: str = None, jinja: str = None,
                 max_tokens: int = 256, verbose: bool = False):
        self.cli = cli or DEFAULT_LLAMA_CLI
        self.model = model or DEFAULT_MODEL
        self.jinja = jinja or DEFAULT_JINJA
        self.max_tokens = max_tokens
        self.verbose = verbose
        for p in (self.cli, self.model, self.jinja):
            if not os.path.exists(p):
                raise FileNotFoundError(f"MiniCPM 依赖缺失: {p}")

    # ------------------------------------------------------------------
    def generate(self, prompt: str, seed: int = 42, max_tokens: int = None) -> str:
        """单轮生成: 返回模型最终回答 (剥离 prompt 回显/思考块/交互噪音)"""
        prompt_path = os.path.join(tempfile.gettempdir(), "cg_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8", newline="") as f:
            f.write(prompt)
        cmd = [
            self.cli, "-m", self.model,
            "--chat-template-file", self.jinja, "--jinja",
            "-f", prompt_path, "-st", "--no-display-prompt",
            "-rea", "off",  # 关闭 MiniCPM5 思考模式, 直接输出 JSON
            "-n", str(max_tokens or self.max_tokens),
            "-ngl", "99", "-c", "2048",
            "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
            "--repeat-penalty", "1.1", "--seed", str(seed), "--no-warmup",
        ]
        if self.verbose:
            print("[MiniCPM] " + " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=180,
                                  text=True, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return ""
        return self._extract_answer(proc.stdout, prompt)

    @staticmethod
    def _extract_answer(raw: str, prompt_text: str = "") -> str:
        """从 llama-cli 输出提取助手回答。

        llama-cli -f 文件单轮模式会在 "> " 提示符后回显 prompt 全文,
        随后是 [Start thinking] 思考块 + 最终回答 + 性能统计/退出提示。
        """
        # 1. 剥离回显的 prompt (已知文本精确匹配)
        if prompt_text and prompt_text in raw:
            idx = raw.find(prompt_text)
            raw = raw[:idx] + raw[idx + len(prompt_text):]
        # 2. 行过滤: 加载横幅 / ASCII 画 / 命令块 / 性能统计 / 退出提示
        skip_kw = ("Loading model", "modalities", "available commands",
                   "/exit", "/regen", "/clear", "/read", "/glob",
                   "Prompt:", "Generation:", "Exiting", "build", "ftype",
                   "model      :", "model :")
        lines = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            if any(k in s for k in skip_kw):
                continue
            if any(ch in s for ch in "▄▀█"):
                continue
            lines.append(s)
        raw = "\n".join(lines)
        # 3. 去思考块 [Start thinking] ... [End thinking] 或直至末尾
        raw = re.sub(r"\[Start thinking\][\s\S]*?\[/?End thinking\]",
                     "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\[Start thinking\][\s\S]*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\[/?End thinking\]", "", raw, flags=re.IGNORECASE)
        # 4. 去残留提示符
        raw = re.sub(r"^>\s*", "", raw, flags=re.MULTILINE)
        return raw.strip()

    # ------------------------------------------------------------------
    def generate_event_chain(self, scene_desc: str) -> Optional[List[dict]]:
        """MiniCPM 生成"场景 → 情感反应链" (简单逻辑链条)。

        返回结构化事件列表 [{event_type, intensity}] 或 None (解析/校验失败)。
        """
        prompt = (
            "你是一名学生心理与情感顾问。给定一个学生生活场景，按时间因果顺序"
            "推断该情境下学生可能经历的情感事件序列（2-4 个事件）。\n"
            "事件类型只允许使用以下 11 种之一：\n"
            "  novelty(新奇/初次经历), achievement(成就/成功), praise(表扬/认可), "
            "social_bond(社交联结/温暖), question(困惑/提问), "
            "threat_social(社交威胁/羞辱), social_loss(社交损失/失落), "
            "criticism(批评/责备), "
            "food_tasty(美食/味觉愉悦), food_bland(平淡乏味), threat_physical(身体不适/危险)\n"
            "【重要】不要思考过程，不要解释，不要任何多余文字，直接输出 JSON 数组。\n"
            "格式（示例）：\n"
            '["novelty", "achievement", "praise"]\n'
            f"场景：{scene_desc}\n"
        )
        raw = self.generate(prompt)
        if not raw:
            return None
        events = self._parse_chain(raw)
        if self.verbose and events is None:
            print(f"[MiniCPM] 链解析失败, 原文: {raw[:200]}")
        return events

    @staticmethod
    def _parse_chain(raw: str) -> Optional[List[dict]]:
        """解析 MiniCPM 输出的事件链。

        兼容两种格式 (1B 模型对严格 JSON 对象服从性弱, 放宽到字符串数组):
          a) 对象数组: [{"event_type": "achievement", "intensity": 35}, ...]
          b) 字符串数组: ["novelty", "achievement", "praise"] → 强度由软件补默认值
        """
        # 1. 整体解析
        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        # 2. 回退: 提取"最后一个"JSON 数组 (回答在输出末尾)
        if data is None:
            matches = list(re.finditer(r"\[.*?\]", raw, flags=re.DOTALL))
            for m in reversed(matches):
                try:
                    data = json.loads(m.group(0))
                    break
                except json.JSONDecodeError:
                    continue
        if not isinstance(data, list) or not (2 <= len(data) <= 4):
            return None
        events = []
        if all(isinstance(x, str) for x in data):          # 格式 b: 类型数组
            for t in data:
                if t not in VALID_EVENTS:
                    return None
                events.append({"event_type": t,
                               "intensity": EVENT_DEFAULT_INTENSITY[t]})
        elif all(isinstance(x, dict) for x in data):        # 格式 a: 对象数组
            for e in data:
                t = e.get("event_type", "")
                i = e.get("intensity")
                if t not in VALID_EVENTS or not isinstance(i, int) or not (-40 <= i <= 50):
                    return None
                events.append({"event_type": t, "intensity": i})
        else:
            return None
        return events
