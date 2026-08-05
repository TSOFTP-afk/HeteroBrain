# -*- coding: utf-8 -*-
"""导出场景库为可读中文清单 (out/scenes_list.md)

用法: python export_scenes.py [--output out/scenes_list.md]
"""
import argparse
import json
import os
from collections import Counter, defaultdict

from engine.scene_builder import LEVELS, RELATIONS, THEMES, TONE_CN

BASE = os.path.dirname(os.path.abspath(__file__))

REL_CN = {rid: cn for rid, cn, _ in RELATIONS}
LVL_CN = {lid: cn for lid, cn, _ in LEVELS}
THEME_NAME = {}
for (dom, theme), td in THEMES.items():
    THEME_NAME[(dom, theme)] = td["name"]

EVT_CN = {"novelty": "新奇", "achievement": "成就", "praise": "表扬",
          "social_bond": "社交联结", "question": "提问",
          "threat_social": "社交威胁", "social_loss": "社交损失", "criticism": "批评",
          "food_tasty": "美食", "food_bland": "平淡", "threat_physical": "身体威胁"}
TOOL_CN = {0: "生成器", 1: "计算器", 2: "草稿记录", 3: "记忆检索",
           4: "知识查询", 5: "时钟", 6: "不调用"}
DOMAIN_CN = {"academic": "学业", "social": "社交", "family": "家庭",
             "interest": "兴趣", "daily": "日常", "health": "健康",
             "environment": "环境", "self": "自我"}


def slot_text(slot) -> str:
    parts = []
    for cand in slot["candidates"]:
        lo, hi = cand["intensity_range"]
        parts.append(f'{EVT_CN.get(cand["type"], cand["type"])}({lo}~{hi})')
    return "/".join(parts)


def scene_line(s) -> str:
    slots = " → ".join(f'{slot["role"]}=[{slot_text(slot)}]' for slot in s["events"])
    tool = TOOL_CN.get(s["tool"], "?")
    return (f'- {REL_CN[s["relation"]]}情境·{THEME_NAME[(s["domain"], s["theme"])]}·'
            f'{TONE_CN[s["tone"]]}({LVL_CN[s["level"]]}): {slots} | 工具={tool}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", default=os.path.join(BASE, "out", "scenes_list.md"))
    args = ap.parse_args()

    with open(os.path.join(BASE, "kb", "scenes.json"), encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]

    L = []
    L.append("# 学生生活场景库总览\n")
    L.append(f"**共 {len(scenes)} 个场景模板**\n")
    L.append(f"维度: 领域({len(DOMAIN_CN)}) × 主题({len(THEMES)}) × 关系({len(RELATIONS)}) × "
             f"强度档({len(LEVELS)}) × 基调({len(TONE_CN)})\n")

    for stage in ("middle_school", "high_school"):
        ss = [s for s in scenes if s["stage"] == stage]
        L.append(f"\n## {'初中' if stage == 'middle_school' else '高中'}（{len(ss)} 个场景）\n")
        by_dom = defaultdict(list)
        for s in ss:
            by_dom[s["domain"]].append(s)
        for dom in sorted(by_dom, key=lambda d: DOMAIN_CN.get(d, d)):
            L.append(f"\n### {DOMAIN_CN.get(dom, dom)}（{len(by_dom[dom])} 个）\n")
            by_theme = defaultdict(list)
            for s in by_dom[dom]:
                by_theme[s["theme"]].append(s)
            for theme in sorted(by_theme, key=lambda t: THEME_NAME.get((dom, t), t)):
                ts = by_theme[theme]
                tname = THEME_NAME.get((dom, theme), theme)
                tools = {TOOL_CN.get(t["tool"], "?") for t in ts}
                tool_str = next(iter(tools)) if len(tools) == 1 else "多种"
                L.append(f"\n**{tname}**（{len(ts)} 个，工具={tool_str}）\n")
                by_tone = defaultdict(list)
                for s in ts:
                    by_tone[s["tone"]].append(s)
                for tone in ("pos", "mixed", "neg"):
                    if tone not in by_tone:
                        continue
                    L.append(f"- 【{TONE_CN[tone]}】\n")
                    for s in by_tone[tone]:
                        L.append(scene_line(s) + "\n")

    # 统计
    L.append("\n---\n## 统计\n")
    L.append(f"- 场景总数: {len(scenes)}\n")
    for stage in ("middle_school", "high_school"):
        ss = [s for s in scenes if s["stage"] == stage]
        L.append(f"- {stage}: {len(ss)}\n")
    L.append("- 按领域: ")
    cnt = Counter(DOMAIN_CN.get(s["domain"], s["domain"]) for s in scenes)
    L.append("、".join(f"{k}={v}" for k, v in sorted(cnt.items())) + "\n")
    L.append("- 按基调: ")
    cnt = Counter(TONE_CN.get(s["tone"], s["tone"]) for s in scenes)
    L.append("、".join(f"{k}={v}" for k, v in sorted(cnt.items())) + "\n")
    L.append("- 按关系: ")
    cnt = Counter(REL_CN.get(s["relation"], s["relation"]) for s in scenes)
    L.append("、".join(f"{k}={v}" for k, v in sorted(cnt.items())) + "\n")
    L.append("- 按强度档: ")
    cnt = Counter(LVL_CN.get(s["level"], s["level"]) for s in scenes)
    L.append("、".join(f"{k}={v}" for k, v in sorted(cnt.items())) + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"场景清单已导出: {args.output} ({len(scenes)} 个场景, {len(L)} 行)")


if __name__ == "__main__":
    main()
