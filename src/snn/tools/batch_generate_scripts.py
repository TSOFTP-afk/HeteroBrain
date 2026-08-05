#!/usr/bin/env python3
"""Qwen 批量化剧本生成器 — few-shot 引导 + 校验 + target 计算
============================================================
调用本地 heterobrain_engine serve (OpenAI 兼容端点), 用 few-shot 范例
引导 Qwen3-4B 生成符合规范的剧本段, 校验后计算调质 target。

用法:
  python batch_generate_scripts.py --arc 值日冲突 --n 5 \
      --out data/events/generated_arcs.jsonl

规范:
  - 事件类型: 11 种枚举 (food_tasty/food_bland/threat_physical/threat_social/
    praise/criticism/social_bond/social_loss/achievement/novelty/question)
  - 强度: -25~+25, 5 的倍数; 消极事件负号, 积极事件正号
  - step_offset: 100/200/300 (每段 3 个时间点, 可并行挂多个)
  - 叙事: 双视角 (snn_view 第一人称 + sandbox_view 第三人称),
    去心理化 (行为/环境/对话/生理), 平滑直白无文学修辞
"""

import json, os, sys, time, urllib.request, re

API = "http://127.0.0.1:8899/v1/chat/completions"
API_KEY = "thetrueai"
MODEL = "thetrueai"

sys.path.insert(0, os.path.dirname(__file__))
from longarc_stories import ARC_DUTY_CONFLICT, ARC_OLYMPIAD, ARC_FAMILY_STORM
from generate_serial_curriculum import simulate_segment, sanitize_first_person, BASELINE
from generate_curriculum_data import ConcentrationSimulator, target_pad_from_conc, clamp_mod

EVENT_TYPES_CN = {
    "food_tasty": "美食", "food_bland": "淡食",
    "threat_physical": "身体威胁(应激)", "threat_social": "社交威胁(当众难堪/压力)",
    "praise": "表扬", "criticism": "批评", "social_bond": "社交联结(友谊/温暖)",
    "social_loss": "社交丧失(孤独/失去)", "achievement": "成就达成",
    "novelty": "新奇", "question": "认知问题(思考/学习/疑问)",
}

# few-shot 范例: 1 负性段 + 1 正性段 + 1 竞争段 (覆盖多样弧线)
FEW_SHOT = [
    {
        "title": "同学不满",
        "time_label": "12月第2周 周三",
        "snn_view": "周三值日，李明扫到一半说这组排得不对，他上周扫过教室了。我说值日表是按名单顺序排的，不记得上周谁扫过。他说你当生活委员连这个都不知道。我站在那，没说话。旁边有人笑了一声。",
        "sandbox_view": "值日时李明对安排提出异议，称上周已扫过。维塔表示按名单顺序排列，未记录历史。李明当众质疑其履职，周围有笑声。",
        "events": [
            {"step_offset": 100, "event_type": "criticism", "intensity": -20,
             "description": "李明当众质疑值日安排不公"},
            {"step_offset": 200, "event_type": "threat_social", "intensity": -15,
             "description": "被当众说当生活委员不称职"},
            {"step_offset": 300, "event_type": "social_loss", "intensity": -10,
             "description": "被嘲笑后沉默"},
        ],
    },
    {
        "title": "和解",
        "time_label": "12月第4周 周五",
        "snn_view": "周五值日，李明扫完教室来跟我对值日表，说下周该轮他了，我记一下。我拿起笔记上。他说你这回记住就行。我们没再提上周的事。放学的时候，他那桌有人喊他一起走，他回头看了我一眼，没说什么。",
        "sandbox_view": "李明主动与维塔核对下周值日安排，维塔记录。两人关系回归正常。",
        "events": [
            {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
             "description": "李明主动核对值日安排"},
            {"step_offset": 200, "event_type": "achievement", "intensity": 10,
             "description": "值日安排确认无误"},
            {"step_offset": 300, "event_type": "social_bond", "intensity": 10,
             "description": "关系修复，回归正常"},
        ],
    },
    {
        "title": "教练批评",
        "time_label": "10月第4周 周三",
        "snn_view": "周三集训，老师让大家轮流上台讲题。轮到我，我在黑板上写了半道，卡住了。老师说，思路不完整，回去把这道题吃透。我走下讲台的时候，底下很安静。后半节课我没怎么听进去，盯着讲义上的字。",
        "sandbox_view": "集训课上维塔上台讲题中途卡住，教师当众指出思路不完整。维塔课后状态受挫。",
        "events": [
            {"step_offset": 100, "event_type": "criticism", "intensity": -20,
             "description": "讲题卡住被教练指出"},
            {"step_offset": 200, "event_type": "threat_social", "intensity": -15,
             "description": "当众受挫"},
            {"step_offset": 300, "event_type": "question", "intensity": 10,
             "description": "课后复盘题目"},
        ],
    },
]


def build_prompt(theme, prev_summary, segment_idx, generated_so_far=None):
    """构建 few-shot 生成 prompt (generated_so_far: 已生成段列表, 用于防重复)"""
    few_shot_text = "\n\n".join(
        json.dumps(ex, ensure_ascii=False, indent=1)
        for ex in FEW_SHOT)

    event_types_text = "\n".join(
        f"  {k}: {v}" for k, v in EVENT_TYPES_CN.items())

    done_text = ""
    if generated_so_far:
        lines = [
            f"  - {s.get('time_label', '?')}《{s.get('title', '?')}》: {s.get('snn_view', '')[:60]}……"
            for s in generated_so_far]
        done_text = ("【已写场景清单（新场景必须与以下全部场景不同：地点、事件、对话、人物互动都不得重复）】\n"
                     + "\n".join(lines) + "\n\n")

    prompt = f"""你是一名初中校园生活剧本作者，为 SNN 情感模拟系统生成剧本片段。

【写作规范】
1. 只用陈述句直白描述发生了什么，不用比喻、拟人、象征、气氛渲染等文学手法
2. snn_view 是第一人称"我"（角色名维塔，初中生）的视角，只写可观察行为、环境、对话、生理信号；
   不要写心理状态标签（如"我很紧张""我觉得高兴"），心理状态由系统推断
3. sandbox_view 是第三人称客观描述（用"维塔"）
4. 场景之间用事件因果衔接

【事件类型枚举】（每个事件必须用其一）
{event_types_text}

【事件规则】
- step_offset ∈ {{100, 200, 300}}（段内 3 个时间点，可同一时间点挂多个事件=同时发生）
- intensity ∈ [-25, +25]，5 的倍数；消极事件（批评/社交威胁/社交丧失/身体威胁）用负号，积极事件用正号
- description 是给沙盒日志的客观事件描述

【创作约束（必须遵守）】
- 必须创作全新场景：不能照抄范例内容（地点、对话、人物互动都要变），范例只是格式参考
- 每段至少包含 1 个与上一段不同的事件类型（避免重复同一事件）
- time_label 必须在上一段时间之后（时间线向前推进，不得倒退）
- 围绕给定主题展开，每段推进剧情（问题出现→加剧→缓和→解决→后果）
- 每段的事件组合要多样：不要每段都只有"成就+社交联结"，可以混入批评/社交威胁/丧失等负性事件
- 输出必须是一个完整闭合的 JSON 对象，以最后一个 }} 结尾，中间不得截断、不得省略任何字段

{done_text}【输出格式】只输出一个 JSON 对象，包含字段（注意字数限制）：
- title: 段名, 4-8 字
- time_label: 时间标记, 如"12月第3周 周三"（必须在上一段之后）
- snn_view: 第一人称"我"，60-100 字（只写行为/环境/对话/生理，不写心理标签）
- sandbox_view: 第三人称，30-60 字
- events: 数组，恰好 3 个事件（step_offset 分别为 100/200/300）

【范例】（只模仿格式与风格，不得复制内容）
{few_shot_text}

【本段任务】
故事主题：{theme}
当前是第 {segment_idx} 段。
上一段剧情摘要：{prev_summary or "（第一段，开场即可）"}

请生成下一段（全新场景，时间推进，与主题和上一段连贯）："""

    return prompt


def call_llm(prompt, max_tokens=2048, temperature=0.9):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {API_KEY}"})
    for attempt in range(3):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=180).read().decode("utf-8"))
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(3)


def extract_json(text):
    """从 LLM 输出提取 JSON (处理 <think> 前缀 / ```json 包裹 / 前后杂讯 / 语法错误)"""
    import ast
    # 去掉 think 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # 去 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    # 找第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise ValueError(f"无 JSON 内容: {text[:200]}")
    if end <= start:
        raise ValueError(f"输出被截断 (JSON 未闭合): ...{text[-200:]}")
    body = text[start:end + 1]

    # 容错链 1: 直接解析
    try:
        return json.loads(body)
    except Exception:
        pass

    # 容错链 2: 修复尾随逗号 (对象/数组末尾多余逗号)
    body2 = re.sub(r",\s*([\]}])", r"\1", body)
    try:
        return json.loads(body2)
    except Exception:
        pass

    # 容错链 3: 字符串外裸换行压平 + 缺失逗号补位
    #   缺失逗号常见形态: 值后紧跟下一键的引号且中间只有空白
    body3 = body2
    body3 = re.sub(r'(\})\s+("(?:[^"\\]|\\.)*")\s*:', r'\1,\2:', body3)  # } "k":
    body3 = re.sub(r'(])\s+("(?:[^"\\]|\\.)*")\s*:', r'\1,\2:', body3)  # ] "k":
    body3 = re.sub(r'("(?:[^"\\]|\\.)*")\s+("(?:[^"\\]|\\.)*")\s*:', r'\1,\2:', body3)  # "v" "k":
    try:
        return json.loads(body3)
    except Exception:
        pass

    # 容错链 4: ast.literal_eval (Python 语法宽容: 单引号/尾随逗号/True-False)
    try:
        return ast.literal_eval(body2)
    except Exception as e4:
        raise ValueError(f"JSON 解析失败: {e4} | 片段: {body[:300]}")


TIME_RE = re.compile(r"(\d{1,2})月第(\d)周")


def time_key(label):
    """时间标签 → 排序键 (月, 周); 无法解析返回 None"""
    m = TIME_RE.search(label or "")
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def is_duplicate(seg, samples):
    """内容去重: snn_view 字符二元组 Jaccard 重叠 > 0.45 视为重复"""
    def ngrams(s, n=2):
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    a = ngrams(seg.get("snn_view", ""))
    if not a:
        return False
    for s in samples:
        b = ngrams(s.get("snn_view", ""))
        inter = len(a & b)
        if inter / min(len(a), len(b)) > 0.45:
            return True
    return False


VALID_EVENTS = set(EVENT_TYPES_CN.keys())
PSYCH_WORDS = ["紧张", "高兴", "低落", "难过", "害怕", "担心", "忐忑", "心情",
               "觉得", "心里", "踏实", "难受", "不安", "激动", "兴奋", "怕"]


def validate_and_fix(seg):
    """校验并修正字段 (事件类型/强度/offset/去心理化)"""
    seg = dict(seg)
    seg["events"] = [dict(e) for e in seg.get("events", [])]
    # 事件类型白名单
    seg["events"] = [e for e in seg["events"] if e.get("event_type") in VALID_EVENTS]
    if not seg["events"]:
        raise ValueError("无合法事件")
    # 强度: 5 的倍数, |.|<=25; 符号按事件极性 (负性事件负号)
    neg_types = {"criticism", "threat_social", "social_loss", "threat_physical"}
    for e in seg["events"]:
        try:
            v = int(e.get("intensity", 0))
        except (TypeError, ValueError):
            v = 0
        v = max(-25, min(25, round(v / 5) * 5))
        if e["event_type"] in neg_types:
            e["intensity"] = -abs(v) if v != 0 else -5
        else:
            e["intensity"] = abs(v) if v != 0 else 5
        e["step_offset"] = int(e.get("step_offset", 100))
        if e["step_offset"] not in (100, 200, 300):
            e["step_offset"] = 100
        e.setdefault("description", "")
    # 双视角必填
    seg.setdefault("snn_view", "")
    seg.setdefault("sandbox_view", "")
    seg.setdefault("title", "片段")
    seg.setdefault("time_label", "")
    # 去心理化 (复用净化器)
    seg["snn_view"] = sanitize_first_person(seg["snn_view"])
    return seg


def compute_target(seg):
    """用连续模拟计算 target (孤立段, 从 baseline 起算)"""
    sim = ConcentrationSimulator()
    conc = simulate_segment(sim, seg["events"])
    return [round(v, 4) for v in conc], target_pad_from_conc(conc)


def main():
    out_path = "data/events/generated_arcs.jsonl"
    theme = "班级值日冲突"
    n_segments = 3
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--arc" and i + 1 < len(args):
            theme, i = args[i + 1], i + 2
        elif args[i] == "--n" and i + 1 < len(args):
            n_segments, i = int(args[i + 1]), i + 2
        elif args[i] == "--out" and i + 1 < len(args):
            out_path, i = args[i + 1], i + 2
        else:
            i += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prev_summary = ""
    prev_time = None
    samples = []
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in range(1, n_segments + 1):
            ok = False
            last_err = ""
            for attempt in range(1, 4):
                prompt = build_prompt(theme, prev_summary, idx, samples)
                if attempt > 1 and last_err:
                    prompt += (f"\n\n注意：上一版输出被拒绝，原因：{last_err}。"
                               f"请重新生成，严格输出合法 JSON。")
                print(f"[{idx}/{n_segments}] 生成中 (尝试 {attempt}/3)...", flush=True)
                try:
                    # 重试时变化温度 (固定 seed 下同参数请求输出完全相同, 重试必须扰动)
                    temp = 0.9 if attempt == 1 else (0.8 if attempt == 2 else 0.95)
                    raw = call_llm(prompt, temperature=temp)
                    seg = extract_json(raw)
                    seg = validate_and_fix(seg)
                    # 时间推进校验 (第一段不强制, 之后必须严格递增)
                    k = time_key(seg.get("time_label", ""))
                    if prev_time is not None and (k is None or k <= prev_time):
                        raise ValueError(
                            f"时间线未推进: '{seg.get('time_label')}' 不晚于上一段")
                    # 内容去重
                    if is_duplicate(seg, samples):
                        raise ValueError("内容与已生成段重复 (场景/对话雷同)")
                    mods, pad = compute_target(seg)
                    sample = {
                        "sample_id": idx,
                        "arc_id": f"generated_{theme}",
                        "segment_idx": idx - 1,
                        "prev_segment": idx - 2 if idx > 1 else None,
                        "title": seg["title"],
                        "time_label": seg["time_label"],
                        "snn_view": seg["snn_view"],
                        "sandbox_view": seg["sandbox_view"],
                        "events": seg["events"],
                        "target_modulators": mods,
                        "target_pad": pad,
                        "target_tool": 6,
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f.flush()
                    samples.append(sample)
                    prev_summary = seg["snn_view"][:80] + "……"
                    if k is not None:
                        prev_time = k
                    print(f"  ✓ 《{seg['title']}》({seg['time_label']}) "
                          f"{len(seg['events'])} 事件 Oxy={mods[5]:.3f}", flush=True)
                    ok = True
                    break
                except Exception as e:
                    last_err = str(e)
                    print(f"  ✗ {e}", flush=True)
                    time.sleep(2)
            if not ok:
                print(f"  ✗✗ 第 {idx} 段 3 次尝试均失败, 跳过", flush=True)

    print(f"\n完成: {len(samples)}/{n_segments} 段 → {out_path}")


if __name__ == "__main__":
    main()
