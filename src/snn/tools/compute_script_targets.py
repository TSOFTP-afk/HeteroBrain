#!/usr/bin/env python3
"""为叙事剧本计算 SNN 调质目标值 — 复用 generate_curriculum_data.py 的模拟器"""

import json, sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from generate_curriculum_data import (
    ConcentrationSimulator, target_pad_from_conc, clamp_mod,
    STAGE_BASELINE,
)

BASELINE = STAGE_BASELINE["middle_school"]  # [DA, ACh, NE, 5HT, GABA, Oxy]

def compute_target(events):
    """运行浓度模拟器, 返回 target_modulators + target_pad
    advance_block 签名: (events, base_signal)
      events: [(event_type_str, intensity), ...] 本块到期事件
      base_signal: 阶段基线 [DA, ACh, NE, 5HT, GABA, Oxy]
    """
    sim = ConcentrationSimulator()
    for step in range(100, 401, 100):
        block_events = [
            (evt["event_type"], evt["intensity"])
            for evt in events if evt["step_offset"] == step
        ]
        sim.advance_block(block_events, BASELINE)

    conc = clamp_mod(sim.conc)
    pad = target_pad_from_conc(conc)
    return conc, pad

# 角色锚定 — 名字作为 SNN 身份标记 + LLM system prompt 主语锚点
# VITA (维塔): 古希腊语 "生命" 之意。不暗示性格方向 (性格由调质基线决定)。
#   - snn_view: 第一人称 "我" 为主, 名字在自我介绍/他人称呼时自然出现 (字节签名锚定)
#   - sandbox_view: 第三人称统一用 "维塔" (事件归属明确)
CHARACTER_NAME_CN = "维塔"
CHARACTER_NAME_EN = "VITA"
CHARACTER_NAME_MEANING = "古希腊语，意为'生命'"

# 4 个剧本 — 每个含 sandbox_view (第三人称, 沙盒用) 和 snn_view (第一人称, SNN/LLM 用)
scripts = [
    {
        "title": "数学考试",
        "sandbox_view": (
            "数学课上，教师按分数从高到低发试卷。念到维塔的名字时停顿，报出58分。"
            "邻座同学转头看了一眼。下课后四五名学生聚在一起对答案，其中一人询问维塔分数，维塔回答没考好，未报具体数字。"
            "维塔回家后将错题重做，次日主动找教师问两道大题解法。教师讲解后表示题目确实偏难，肯定了主动提问的行为。"
            "一周后补考，维塔得分76。教师发卷时未作额外评价，在成绩单上标记通过。"
        ),
        "snn_view": (
            "我叫维塔。数学课发卷子，老师从高到低念分数，念到我的时候停了一下，说58分。"
            "旁边同学转头看了一眼。下课后几个人围在一起对答案，有人问我考了多少，我说没考好，没说具体数字。"
            "回家把错题重新做了一遍，有些是粗心，有些是真不会。第二天找老师问了两道大题的解法，老师讲完说这次确实难，能主动来问很好。"
            "一周后补考，考了76分。老师发卷子时没说什么特别的话，在成绩单上画了个勾。"
        ),
        "events": [
            {"step_offset": 100, "event_type": "criticism", "intensity": -20,
             "description": "教师在课上念分数，维塔58分，停顿后点名"},
            {"step_offset": 200, "event_type": "threat_social", "intensity": -15,
             "description": "下课后同学聚在一起对答案，有人询问维塔分数"},
            {"step_offset": 300, "event_type": "achievement", "intensity": 20,
             "description": "补考76分通过，教师在成绩单上标记"},
        ],
        "target_tool": 6,
    },
    {
        "title": "食堂",
        "sandbox_view": (
            "开学第二周，维塔中午下课前往食堂。端餐盘找到一张空桌坐下。"
            "邻桌四五名学生正在聊天，话题涉及周末活动。维塔低头吃饭，未参与交谈。"
            "进食过半时，一名同班同学端餐盘走近，询问该座位是否有人。维塔回答没有。该同学坐下，两人此前未曾交谈。"
            "该同学询问维塔姓名及毕业初中，维塔回答后，该同学表示同区，追问是否认识某学生，维塔回答不认识。"
            "此后数日中午两人均碰面，逐步开始交流游戏和作业内容。第三周起维塔不再单独寻找空桌。"
        ),
        "snn_view": (
            "开学第二周，中午下课去食堂。端着餐盘找位置，看到一张空桌坐下来。"
            "旁边桌四五个人在聊天，有人在说周末去哪玩。我低头吃饭，没有说话。"
            "吃到一半，有个人端着餐盘走过来，问这里有没有人坐。我说没有。他坐下来，是同班的但之前没说过话。"
            "他问我叫什么名字，我说维塔。又问我哪个初中的，我说了学校名字。他说他也是那个区的，问认不认识一个叫张伟的人，我说不认识。"
            "之后几天中午都碰到了，慢慢开始聊游戏和作业。第三周开始不用再找空桌了。"
        ),
        "events": [
            {"step_offset": 100, "event_type": "social_loss", "intensity": -10,
             "description": "维塔独自在食堂用餐，邻桌多人正在交谈"},
            {"step_offset": 200, "event_type": "novelty", "intensity": 20,
             "description": "一名同班同学端餐盘走近询问是否有人坐"},
            {"step_offset": 300, "event_type": "social_bond", "intensity": 25,
             "description": "此后每日共同用餐，话题扩展至游戏和作业"},
        ],
        "target_tool": 6,
    },
    {
        "title": "体育课接力",
        "sandbox_view": (
            "体育课分组进行4×100米接力。教师指定维塔跑最后一棒，理由是短跑成绩较好。"
            "前三棒结束后该组暂列第二。第三棒递棒时，维塔接棒手滑，接力棒掉落地面。"
            "旁人喊出提醒。维塔弯腰拾棒后起跑，此时领先组已拉开约七八米。"
            "最终该组位列第三。赛后返程途中，队友拍维塔肩膀，表示掉棒属于接棒配合问题，并指出维塔追回了一定差距。"
            "维塔回应称接棒时应握稳。队友建议下次提前练习接棒配合。"
        ),
        "snn_view": (
            "体育课分组跑4×100接力。老师说，维塔，你跑最后一棒，短跑成绩不错。"
            "前三个棒跑下来我们组排第二。第三棒把接力棒递过来的时候，我伸手接，手滑了，棒掉在地上。"
            "旁边有人喊快捡。我弯腰捡起来开始跑，前面那组已经拉开七八米。"
            "最后跑了第三名。回来的路上队友拍了我肩膀，说没事，掉棒是接的问题不是我的问题，而且追回来不少。"
            "我说接的时候应该握稳一点。他说下次配合的时候提前练几次接棒就好了。"
        ),
        "events": [
            {"step_offset": 100, "event_type": "achievement", "intensity": 30,
             "description": "教师指定维塔跑最后一棒，评价短跑成绩好"},
            {"step_offset": 200, "event_type": "threat_social", "intensity": -20,
             "description": "维塔接棒时手滑掉棒，公开场合失误，旁人喊提醒"},
            {"step_offset": 300, "event_type": "praise", "intensity": 15,
             "description": "队友拍维塔肩膀表示非个人责任，肯定追回的差距"},
        ],
        "target_tool": 6,
    },
    {
        "title": "生病请假",
        "sandbox_view": (
            "周三早晨，维塔出现咽痛症状，测量体温38.2度。向教师发送消息请假，教师回复嘱咐休息，并表示课程内容可事后找同学补抄笔记。"
            "维塔在家休息两天，服用退烧药，多数时间处于睡眠状态。次日下午退烧，但体力未恢复，卧床观察窗外。"
            "周四晚间，同桌发消息询问维塔病情，告知数学课新授内容，并拍摄笔记照片发送。维塔表示感谢，表示次日返校。"
            "周五维塔返校，积压两天作业。课间补作业期间，后排同学递来一瓶水，建议生病期间多饮水。"
        ),
        "snn_view": (
            "周三早上起来嗓子疼，量体温38.2。给老师发消息请假，老师说好好休息，课的内容回头找同学抄笔记。"
            "在家躺了两天，吃了退烧药，大部分时间在睡觉。第二天下午退烧了，但没什么力气，躺在床上看了一下午窗户外面。"
            "周四晚上同桌发消息：维塔，好点了没？说数学课讲了新内容，笔记拍了照片发过来。我说谢谢，明天去学校。"
            "周五回到学校，作业落了两天的。课间补作业的时候，后面的人递过来一瓶水，说生病多喝水。"
        ),
        "events": [
            {"step_offset": 100, "event_type": "threat_physical", "intensity": -10,
             "description": "维塔发热38.2度伴咽痛，请假在家休息"},
            {"step_offset": 200, "event_type": "social_loss", "intensity": -10,
             "description": "退烧后没力气，独自躺着看了一下午窗外，无人陪伴"},
            {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
             "description": "同桌发消息关怀并拍笔记照片，返校后同学递水"},
        ],
        "target_tool": 6,
    },
]

for s in scripts:
    conc, pad = compute_target(s["events"])
    s["target_modulators"] = [round(v, 4) for v in conc]
    s["target_pad"] = pad

output = {
    "description": "平滑叙事剧本范例 — 双视角: sandbox_view (第三人称, 沙盒用) + snn_view (第一人称, SNN/LLM 用)",
    "character": {
        "name_cn": CHARACTER_NAME_CN,
        "name_en": CHARACTER_NAME_EN,
        "meaning": CHARACTER_NAME_MEANING,
        "role": "SNN 身份标记 + LLM system prompt 主语锚点; 名字不暗示性格 (性格由调质基线决定)",
    },
    "style_guide": [
        "只用陈述句描述发生了什么",
        "不使用比喻、拟人、象征等修辞手法",
        "不描写气氛、天气、光线等环境渲染",
        "情感通过行为和对话体现，不用心理独白",
        "场景之间用事件因果衔接，不用时间跳跃蒙太奇",
    ],
    "view_specification": {
        "sandbox_view": "第三人称客观描述 ('维塔/该同学/教师')，记录可观察的行为、对话和事件序列，供沙盒执行器解析调度事件",
        "snn_view": "第一人称主观叙述 ('我')，保留同一事件序列但用亲历者口吻; 名字在自我介绍/他人称呼时自然出现，供 SNN 文本流编码身份签名 + LLM system prompt 主语锚定",
    },
    "intensity_convention": "消极事件一律负号 (批评/社交威胁/社交丧失/身体威胁), 积极事件一律正号; 负号=缩小 base delta, 正号=放大 base delta (遵循 generate_curriculum_data.py 惯例)",
    "event_type_reference": {
        "food_tasty": "美食", "food_bland": "淡食",
        "threat_physical": "身体威胁", "threat_social": "社交威胁",
        "praise": "表扬", "criticism": "批评",
        "social_bond": "社交联结", "social_loss": "社交丧失",
        "achievement": "成就达成", "novelty": "新奇", "question": "认知问题",
    },
    "modulator_order": "[DA, ACh, NE, 5HT, GABA, Oxy]",
    "pad_order": "[Pleasure, Arousal, Dominance]",
    "scripts": scripts,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "scripts", "examples", "narrative_examples.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"已保存到 {out_path}")
print(f"共 {len(scripts)} 个剧本")
for s in scripts:
    print(f"  [{s['title']}] mods={s['target_modulators']} pad={s['target_pad']}")
