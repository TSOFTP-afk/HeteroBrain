#!/usr/bin/env python3
"""长线剧本生成器 v2 — 50 段全连续叙事 (20K 步长, 最高规格)
============================================================
规格:
  - 50 段时间窗口 × 400 步 = 20,000 步 (一个学期: 9月开学 → 1月散学, 每段 ≈ 3-4 天)
  - 全部连续叙事: 段间浓度模拟器不 reset (远程因果, 慢通道残留传导)
  - 最高规格: 每段详细双视角叙事 (snn_view 第一人称 + sandbox_view 第三人称)
  - 并行事件: 同一 step_offset 多事件 = 同时发生
  - 事件与文本同源: --emit-text 把 50 段 snn_view 拼接成文本流文件

视角规范 (2026-08-05, 关键):
  SNN 的文本流输入 (snn_view) 必须"去心理化" — 只含可观察行为、环境、
  对话、生理信号; 不含心理状态标签 ("我紧张""我高兴""我觉得")。
  理由: 心理状态必须是 SNN 的输出 (调质/PAD 预测), 不能作为输入注入,
  否则 SNN 退化为"文本标签→调质"的查表器, 丧失情感自主性。
  净化由 sanitize_first_person() 执行 (PSYCH_REWRITES 规则表)。

用法:
  python generate_serial_curriculum.py                        # 输出 curriculum_long_arc_50.jsonl
  python generate_serial_curriculum.py --emit-text            # 同时输出净化后文本流 story_text_50.txt
  python generate_serial_curriculum.py --emit-raw             # 输出原始(含心理)文本流 story_text_50_raw.txt

输出:
  data/events/curriculum_long_arc_50.jsonl   (课程数据, CurriculumLoader 兼容)
  data/scripts/story_text_50.txt             (净化文本流, SNN 输入用)
  data/scripts/story_text_50_raw.txt         (原始文本流, 仅对比)
"""

import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_curriculum_data import (
    ConcentrationSimulator, target_pad_from_conc, clamp_mod,
    STAGE_BASELINE,
)

BASELINE = STAGE_BASELINE["middle_school"]  # [DA, ACh, NE, 5HT, GABA, Oxy]
STEPS_PER_BLOCK = 100


def simulate_segment(sim, events):
    """推进一个 400 步段 (4×100 块), 段末返回 conc (并行事件同块累加)"""
    for blk in range(4):
        rel = (blk + 1) * STEPS_PER_BLOCK
        block_events = [
            (evt["event_type"], evt["intensity"])
            for evt in events if evt["step_offset"] == rel
        ]
        sim.advance_block(block_events, BASELINE)
    return clamp_mod(sim.conc)


# =============================================================================
# 第一人称文本净化 (2026-08-05) — 剥离心理状态标签
# =============================================================================
# 原则: 心理状态 (紧张/高兴/低落/怕/觉得...) 必须是 SNN 的输出, 不能是输入。
# 规则表: 精确句子级替换, 长句优先 (str.replace 顺序执行)。
#   心理标签 → 可观察行为 / 环境 / 对话 / 生理信号。
PSYCH_REWRITES = [
    # 段49 等成绩: 情绪评估 → 具体行为 (算分次数与结果)
    ("考完等成绩的两天，比考试本身还难受。周二晚上我算了三遍数学，"
     "觉得自己能考80上下，但又怕算错。周三白天上课，老师没提成绩。"
     "晚上我写作业的时候，脑子里老是有个声音在算分。",
     "考完等成绩的两天，周二晚上我算了三遍数学，一遍80，一遍78，一遍81。"
     "周三白天上课，老师没提成绩。晚上写作业的时候，我又算了一遍分。"),
    # 段13 赛前夜: 担忧 → 失眠行为 + 训练事实
    ("周日晚上我在床上翻来覆去。想着明天运动会，最后一棒要是又掉棒怎么办。"
     "上次练到第八次才顺，万一比赛的时候手滑。我起来喝了口水，又躺下。",
     "周日晚上我在床上翻来覆去。想着明天的接力，接棒的位置练了八次才顺。"
     "我起来喝了口水，又躺下。"),
    # 段25 犹豫: 担忧 → 行为 (话到嘴边没说出口)
    ("周二放学路上我在想竞选的事。想报名，又怕当不好，怕演讲的时候忘词。",
     "周二放学路上我想着竞选的事，报名的话到嘴边没说出口。"),
    # 段44 期末复习: 高兴 → 行为 (夹卷子)
    ("很多了，我心里高兴，但没表现出来。晚上回家", "很多了。晚上回家"),
    # 段44 期末复习: 强度描述去心理化
    ("期末复习比期中紧张。", "期末复习比期中量更大。"),
    # 段4 课堂提问: 心理计算 → 行为 (没出声)
    ("心里算了一遍，答案是对的，但没敢说", "算了一遍，答案是对的，没出声"),
    # 段4 课堂提问: 心情 → 行为 (脚步快)
    ("那天我心情不错，放学路上脚步快了一些", "放学路上我脚步快了一些"),
    # 段11 运动会报名: 怕 → 自我能力陈述
    ("我说我怕跑不好", "我说我跑得慢"),
    # 段16 赛后: 心理评估 → 回忆行为
    ("我躺在床上想，白天的事好像没那么糟", "我躺在床上，想起白天掉棒的事"),
    # 段18 期中动员: 联想 → 沉默行为
    ("我想到上次摸底58分，没说话", "我没说话"),
    # 段3 周末: 不确定 → 删除
    ("我说不用，我自己能跟上。说完又有点不确定。", "我说不用，我自己能跟上。"),
    # 段24 竞选公告: 判断词 → 删除
    ("觉得这个我能干，但没跟别人说", "这个我能干，但没跟别人说"),
    # 段26 报名: 怕 → 删除
    ("我说我想报生活委员，怕当不好。他说你可以试试", "我说我想报生活委员。他说你可以试试"),
    # 段29 上任: 踏实 → 删除
    ("一天下来，做了好几件事，心里有点踏实。", "一天下来，做了好几件事。"),
    # 段32 卫生检查: 预期批评 → 行为 (低头)
    ("我以为会被当众批评，低着头", "我低着头"),
    # 段33 同学分担: 感受 → 行为 (同行)
    ("回家的路上，我觉得自己没被孤立。", "回家的路上，我和值日的同学一起走了一段。"),
    # 段40 布置教室: 判断词 → 删除
    ("我站在门口看了一眼教室，觉得是我们布置的。", "我站在门口看了一眼教室，是我们布置的。"),
    # 段43 期末动员: 不敢 → 行为 (不出声)
    ("我们从对答案都不太敢", "我们从对答案都不出声"),
]


def sanitize_first_person(text: str) -> str:
    """净化第一人称感知文本: 剥离心理状态标签, 保留行为/环境/对话/生理"""
    out = text
    for old, new in PSYCH_REWRITES:
        if old in out:
            out = out.replace(old, new)
    return out


# =============================================================================
# 长线剧本 v2: 《维塔的转学第一学期》 50 段
# =============================================================================
# 时间线: 9月1日 → 1月20日, 每段 ≈ 3-4 天 (20K 步 = 50 段 × 400 步)
# 远程因果链 (贯穿 50 段):
#   A. 友谊线: 段1 结识陈默 → 段11 听其家事 → 段15-16 运动会同队
#      → 段27 鼓励竞选 → 段33 分担责任 → 段37 生病关怀 → 段49 约定下学期
#   B. 学业线: 段6 摸底 → 段7 58分 → 段9 补考76 → 段23 期中82
#      → 段45-46 期末 → 段48 88分前25%
#   C. 家庭线: 段3 妈妈问成绩 → 段20 妈妈加班孤独 → 段43 妈妈回归
LONG_ARC = {
    "arc_id": "transfer_semester_v2",
    "title": "转学的第一学期 (50 段)",
    "time_span": "9月1日 → 1月20日, 50 段 × 400 步 = 20,000 步",
    "segments": [
        # ================= 9月: 适应 (段0-5) =================
        {
            "title": "开学报到",
            "time_label": "9月1日 周一",
            "snn_view": "开学第一天，我转到新学校。上午到教导处报到，领了课本，分到初一三班。教室在二楼走廊尽头，座位在第二排靠窗。班主任姓周，让我们轮流自我介绍。我站起来说了名字，声音不大，说了两句就坐下。课间没有人跟我说话，我坐在位子上看窗外。放学时在校门口站了一会儿才走。",
            "sandbox_view": "9月1日，转学生维塔到新学校报到，编入初一三班，座位在二楼教室第二排靠窗。自我介绍环节声音较小。课间独处，放学在校门口停留片刻后离开。",
            "events": [
                {"step_offset": 100, "event_type": "novelty", "intensity": 20,
                 "description": "新学校报到领课本，环境完全陌生"},
                {"step_offset": 100, "event_type": "threat_social", "intensity": -10,
                 "description": "课上自我介绍紧张，声音小"},
                {"step_offset": 300, "event_type": "social_loss", "intensity": -10,
                 "description": "课间无人交谈，独自看窗外"},
            ],
        },
        {
            "title": "食堂结识",
            "time_label": "9月2-4日 周二至周四",
            "snn_view": "第二天中午下课去食堂，我端着餐盘找了一张空桌坐下。旁边桌四五个人在聊天，我低头吃饭。吃到一半，有个人端着餐盘过来问这里有没有人坐。我说没有。他坐下，是同班的，叫陈默。他问我哪个初中转来的，我说了学校名字。他说他住这个区，问我认不认识一个叫张伟的人，我说不认识。之后两天中午都碰到，开始聊游戏和作业。",
            "sandbox_view": "第二日起维塔午间独自在食堂用餐。进食中同班同学陈默询问座位后坐下，两人首次交谈。此后两日多次共餐，话题扩展至游戏与作业。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -10,
                 "description": "独自在食堂用餐，邻桌多人交谈"},
                {"step_offset": 200, "event_type": "novelty", "intensity": 15,
                 "description": "陈默端餐盘走近询问座位，首次交谈"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 25,
                 "description": "此后每日共餐，话题扩展至游戏作业"},
            ],
        },
        {
            "title": "第一周课程",
            "time_label": "9月5日 周五",
            "snn_view": "周五上了英语和数学。英语老师点我起来读课文，我读得慢，有几个词读错了，同学没笑。数学课讲有理数，我听了一半，后半程有点跟不上。下课问同桌陈默借笔记，他翻到中间一页给我。放学妈妈问我这周适应得怎么样，我说还行。",
            "sandbox_view": "周五英语课维塔被点名读课文，语速慢且个别发音有误，课堂未出现嘲笑。数学课后半程理解吃力，课后向陈默借阅笔记。放学后母亲询问适应情况，维塔表示尚可。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -15,
                 "description": "英语课被点名读课文，读得慢有错音"},
                {"step_offset": 200, "event_type": "question", "intensity": 15,
                 "description": "数学课跟不上，借陈默笔记"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 10,
                 "description": "放学与妈妈聊适应情况"},
            ],
        },
        {
            "title": "周末独处",
            "time_label": "9月6-7日 周末",
            "snn_view": "周末两天在家。妈妈周六加班，我一个人写作业，写完看了一会儿电视，没什么好看的，又关掉了。周日妈妈带我去买了新书包，旧的用了三年。晚上妈妈问我下个月要不要报个补习班，我说不用，我自己能跟上。说完又有点不确定。",
            "sandbox_view": "周末维塔独居多时，母亲周六加班。周日母亲为其购置新书包。晚间母亲提议报补习班，维塔婉拒并表示能自行跟上课程，语气略有不确定。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -10,
                 "description": "周末独处，妈妈加班"},
                {"step_offset": 200, "event_type": "novelty", "intensity": 10,
                 "description": "买新书包"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "婉拒补习班，自我评估"},
            ],
        },
        {
            "title": "课堂提问",
            "time_label": "9月8-10日 第二周",
            "snn_view": "第二周数学开始讲绝对值。老师问谁会做第三题，我低头没举手，心里算了一遍，答案是对的，但没敢说。老师叫了别人。下课陈默说那道题他也不会，问我，我讲给他听，他说我讲得挺清楚。那天我心情不错，放学路上脚步快了一些。",
            "sandbox_view": "第二周数学课老师提问第三题时，维塔虽已心算得出正确结果但未举手。课后向陈默讲解该题，获认可。当日放学步伐较往常轻快。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -10,
                 "description": "心里算对答案但没敢举手"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 10,
                 "description": "给陈默讲题，讲解清楚"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 10,
                 "description": "讲解获认可，心情转好"},
            ],
        },
        {
            "title": "摸底测试",
            "time_label": "9月11-12日 周四至周五",
            "snn_view": "周四数学老师宣布下周一摸底测试，范围是开学以来讲的内容。我周末把课本翻了一遍，练习题做了一部分。周五下午老师又提醒了一次。考场上有些题看着眼熟，但不确定做得对不对。交卷前检查了一遍，改了两处。",
            "sandbox_view": "周四周五数学教师两次提醒下周一摸底测试。维塔周末复习课本并完成部分练习。测试中部分题目不确定，交卷前修改了两处作答。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "复习摸底范围，完成部分练习"},
                {"step_offset": 200, "event_type": "question", "intensity": 15,
                 "description": "测试中部分题目不确定"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "交卷前检查修改"},
            ],
        },
        {
            "title": "摸底发卷",
            "time_label": "9月15日 周一",
            "snn_view": "周一发摸底卷。老师从高到低念分数，念到我的名字时停了一下，说58分。我上台拿卷子，旁边同学转头看了一眼。整节课后半程我盯着卷子上的红叉没怎么听讲。放学陈默问我考了多少，我说没考好，没说具体数字。",
            "sandbox_view": "摸底卷发放，教师按分数念名，念到维塔时停顿并报出58分。维塔上台领卷，邻座侧目。课间陈默询问分数，维塔未答具体数字。",
            "events": [
                {"step_offset": 100, "event_type": "criticism", "intensity": -20,
                 "description": "发卷念分数58分，停顿后点名"},
                {"step_offset": 200, "event_type": "threat_social", "intensity": -15,
                 "description": "拿卷时同学侧目，被问分数"},
                {"step_offset": 300, "event_type": "social_loss", "intensity": -10,
                 "description": "盯着红叉，情绪低落"},
            ],
        },
        {
            "title": "问老师",
            "time_label": "9月16-17日 周二至周三",
            "snn_view": "周二放学我拿着卷子去数学老师办公室，问那两道大题。老师讲了一遍，我没全懂，又问了一遍，他换了个说法。讲完他说，这次题确实偏难，班上平均分不高，能主动来问是好事。我说谢谢。出了办公室天快黑了。",
            "sandbox_view": "周二放学维塔携卷至数学教师办公室请教两道大题解法。教师讲解两遍，并肯定其主动提问行为。维塔道谢后离开时天色已晚。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 20,
                 "description": "主动找老师问错题，问了两遍"},
                {"step_offset": 200, "event_type": "praise", "intensity": 10,
                 "description": "老师肯定主动提问"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 5,
                 "description": "弄懂两道大题解法"},
            ],
        },
        {
            "title": "补考",
            "time_label": "9月18-22日",
            "snn_view": "周三放学，陈默说他也考得不好，我们一起留下来把错题重做了一遍。他把他的卷子拿出来，我们一题一题对。周四老师宣布下周一补考。周末我又做了一遍错题。周一补考，题目看着都熟。考完感觉比上次好。",
            "sandbox_view": "周三至周末维塔与陈默共同重做错题。教师宣布下周一补考。周一补考完成，自感作答好于首次。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "与陈默一起重做错题"},
                {"step_offset": 200, "event_type": "question", "intensity": 15,
                 "description": "周末复习错题准备补考"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 20,
                 "description": "补考作答顺利"},
            ],
        },
        {
            "title": "补考发卷",
            "time_label": "9月23日 周二",
            "snn_view": "周二补考卷发下来，76分。老师在成绩单上画了个勾，没说什么特别的话。我坐在位子上看了两遍卷子，把错的题又看了一遍，有两道是粗心，一道是真不会。放学陈默问多少，我说76。他说可以啊，比上次强。我说还行。",
            "sandbox_view": "补考卷发放，维塔得分76。教师未作额外评价，仅在成绩单上标记通过。维塔自查卷面，归因两处粗心、一处未掌握。陈默得知分数后给予肯定。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 20,
                 "description": "补考76分通过"},
                {"step_offset": 200, "event_type": "question", "intensity": 5,
                 "description": "自查错题，归因"},
                {"step_offset": 300, "event_type": "praise", "intensity": 10,
                 "description": "陈默肯定进步"},
            ],
        },
        {
            "title": "图书馆日",
            "time_label": "10月第1周 周六",
            "snn_view": "周六陈默约我去区图书馆写作业。中午一起吃了面，牛肉面一般，但我们聊了不少。他聊起他爸妈离婚的事，说他跟着爸爸住，爸爸开出租，晚上常不在家。我不知道该说什么，就说嗯。他说没事，习惯了。下午作业写完，去操场走了两圈才回家。",
            "sandbox_view": "周末维塔与陈默在区图书馆完成作业，午间共同用餐。陈默提及父母离异、随父居住，维塔未深入回应。下午操场散步后各自回家。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "约图书馆写作业"},
                {"step_offset": 100, "event_type": "food_tasty", "intensity": 10,
                 "description": "午间一起用餐"},
                {"step_offset": 200, "event_type": "social_loss", "intensity": -10,
                 "description": "陈默聊父母离异"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "操场散步，关系加深"},
            ],
        },
        {
            "title": "运动会报名",
            "time_label": "10月第1周 周二至周四",
            "snn_view": "班主任说月底开秋季运动会，问谁报名。课间我看到体育委员拿了张表在班里转。陈默说他要跑4×100接力，问我要不要一起。我说我怕跑不好。他说没事，一起跑，跑最后一名也没人说什么。我想了想，说行。",
            "sandbox_view": "月底秋季运动会报名期间，陈默报名4×100接力并邀请维塔加入。维塔犹豫后接受。",
            "events": [
                {"step_offset": 100, "event_type": "novelty", "intensity": 15,
                 "description": "运动会报名"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "陈默邀请一起跑接力"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 10,
                 "description": "决定报名参加"},
            ],
        },
        {
            "title": "接力训练",
            "time_label": "10月第2周",
            "snn_view": "这周放学后我们练了三次接力。体育老师看我们练，测了短跑，说我跑得不慢，最后一棒你来。第一次练接棒我手滑了，棒掉在地上，陈默说没事，再来。我们练到第七八次才顺。回家路上手被接力棒磨得有点疼，但我没跟妈妈说。",
            "sandbox_view": "本周维塔与队友练习接力三次。体育教师测试后指定维塔跑最后一棒。首次接棒练习掉棒，经多次练习后配合趋于熟练。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 25,
                 "description": "体育老师指定跑最后一棒"},
                {"step_offset": 200, "event_type": "threat_social", "intensity": -10,
                 "description": "首次接棒练习掉棒"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "反复练习配合"},
            ],
        },
        {
            "title": "赛前紧张",
            "time_label": "10月第2周 周日",
            "snn_view": "周日晚上我在床上翻来覆去。想着明天运动会，最后一棒要是又掉棒怎么办。上次练到第八次才顺，万一比赛的时候手滑。我起来喝了口水，又躺下。第二天早上出门前，把接力棒的手感在脑子里过了一遍。",
            "sandbox_view": "运动会前夜维塔睡眠不佳，反复思虑接棒失误可能。次日出门前自行回顾接棒要领。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -15,
                 "description": "赛前夜失眠，担心掉棒"},
                {"step_offset": 200, "event_type": "question", "intensity": 5,
                 "description": "回顾接棒要领"},
                {"step_offset": 300, "event_type": "novelty", "intensity": 10,
                 "description": "运动会当天清晨"},
            ],
        },
        {
            "title": "运动会上午",
            "time_label": "10月第3周 周一上午",
            "snn_view": "运动会上午是短跑和跳远。我在看台上看同学比赛，班里有人拿了名次，大家鼓掌。我也跟着拍了几下手。广播里报成绩的声音很响。下午才轮到接力，我坐在看台上有点坐不住，来回看了几次表。",
            "sandbox_view": "运动会上午为短跑、跳远项目。维塔在看台观赛，参与班级掌声。接力项目在下午，维塔等待期间多次查看时间。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "上午观赛"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 10,
                 "description": "与同学一起观赛鼓掌"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -10,
                 "description": "等待接力，坐立不安"},
            ],
        },
        {
            "title": "运动会接力",
            "time_label": "10月第3周 周一下午",
            "snn_view": "下午4×100接力。我们组前三个棒跑完排第二。第三棒把接力棒递过来的时候，我伸手接，手滑了，棒掉在地上。旁边有人喊快捡。我弯腰捡起来开始跑，前面那组已经拉开七八米。最后跑了第三名。陈默跑完过来拍我肩膀，说没事，掉棒是接的问题不是你的问题。我们组拿了第三，同学说还行。",
            "sandbox_view": "接力赛中维塔接棒手滑掉棒，旁人提醒，弯腰拾棒后追回部分差距，该组最终列第三。赛后陈默拍肩安慰。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "前三棒跑完组内列第二"},
                {"step_offset": 200, "event_type": "threat_social", "intensity": -25,
                 "description": "接棒手滑掉棒，公开失误"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 20,
                 "description": "陈默拍肩安慰"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 10,
                 "description": "组内列第三，同学认可"},
            ],
        },
        {
            "title": "赛后",
            "time_label": "10月第3周 周一晚",
            "snn_view": "晚上回家，妈妈问运动会怎么样。我说跑了第三名。她说也挺好。我说我掉棒了。她说那也没关系，以后注意就行。我躺在床上想，白天的事好像没那么糟。同桌和妈妈都没怪我。第二天去学校，同学跟我打招呼，我愣了一下，也回了。",
            "sandbox_view": "当晚维塔向母亲提及掉棒一事，母亲表示无碍。次日上学有同学主动与维塔打招呼，维塔稍作停顿后回应。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -5,
                 "description": "回家谈及掉棒"},
                {"step_offset": 200, "event_type": "praise", "intensity": 10,
                 "description": "妈妈安慰"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "次日同学主动打招呼"},
            ],
        },
        {
            "title": "陈默请假",
            "time_label": "10月第3周 周三至周五",
            "snn_view": "周三陈默没来上学，问了同桌说感冒请假了。中午我一个人去食堂，坐的还是老位置，对面空着。饭吃了一半，想起他平时坐对面说游戏通关攻略的样子。周五他回来了，说就是普通感冒。我说明天周六，要不要去图书馆。他说去。",
            "sandbox_view": "陈默因病请假两日，维塔独自用餐，坐平时共餐位置。陈默周五返校，维塔主动约周末图书馆，获应允。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -10,
                 "description": "陈默请假，独自用餐"},
                {"step_offset": 200, "event_type": "question", "intensity": 5,
                 "description": "回想平时共餐情形"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "陈默返校，主动约周末"},
            ],
        },
        {
            "title": "期中动员",
            "time_label": "10月第4周 周一",
            "snn_view": "班主任在班会课上宣布，11月中旬期中考试，语数英三门。她让大家从现在开始复习。下课陈默问我期中打算考多少，我说不知道，尽力吧。他说他数学目标是及格。我想到上次摸底58分，没说话。",
            "sandbox_view": "班会课宣布期中考试时间与科目。陈默课间与维塔谈及考试目标，维塔未明确回应。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "期中动员，明确范围"},
                {"step_offset": 200, "event_type": "threat_social", "intensity": -10,
                 "description": "谈及考试目标，联想摸底失利"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "开始规划期中复习"},
            ],
        },
        {
            "title": "复习开始",
            "time_label": "10月第4周 周二至周五",
            "snn_view": "这周开始每天复习。数学我把补考卷子又做了一遍，英语背了单词，语文背古文。晚上写到十点，妈妈十点前进我房间看了一眼，说早点睡。作业比平时多，有两晚写到十点半，第二天早上有点困。",
            "sandbox_view": "维塔本周开始系统复习期中内容，晚间学习至十点后，有两晚至十点半。母亲进房提醒早睡。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 20,
                 "description": "系统复习三门功课"},
                {"step_offset": 200, "event_type": "threat_physical", "intensity": -5,
                 "description": "连续熬夜，次日犯困"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 5,
                 "description": "妈妈提醒早睡"},
            ],
        },
        {
            "title": "妈妈加班",
            "time_label": "11月第1周 周一至周四",
            "snn_view": "这周妈妈开始加班，说单位要赶一个项目。晚饭我自己热了前一天剩的饭吃。周二晚上写完作业，客厅灯黑着，我坐在沙发上发了一会儿呆。周四妈妈九点多才回来，问我作业写完没，我说写了。她坐了十分钟又回房间开电脑。",
            "sandbox_view": "本周母亲加班频繁，维塔独自热饭用餐。某晚独坐客厅发呆。周四母亲短暂交流后继续工作。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -15,
                 "description": "妈妈加班，晚饭独自热饭"},
                {"step_offset": 200, "event_type": "social_loss", "intensity": -10,
                 "description": "晚上独坐客厅发呆"},
                {"step_offset": 300, "event_type": "question", "intensity": 5,
                 "description": "妈妈短暂询问作业"},
            ],
        },
        {
            "title": "期中第一天",
            "time_label": "11月第2周 周一",
            "snn_view": "期中考试第一天，上午语文，下午英语。早上出门前我把笔袋检查了两遍。语文卷子比平时练习的长，古文默写有两句拿不准。下午英语听力，广播有点杂音，我凑近听。考完陈默问我对答案，我说不想对。",
            "sandbox_view": "期中考试首日进行语文与英语。维塔考前检查文具，古文默写两句不确定，英语听力受广播杂音影响。考后未参与对答案。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -10,
                 "description": "考前紧张，检查文具"},
                {"step_offset": 200, "event_type": "question", "intensity": 15,
                 "description": "古文默写不确定，听力有杂音"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -5,
                 "description": "考后拒绝对答案"},
            ],
        },
        {
            "title": "期中第二天",
            "time_label": "11月第2周 周二",
            "snn_view": "第二天上午数学。发卷前我深吸了一口气。题目大部分会做，有一道应用题卡了一会儿，最后按自己的理解写了。交卷后我算了算，大概能及格。下午放假半天。回家路上经过学校门口的小店，买了两根棒棒糖，一根放自己书包，一根放桌上没送出去。",
            "sandbox_view": "期中数学考试维塔大部分作答顺利，一道应用题卡顿后按自身理解完成，自评可能及格。下午放假。途经小店购棒棒糖，留一根未送出。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "数学考试，应用题卡顿"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 15,
                 "description": "自评可能及格，较摸底进步"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 5,
                 "description": "买棒棒糖未送出"},
            ],
        },
        {
            "title": "期中成绩",
            "time_label": "11月第2周 周四",
            "snn_view": "周四期中成绩出来。数学82，英语71，语文68。班主任念到我的分数，说比摸底进步明显。我坐在位子上，手心有点出汗。下课后几个同学围过来问分数，我说数学82。有人啧了一声说可以啊。陈默说他数学刚好及格，61分，我说下次一起进步。",
            "sandbox_view": "期中成绩公布：数学82分较摸底进步24分，教师点名肯定进步。课间有同学询问成绩。陈默数学61分及格，两人约定共同进步。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 25,
                 "description": "期中数学82分，年级前30%"},
                {"step_offset": 200, "event_type": "praise", "intensity": 20,
                 "description": "班主任点名表扬进步"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "与陈默约定共同进步"},
            ],
        },
        # ================= 11月: 竞选班委 (段24-29) =================
        {
            "title": "竞选公告",
            "time_label": "11月第3周 周一",
            "snn_view": "班主任在班里说，下周竞选班委，每人准备三分钟演讲。课间大家在讨论报什么，我听见有人说要当班长。我其实想报生活委员，管卫生和班级用品，觉得这个我能干，但没跟别人说。",
            "sandbox_view": "教师宣布下周班委竞选，需三分钟演讲。维塔考虑报名生活委员，未向他人提及。",
            "events": [
                {"step_offset": 100, "event_type": "novelty", "intensity": 15,
                 "description": "竞选公告"},
                {"step_offset": 200, "event_type": "question", "intensity": 10,
                 "description": "考虑报名生活委员"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -10,
                 "description": "未向他人透露意向"},
            ],
        },
        {
            "title": "犹豫",
            "time_label": "11月第3周 周二至周三",
            "snn_view": "周二放学路上我在想竞选的事。想报名，又怕当不好，怕演讲的时候忘词。周三课间看到班主任办公室的门，走过去又走回来，没进去。晚上写作业的时候走神，妈妈问我想什么，我说老师在选班委。妈妈说你初中了，想当就当，当不好也是经验。",
            "sandbox_view": "维塔连续两日犹豫是否报名，曾行至教师办公室门口未入。晚间母亲鼓励其尝试。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -15,
                 "description": "犹豫竞选，怕忘词"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 10,
                 "description": "妈妈鼓励尝试"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "权衡竞选利弊"},
            ],
        },
        {
            "title": "报名",
            "time_label": "11月第3周 周四",
            "snn_view": "周四中午，陈默问我你报不报。我说我想报生活委员，怕当不好。他说你可以试试，反正生活委员就是管卫生和发作业，不难当。我吃完午饭，走到班主任办公室门口，站了一会儿，敲门进去，说我想报名生活委员。老师说好，周四下午交演讲。",
            "sandbox_view": "周四陈默鼓励维塔报名生活委员。午饭后维塔进入教师办公室报名，教师确认并安排演讲稿提交。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "陈默鼓励报名"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 15,
                 "description": "成功报名生活委员"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "准备演讲稿"},
            ],
        },
        {
            "title": "演讲稿",
            "time_label": "11月第3周 周五",
            "snn_view": "周五晚上我写演讲稿，写了改，改了写。先写了一版太短，念了一遍只有一分钟。又加了一段，写我为什么想当生活委员，因为我做事细，会记清楚班级里的东西放哪。写完让妈妈听了一遍，她说可以，就是别念太快。我练到十点。",
            "sandbox_view": "维塔晚间撰写并修改演讲稿两版。请母亲试听，母亲建议放慢语速。练习至十点。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "撰写修改演讲稿"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 5,
                 "description": "请妈妈试听"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -5,
                 "description": "反复练习，担心忘词"},
            ],
        },
        {
            "title": "竞选演讲",
            "time_label": "11月第4周 周一",
            "snn_view": "竞选那天，班里很安静。轮到我上讲台，我拿着稿子，念了开头两句就忘了词，站在台上没说话。教室里安静了几秒。底下有人小声说加油。我低头看了一眼稿子，找到地方接着念，后面慢慢顺了。讲完下来，陈默说讲得挺好。投票结果出来，我当选了生活委员。",
            "sandbox_view": "竞选演讲中维塔开场忘词，有同学低声鼓励，查阅稿子后完成演讲。陈默给予肯定。投票结果维塔当选生活委员。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -20,
                 "description": "演讲开场忘词"},
                {"step_offset": 200, "event_type": "praise", "intensity": 15,
                 "description": "同学鼓励，陈默肯定"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 25,
                 "description": "当选生活委员"},
            ],
        },
        {
            "title": "上任",
            "time_label": "11月第4周 周二至周五",
            "snn_view": "当选后，班主任交代了生活委员的工作：早上检查值日，课间管饮水机，放学检查门窗。周二我提前十分钟到教室，检查值日生扫的地。有一处没扫干净，我指了指，对方扫了。中午饮水机没水了，我去找老师换了桶。一天下来，做了好几件事，心里有点踏实。",
            "sandbox_view": "维塔上任生活委员，职责包括检查值日、管理饮水机、检查门窗。首日按流程执行各项职责。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 15,
                 "description": "首日履职检查值日"},
                {"step_offset": 200, "event_type": "question", "intensity": 10,
                 "description": "处理饮水机换水"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 10,
                 "description": "完成多项职责"},
            ],
        },
        # ================= 12月: 班会·班级·生病 (段30-37) =================
        {
            "title": "班会筹备",
            "time_label": "12月第1周 周三",
            "snn_view": "班主任说周五班会，让生活委员组织，内容自定。周三我列了节目单，猜词游戏、才艺展示、讲笑话。跟几个同学分了工，有人负责布置，有人负责记分。有个同学说周五他要表演魔方，我把他加进节目单。",
            "sandbox_view": "教师指定维塔组织周五班会。维塔列出节目单并分工，接纳同学新增魔方表演。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "列节目单"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "与同学分工合作"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 5,
                 "description": "确定节目单"},
            ],
        },
        {
            "title": "班会成功",
            "time_label": "12月第1周 周五",
            "snn_view": "周五班会。猜词游戏我出题，同学分组比划，有一组把茄子猜成了香蕉，大家都笑了。中途有同学说要换节目顺序，我按他说的调了。魔方表演很顺利。班会结束，班主任说这周班会组织得不错，问是谁组织的，我说是我。陈默在底下鼓掌。",
            "sandbox_view": "周五班会顺利开展。猜词游戏环节气氛活跃，维塔按同学建议调整节目顺序。班会后教师当众肯定班会组织。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "主持猜词游戏"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "按同学建议调整"},
                {"step_offset": 300, "event_type": "praise", "intensity": 15,
                 "description": "班主任肯定组织能力"},
            ],
        },
        {
            "title": "卫生检查",
            "time_label": "12月第2周 周二",
            "snn_view": "周二学校检查卫生，我们班教室角落没打扫干净，被扣了分，在全校通报里点名了。班主任在班会上问卫生是谁负责的。我说是我，安排值日的时候漏了角落。我以为会被当众批评，低着头。结果班主任说，下次注意，把值日表再细化一下。",
            "sandbox_view": "学校卫生检查中班级角落未净被扣分并通报。班主任询问责任归属，维塔承认安排值日遗漏角落。教师要求细化值日表。",
            "events": [
                {"step_offset": 100, "event_type": "criticism", "intensity": -20,
                 "description": "班级卫生扣分被通报"},
                {"step_offset": 200, "event_type": "threat_social", "intensity": -15,
                 "description": "当众承认责任"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "细化值日表"},
            ],
        },
        {
            "title": "同学分担",
            "time_label": "12月第2周 周三",
            "snn_view": "周三放学，值日的一个同学说，那天角落他也忘了提醒我，责任不能都算我头上。他说以后他值日的时候会重点检查角落。另外两个同学也说，角落那个位置平时不放东西，谁都会漏。我听了，说谢谢。回家的路上，我觉得自己没被孤立。",
            "sandbox_view": "次日有值日同学主动分担卫生疏漏责任，另有两名同学为维塔解释。维塔致谢，归途自感未被孤立。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -10,
                 "description": "独自承担疏漏"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 20,
                 "description": "同学主动分担责任"},
                {"step_offset": 300, "event_type": "praise", "intensity": 5,
                 "description": "感受到支持"},
            ],
        },
        {
            "title": "感冒初起",
            "time_label": "12月第2周 周五",
            "snn_view": "周五下午开始嗓子有点疼，我以为是喊班会喊的。放学回家喝了很多水。晚上睡觉前有点发冷，我多盖了一层被子。周六早上起来，头疼，量了体温37.8，有点烧。妈妈给我找了退烧药，说先吃一片看看。",
            "sandbox_view": "周五维塔开始咽痛，周末体温升至37.8度，出现头痛。母亲提供退烧药观察。",
            "events": [
                {"step_offset": 100, "event_type": "threat_physical", "intensity": -5,
                 "description": "嗓子疼，开始感冒"},
                {"step_offset": 200, "event_type": "threat_physical", "intensity": -10,
                 "description": "发烧37.8度"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 5,
                 "description": "妈妈照顾吃药"},
            ],
        },
        {
            "title": "发烧请假",
            "time_label": "12月第3周 周一",
            "snn_view": "周一早上烧到38.2，嗓子更疼了。妈妈帮我给班主任发了消息请假。班主任回：好好休息，课的内容回头找同学抄笔记。我在床上躺了一天，吃了药就睡，醒了就喝水。下午退了一点，但还是没力气。窗外的天灰蒙蒙的，我盯着看了很久。",
            "sandbox_view": "周一维塔体温升至38.2度，母亲代为向班主任请假。卧床休息，下午体温略降但体力未复，长时间望窗外。",
            "events": [
                {"step_offset": 100, "event_type": "threat_physical", "intensity": -15,
                 "description": "高烧38.2度，请假在家"},
                {"step_offset": 200, "event_type": "social_loss", "intensity": -10,
                 "description": "独自卧床看窗外"},
                {"step_offset": 300, "event_type": "question", "intensity": 5,
                 "description": "惦记落课"},
            ],
        },
        {
            "title": "病中孤独",
            "time_label": "12月第3周 周二",
            "snn_view": "周二还是没退烧，妈妈上午请假在家陪了我半天，下午去上班了。我一个人躺着，手机没什么消息。班里群聊了几条消息，我没点开。下午睡醒，房间里很安静，我打开电视，声音调到最小，看了会儿综艺，没看进去。",
            "sandbox_view": "周二维塔仍发热，母亲上午陪同、下午返工。维塔独处，未参与班级群聊，电视开启但未投入观看。",
            "events": [
                {"step_offset": 100, "event_type": "social_loss", "intensity": -15,
                 "description": "母亲返工，独自在家"},
                {"step_offset": 200, "event_type": "social_loss", "intensity": -10,
                 "description": "群里消息未点开"},
                {"step_offset": 300, "event_type": "question", "intensity": 5,
                 "description": "想回学校"},
            ],
        },
        {
            "title": "陈默关怀",
            "time_label": "12月第3周 周四晚",
            "snn_view": "周四晚上，手机响了一下，是陈默发的消息：维塔，好点了没？我说好多了，明天去学校。他说数学课讲了新内容，笔记拍了照片发给你。他发过来三张照片，字迹工整。我说谢谢。他说客气什么，你落了两天课，周五我给你讲讲重点。我看了那条消息很久。",
            "sandbox_view": "周四晚陈默发消息问候病情，并拍摄数学课堂笔记三张发送，主动提出周五为维塔讲解落课重点。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 20,
                 "description": "陈默发消息问候，拍笔记"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "约定周五补课"},
                {"step_offset": 300, "event_type": "praise", "intensity": 5,
                 "description": "被记挂，情绪回暖"},
            ],
        },
        # ================= 12月-1月: 返校·联欢·期末 (段38-49) =================
        {
            "title": "返校",
            "time_label": "12月第3周 周五",
            "snn_view": "周五回到学校。落了两天课，数学笔记借了陈默的抄，抄了一节课。课间补作业，后排的同学递过来一瓶水，说生病多喝水。我说谢谢。中午陈默给我讲了新内容的重点，讲了一遍我没全懂，他又讲了一遍。下午的课有点累，但撑着听完了。",
            "sandbox_view": "周五维塔返校，补齐数学笔记与作业。后排同学递水关心。午间陈默讲解落课重点两遍。维塔带病坚持完成当天课程。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "返校补齐笔记作业"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "同学递水，陈默讲解"},
                {"step_offset": 300, "event_type": "threat_physical", "intensity": -5,
                 "description": "身体未痊愈撑着听课"},
            ],
        },
        {
            "title": "联欢筹备",
            "time_label": "12月第4周 周二",
            "snn_view": "班主任说元旦班级联欢，让班委组织，生活委员负责布置。周二放学我去文具店买了拉花和气球，一共花了二十多块，回去跟班长对账。班长说买多了，气球用不了这么多，我说留着下次用。晚上我把拉花在客厅挂了一下试效果，妈妈说我弄得挺像回事。",
            "sandbox_view": "元旦联欢筹备中，维塔采购拉花气球等布置用品，与班长核对开销。晚间在家试挂拉花，获母亲认可。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "采购布置用品"},
                {"step_offset": 200, "event_type": "question", "intensity": 10,
                 "description": "规划布置方案"},
                {"step_offset": 300, "event_type": "praise", "intensity": 5,
                 "description": "试挂拉花获认可"},
            ],
        },
        {
            "title": "布置教室",
            "time_label": "12月第4周 周四",
            "snn_view": "周四放学后布置教室。我们四个人，一个打气球，两个贴拉花，我踩着凳子挂窗花。挂到一半，气球的绳子松了，飞了一个，大家笑。有人重新打了一个。布置完，教室跟平时不一样了，彩色的。班长说辛苦大家，周五联欢。我站在门口看了一眼教室，觉得是我们布置的。",
            "sandbox_view": "周四放学后维塔与三名同学布置教室。过程中一个气球飞脱引发笑声。布置完成后教室焕然一新。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "与同学分工布置"},
                {"step_offset": 200, "event_type": "novelty", "intensity": 10,
                 "description": "气球飞脱"},
                {"step_offset": 300, "event_type": "achievement", "intensity": 10,
                 "description": "布置完成"},
            ],
        },
        {
            "title": "元旦联欢",
            "time_label": "12月31日 周五",
            "snn_view": "联欢那天，上午正常上课，下午联欢。节目有唱歌、魔方、猜谜。陈默上台弹了吉他，弹的是《小星星》加花，底下有人跟着拍手。我坐在第一排，鼓掌鼓得手有点红。班主任说这学期班委辛苦了，尤其是生活委员，每月检查值日没断过。大家鼓掌，我有点不好意思。",
            "sandbox_view": "元旦联欢下午举行。陈默吉他弹奏获掌声，维塔坐第一排。教师当众肯定班委工作并特别提及生活委员职责履行。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "联欢顺利开展"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 15,
                 "description": "陈默弹吉他"},
                {"step_offset": 300, "event_type": "praise", "intensity": 20,
                 "description": "班主任肯定班委工作"},
            ],
        },
        {
            "title": "元旦假期",
            "time_label": "1月1-3日 元旦",
            "snn_view": "元旦假期三天。第一天跟陈默在图书馆写了半天作业，聊起寒假去哪玩，他说可能回老家。我说那下学期见。第二天妈妈带我去吃了火锅，问我这学期过得怎么样。我说挺好的，认识了一个朋友，当了生活委员。妈妈听了，说那就好。晚上我躺在被窝里，想这半年的事，从开学报到那天想起。",
            "sandbox_view": "元旦假期维塔与陈默在图书馆写作业，陈默提及寒假回老家。次日与母亲吃火锅，维塔回顾本学期收获。晚间回想开学以来的经历。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "假期与陈默学习"},
                {"step_offset": 200, "event_type": "food_tasty", "intensity": 10,
                 "description": "与妈妈吃火锅"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "回顾一学期经历"},
            ],
        },
        {
            "title": "期末动员",
            "time_label": "1月第1周 周一",
            "snn_view": "假期回来，班主任说月底期末考，范围是整学期，比期中难一些。我算了算时间，还有三周。课间陈默说，期末他数学想考70以上。我说我试试80。说完我们互相看了一眼，都笑了。这学期的数学课，我们从对答案都不太敢，到现在敢说目标了。",
            "sandbox_view": "期末动员宣布考试范围与时间。维塔与陈默互提期末目标，对比期中时的回避态度明显主动。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "期末动员"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 10,
                 "description": "与陈默互提目标"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 10,
                 "description": "互相鼓励"},
            ],
        },
        {
            "title": "期末复习",
            "time_label": "1月第1周 周二至周五",
            "snn_view": "期末复习比期中紧张。数学我把整学期的卷子翻出来，错题重新做。英语单词背了三遍，语文古文默写了四遍。周五数学老师发了一张综合练习，我做了85分，老师说比上次摸底进步很多了，我心里高兴，但没表现出来。晚上回家我把那张卷子夹在课本里。",
            "sandbox_view": "期末复习周维塔系统重做全学期错题、背诵英语单词与古文。周五综合练习得85分，教师肯定其进步。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 20,
                 "description": "系统复习"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 15,
                 "description": "综合练习85分"},
                {"step_offset": 300, "event_type": "praise", "intensity": 10,
                 "description": "老师肯定进步"},
            ],
        },
        {
            "title": "妈妈回归",
            "time_label": "1月第2周 周一至周三",
            "snn_view": "这周妈妈的项目做完了，不加班了。周一晚上她做了饭等我回家，红烧排骨，炒青菜。我们一边吃饭一边聊，她说这阵子忙，没顾上我，问我期末复习得怎么样。我说还行。她说妈妈以后尽量不加班了。我说嗯。吃完饭我写作业，她坐在客厅，没开电视，就坐着。",
            "sandbox_view": "母亲结束加班恢复常态，为维塔做饭并询问复习情况，表示今后减少加班。维塔写作业期间母亲静坐客厅陪伴。",
            "events": [
                {"step_offset": 100, "event_type": "social_bond", "intensity": 15,
                 "description": "妈妈结束加班做晚饭"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 10,
                 "description": "聊学期近况，妈妈表态不加班"},
                {"step_offset": 300, "event_type": "question", "intensity": 5,
                 "description": "安心复习"},
            ],
        },
        {
            "title": "期末前一天",
            "time_label": "1月第2周 周四",
            "snn_view": "期末考试前一天，我晚上把错题本又翻了一遍。有几道题第一次做错，现在看着会了。妈妈端了杯牛奶进来，说明天好好考，考多少都行。我说嗯。她出去的时候把门轻轻带上。我坐在桌前，把笔袋拉好，放到书包最上面一层。",
            "sandbox_view": "期末考前夜维塔复习错题本，几道原错题已能独立解答。母亲送牛奶并淡化成绩压力。维塔提前整理文具。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "复习错题本"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 10,
                 "description": "妈妈送牛奶，淡化压力"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -5,
                 "description": "考前准备，轻微紧张"},
            ],
        },
        {
            "title": "期末考第一天",
            "time_label": "1月第2周 周五",
            "snn_view": "期末考第一天，上午语文，下午英语。语文作文题目是《我的一个学期》，我写了转学的事，写了食堂、接力、班会，写了陈默。写到最后一段，我顿了顿，把笔放下缓了一下，又接着写。下午英语比期中顺，听力这次听清了。",
            "sandbox_view": "期末首日语文、英语。语文作文题为《我的一个学期》，维塔以转学经历为素材完成作文。英语听力作答顺畅。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "语文考试，作文写学期经历"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 10,
                 "description": "作文素材充实，写作顺畅"},
                {"step_offset": 300, "event_type": "question", "intensity": 10,
                 "description": "英语听力顺畅"},
            ],
        },
        {
            "title": "期末考第二天",
            "time_label": "1月第3周 周一",
            "snn_view": "期末考第二天，上午数学。卷子发下来，我先扫了一遍，大部分题型都练过。有一道几何题绕了点，我画了辅助线，试了两条，第二条做出来了。交卷前检查，改了两处粗心。出考场，陈默在走廊等我，问我怎么样，我说应该比期中好。他说他也是。",
            "sandbox_view": "期末数学考试，维塔作答大部分顺利，一道几何题经两条辅助线尝试后解出。交卷前修正两处粗心。出考场与陈默互报感受，均自评好于期中。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 15,
                 "description": "数学考试，几何题卡顿"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 15,
                 "description": "解出几何题，自评好于期中"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 10,
                 "description": "与陈默互报考试感受"},
            ],
        },
        {
            "title": "等成绩",
            "time_label": "1月第3周 周二至周三",
            "snn_view": "考完等成绩的两天，比考试本身还难受。周二晚上我算了三遍数学，觉得自己能考80上下，但又怕算错。周三白天上课，老师没提成绩。晚上我写作业的时候，脑子里老是有个声音在算分。我放下笔，去客厅倒了杯水，回来继续写。",
            "sandbox_view": "成绩公布前两日维塔反复估算数学得分，存在明显紧张情绪，期间有注意力分散。",
            "events": [
                {"step_offset": 100, "event_type": "threat_social", "intensity": -10,
                 "description": "反复估算分数"},
                {"step_offset": 200, "event_type": "question", "intensity": 10,
                 "description": "等待成绩，注意力分散"},
                {"step_offset": 300, "event_type": "threat_social", "intensity": -5,
                 "description": "调整状态继续复习"},
            ],
        },
        {
            "title": "期末成绩",
            "time_label": "1月第3周 周四",
            "snn_view": "周四成绩出来。数学88，英语79，语文74，总分进了年级前25%。班主任在成绩单上写了评语：进步显著，继续保持。我看了好几遍。陈默数学72，他说比我定的70目标高。我说那就好。放学路上我们一路走一路聊，聊到校门口分开。",
            "sandbox_view": "期末成绩公布：数学88分，总分年级前25%，教师评语肯定进步。陈默数学72分达成目标。两人放学同行至校门口。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 25,
                 "description": "期末数学88分，年级前25%"},
                {"step_offset": 200, "event_type": "praise", "intensity": 15,
                 "description": "班主任评语肯定进步"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "与陈默互相祝贺"},
            ],
        },
        {
            "title": "散学典礼",
            "time_label": "1月第3周 周五",
            "snn_view": "散学典礼，操场站满了人，校长讲话，然后各班发成绩单和寒假作业。回教室，班主任说了几句，让大家寒假注意安全。放学时陈默说，下学期还要一起吃饭。我说好。他挥了挥手走了。我在教室门口站了一下，看了一会儿空教室，才下楼。",
            "sandbox_view": "散学典礼发放成绩单与寒假作业。陈默与维塔约定下学期继续共餐。维塔在教室门口停留后离校。",
            "events": [
                {"step_offset": 100, "event_type": "achievement", "intensity": 10,
                 "description": "散学典礼，领取成绩单"},
                {"step_offset": 200, "event_type": "social_bond", "intensity": 20,
                 "description": "陈默约定下学期共餐"},
                {"step_offset": 300, "event_type": "question", "intensity": 5,
                 "description": "回望教室，学期结束"},
            ],
        },
        {
            "title": "学期结束",
            "time_label": "1月第3周 周六",
            "snn_view": "周六在家收拾书包，把这学期的卷子一张张叠好。妈妈在客厅问，下学期还要不要转回原来的学校。我想了一下，说不想，这里挺好的。妈妈问为什么。我说有朋友，还当了班委。她没再问。晚上我给陈默发了条消息，说明天要不要去图书馆。他说好。",
            "sandbox_view": "学期结束后维塔整理一学期卷子。母亲询问是否转回原校，维塔明确表示不愿转学，理由为已建立友谊并担任班委。晚间约陈默图书馆，获应允。",
            "events": [
                {"step_offset": 100, "event_type": "question", "intensity": 10,
                 "description": "整理卷子，回顾学期"},
                {"step_offset": 200, "event_type": "achievement", "intensity": 10,
                 "description": "明确决定不转学"},
                {"step_offset": 300, "event_type": "social_bond", "intensity": 15,
                 "description": "约陈默图书馆，友谊延续"},
            ],
        },
    ],
}


def generate_arc(arc):
    """长线剧本: 段间连续模拟 (不 reset), 输出全部样本"""
    sim = ConcentrationSimulator()  # 故事线起点冷启动
    samples = []
    prev = None
    for seg_idx, seg in enumerate(arc["segments"]):
        conc = simulate_segment(sim, seg["events"])  # 继承 sim 状态
        pad = target_pad_from_conc(conc)
        causal_links = []
        if prev is not None:
            causal_links.append({
                "prev_segment": prev,
                "relation": "跨窗口状态延续 (慢通道残留传导)",
            })
        samples.append({
            "sample_id": seg_idx + 1,
            "arc_id": arc["arc_id"],
            "segment_idx": seg_idx,
            "prev_segment": prev,
            "time_label": seg["time_label"],
            "causal_links": causal_links,
            "events": seg["events"],
            "target_modulators": [round(v, 4) for v in conc],
            "target_pad": pad,
            "target_tool": 6,
        })
        prev = seg_idx
    return samples


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data",
                            "events", "curriculum_long_arc_50.jsonl")
    emit_text = False
    emit_raw = False
    args = [a for a in sys.argv[1:]]
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        elif args[i] == "--emit-text":
            emit_text = True
            i += 1
        elif args[i] == "--emit-raw":
            emit_raw = True
            i += 1
        else:
            i += 1

    samples = generate_arc(LONG_ARC)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"已保存到 {out_path}")
    print(f"共 {len(samples)} 段 / 1 条长线 (arc={LONG_ARC['arc_id']})")
    print(f"时间跨度: {LONG_ARC['time_span']}")

    if emit_raw:
        # 原始文本流 (含心理状态, 仅对比用)
        lines = []
        for s in samples:
            seg = LONG_ARC["segments"][s["segment_idx"]]
            lines.append(f"[{seg['time_label']}]")
            lines.append(seg["snn_view"])
            lines.append("")
        raw_path = os.path.join(os.path.dirname(out_path), "..", "scripts", "story_text_50_raw.txt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"原始文本流已保存到 {raw_path}")

    if emit_text:
        # 净化文本流 (SNN 输入: 剥离心理状态, 保留行为/环境/对话/生理)
        lines, n_rewrite = [], 0
        for s in samples:
            seg = LONG_ARC["segments"][s["segment_idx"]]
            pure = sanitize_first_person(seg["snn_view"])
            if pure != seg["snn_view"]:
                n_rewrite += 1
            lines.append(f"[{seg['time_label']}]")
            lines.append(pure)
            lines.append("")
        text_path = os.path.join(os.path.dirname(out_path), "..", "scripts", "story_text_50.txt")
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_bytes = os.path.getsize(text_path)
        n_chars = sum(len(l) for l in lines)
        print(f"净化文本流已保存到 {text_path} (重写 {n_rewrite}/{len(samples)} 段)")
        print(f"  {n_chars} 字符 / {n_bytes} 字节 (20K 步消耗 6666 字节 → 覆盖 {n_bytes/6666*100:.0f}%)")
        print()
        print("=== 净化对比示例 (前 8 处改写) ===")
        shown = 0
        for s in samples:
            seg = LONG_ARC["segments"][s["segment_idx"]]
            raw = seg["snn_view"]
            pure = sanitize_first_person(raw)
            if raw != pure and shown < 8:
                print(f"--- [{s['segment_idx']}] {seg['time_label']}")
                print(f"  原: {raw}")
                print(f"  净: {pure}")
                shown += 1

    for s in samples:
        evts = s["events"]
        offsets = sorted(set(e["step_offset"] for e in evts))
        par = sum(1 for o in offsets if sum(1 for e in evts if e["step_offset"] == o) > 1)
        print(f"  [{s['segment_idx']:02d}] {s['time_label']:<18s} "
              f"Oxy={s['target_modulators'][5]:.3f} "
              f"PAD=({s['target_pad'][0]:+.2f},{s['target_pad'][1]:+.2f},{s['target_pad'][2]:+.2f}) "
              f"并行={par}")


if __name__ == "__main__":
    main()
