"""curriculum_generator — 学生生活样本生成软件。

确定性逻辑 (完整链条 + 数据库调动): 场景库 → 因果组合 → 浓度模拟 → 校验
生成式逻辑 (本地 MiniCPM, 零远程 API 费用): 场景 → 情感反应链 (简单逻辑链条)

用法:
  python main.py --stage middle_school --samples 20000 --rebuild-scenes
  python main.py --stage middle_school --samples 5000 --use-llm --llm-frac 0.3
"""
import argparse
import json
import os

from engine.chain_engine import generate_samples
from engine.scene_builder import (build_scene_library, load_scene_library,
                                  save_scene_library)
from engine.simulator import EventKB, load_event_kb
from engine.validator import print_report, validate_samples

STAGE_BASELINE = {
    # GENE_MAP 列顺序 [DA, ACh, NE, 5HT, GABA, Oxy]
    "middle_school": [0.22, 0.28, 0.25, 0.15, 0.18, 0.20],
    "high_school":   [0.20, 0.25, 0.22, 0.18, 0.22, 0.22],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "kb")
OUT_DIR = os.path.join(BASE_DIR, "out")


def main():
    ap = argparse.ArgumentParser(description="学生生活课程样本生成软件")
    ap.add_argument("--stage", choices=["middle_school", "high_school"],
                    default="middle_school")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-events", type=int, default=3)
    ap.add_argument("--rebuild-scenes", action="store_true",
                    help="重建场景库 (维度组合 → kb/scenes.json)")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--use-llm", action="store_true", help="启用本地 MiniCPM 生成事件链")
    ap.add_argument("--llm-frac", type=float, default=0.3,
                    help="LLM 生成样本占比 (0-1), 其余走规则引擎")
    ap.add_argument("--llm-verbose", action="store_true")
    args = ap.parse_args()

    # 1. 事件知识库
    event_kb = EventKB(load_event_kb(os.path.join(KB_DIR, "events.json")))

    # 2. 场景库 (数据库: kb/scenes.json, 可查可改)
    scenes_path = os.path.join(KB_DIR, "scenes.json")
    if args.rebuild_scenes or not os.path.exists(scenes_path):
        scenes = build_scene_library(args.stage)
        save_scene_library(scenes, scenes_path)
        print(f"[SceneBuilder] 场景库重建: {len(scenes)} 个场景模板 → {scenes_path}")
    else:
        scenes = load_scene_library(scenes_path)
        print(f"[SceneBuilder] 加载场景库: {len(scenes)} 个场景模板 (来自 {scenes_path})")
    stage_scenes = [s for s in scenes if s["stage"] == args.stage]
    if not stage_scenes:
        # 场景库缺少当前阶段 → 构建该阶段并合并保存 (scenes.json 为多阶段合并库)
        print(f"[SceneBuilder] 场景库缺少 {args.stage}, 构建该阶段并合并...")
        stage_scenes = build_scene_library(args.stage)
        merged = scenes + [s for s in stage_scenes
                           if s["stage"] not in {x["stage"] for x in scenes}]
        save_scene_library(merged, scenes_path)
        print(f"[SceneBuilder] 合并后场景库: {len(merged)} 个场景模板")

    # 3. LLM 链生成器 (可选, 本地 MiniCPM)
    llm_chain_fn = None
    if args.use_llm:
        from engine.llm_client import MiniCPMClient
        client = MiniCPMClient(verbose=args.llm_verbose)
        print(f"[MiniCPM] 本地推理已启用: {os.path.basename(client.model)}")

        def llm_chain(scene):
            chain = client.generate_event_chain(scene["desc"])
            if chain is None or len(chain) > 3:
                return None  # 非法链回退规则引擎 (offset 池仅 {100,200,300})
            # 场景基调一致性: LLM 链极性须与场景基调相符, 否则回退规则引擎
            ints = [e["intensity"] for e in chain]
            has_pos = any(i > 0 for i in ints)
            has_neg = any(i < 0 for i in ints)
            tone = scene["tone"]
            if tone == "pos" and not has_pos:
                return None
            if tone == "neg" and not has_neg:
                return None
            if tone == "mixed" and not (has_pos and has_neg):
                return None
            # 补 offset: 按因果顺序分配 {100,200,300}
            import random
            offsets = sorted(random.Random(scene["scene_id"]).sample([100, 200, 300], len(chain)))
            for e, off in zip(chain, offsets):
                e["step_offset"] = off
            return chain

        llm_chain_fn = llm_chain

    # 4. 生成样本
    samples = generate_samples(stage_scenes, event_kb, STAGE_BASELINE,
                               args.samples, args.seed, args.max_events,
                               llm_chain_fn=llm_chain_fn,
                               use_llm_frac=args.llm_frac if args.use_llm else 0.0)

    # 5. 校验 + 输出
    report = validate_samples(samples, set(event_kb.events.keys()))
    print(f"\n[generate] {args.stage} 生成完成:")
    print_report(report)
    if not report["ok"]:
        print("[generate] 校验未通过, 仍写出样本供检查")

    out_path = args.output or os.path.join(OUT_DIR, f"curriculum_{args.stage}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  输出文件: {out_path}")


if __name__ == "__main__":
    main()
