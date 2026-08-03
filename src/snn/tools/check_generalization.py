# -*- coding: utf-8 -*-
"""
诚实检验: 课程数据多样性 + 分类器是否真泛化 (2026-08-03)
1. 独特样本模式去重: (事件类型+强度+情绪) 组合去重后剩多少 → 数据真实多样性
2. 按 chain 留出验证: 训练时完全未见过的 chain, 分类器能否推断 → 真泛化 vs 记忆
"""
import json
import collections
import numpy as np

DATA = r"f:\thetrueai\data\events\curriculum_middle_school.jsonl"

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]

# ---- 1. 多样性统计 ----
print("=== 1. 数据真实多样性 ===")
chains = sorted({r["chain"] for r in rows})
uniq_events = collections.Counter()          # 事件序列 (type,inten) 去重
uniq_evmod = collections.Counter()           # 事件序列 + 情绪目标 去重
uniq_chain_combo = collections.Counter()     # chain 内事件组合变体
for r in rows:
    ev_seq = tuple(sorted((e["event_type"], e["intensity"]) for e in r["events"]))
    uniq_events[ev_seq] += 1
    uniq_evmod[(ev_seq, tuple(r["target_modulators"]), tuple(r["target_pad"]))] += 1
    uniq_chain_combo[(r["chain"], ev_seq)] += 1

print(f"总样本: {len(rows)}")
print(f"独特事件序列(类型+强度): {len(uniq_events)}  ({len(uniq_events)}/{len(rows)} 占比 {len(uniq_events)/len(rows)*100:.1f}%)")
print(f"独特(事件+情绪)组合: {len(uniq_evmod)}")
print(f"chain 内事件组合变体数: 每 chain 平均 {np.mean([len(set(uniq_chain_combo[k] for k in uniq_chain_combo if k[0]==ch)) for ch in chains]):.1f} 种")

# ---- 2. 按 chain 留出验证 ----
print("\n=== 2. 按 chain 留出验证 (真泛化测试) ===")
# 用与 train_tool_classifier 相同的 MLP + 特征

def build_features(rows, event_types):
    X, y = [], []
    for r in rows:
        ev = collections.defaultdict(list)
        for e in r["events"]:
            ev[e["event_type"]].append(e["intensity"])
        feat = []
        for t in event_types:
            vals = ev.get(t, [])
            feat += [len(vals), sum(vals), max(vals) if vals else 0.0]
        feat.append(sum(len(v) for v in ev.values()))
        feat += r["target_modulators"] + r["target_pad"]
        X.append(feat)
        y.append(r["target_tool"])
    return np.array(X), np.array(y)

event_types = sorted({e["event_type"] for r in rows for e in r["events"]})
chains = sorted({r["chain"] for r in rows})

# 留出 2 个知识链 + 2 个情感链 (共 4 个未见 chain)
held_out = ["math_problem_chain", "writing_task_chain",
            "peer_acceptance_chain", "exam_failure_recovery_chain"]
tr_mask = np.array([r["chain"] not in held_out for r in rows])
te_mask = ~tr_mask
print(f"训练: {tr_mask.sum()} 样本 ({len(chains)-len(held_out)} 个 chain), "
      f"测试(未见 chain): {te_mask.sum()} 样本 ({len(held_out)} 个 chain)")

# 复用 MLP 定义 (从 train_tool_classifier 导入避免重复)
import importlib.util
spec = importlib.util.spec_from_file_location("tcl", r"f:\thetrueai\src\snn\tools\train_tool_classifier.py")
tcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcl)

X, y = build_features(rows, event_types)
mu, sd = X[tr_mask].mean(axis=0), X[tr_mask].std(axis=0) + 1e-8
Xtr, Xte = (X[tr_mask] - mu) / sd, (X[te_mask] - mu) / sd
ytr, yte = y[tr_mask], y[te_mask]

mlp = tcl.MLP(Xtr.shape[1], 48, 7)
rng = np.random.RandomState(0)
n = len(ytr)
for ep in range(800):
    idx = rng.permutation(n)
    for s in range(0, n, 64):
        mb = idx[s:s + 64]
        mlp.forward(Xtr[mb])
        mlp.backward(Xtr[mb], ytr[mb], np.ones(7), 0.02)
    if ep % 200 == 0:
        te_acc = np.mean(mlp.predict(Xte) == yte)
        tr_acc = np.mean(mlp.predict(Xtr) == ytr)
        print(f"  ep={ep:4d}  train_acc={tr_acc:.4f}  unseen_chain_test_acc={te_acc:.4f}")

print(f"\n结果: 未见 chain 测试准确率 {np.mean(mlp.predict(Xte) == yte):.4f} "
      f"(若远低于 100% → 之前 100% 是记忆而非泛化)")
