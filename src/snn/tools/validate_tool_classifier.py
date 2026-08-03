# -*- coding: utf-8 -*-
"""
工具分类可分性验证 (2026-08-03)
目的: 判断"情境 -> 工具调用"映射是否可以被轻量分类器学习,
      为"工具分类外包给小模型"决策提供依据。

方法:
  1. 零模型基线: chain × tool 交叉表 (若 chain 与 tool 高度对应 → 映射结构化)
  2. 事件特征分类: 纯 numpy 逻辑回归 (one-vs-rest), 特征 = 事件类型统计
  3. 情绪特征增强: 特征加 target_modulators (模拟 SNN 情绪快照), 看是否提升
  4. 按类别评估 (排除类别不平衡假象)
"""
import json
import collections
import numpy as np

DATA = r"f:\thetrueai\data\events\curriculum_middle_school.jsonl"

# 工具语义 (与 curriculum 数据一致)
TOOL_NAMES = {0: "生成器", 1: "计算器", 2: "草稿记录", 3: "长程检索", 4: "知识库查询", 5: "工具5", 6: "不调用"}


def load_rows():
    with open(DATA, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def event_type_set(rows):
    s = set()
    for r in rows:
        for e in r["events"]:
            s.add(e["event_type"])
    return sorted(s)


def build_features(rows, event_types, use_mod=False):
    """特征: 每事件类型计数 + 强度总和 + 是否出现; 可选 + target_modulators[6]"""
    X, y = [], []
    for r in rows:
        ev_counts = collections.Counter()
        ev_inten = collections.Counter()
        for e in r["events"]:
            ev_counts[e["event_type"]] += 1
            ev_inten[e["event_type"]] += e["intensity"]
        feat = []
        for t in event_types:
            feat.append(ev_counts[t])
            feat.append(ev_inten[t])
        if use_mod:
            feat.extend(r["target_modulators"])
        X.append(feat)
        y.append(r["target_tool"])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def fit_logreg(X, y, classes, lr=0.1, epochs=300, seed=42):
    """one-vs-rest 逻辑回归 (numpy 实现), 返回 W (n_class, n_feat+1)"""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])  # bias
    W = rng.randn(len(classes), d + 1) * 0.01
    for _ in range(epochs):
        for k, c in enumerate(classes):
            yb = (y == c).astype(np.float64)
            p = 1.0 / (1.0 + np.exp(-Xb @ W[k]))
            grad = Xb.T @ (p - yb) / n
            W[k] -= lr * grad
    return W


def predict(X, W):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.argmax(Xb @ W.T, axis=1)


def evaluate(y_true, y_pred, classes, label):
    n = len(y_true)
    acc = np.mean(y_true == y_pred)
    # 每类 recall
    print(f"\n=== {label} ===")
    print(f"总体准确率: {acc:.4f}  (多数类基线: {max(np.bincount(y_true)) / n:.4f})")
    print(f"{'类':<6}{'样本数':<8}{'正确':<8}{'recall':<10}")
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            print(f"{c}({TOOL_NAMES[c]}){'':<4}0        —        —")
            continue
        corr = np.sum(y_pred[mask] == c)
        print(f"{c}({TOOL_NAMES[c]}){'':<4}{mask.sum():<8}{corr:<8}{corr / mask.sum():.4f}")
    # 混淆矩阵 (仅显示有样本的类)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c) & (y_pred == c2))
    print("混淆矩阵 (行=真值, 列=预测):")
    print("      " + "".join(f"{c:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>6}" + "".join(f"{v:>6}" for v in cm[i]))
    return acc


def main():
    rows = load_rows()
    event_types = event_type_set(rows)
    classes = sorted(set(r["target_tool"] for r in rows))
    print(f"总样本: {len(rows)}, 事件类型: {event_types}, 工具类别: {classes}")

    # 1) chain × tool 交叉表 (零模型基线)
    ct = collections.defaultdict(collections.Counter)
    for r in rows:
        ct[r["chain"]][r["target_tool"]] += 1
    print(f"\n=== chain × 工具 交叉表 ===")
    single = 0
    for ch, c in sorted(ct.items()):
        if len(c) == 1:
            single += 1
        print(f"  {ch:<32} {dict(sorted(c.items()))}")
    print(f"chain→工具一一对应: {single}/{len(ct)}")

    # 2) 事件特征分类 (无情绪特征)
    X, y = build_features(rows, event_types, use_mod=False)
    print(f"\n特征维度 (仅事件): {X.shape[1]}")
    W = fit_logreg(X, y, classes)
    y_pred = predict(X, W)
    evaluate(y, y_pred, classes, "事件特征逻辑回归 (训练集内)")

    # 3) 事件 + 情绪目标特征 (模拟 SNN 情绪快照加入后的增益)
    Xm, _ = build_features(rows, event_types, use_mod=True)
    print(f"\n特征维度 (事件+情绪): {Xm.shape[1]}")
    Wm = fit_logreg(Xm, y, classes)
    y_pred_m = predict(Xm, Wm)
    evaluate(y, y_pred_m, classes, "事件+情绪特征逻辑回归 (训练集内)")

    # 4) 随机划分验证 (70/30, 避免纯拟合假象)
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    Ws = fit_logreg(X[tr], y[tr], classes)
    y_pred_te = predict(X[te], Ws)
    acc_ev = np.mean(y_pred_te == y[te])
    Ws_m = fit_logreg(Xm[tr], y[tr], classes)
    y_pred_te_m = predict(Xm[te], Ws_m)
    acc_m = np.mean(y_pred_te_m == y[te])
    print(f"\n=== 70/30 划分测试集 ===")
    print(f"仅事件特征: 测试准确率 {acc_ev:.4f}")
    print(f"事件+情绪特征: 测试准确率 {acc_m:.4f}")


if __name__ == "__main__":
    main()
