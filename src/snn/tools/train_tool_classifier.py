# -*- coding: utf-8 -*-
"""
工具分类器训练 (2026-08-03)
输入: curriculum_middle_school.jsonl (2000 样本)
特征: 每事件类型 {count, 强度总和, 最大强度} + 事件总数 + target_modulators[6] + target_pad[3]
模型: 2 层 MLP (numpy 实现, ReLU + softmax CE), 可选类别加权
输出: 70/30 测试准确率 + 每类 recall + 混淆矩阵; 保存模型 .npz (供 C++ 集成)

可分性依据: chain→tool 确定性 12/12, 5 知识链靠 (question,achievement) 强度组合唯一区分
"""
import json
import collections
import numpy as np

DATA = r"f:\thetrueai\data\events\curriculum_middle_school.jsonl"
MODEL_OUT = r"f:\thetrueai\data\events\tool_classifier.npz"

TOOL_NAMES = {0: "生成器", 1: "计算器", 2: "草稿记录", 3: "长程检索", 4: "知识库查询", 5: "工具5", 6: "不调用"}


def load_rows():
    with open(DATA, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_features(rows, event_types):
    """特征: 每事件类型 [count, inten_sum, inten_max] + 事件总数 + mod[6] + pad[3]"""
    X, y = [], []
    for r in rows:
        ev = collections.defaultdict(list)
        for e in r["events"]:
            ev[e["event_type"]].append(e["intensity"])
        feat = []
        for t in event_types:
            vals = ev.get(t, [])
            feat.append(len(vals))                       # count
            feat.append(sum(vals))                       # inten_sum
            feat.append(max(vals) if vals else 0.0)      # inten_max
        feat.append(sum(len(v) for v in ev.values()))    # 事件总数
        feat.extend(r["target_modulators"])              # 6
        feat.extend(r["target_pad"])                     # 3
        X.append(feat)
        y.append(r["target_tool"])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


class MLP:
    def __init__(self, d_in, d_hidden, n_class, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_in, d_hidden) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.randn(d_hidden, n_class) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(n_class)

    def forward(self, X):
        self.h1 = np.maximum(0, X @ self.W1 + self.b1)     # relu
        self.logits = self.h1 @ self.W2 + self.b2
        m = self.logits.max(axis=1, keepdims=True)
        e = np.exp(self.logits - m)
        self.p = e / e.sum(axis=1, keepdims=True)
        return self.p

    def backward(self, X, y, class_w, lr):
        n = X.shape[0]
        # 加权 CE 梯度: dL/dz = w_y · (p - y)
        onehot = np.zeros_like(self.p)
        onehot[np.arange(n), y] = 1.0
        w = class_w[y][:, None]
        dlogits = w * (self.p - onehot) / n
        dW2 = self.h1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh1 = (dlogits @ self.W2.T) * (self.h1 > 0)
        dW1 = X.T @ dh1
        db1 = dh1.sum(axis=0)
        # SGD + 动量
        self.vW1 = 0.9 * getattr(self, "vW1", 0) + lr * dW1
        self.vb1 = 0.9 * getattr(self, "vb1", 0) + lr * db1
        self.vW2 = 0.9 * getattr(self, "vW2", 0) + lr * dW2
        self.vb2 = 0.9 * getattr(self, "vb2", 0) + lr * db2
        self.W1 -= self.vW1
        self.b1 -= self.vb1
        self.W2 -= self.vW2
        self.b2 -= self.vb2

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def loss(self, X, y, class_w):
        p = self.forward(X)
        w = class_w[y]
        return np.mean(-w * np.log(p[np.arange(len(y)), y] + 1e-12))


def train(Xtr, ytr, Xte, yte, class_w, epochs=800, lr=0.02, batch=64, seed=42):
    mlp = MLP(Xtr.shape[1], 48, int(ytr.max()) + 1, seed)
    rng = np.random.RandomState(0)
    n = len(ytr)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            mb = idx[s:s + batch]
            mlp.forward(Xtr[mb])
            mlp.backward(Xtr[mb], ytr[mb], class_w, lr)
        if ep % 200 == 0:
            tr_loss = mlp.loss(Xtr, ytr, class_w)
            te_acc = np.mean(mlp.predict(Xte) == yte)
            print(f"  ep={ep:4d}  train_loss={tr_loss:.4f}  test_acc={te_acc:.4f}")
    return mlp


def evaluate(y_true, y_pred, classes, label):
    acc = np.mean(y_true == y_pred)
    print(f"\n=== {label} ===")
    print(f"总体准确率: {acc:.4f}  (多数类基线: {np.bincount(y_true).max() / len(y_true):.4f})")
    print(f"{'类':<6}{'样本数':<8}{'正确':<8}{'recall':<10}")
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            print(f"{c}({TOOL_NAMES[c]}) 0        —        —")
            continue
        corr = np.sum(y_pred[mask] == c)
        print(f"{c}({TOOL_NAMES[c]}){mask.sum():<10}{corr:<8}{corr / mask.sum():.4f}")
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
    event_types = sorted({e["event_type"] for r in rows for e in r["events"]})
    classes = sorted(set(r["target_tool"] for r in rows))
    X, y = build_features(rows, event_types)
    print(f"样本: {len(rows)}, 特征维度: {X.shape[1]}, 事件类型: {event_types}")

    # 类别权重: inverse-frequency 归一化 (平均=1), 按工具值 0-6 索引 (类 5 无样本设 1)
    counts = np.bincount(y, minlength=7).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    class_w = inv / inv.mean()

    # 70/30 划分 + 标准化
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(y))
    tr, te = idx[: int(0.7 * len(y))], idx[int(0.7 * len(y)):]
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = y[tr], y[te]

    # 方案 1: 标准 CE (class_w 全 1, 7 槽)
    print("\n--- 标准 CE 训练 ---")
    mlp1 = train(Xtr, ytr, Xte, yte, np.ones(7))
    yp1 = mlp1.predict(Xte)
    acc1 = evaluate(yte, yp1, classes, "标准 CE (测试集)")

    # 方案 2: 类别加权 CE
    print("\n--- 类别加权 CE 训练 ---")
    mlp2 = train(Xtr, ytr, Xte, yte, class_w)
    yp2 = mlp2.predict(Xte)
    acc2 = evaluate(yte, yp2, classes, "类别加权 CE (测试集)")

    # 保存加权模型 (若更优) 或标准模型
    best = mlp2 if acc2 >= acc1 else mlp1
    np.savez(MODEL_OUT,
             W1=best.W1, b1=best.b1, W2=best.W2, b2=best.b2,
             feat_mean=mu, feat_std=sd, classes=np.array(classes),
             event_types=np.array(event_types))
    print(f"\n模型已保存: {MODEL_OUT} (acc={max(acc1, acc2):.4f})")


if __name__ == "__main__":
    main()
