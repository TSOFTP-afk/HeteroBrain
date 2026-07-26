#!/usr/bin/env python3
"""Prepare WikiText-2 BPE token stream for Stage 2e SNN training.

使用 HuggingFace distilgpt2 tokenizer 将 WikiText-2 编码为 int32 token id 序列,
输出供 SNN BPTT 代理梯度训练使用的二进制流和元数据 JSON。

用法:
    python prepare_bpe_data.py --dataset wikitext-2 --output-dir . --tokenizer distilgpt2

依赖:
    pip install transformers datasets numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# WikiText-2 在 HuggingFace datasets 中的标识
WIKITEXT_DATASET_NAME = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"

# 输出文件名
BIN_FILENAME = "wikitext2_bpe.bin"
META_FILENAME = "wikitext2_bpe_meta.json"


def load_tokenizer(tokenizer_name: str):
    """加载 HuggingFace tokenizer, 返回 (tokenizer, vocab_size)。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    # 显式打印, 方便排查加载错误
    print(f"Loaded tokenizer: {tokenizer_name} (vocab_size={vocab_size})")
    return tokenizer, vocab_size


def load_dataset_splits(dataset: str):
    """加载 WikiText-2 数据集, 返回 dict {split: list[str]}。"""
    from datasets import load_dataset

    if dataset != "wikitext-2":
        raise ValueError(
            f"目前只支持 'wikitext-2', 收到: {dataset!r}"
        )

    ds = load_dataset(WIKITEXT_DATASET_NAME, WIKITEXT_CONFIG)
    print(f"Loaded dataset: {WIKITEXT_CONFIG}")

    splits = {
        "train": list(ds["train"]["text"]),
        "validation": list(ds["validation"]["text"]),
        "test": list(ds["test"]["text"]),
    }
    return splits


def encode_split(tokenizer, texts, batch_size: int = 1000, max_length: int = 1024):
    """把一组文本编码为 BPE token id 序列 (扁平化拼接)。

    - 跳过空文本 (wikitext-2 中有空行作为文档分隔)
    - 对单条超长文本做截断 (max_length), 避免单条文档过大
    - 逐 batch 编码, 减少峰值内存占用
    返回: list[int]
    """
    all_ids: list[int] = []

    # 1) 过滤空文本与纯空白文本
    filtered = [t for t in texts if t and t.strip()]

    # 2) 分批编码
    total = len(filtered)
    for start in range(0, total, batch_size):
        batch = filtered[start:start + batch_size]
        # add_special_tokens=False: 我们要的是连续 BPE 流, 不要 EOS/分隔符
        # truncation=True, max_length=max_length: 截断超长文档
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        for ids in enc["input_ids"]:
            all_ids.extend(ids)
        # 进度提示 (每 10 个 batch 一次)
        if (start // batch_size) % 10 == 0:
            sys.stdout.flush()

    return all_ids


def save_bin(tokens, out_path: Path) -> int:
    """把 token id 列表保存为 int32 little-endian 二进制流, 返回字节数。"""
    import numpy as np

    arr = np.asarray(tokens, dtype=np.int32)
    arr.tofile(out_path)
    return arr.nbytes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 WikiText-2 编码为 distilgpt2 BPE token 流, 供 SNN 训练使用。"
    )
    parser.add_argument(
        "--dataset",
        default="wikitext-2",
        help="数据集名称, 目前只支持 wikitext-2 (默认: wikitext-2)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="输出目录 (默认: 当前目录)",
    )
    parser.add_argument(
        "--tokenizer",
        default="distilgpt2",
        help="HuggingFace tokenizer 名称 (默认: distilgpt2)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 tokenizer
    tokenizer, vocab_size = load_tokenizer(args.tokenizer)

    # 2) 加载 WikiText-2
    splits = load_dataset_splits(args.dataset)

    # 3) 逐 split 编码
    print("Encoding train split...")
    train_ids = encode_split(tokenizer, splits["train"])
    print(f"Train: {len(train_ids)} tokens")

    print("Encoding validation split...")
    validation_ids = encode_split(tokenizer, splits["validation"])
    print(f"Validation: {len(validation_ids)} tokens")

    print("Encoding test split...")
    test_ids = encode_split(tokenizer, splits["test"])
    print(f"Test: {len(test_ids)} tokens")

    # 4) 保存 train split 为 int32 binary
    bin_path = out_dir / BIN_FILENAME
    bin_bytes = save_bin(train_ids, bin_path)
    bin_mb = bin_bytes / (1024 * 1024)
    print(f"Saved: {BIN_FILENAME} ({bin_mb:.1f} MB)")

    # 5) 保存元数据 JSON
    total_tokens = len(train_ids) + len(validation_ids) + len(test_ids)
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dataset": WIKITEXT_CONFIG,
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "test_tokens": len(test_ids),
        "total_tokens": total_tokens,
    }
    meta_path = out_dir / META_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved: {META_FILENAME}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
