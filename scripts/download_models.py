#!/usr/bin/env python3
"""
vita 模型权重下载脚本

下载 MiniCPM5-1B INT4 GGUF 等模型到 models/ 目录。
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "minicpm5-1b-int4": {
        "url": "https://modelscope.cn/api/v1/models/OpenBMB/MiniCPM5-1B-GGUF/repo?Revision=master&FilePath=MiniCPM5-1B-Q4_K_M.gguf",
        "filename": "MiniCPM5-1B-Q4_K_M.gguf",
        "size_gb": 0.65,
        "description": "MiniCPM5-1B INT4 量化版本 (Q4_K_M), ~656MB, AA-Index 小模型第一",
    },
    "qwen3-0.6b-int4": {
        "url": "https://modelscope.cn/api/v1/models/Qwen/Qwen3-0.6B-GGUF/repo?Revision=master&FilePath=Qwen3-0.6B-Q4_K_M.gguf",
        "filename": "Qwen3-0.6B-Q4_K_M.gguf",
        "size_gb": 0.4,
        "description": "Qwen3-0.6B INT4 替代方案, 0.4GB, 双模式 (思考/非思考)",
    },
}


def download(url: str, dest: Path, size_gb: float) -> None:
    """流式下载, 显示进度"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}")
    print(f"目标: {dest}")
    print(f"预计大小: {size_gb} GB")

    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  进度: {downloaded / 1024 / 1024:.1f} MB / {total / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print(f"\n完成: {dest}")


def main():
    parser = argparse.ArgumentParser(description="vita 模型下载")
    parser.add_argument("--model", required=True, choices=MODELS.keys(), help="模型名称")
    parser.add_argument("--output-dir", default="models", help="输出目录")
    args = parser.parse_args()

    spec = MODELS[args.model]
    dest = Path(args.output_dir) / spec["filename"]

    if dest.exists():
        print(f"已存在: {dest}, 跳过下载")
        return

    print(f"=== {spec['description']} ===")
    try:
        download(spec["url"], dest, spec["size_gb"])
    except Exception as e:
        print(f"\n下载失败: {e}", file=sys.stderr)
        print(f"请手动下载: {spec['url']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
