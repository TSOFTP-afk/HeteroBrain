# 数据目录

本目录存放 vita 训练与评测所需的数据集。**大文件不入库**，请用 `scripts/` 下的脚本下载或预处理。

## 数据集

| 数据集 | 大小 | 用途 | 获取方式 |
|---|---|---|---|
| `lccc_base.txt` | 829MB | 中文对话预训练 (从 legacy 继承) | `scripts/prepare_lccc.py` |
| `wikitext2_bpe.bin` | ~12MB | BPE token 流 (SNN 训练) | `scripts/prepare_bpe_data.py` |
| `ceval_subset.json` | ~5MB | 中文评测 | `scripts/download_ceval.py` |

## 旧数据归档

上一代项目（pure-snn-language）的训练日志、PCA 聚类 CSV 等已归档到 `docs/archive/data_logs/`，仅供参考。

## 数据格式约定

- **BPE stream**: int32 little-endian binary, 4 字节/token, 与 legacy/stage2e/tools/prepare_bpe_data.py 兼容
- **对话语料**: UTF-8 纯文本, 每行一轮对话
- **评测集**: JSONL, 字段 `{"input": str, "reference": str, "category": str}`
