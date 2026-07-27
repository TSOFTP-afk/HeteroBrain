# 模型权重目录

本目录存放 HeteroBrain 运行所需的模型权重。**大文件不入库**，请用 `scripts/download_models.py` 下载。

## 必需文件

| 文件 | 大小 | 来源 | 用途 |
|---|---|---|---|
| `MiniCPM5-1B-Q4_K_M.gguf` | ~656MB | [ModelScope](https://modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF) / HuggingFace | LLM 子系统 |
| `snn_60k_v3.snn2e` | ~32MB | 从 legacy/stage2e 训练产物复制 | SNN 子系统 (Phase 2) |
| `spike_to_embedding.pt` | ~8MB | PyTorch 离线训练 (Phase 3) | Bridge 转换层 |

## 下载命令

```powershell
# MiniCPM5-1B INT4 GGUF (Phase 1 必需)
python scripts/download_models.py --model minicpm5-1b-int4

# SNN 60K 神经元权重 (Phase 2)
python scripts/download_models.py --model snn-60k

# Bridge 投影矩阵 (Phase 3, 需先训练)
python scripts/train_bridge_projection.py --output models/spike_to_embedding.pt
```

## 替代模型

如果 MiniCPM5-1B 不可用，可使用以下替代：

| 模型 | 大小 | 中文能力 | 备注 |
|---|---|---|---|
| Qwen3-0.6B Q4_K_M | ~0.4GB | AA | 28 层, 双模式 |
| Phi-3.5-mini Q4_K_M | ~1.8GB | A | 英文优先, 中文可用 |
