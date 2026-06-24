"""
cross_modal_translation.py — scGPT 跨模态翻译评估脚本

功能：
  1. 加载微调后的模型权重（scGPT/result/best_finetuned.pt）
  2. 推断时将 ATAC 模态 token 的值全部遮蔽（mask_ratio=1.0 for ATAC）
  3. 让模型从 RNA 信息重建 ATAC 信号
  4. 将模型输出的分箱 logits 还原为连续值（argmax + bin center 映射）
  5. 与真实 ATAC 信号对比，计算跨模态翻译指标：
     - Reconstruction MSE（均方误差）
     - Pearson 相关系数（全局与逐细胞）
     - Cell-state cosine similarity（真实 vs 预测 ATAC 嵌入的余弦相似度）
  6. 结果输出到 scGPT/result/translation_metrics.csv

运行方式：
  python scGPT/cross_modal_translation.py
  python scGPT/cross_modal_translation.py --batch_size 64
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from finetune_integration import MultiomeDataset, build_model

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "scGPT" / "data" / "pbmc_scgpt_input_labeled.h5ad"
ATAC_PATH = PROJECT_ROOT / "outputs" / "pbmc_granulocyte_sorted_10k" / "atac_processed.h5ad"
CHECKPOINT_PATH = PROJECT_ROOT / "scGPT" / "result" / "best_finetuned.pt"
RESULT_DIR = PROJECT_ROOT / "scGPT" / "result"

BATCH_SIZE = 64
MAX_SEQ_LEN = 1200


# ===========================================================================
# 辅助函数
# ===========================================================================

def bin_centers(n_bins: int) -> np.ndarray:
    """
    返回每个分箱区间的"中心值"，用于将 argmax 离散预测还原为连续值。

    分箱规则与 prepare_data.py 的 binning_expression 一致：
      - bin 0 → 值为 0（未检测）
      - bin k (k>=1) → 映射到 k / (n_bins - 1)（归一化到 [0,1]，再由下游比较）

    这里返回归一化后的连续值，方便与真实二值 ATAC 信号（0 or 1）对比。
    """
    centers = np.zeros(n_bins, dtype=np.float32)
    for k in range(1, n_bins):
        centers[k] = k / (n_bins - 1)
    return centers


@torch.no_grad()
def predict_atac(
    model: nn.Module,
    loader: DataLoader,
    adata: ad.AnnData,
    n_bins: int,
    device: torch.device,
) -> np.ndarray:
    """
    用 RNA-only 条件重建 ATAC 信号。

    推断时：
      - 将 ATAC 位置（modality_id == 1）的值设为 0（即"完全遮蔽"）
      - 保留 RNA 值不变
      - 取模型输出 logits 的 argmax 作为预测分箱 id
      - 通过 bin_centers 映射为连续值

    返回：
      predicted_atac: (n_cells, n_atac_features) 的预测值矩阵
    """
    model.eval()
    centers = bin_centers(n_bins)
    n_atac = adata.uns["n_atac_features"]
    n_rna = adata.uns["n_rna_features"]
    all_predictions = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        values = batch["values"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        B, L = values.shape

        # 完全遮蔽 ATAC token 的值
        atac_pos = modality_ids == 1
        values_masked = values.clone()
        values_masked[atac_pos] = 0

        # 前向推断
        try:
            output = model(
                src=token_ids,
                values=values_masked,
                src_key_padding_mask=padding_mask,
                batch_labels=None,
                CLS=False, CCE=False, MVC=True, ECS=False,
            )
            logits = output["mvc_output"]  # (B, L, n_bins)
        except Exception:
            logits = model(token_ids, values_masked, modality_ids, padding_mask)

        # argmax → bin id → 连续值
        pred_bin_ids = logits.argmax(dim=-1).cpu().numpy()  # (B, L)
        pred_values = centers[pred_bin_ids]                  # (B, L) float

        # 仅提取 ATAC 位置的预测值
        # 注意：ATAC 特征位于序列的后 n_atac 位（对应 modality_id==1）
        # 实际索引取决于 feature_indices 采样，因此用 modality_ids 定位
        atac_pos_cpu = atac_pos.cpu().numpy()  # (B, L)
        batch_atac_preds = []
        for b in range(B):
            atac_idx = np.where(atac_pos_cpu[b])[0]
            if len(atac_idx) == 0:
                # 当 ATAC 全部被 padding 时，用零填充
                batch_atac_preds.append(np.zeros(n_atac, dtype=np.float32))
            else:
                pred = pred_values[b, atac_idx]
                # 若 atac_idx 长度不等于 n_atac（因 max_seq_len 截断），截断或补零
                if len(pred) > n_atac:
                    pred = pred[:n_atac]
                elif len(pred) < n_atac:
                    pred = np.pad(pred, (0, n_atac - len(pred)))
                batch_atac_preds.append(pred)

        all_predictions.append(np.stack(batch_atac_preds, axis=0))

    return np.concatenate(all_predictions, axis=0)  # (n_cells, n_atac)


# ===========================================================================
# 评估指标
# ===========================================================================

def compute_reconstruction_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """全局均方误差（所有细胞 × 所有峰）。"""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_pearson_global(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """全局 Pearson 相关系数（将矩阵展平后计算）。"""
    t = y_true.flatten()
    p = y_pred.flatten()
    corr = np.corrcoef(t, p)[0, 1]
    return float(corr)


def compute_pearson_per_cell(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    逐细胞 Pearson 相关系数，取中位数作为汇总指标。
    （衡量每个细胞的 ATAC 信号分布是否被准确还原）
    """
    n_cells = y_true.shape[0]
    per_cell_corr = []
    for i in range(n_cells):
        t = y_true[i]
        p = y_pred[i]
        if t.std() < 1e-6 or p.std() < 1e-6:
            # 标准差接近 0 时跳过（全 0 行无意义）
            continue
        corr = np.corrcoef(t, p)[0, 1]
        per_cell_corr.append(corr)
    return float(np.median(per_cell_corr)) if per_cell_corr else 0.0


def compute_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cell-state cosine similarity：逐细胞计算真实与预测 ATAC 向量的余弦相似度，取均值。
    """
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(y_true, y_pred)
    # cos_sim 是 (n_cells, n_cells) 矩阵，取对角线即逐细胞自身相似度
    per_cell = np.diag(cos_sim)
    return float(np.mean(per_cell))


def compute_auprc_per_peak(y_true_binary: np.ndarray, y_pred: np.ndarray) -> float:
    """
    逐峰 AUPRC（Area Under Precision-Recall Curve），取均值。
    衡量模型是否能将"开放"峰（y=1）的预测值排在"关闭"峰（y=0）之前。
    """
    from sklearn.metrics import average_precision_score
    n_peaks = y_true_binary.shape[1]
    auprcs = []
    for j in range(n_peaks):
        t = y_true_binary[:, j]
        p = y_pred[:, j]
        if t.sum() == 0 or t.sum() == len(t):
            # 该峰全 0 或全 1，AUPRC 无意义
            continue
        auprcs.append(average_precision_score(t, p))
    return float(np.mean(auprcs)) if auprcs else 0.0


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 跨模态翻译评估脚本")
    parser.add_argument("--data_path", type=Path, default=DATA_PATH)
    parser.add_argument("--atac_path", type=Path, default=ATAC_PATH)
    parser.add_argument("--checkpoint_path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--result_dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    # 检查前置文件
    for p, hint in [
        (args.data_path, "请先运行：python scGPT/annotate_cells.py"),
        (args.checkpoint_path, "请先运行：python scGPT/finetune_integration.py"),
    ]:
        if not p.exists():
            print(f"❌ 文件不存在：{p}\n   {hint}")
            sys.exit(1)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 使用设备：{device}")

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    adata = ad.read_h5ad(args.data_path)
    n_bins = adata.uns.get("n_bins", 51)
    n_atac = adata.uns["n_atac_features"]
    n_rna = adata.uns["n_rna_features"]
    print(f"  {adata.n_obs} 细胞 | RNA 特征：{n_rna} | ATAC 特征：{n_atac}")

    # 获取真实 ATAC 信号（从 adata 的 values 层提取 ATAC 列）
    atac_mask = adata.var["modality_type"] == "atac"
    if sp.issparse(adata.layers["values"]):
        true_atac = adata.layers["values"][:, atac_mask].toarray().astype(np.float32)
    else:
        true_atac = np.asarray(adata.layers["values"][:, atac_mask], dtype=np.float32)
    print(f"  真实 ATAC 矩阵：{true_atac.shape}，均值开放率：{true_atac.mean():.4f}")

    # 2. 加载模型
    print("\n[2/5] 加载微调模型...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    vocab_size = checkpoint.get("vocab_size", 60000)
    saved_args = checkpoint.get("args", {})
    embed_dim = saved_args.get("embed_dim", 512)
    n_layers = saved_args.get("n_layers", 12)
    n_heads = saved_args.get("n_heads", 8)

    model = build_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_bins=n_bins,
        pretrained_path=None,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("  ✅ 模型权重加载完成")

    # 3. 构建 DataLoader
    dataset = MultiomeDataset(adata, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. 模型预测 ATAC
    print(f"\n[3/5] 使用 RNA 信号预测 ATAC（{adata.n_obs} 细胞）...")
    t0 = time.time()
    pred_atac = predict_atac(model, loader, adata, n_bins, device)
    pred_time = time.time() - t0
    print(f"  ✅ 预测完成，耗时：{pred_time:.1f} s")
    print(f"  预测 ATAC 矩阵：{pred_atac.shape}")

    # 保存预测值
    np.save(args.result_dir / "predicted_atac.npy", pred_atac)
    print(f"  ✅ 预测值已保存：{args.result_dir / 'predicted_atac.npy'}")

    # 5. 计算评估指标
    print("\n[4/5] 计算跨模态翻译指标...")
    metrics: dict[str, float] = {}

    metrics["reconstruction_mse"] = compute_reconstruction_mse(true_atac, pred_atac)
    print(f"  Reconstruction MSE          : {metrics['reconstruction_mse']:.6f}")

    metrics["pearson_global"] = compute_pearson_global(true_atac, pred_atac)
    print(f"  Pearson（全局）              : {metrics['pearson_global']:.4f}")

    metrics["pearson_per_cell_median"] = compute_pearson_per_cell(true_atac, pred_atac)
    print(f"  Pearson（逐细胞中位数）       : {metrics['pearson_per_cell_median']:.4f}")

    metrics["cosine_similarity"] = compute_cosine_similarity(true_atac, pred_atac)
    print(f"  Cell-state cosine similarity : {metrics['cosine_similarity']:.4f}")

    # AUPRC（二值 ATAC 标签）
    true_atac_binary = (true_atac > 0).astype(np.float32)
    metrics["auprc_per_peak_mean"] = compute_auprc_per_peak(true_atac_binary, pred_atac)
    print(f"  AUPRC（逐峰均值）             : {metrics['auprc_per_peak_mean']:.4f}")

    # 6. 保存结果
    print("\n[5/5] 保存评估指标...")
    out_csv = args.result_dir / "translation_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"  ✅ 指标已保存：{out_csv}")

    print("\n🎉 跨模态翻译评估完成！")
    print(f"   指标汇总：")
    for k, v in metrics.items():
        print(f"     {k:<35s}: {v:.4f}")


if __name__ == "__main__":
    main()
