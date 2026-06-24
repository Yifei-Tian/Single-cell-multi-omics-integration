"""
visualize.py — scGPT 结果可视化脚本

功能：
  1. 在联合嵌入上做 UMAP 降维
  2. 绘制并保存以下图（300 dpi PNG）：
     - umap_cell_type.png    ：按细胞类型着色的 UMAP
     - umap_modality.png     ：按模态来源（RNA / ATAC 嵌入）着色的 UMAP
     - umap_leiden.png       ：按 leiden 簇着色的 UMAP
     - translation_scatter.png：真实 ATAC vs 预测 ATAC 散点图（子采样）

前置条件：
  - scGPT/result/adata_with_emb.h5ad  已存在
  - scGPT/result/predicted_atac.npy   已存在（可选，用于翻译散点图）

运行方式：
  python scGPT/visualize.py
  python scGPT/visualize.py --n_sample_peaks 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp

matplotlib.use("Agg")  # 无 GUI 模式，适合服务器环境

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADATA_PATH = PROJECT_ROOT / "scGPT" / "result" / "adata_with_emb.h5ad"
PRED_ATAC_PATH = PROJECT_ROOT / "scGPT" / "result" / "predicted_atac.npy"
RESULT_DIR = PROJECT_ROOT / "scGPT" / "result"

DPI = 300
FIG_SIZE_UMAP = (8, 7)
FIG_SIZE_SCATTER = (7, 6)

# PBMC 细胞类型调色板（与 Human Cell Atlas 配色风格一致）
CELL_TYPE_PALETTE = {
    "CD4 T cell":        "#E41A1C",
    "CD8 T cell":        "#FF7F00",
    "NK cell":           "#4DAF4A",
    "B cell":            "#377EB8",
    "CD14 Monocyte":     "#984EA3",
    "FCGR3A Monocyte":   "#A65628",
    "Dendritic cell":    "#F781BF",
    "Platelet":          "#999999",
    "Erythrocyte":       "#FFFF33",
    "unknown":           "#CCCCCC",
}


# ===========================================================================
# UMAP 计算
# ===========================================================================

def compute_umap(
    adata: ad.AnnData,
    embed_key: str = "X_scgpt",
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> ad.AnnData:
    """
    在 scGPT 联合嵌入上计算 UMAP 坐标。
    将结果存储在 adata.obsm["X_umap"]。
    """
    print("  计算 UMAP...")
    sc.pp.neighbors(adata, use_rep=embed_key, n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    print(f"  ✅ UMAP 坐标计算完成：shape = {adata.obsm['X_umap'].shape}")
    return adata


# ===========================================================================
# UMAP 绘图
# ===========================================================================

def _make_color_list(labels: np.ndarray, palette: dict[str, str]) -> list[str]:
    """根据标签列表生成颜色列表，未知类型用灰色。"""
    return [palette.get(str(lb), "#CCCCCC") for lb in labels]


def plot_umap_cell_type(
    adata: ad.AnnData,
    label_key: str = "cell_type",
    out_path: Path = RESULT_DIR / "umap_cell_type.png",
) -> None:
    """按细胞类型着色的 UMAP 图。"""
    print("  绘制 umap_cell_type.png...")
    umap = adata.obsm["X_umap"]
    labels = adata.obs.get(label_key, pd.Series(["unknown"] * adata.n_obs)).astype(str).values
    unique_types = sorted(set(labels))

    fig, ax = plt.subplots(figsize=FIG_SIZE_UMAP)
    for ct in unique_types:
        mask = labels == ct
        color = CELL_TYPE_PALETTE.get(ct, "#AAAAAA")
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color, label=ct, s=3, alpha=0.7, linewidths=0, rasterized=True,
        )

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        markerscale=4, fontsize=9, frameon=False,
    )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("scGPT 联合嵌入 — 细胞类型", fontsize=14, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已保存：{out_path}")


def plot_umap_leiden(
    adata: ad.AnnData,
    leiden_key: str = "leiden",
    out_path: Path = RESULT_DIR / "umap_leiden.png",
) -> None:
    """按 leiden 聚类簇着色的 UMAP 图。"""
    print("  绘制 umap_leiden.png...")
    umap = adata.obsm["X_umap"]

    if leiden_key not in adata.obs.columns:
        leiden_key = "leiden_eval" if "leiden_eval" in adata.obs.columns else None

    if leiden_key is None:
        print("    ⚠️  没有找到 leiden 聚类结果，跳过")
        return

    labels = adata.obs[leiden_key].astype(str).values
    unique_clusters = sorted(set(labels), key=lambda x: int(x) if x.isdigit() else 0)
    cmap = plt.cm.get_cmap("tab20", len(unique_clusters))
    color_map = {c: matplotlib.colors.to_hex(cmap(i)) for i, c in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=FIG_SIZE_UMAP)
    for cluster in unique_clusters:
        mask = labels == cluster
        ax.scatter(
            umap[mask, 0], umap[mask, 1],
            c=color_map[cluster], label=f"Cluster {cluster}",
            s=3, alpha=0.7, linewidths=0, rasterized=True,
        )

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        markerscale=4, fontsize=8, frameon=False, ncol=2,
    )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("scGPT 联合嵌入 — leiden 聚类", fontsize=14, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已保存：{out_path}")


def plot_umap_modality(
    adata: ad.AnnData,
    out_path: Path = RESULT_DIR / "umap_modality.png",
) -> None:
    """
    对比 RNA 嵌入 UMAP 和 ATAC 嵌入 UMAP（并排双图）。

    左图：RNA 嵌入做 UMAP，按细胞类型着色
    右图：ATAC 嵌入做 UMAP，按细胞类型着色
    用于直观展示两种模态嵌入的分布差异。
    """
    print("  绘制 umap_modality.png...")

    if "X_scgpt_rna" not in adata.obsm or "X_scgpt_atac" not in adata.obsm:
        print("    ⚠️  X_scgpt_rna 或 X_scgpt_atac 不在 obsm 中，跳过模态 UMAP")
        return

    label_key = "cell_type" if "cell_type" in adata.obs.columns else None
    labels = (
        adata.obs[label_key].astype(str).values
        if label_key else np.array(["unknown"] * adata.n_obs)
    )
    unique_types = sorted(set(labels))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mod_key, title in zip(
        axes,
        ["X_scgpt_rna", "X_scgpt_atac"],
        ["RNA 嵌入", "ATAC 嵌入"],
    ):
        # 为每个模态单独计算 UMAP
        tmp = ad.AnnData(X=adata.obsm[mod_key], obs=adata.obs.copy())
        sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=42)
        sc.tl.umap(tmp, min_dist=0.3, random_state=42)
        umap_coords = tmp.obsm["X_umap"]

        for ct in unique_types:
            mask = labels == ct
            color = CELL_TYPE_PALETTE.get(ct, "#AAAAAA")
            ax.scatter(
                umap_coords[mask, 0], umap_coords[mask, 1],
                c=color, label=ct, s=3, alpha=0.7, linewidths=0, rasterized=True,
            )

        ax.set_title(f"scGPT {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sns.despine(ax=ax, left=True, bottom=True)

    # 共享图例放在最右侧
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CELL_TYPE_PALETTE.get(ct, "#AAAAAA"),
                   markersize=8, label=ct)
        for ct in unique_types
    ]
    fig.legend(
        handles=handles, loc="center right",
        bbox_to_anchor=(1.12, 0.5), fontsize=9, frameon=False,
    )

    plt.suptitle("scGPT — RNA 嵌入 vs ATAC 嵌入的 UMAP 对比", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已保存：{out_path}")


# ===========================================================================
# 跨模态翻译散点图
# ===========================================================================

def plot_translation_scatter(
    adata: ad.AnnData,
    pred_atac_path: Path,
    n_sample_peaks: int = 300,
    n_sample_cells: int = 1000,
    out_path: Path = RESULT_DIR / "translation_scatter.png",
) -> None:
    """
    真实 ATAC 值 vs 预测 ATAC 值的散点图。

    随机抽取 n_sample_peaks 个峰和 n_sample_cells 个细胞，
    展示模型重建 ATAC 信号的准确性。
    额外绘制 y=x 参考线和 Pearson 相关系数。
    """
    print("  绘制 translation_scatter.png...")
    if not pred_atac_path.exists():
        print(f"    ⚠️  预测 ATAC 文件不存在：{pred_atac_path}，跳过")
        return

    pred_atac = np.load(pred_atac_path)  # (n_cells, n_atac)

    # 获取真实 ATAC 值
    atac_mask = adata.var["modality_type"] == "atac"
    if sp.issparse(adata.layers["values"]):
        true_atac = adata.layers["values"][:, atac_mask].toarray().astype(np.float32)
    else:
        true_atac = np.asarray(adata.layers["values"][:, atac_mask], dtype=np.float32)

    # 对齐形状（ATAC 特征数可能因 max_seq_len 截断而不同）
    n_atac = min(true_atac.shape[1], pred_atac.shape[1])
    true_atac = true_atac[:, :n_atac]
    pred_atac = pred_atac[:, :n_atac]

    # 子采样（避免散点图过于拥挤）
    n_cells_actual = min(n_sample_cells, true_atac.shape[0])
    n_peaks_actual = min(n_sample_peaks, n_atac)
    np.random.seed(42)
    cell_idx = np.random.choice(true_atac.shape[0], n_cells_actual, replace=False)
    peak_idx = np.random.choice(n_atac, n_peaks_actual, replace=False)

    y_true = true_atac[np.ix_(cell_idx, peak_idx)].flatten()
    y_pred = pred_atac[np.ix_(cell_idx, peak_idx)].flatten()

    # Pearson 相关系数
    if y_true.std() > 1e-6 and y_pred.std() > 1e-6:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0

    fig, ax = plt.subplots(figsize=FIG_SIZE_SCATTER)
    ax.scatter(y_true, y_pred, alpha=0.15, s=5, c="#2196F3", linewidths=0, rasterized=True)

    # y=x 参考线
    lim = max(y_true.max(), y_pred.max(), 1.0)
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="y = x（完美预测）")

    ax.set_xlabel("真实 ATAC 信号（二值化分箱值）", fontsize=12)
    ax.set_ylabel("预测 ATAC 信号（模型重建值）", fontsize=12)
    ax.set_title(
        f"scGPT 跨模态翻译：RNA → ATAC\n"
        f"Pearson r = {pearson:.4f}  "
        f"（采样：{n_cells_actual} 细胞 × {n_peaks_actual} 峰）",
        fontsize=12,
    )
    ax.legend(fontsize=10, frameon=False)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已保存：{out_path}  （Pearson r = {pearson:.4f}）")


# ===========================================================================
# 评估指标热力图（可选）
# ===========================================================================

def plot_metrics_summary(
    metrics_csv: Path,
    out_path: Path = RESULT_DIR / "metrics_summary.png",
) -> None:
    """将 evaluation_metrics.csv 中的主要指标绘制为水平条形图。"""
    if not metrics_csv.exists():
        print(f"    ⚠️  指标文件不存在：{metrics_csv}，跳过指标图")
        return

    print("  绘制 metrics_summary.png...")
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    # 仅显示整合质量指标（排除效率指标）
    quality_keys = ["ARI", "NMI", "silhouette_score", "cLISI", "iLISI",
                    "graph_connectivity", "FOSCTTM"]
    df_plot = df[df["metric"].isin(quality_keys)].copy()
    df_plot["value"] = pd.to_numeric(df_plot["value"], errors="coerce")
    df_plot = df_plot.dropna(subset=["value"])

    if df_plot.empty:
        print("    ⚠️  没有可绘制的指标，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df_plot["value"]]
    bars = ax.barh(df_plot["metric"], df_plot["value"], color=colors, edgecolor="white")

    for bar, v in zip(bars, df_plot["value"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=10,
        )

    ax.set_xlim(-0.05, max(df_plot["value"].max() * 1.15, 1.0))
    ax.set_xlabel("指标值", fontsize=12)
    ax.set_title("scGPT 整合质量指标汇总", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✅ 已保存：{out_path}")


# ===========================================================================
# 主函数
# ===========================================================================

import pandas as pd  # noqa: E402（放在函数体之外供 plot_metrics_summary 使用）


def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 结果可视化脚本")
    parser.add_argument("--adata_path", type=Path, default=ADATA_PATH)
    parser.add_argument("--pred_atac_path", type=Path, default=PRED_ATAC_PATH)
    parser.add_argument("--result_dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--n_sample_peaks", type=int, default=300,
                        help="散点图采样峰数（默认 300）")
    parser.add_argument("--n_sample_cells", type=int, default=1000,
                        help="散点图采样细胞数（默认 1000）")
    parser.add_argument("--embed_key", type=str, default="X_scgpt")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    if not args.adata_path.exists():
        print(f"❌ 文件不存在：{args.adata_path}")
        print("   请先运行：python scGPT/extract_embeddings.py")
        sys.exit(1)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 0

    print("\n[1/6] 加载带嵌入的 AnnData...")
    adata = ad.read_h5ad(args.adata_path)
    print(f"  {adata.n_obs} 细胞  |  嵌入维度：{adata.obsm[args.embed_key].shape[1]}")

    print("\n[2/6] 计算 UMAP（联合嵌入）...")
    adata = compute_umap(adata, embed_key=args.embed_key, random_state=args.random_state)

    print("\n[3/6] 绘制细胞类型 UMAP...")
    plot_umap_cell_type(
        adata,
        label_key="cell_type" if "cell_type" in adata.obs.columns else "leiden",
        out_path=args.result_dir / "umap_cell_type.png",
    )

    print("\n[4/6] 绘制 leiden 聚类 UMAP...")
    plot_umap_leiden(adata, out_path=args.result_dir / "umap_leiden.png")

    print("\n[5/6] 绘制模态对比 UMAP（RNA vs ATAC 嵌入）...")
    plot_umap_modality(adata, out_path=args.result_dir / "umap_modality.png")

    print("\n[6/6] 绘制跨模态翻译散点图...")
    plot_translation_scatter(
        adata,
        args.pred_atac_path,
        n_sample_peaks=args.n_sample_peaks,
        n_sample_cells=args.n_sample_cells,
        out_path=args.result_dir / "translation_scatter.png",
    )

    # 额外：指标汇总图
    metrics_csv = args.result_dir / "evaluation_metrics.csv"
    plot_metrics_summary(metrics_csv, out_path=args.result_dir / "metrics_summary.png")

    print("\n🎉 可视化完成！所有图像已保存至：")
    for png in sorted(args.result_dir.glob("*.png")):
        print(f"   {png.name}")


if __name__ == "__main__":
    main()
