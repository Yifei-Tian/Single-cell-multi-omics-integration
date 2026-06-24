"""
annotate_cells.py — PBMC 细胞类型自动标注脚本

功能：
  1. 读取 scGPT/data/pbmc_scgpt_input.h5ad
  2. 从 RNA 层（高变基因表达矩阵）提取 PCA 嵌入，做 leiden 粗聚类
  3. 基于 PBMC 已知 marker 基因为每个簇赋予 cell_type 标签
  4. 输出带标注的 AnnData：scGPT/data/pbmc_scgpt_input_labeled.h5ad

PBMC Marker 基因参考（Human Cell Atlas）：
  - T cell (CD4+):   CD3D, CD3E, CD4, IL7R
  - T cell (CD8+):   CD3D, CD3E, CD8A, CD8B
  - B cell:          MS4A1, CD79A, CD79B
  - NK cell:         GNLY, NKG7, KLRD1
  - Monocyte (CD14): CD14, LYZ, S100A8, S100A9
  - Monocyte (FCGR3A/CD16): FCGR3A, MS4A7
  - Dendritic cell:  FCER1A, CST3, CLEC4C
  - Platelet:        PPBP, GP1BB

运行方式：
  python scGPT/annotate_cells.py
  python scGPT/annotate_cells.py --input_path scGPT/data/pbmc_scgpt_input.h5ad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "scGPT" / "data" / "pbmc_scgpt_input.h5ad"
OUTPUT_PATH = PROJECT_ROOT / "scGPT" / "data" / "pbmc_scgpt_input_labeled.h5ad"

# ---------------------------------------------------------------------------
# PBMC Marker 基因字典（用于簇标注）
# key = 细胞类型名，value = marker 基因列表（表达越高优先级越高）
# ---------------------------------------------------------------------------
PBMC_MARKERS: dict[str, list[str]] = {
    "CD4 T cell":   ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
    "CD8 T cell":   ["CD3D", "CD3E", "CD8A", "CD8B", "GZMK"],
    "NK cell":      ["GNLY", "NKG7", "KLRD1", "KLRF1", "NCAM1"],
    "B cell":       ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "CD14 Monocyte":["CD14", "LYZ", "S100A8", "S100A9", "CSF1R"],
    "FCGR3A Monocyte": ["FCGR3A", "MS4A7", "IFITM3", "CX3CR1"],
    "Dendritic cell":["FCER1A", "CST3", "CLEC4C", "LILRA4", "ITGAX"],
    "Platelet":     ["PPBP", "GP1BB", "PF4", "ITGA2B"],
    "Erythrocyte":  ["HBB", "HBA1", "HBA2"],
}


def extract_rna_layer(adata: ad.AnnData) -> ad.AnnData:
    """
    从合并的 AnnData 中提取 RNA 模态特征，构建用于聚类的子 AnnData。

    使用 layers["values"] 中 modality_type=="rna" 的列。
    """
    rna_mask = adata.var["modality_type"] == "rna"
    rna_features = adata.var_names[rna_mask].tolist()

    if sp.issparse(adata.layers["values"]):
        rna_matrix = adata.layers["values"][:, rna_mask].toarray().astype(np.float32)
    else:
        rna_matrix = np.asarray(adata.layers["values"][:, rna_mask], dtype=np.float32)

    rna_adata = ad.AnnData(
        X=sp.csr_matrix(rna_matrix),
        obs=adata.obs.copy(),
        var=adata.var[rna_mask].copy(),
    )
    rna_adata.var_names = pd.Index(rna_features)
    print(f"  RNA 子矩阵：{rna_adata.n_obs} 细胞 × {rna_adata.n_vars} 基因")
    return rna_adata


def cluster_cells(
    rna_adata: ad.AnnData,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    leiden_resolution: float = 0.5,
    random_state: int = 42,
) -> ad.AnnData:
    """
    标准 Scanpy 聚类流程：PCA → KNN → Leiden

    返回：带 obs["leiden"] 标签的 AnnData
    """
    sc.settings.verbosity = 1
    sc.pp.pca(rna_adata, n_comps=n_pcs, random_state=random_state)
    sc.pp.neighbors(rna_adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    sc.tl.leiden(rna_adata, resolution=leiden_resolution, random_state=random_state)

    n_clusters = rna_adata.obs["leiden"].nunique()
    print(f"  leiden 聚类完成：{n_clusters} 个簇（resolution={leiden_resolution}）")
    return rna_adata


def score_clusters(
    rna_adata: ad.AnnData,
    markers: dict[str, list[str]],
) -> pd.DataFrame:
    """
    对每个 leiden 簇，计算各细胞类型的 marker 基因平均表达分数。

    使用 scanpy.tl.score_genes，返回 (n_clusters × n_cell_types) 分数矩阵。
    """
    cluster_ids = rna_adata.obs["leiden"].unique().tolist()
    cell_types = list(markers.keys())
    score_matrix = pd.DataFrame(index=cluster_ids, columns=cell_types, dtype=float)

    available_genes = set(rna_adata.var_names.tolist())

    for ct, ct_markers in markers.items():
        # 过滤掉数据集中不存在的 marker 基因
        valid_markers = [g for g in ct_markers if g in available_genes]
        if not valid_markers:
            score_matrix[ct] = 0.0
            continue

        # sc.tl.score_genes 为每个细胞计算 marker 基因的平均表达与背景对照的差值
        try:
            sc.tl.score_genes(
                rna_adata,
                gene_list=valid_markers,
                score_name=f"_score_{ct}",
                random_state=42,
            )
            # 每个簇取该分数的中位数作为代表值
            for cluster in cluster_ids:
                mask = rna_adata.obs["leiden"] == cluster
                score_matrix.loc[cluster, ct] = rna_adata.obs.loc[mask, f"_score_{ct}"].median()
        except Exception as e:
            print(f"  ⚠️  {ct} 评分失败：{e}")
            score_matrix[ct] = 0.0

    return score_matrix


def assign_cell_types(score_matrix: pd.DataFrame) -> dict[str, str]:
    """
    为每个 leiden 簇分配得分最高的细胞类型标签。

    返回：{cluster_id: cell_type} 映射字典
    """
    cluster2type: dict[str, str] = {}
    for cluster in score_matrix.index:
        scores = score_matrix.loc[cluster]
        best_type = scores.idxmax()
        cluster2type[cluster] = best_type
    return cluster2type


def main() -> None:
    parser = argparse.ArgumentParser(description="PBMC 细胞类型自动标注脚本")
    parser.add_argument("--input_path", type=Path, default=INPUT_PATH,
                        help="pbmc_scgpt_input.h5ad 路径")
    parser.add_argument("--output_path", type=Path, default=OUTPUT_PATH,
                        help="输出带标注的 h5ad 路径")
    parser.add_argument("--leiden_resolution", type=float, default=0.5,
                        help="leiden 聚类分辨率（默认 0.5）")
    parser.add_argument("--n_pcs", type=int, default=30, help="PCA 维数（默认 30）")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"❌ 输入文件不存在：{args.input_path}")
        print("   请先运行：python scGPT/prepare_data.py")
        sys.exit(1)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("[1/5] 加载 scGPT 输入数据...")
    adata = ad.read_h5ad(args.input_path)
    print(f"  {adata.n_obs} 细胞 × {adata.n_vars} 特征")

    # 2. 提取 RNA 层做聚类
    print("\n[2/5] 提取 RNA 模态进行聚类...")
    rna_adata = extract_rna_layer(adata)

    # 3. 聚类
    print("\n[3/5] PCA + KNN + leiden 聚类...")
    rna_adata = cluster_cells(
        rna_adata,
        n_pcs=args.n_pcs,
        leiden_resolution=args.leiden_resolution,
        random_state=args.random_state,
    )

    # 4. 为每个簇打分并标注细胞类型
    print("\n[4/5] 基于 marker 基因评分，为 leiden 簇分配细胞类型...")
    score_matrix = score_clusters(rna_adata, PBMC_MARKERS)
    cluster2type = assign_cell_types(score_matrix)

    # 打印簇→类型映射
    print("\n  leiden 簇 → 细胞类型 映射：")
    for cluster, cell_type in sorted(cluster2type.items(), key=lambda x: int(x[0])):
        n_cells = (rna_adata.obs["leiden"] == cluster).sum()
        print(f"    簇 {cluster:>2s}  ({n_cells:>5d} 细胞)  →  {cell_type}")

    # 5. 将标注写回原 AnnData 并保存
    print("\n[5/5] 写入 cell_type 标签并保存...")
    adata.obs["leiden"] = rna_adata.obs["leiden"].values
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster2type)

    # 同时保存 PCA 和 UMAP 供后续可视化使用
    adata.obsm["X_pca"] = rna_adata.obsm["X_pca"]

    adata.write_h5ad(args.output_path)
    print(f"  ✅ 标注后的 AnnData 已保存：{args.output_path}")

    # 打印细胞类型分布
    type_counts = adata.obs["cell_type"].value_counts()
    print("\n  细胞类型分布：")
    for ct, cnt in type_counts.items():
        pct = cnt / adata.n_obs * 100
        print(f"    {ct:<25s}  {cnt:>5d} 细胞  ({pct:.1f}%)")

    print("\n🎉 细胞类型标注完成！")


if __name__ == "__main__":
    main()
