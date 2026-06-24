"""
evaluate.py — scGPT 整合质量全套评估脚本

功能：
  计算 Proposal §6 中定义的全部评估指标并保存到 CSV：

  整合质量指标（§6.1）：
    - ARI  (Adjusted Rand Index)：leiden 聚类 vs cell_type 标签
    - NMI  (Normalized Mutual Information)：同上
    - cLISI（cell-type LISI）：嵌入空间中细胞类型的局部多样性
    - iLISI（integration LISI）：嵌入空间中批次的局部多样性
    - silhouette score（细胞类型分离度）
    - graph connectivity（同类型细胞在 KNN 图中的连通性）
    - FOSCTTM（RNA vs ATAC 嵌入的最近邻重叠率，越低越好）

  计算效率指标（§6.3，读取已记录的值）：
    - 推断时间（秒/1000 细胞）
    - GPU 显存峰值（GB）
    - 参数量（M）

  输出：
    scGPT/result/evaluation_metrics.csv

前置条件：
  - scGPT/result/adata_with_emb.h5ad 已存在（含 obsm["X_scgpt"]、obsm["X_scgpt_rna"]、obsm["X_scgpt_atac"]）
  - adata.obs["cell_type"] 已标注

运行方式：
  python scGPT/evaluate.py
  python scGPT/evaluate.py --embed_key X_scgpt
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADATA_PATH = PROJECT_ROOT / "scGPT" / "result" / "adata_with_emb.h5ad"
RESULT_DIR = PROJECT_ROOT / "scGPT" / "result"


# ===========================================================================
# 整合质量指标
# ===========================================================================

def compute_ari_nmi(
    adata: ad.AnnData,
    embed_key: str = "X_scgpt",
    label_key: str = "cell_type",
    leiden_resolution: float = 0.5,
    random_state: int = 42,
) -> dict[str, float]:
    """
    基于 scGPT 嵌入做 leiden 聚类，与真实 cell_type 标签对比，计算 ARI 和 NMI。
    """
    print("  计算 ARI / NMI...")
    sc.pp.neighbors(adata, use_rep=embed_key, random_state=random_state)
    sc.tl.leiden(adata, resolution=leiden_resolution, random_state=random_state, key_added="leiden_eval")

    labels_true = adata.obs[label_key].astype(str).values
    labels_pred = adata.obs["leiden_eval"].astype(str).values

    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print(f"    ARI = {ari:.4f},  NMI = {nmi:.4f}")
    return {"ARI": ari, "NMI": nmi}


def compute_silhouette(
    adata: ad.AnnData,
    embed_key: str = "X_scgpt",
    label_key: str = "cell_type",
) -> dict[str, float]:
    """
    在嵌入空间中计算 silhouette score（细胞类型分离度）。
    值域 [-1, 1]，越高越好。
    """
    print("  计算 silhouette score...")
    X = adata.obsm[embed_key]
    labels = adata.obs[label_key].astype(str).values
    try:
        score = silhouette_score(X, labels, metric="cosine", sample_size=min(5000, len(X)))
    except Exception as e:
        print(f"    ⚠️  silhouette 计算失败：{e}")
        score = float("nan")
    print(f"    silhouette = {score:.4f}")
    return {"silhouette_score": score}


def compute_lisi(
    adata: ad.AnnData,
    embed_key: str = "X_scgpt",
    label_key: str = "cell_type",
    batch_key: str = "batch",
    n_neighbors: int = 90,
) -> dict[str, float]:
    """
    计算 cLISI（细胞类型局部多样性）和 iLISI（批次局部多样性）。
    使用 scib_metrics 库（若可用），否则使用简化版本。

    cLISI 越低越好（细胞类型混合少 = 分离好）
    iLISI 越高越好（批次混合多 = 批次校正好）
    """
    print("  计算 cLISI / iLISI...")

    # 尝试使用 scib_metrics
    try:
        from scib_metrics.benchmark import Benchmarker
        from scib_metrics import lisi_knn

        X = adata.obsm[embed_key]
        ct_labels = adata.obs[label_key].astype("category").cat.codes.values
        batch_labels = (
            adata.obs[batch_key].astype("category").cat.codes.values
            if batch_key in adata.obs.columns
            else np.zeros(adata.n_obs, dtype=int)
        )

        clisi = _compute_lisi_score(X, ct_labels, n_neighbors=n_neighbors)
        ilisi = (
            _compute_lisi_score(X, batch_labels, n_neighbors=n_neighbors)
            if batch_key in adata.obs.columns
            else float("nan")
        )

    except ImportError:
        # 回退：用自定义实现
        X = adata.obsm[embed_key]
        ct_labels = adata.obs[label_key].astype("category").cat.codes.values
        clisi = _compute_lisi_score(X, ct_labels, n_neighbors=n_neighbors)
        ilisi = float("nan")
        warnings.warn("scib_metrics 未安装，iLISI 跳过。安装：pip install scib-metrics")

    print(f"    cLISI = {clisi:.4f},  iLISI = {ilisi}")
    return {"cLISI": clisi, "iLISI": ilisi}


def _compute_lisi_score(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 90,
    perplexity: float = 30.0,
) -> float:
    """
    LISI（Local Inverse Simpson's Index）的简化实现。
    参考：Korsunsky et al., 2019, Nature Methods

    LISI = 每个细胞的局部邻域中，标签的"有效多样性"（基于 Simpson's Index）。
    取所有细胞的中位数。
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    indices = indices[:, 1:]  # 去掉自身

    lisi_scores = []
    n_labels = len(np.unique(labels))
    for i in range(len(X)):
        neighbor_labels = labels[indices[i]]
        # 计算邻居中各类别的频率
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        freqs = counts / counts.sum()
        # Simpson's Index = sum(p^2)，LISI = 1/Simpson
        simpson = np.sum(freqs ** 2)
        lisi = 1.0 / simpson if simpson > 0 else 1.0
        lisi_scores.append(lisi)

    return float(np.median(lisi_scores))


def compute_graph_connectivity(
    adata: ad.AnnData,
    embed_key: str = "X_scgpt",
    label_key: str = "cell_type",
    n_neighbors: int = 15,
) -> dict[str, float]:
    """
    Graph Connectivity：同一细胞类型的细胞在 KNN 图中的连通性。

    计算方法：
      1. 构建 KNN 图
      2. 对每个细胞类型，计算该类型细胞子图中最大连通分量的比例
      3. 取所有类型的加权平均（权重 = 细胞数）
    值域 [0, 1]，越高越好。
    """
    print("  计算 graph connectivity...")
    try:
        import igraph as ig

        X = adata.obsm[embed_key]
        labels = adata.obs[label_key].astype(str).values
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        n_cells = len(X)
        # 构建边列表
        edges = []
        for i in range(n_cells):
            for j in indices[i]:
                if i != j:
                    edges.append((i, j))

        g = ig.Graph(n=n_cells, edges=edges, directed=False)

        unique_types = np.unique(labels)
        gc_scores = []
        weights = []
        for ct in unique_types:
            ct_indices = np.where(labels == ct)[0].tolist()
            if len(ct_indices) < 2:
                continue
            subgraph = g.induced_subgraph(ct_indices)
            components = subgraph.connected_components()
            largest = max(len(c) for c in components)
            gc_scores.append(largest / len(ct_indices))
            weights.append(len(ct_indices))

        if gc_scores:
            gc = float(np.average(gc_scores, weights=weights))
        else:
            gc = float("nan")

    except ImportError:
        print("    ⚠️  igraph 未安装，graph connectivity 跳过。安装：pip install python-igraph")
        gc = float("nan")

    print(f"    graph connectivity = {gc:.4f}")
    return {"graph_connectivity": gc}


def compute_foscttm(
    adata: ad.AnnData,
    rna_embed_key: str = "X_scgpt_rna",
    atac_embed_key: str = "X_scgpt_atac",
    n_neighbors: int = 10,
) -> dict[str, float]:
    """
    FOSCTTM（Fraction Of Samples Closer Than True Match）：
    衡量配对细胞在跨模态嵌入中的对齐程度。

    对每个细胞 i：
      在 ATAC 嵌入空间中，找 RNA 嵌入 i 的最近邻
      记录：有多少 ATAC 样本比"真实配对 ATAC i"更近？
      FOSCTTM_i = 比真实配对更近的比例

    FOSCTTM 越低越好（越接近 0，说明模态对齐越好）。
    """
    print("  计算 FOSCTTM...")

    if rna_embed_key not in adata.obsm or atac_embed_key not in adata.obsm:
        print(f"    ⚠️  {rna_embed_key} 或 {atac_embed_key} 不在 obsm 中，跳过 FOSCTTM")
        return {"FOSCTTM": float("nan")}

    emb_rna = adata.obsm[rna_embed_key]
    emb_atac = adata.obsm[atac_embed_key]
    n_cells = emb_rna.shape[0]

    # 对 RNA 嵌入中的每个细胞，在 ATAC 嵌入空间中查找其距离排名
    from sklearn.metrics.pairwise import cosine_distances

    dist_matrix = cosine_distances(emb_rna, emb_atac)  # (n_cells, n_cells)

    foscttm_scores = []
    for i in range(n_cells):
        true_dist = dist_matrix[i, i]
        # 有多少其他 ATAC 样本比真实配对更近？
        closer = (dist_matrix[i, :] < true_dist).sum() - (dist_matrix[i, i] < true_dist)
        foscttm_i = closer / (n_cells - 1)
        foscttm_scores.append(foscttm_i)

    foscttm = float(np.mean(foscttm_scores))
    print(f"    FOSCTTM = {foscttm:.4f}  （越低越好，0 = 完美对齐）")
    return {"FOSCTTM": foscttm}


# ===========================================================================
# 计算效率指标（从 adata.uns 读取已记录的值）
# ===========================================================================

def read_efficiency_metrics(adata: ad.AnnData, checkpoint_path: Path) -> dict[str, float]:
    """从 adata.uns 和训练日志中读取计算效率指标。"""
    metrics: dict[str, float] = {}
    metrics["inference_time_per_1k_s"] = adata.uns.get("inference_time_per_1k_s", float("nan"))
    metrics["peak_memory_gb"] = adata.uns.get("peak_memory_gb", float("nan"))
    metrics["embed_dim"] = float(adata.uns.get("embed_dim", 512))
    metrics["n_params_M"] = adata.uns.get("n_params_M", float("nan"))

    # 从 training_log.csv 读取总训练时间
    training_log = checkpoint_path.parent / "training_log.csv"
    if training_log.exists():
        total_time = float("nan")
        with open(training_log, "r") as f:
            for line in f:
                if line.startswith("# 总训练时间"):
                    try:
                        total_time = float(line.split("：")[1].split(" ")[0])
                    except Exception:
                        pass
        metrics["training_time_s"] = total_time
    else:
        metrics["training_time_s"] = float("nan")

    return metrics


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 整合质量全套评估脚本")
    parser.add_argument("--adata_path", type=Path, default=ADATA_PATH)
    parser.add_argument("--result_dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--embed_key", type=str, default="X_scgpt",
                        help="联合嵌入的 obsm key（默认 X_scgpt）")
    parser.add_argument("--label_key", type=str, default="cell_type",
                        help="细胞类型标签的 obs key（默认 cell_type）")
    parser.add_argument("--batch_key", type=str, default="batch",
                        help="批次标签的 obs key（默认 batch；若不存在则跳过 iLISI）")
    parser.add_argument("--leiden_resolution", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    if not args.adata_path.exists():
        print(f"❌ 文件不存在：{args.adata_path}")
        print("   请先运行：python scGPT/extract_embeddings.py")
        sys.exit(1)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.verbosity = 0

    print("\n[1/7] 加载带嵌入的 AnnData...")
    adata = ad.read_h5ad(args.adata_path)
    print(f"  {adata.n_obs} 细胞  |  嵌入维度：{adata.obsm[args.embed_key].shape[1]}")

    if args.label_key not in adata.obs.columns:
        print(f"  ⚠️  obs['{args.label_key}'] 不存在，请先运行 annotate_cells.py")
        # 用空标签占位
        adata.obs[args.label_key] = "unknown"

    all_metrics: dict[str, float] = {}

    print("\n[2/7] 计算 ARI & NMI...")
    all_metrics.update(compute_ari_nmi(
        adata, args.embed_key, args.label_key,
        args.leiden_resolution, args.random_state
    ))

    print("\n[3/7] 计算 silhouette score...")
    all_metrics.update(compute_silhouette(adata, args.embed_key, args.label_key))

    print("\n[4/7] 计算 cLISI & iLISI...")
    all_metrics.update(compute_lisi(
        adata, args.embed_key, args.label_key, args.batch_key
    ))

    print("\n[5/7] 计算 graph connectivity...")
    all_metrics.update(compute_graph_connectivity(adata, args.embed_key, args.label_key))

    print("\n[6/7] 计算 FOSCTTM...")
    all_metrics.update(compute_foscttm(adata, "X_scgpt_rna", "X_scgpt_atac"))

    print("\n[7/7] 读取计算效率指标...")
    checkpoint_path = args.result_dir / "best_finetuned.pt"
    all_metrics.update(read_efficiency_metrics(adata, checkpoint_path))
    for k in ["inference_time_per_1k_s", "training_time_s", "peak_memory_gb", "n_params_M"]:
        v = all_metrics.get(k, float("nan"))
        print(f"  {k:<35s}: {v}")

    # 保存结果
    out_csv = args.result_dir / "evaluation_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "description"])
        descriptions = {
            "ARI":                    "Adjusted Rand Index（聚类 vs 真实标签，↑越好）",
            "NMI":                    "Normalized Mutual Information（↑越好）",
            "silhouette_score":       "细胞类型分离度，余弦距离（↑越好）",
            "cLISI":                  "细胞类型局部多样性（↓越好，类型分离）",
            "iLISI":                  "批次局部多样性（↑越好，批次混合）",
            "graph_connectivity":     "同类型细胞 KNN 图连通性（↑越好）",
            "FOSCTTM":                "模态对齐率（↓越好，0=完美对齐）",
            "inference_time_per_1k_s":"每 1000 细胞推断时间（秒）",
            "training_time_s":        "总微调训练时间（秒）",
            "peak_memory_gb":         "GPU 显存峰值（GB）",
            "n_params_M":             "参数量（M）",
            "embed_dim":              "嵌入维度",
        }
        for k, v in all_metrics.items():
            writer.writerow([k, f"{v:.6f}" if not np.isnan(v) else "N/A", descriptions.get(k, "")])

    print(f"\n✅ 全套评估指标已保存：{out_csv}")

    print("\n📊 评估结果汇总：")
    print(f"  {'指标':<35s} {'值':>10s}")
    print(f"  {'-'*47}")
    for k, v in all_metrics.items():
        val_str = f"{v:.4f}" if not np.isnan(v) else "N/A"
        print(f"  {k:<35s} {val_str:>10s}")

    print("\n🎉 评估完成！")


if __name__ == "__main__":
    main()
