"""
prepare_data.py — scGPT 多组学整合数据准备脚本

功能：
  1. 读取已预处理的 RNA（rna_processed.h5ad）和 ATAC（atac_processed.h5ad）数据
  2. 将 RNA 表达量做"分箱"离散化，生成与预训练词表对齐的 token id
  3. 将 ATAC 峰信号作为额外模态 token 拼接，并添加 modality_type 标识
  4. 输出合并后的 AnnData：scGPT/data/pbmc_scgpt_input.h5ad
  5. 输出基因/峰词表映射：scGPT/data/gene2idx.json

前置条件：
  - outputs/pbmc_granulocyte_sorted_10k/rna_processed.h5ad  已存在
  - outputs/pbmc_granulocyte_sorted_10k/atac_processed.h5ad 已存在
  - scGPT/pretrained_models/scGPT_human/vocab.json          已存在（用于对齐预训练词表）
  - 若以上文件缺失，脚本会给出清晰提示

运行方式：
  python scGPT/prepare_data.py
  python scGPT/prepare_data.py --rna_path path/to/rna.h5ad --atac_path path/to/atac.h5ad
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# 默认路径常量（相对于项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RNA_PATH = PROJECT_ROOT / "outputs" / "pbmc_granulocyte_sorted_10k" / "rna_processed.h5ad"
ATAC_PATH = PROJECT_ROOT / "outputs" / "pbmc_granulocyte_sorted_10k" / "atac_processed.h5ad"
VOCAB_PATH = PROJECT_ROOT / "scGPT" / "pretrained_models" / "scGPT_human" / "vocab.json"
OUT_DIR = PROJECT_ROOT / "scGPT" / "data"

# scGPT 表达量分箱数（与预训练设置一致）
N_BINS = 51   # 0 ~ 51，其中 0 表示未检测，1-50 对应表达量区间


def check_prerequisites(rna_path: Path, atac_path: Path, vocab_path: Path) -> None:
    """检查前置文件是否存在，提供清晰的缺失提示。"""
    missing = []
    if not rna_path.exists():
        missing.append(
            f"  [缺失] {rna_path}\n"
            "         → 请先运行：python scripts/preprocess_pbmc_multiome.py"
        )
    if not atac_path.exists():
        missing.append(
            f"  [缺失] {atac_path}\n"
            "         → 请先运行：python scripts/preprocess_pbmc_multiome.py"
        )
    if not vocab_path.exists():
        missing.append(
            f"  [缺失] {vocab_path}\n"
            "         → 请将预训练模型放置于 scGPT/pretrained_models/scGPT_human/\n"
            "         → 文件列表：best_model.pt, vocab.json, args.json"
        )
    if missing:
        print("❌ 前置文件缺失，请先准备以下内容：")
        for msg in missing:
            print(msg)
        sys.exit(1)
    print("✅ 前置文件检查通过")


def load_vocab(vocab_path: Path) -> dict[str, int]:
    """
    加载预训练词表 vocab.json。
    词表格式：{"<gene_name>": token_id, ...}
    """
    with open(vocab_path, "r") as f:
        vocab: dict[str, int] = json.load(f)
    print(f"✅ 预训练词表加载完成：{len(vocab)} 个 token")
    return vocab


def binning_expression(
    expr_matrix: np.ndarray,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """
    将 log1p 归一化后的 RNA 表达量分箱离散化为整数 token id。

    规则（与 scGPT 预训练保持一致）：
      - 值为 0（未检测）→ token id = 0
      - 非零值按 n_bins-1 个等分位数分箱 → token id 1 ~ n_bins-1

    参数：
        expr_matrix: shape (n_cells, n_genes)，已做 log1p 归一化的密集矩阵
        n_bins: 分箱总数（含 0 bins），默认 51

    返回：
        binned: shape (n_cells, n_genes)，dtype int64，值域 [0, n_bins-1]
    """
    binned = np.zeros_like(expr_matrix, dtype=np.int64)
    nonzero_mask = expr_matrix > 0

    if nonzero_mask.any():
        # 仅对非零值计算分位数边界
        nonzero_vals = expr_matrix[nonzero_mask]
        quantiles = np.percentile(nonzero_vals, np.linspace(0, 100, n_bins))
        # 去除重复边界，避免 digitize 异常
        quantiles = np.unique(quantiles)

        bin_ids = np.digitize(nonzero_vals, bins=quantiles[1:], right=False) + 1
        # 确保最大值不超过 n_bins - 1
        bin_ids = np.clip(bin_ids, 1, n_bins - 1)
        binned[nonzero_mask] = bin_ids

    return binned


def align_genes_to_vocab(
    gene_names: np.ndarray,
    vocab: dict[str, int],
    binned_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    将数据集中的基因名映射到预训练词表 token id。

    - 在词表中找到的基因：直接使用 vocab token id
    - 不在词表中的基因：过滤掉（scGPT 不支持 OOV 基因）

    返回：
        token_ids_matrix: shape (n_cells, n_matched_genes)，已映射的 token id 矩阵
        gene_token_ids:   shape (n_matched_genes,)，每个基因对应的 vocab token id
        matched_genes:    List[str]，匹配到的基因名列表
    """
    matched_idx = []
    gene_token_ids = []
    matched_genes = []

    for i, gene in enumerate(gene_names):
        if gene in vocab:
            matched_idx.append(i)
            gene_token_ids.append(vocab[gene])
            matched_genes.append(gene)

    if not matched_idx:
        print("⚠️  警告：数据集中无任何基因匹配到预训练词表，请检查基因命名格式（通常为 HGNC 符号）")
        sys.exit(1)

    matched_idx = np.array(matched_idx)
    gene_token_ids = np.array(gene_token_ids, dtype=np.int64)
    token_ids_matrix = binned_matrix[:, matched_idx]

    print(f"✅ 基因词表对齐：{len(matched_genes)}/{len(gene_names)} 个基因匹配成功")
    return token_ids_matrix, gene_token_ids, matched_genes


def build_atac_tokens(
    atac_adata: ad.AnnData,
    vocab: dict[str, int],
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    为 ATAC 峰构建 token 表示。

    策略：
      - ATAC 峰通常不在 RNA 预训练词表中，因此为每个峰创建新的 token id
      - token id 从 max(vocab.values()) + 1 开始连续编号
      - ATAC 峰的"表达值"直接使用二值矩阵（0=关闭，1=开放），对应 token id (0 or 1)
      - 仅保留 highly_variable 峰（若有标记），以控制序列长度

    返回：
        atac_value_matrix: shape (n_cells, n_peaks)，二值 0/1
        peak_token_ids:    shape (n_peaks,)，每个峰对应的 token id
        peak_names:        List[str]，峰名列表（格式：chr:start-end）
    """
    # 过滤高变峰
    if "highly_variable" in atac_adata.var.columns:
        atac_hv = atac_adata[:, atac_adata.var["highly_variable"]].copy()
    else:
        atac_hv = atac_adata.copy()

    peak_names = atac_hv.var_names.tolist()
    n_peaks = len(peak_names)

    # ATAC 值矩阵（二值，0 or 1）
    if sp.issparse(atac_hv.X):
        atac_value_matrix = np.asarray(atac_hv.X.todense(), dtype=np.int64)
    else:
        atac_value_matrix = np.asarray(atac_hv.X, dtype=np.int64)

    # 分配新 token id（不与 RNA vocab 冲突）
    max_rna_token = max(vocab.values()) if vocab else 0
    peak_token_ids = np.arange(
        max_rna_token + 1,
        max_rna_token + 1 + n_peaks,
        dtype=np.int64,
    )

    print(f"✅ ATAC peaks token 构建：{n_peaks} 个峰，token id 从 {max_rna_token + 1} 开始")
    return atac_value_matrix, peak_token_ids, peak_names


def build_combined_adata(
    rna_adata: ad.AnnData,
    rna_token_ids_matrix: np.ndarray,
    rna_gene_token_ids: np.ndarray,
    rna_matched_genes: list[str],
    atac_value_matrix: np.ndarray,
    peak_token_ids: np.ndarray,
    peak_names: list[str],
) -> ad.AnnData:
    """
    将 RNA 和 ATAC 的 token 信息合并为一个 AnnData 对象。

    AnnData 结构说明：
      - obs：细胞元数据（保留原 RNA 的 obs）
      - var：特征元数据，包含 modality_type（"rna" or "atac"）
      - layers["token_ids"]：(n_cells, n_features) 整数矩阵，存储 token id
      - layers["values"]：(n_cells, n_features) 整数矩阵，存储分箱后的表达/信号值
      - uns["rna_gene_token_ids"]：RNA 基因对应的 vocab token id 数组
      - uns["peak_token_ids"]：ATAC 峰对应的 token id 数组
      - uns["n_rna_features"]：RNA 特征数量
      - uns["n_atac_features"]：ATAC 特征数量
    """
    n_cells = rna_adata.n_obs
    n_rna = len(rna_matched_genes)
    n_atac = len(peak_names)

    # --- 构建合并的 token id 矩阵 ---
    # RNA 部分：每个细胞 × 每个基因 → 对应 vocab token id（作为 feature token）
    rna_feature_tokens = np.tile(rna_gene_token_ids, (n_cells, 1))  # (n_cells, n_rna)

    # ATAC 部分：每个细胞 × 每个峰 → 对应 peak token id
    atac_feature_tokens = np.tile(peak_token_ids, (n_cells, 1))  # (n_cells, n_atac)

    # 拼接为 (n_cells, n_rna + n_atac)
    all_token_ids = np.concatenate([rna_feature_tokens, atac_feature_tokens], axis=1)

    # --- 构建合并的值矩阵 ---
    all_values = np.concatenate(
        [rna_token_ids_matrix, atac_value_matrix], axis=1
    )  # (n_cells, n_rna + n_atac)

    # --- 构建 var 表 ---
    var_df = pd.DataFrame(
        {
            "feature_name": rna_matched_genes + peak_names,
            "modality_type": ["rna"] * n_rna + ["atac"] * n_atac,
            "modality_id": [0] * n_rna + [1] * n_atac,  # 0=RNA, 1=ATAC
        },
        index=rna_matched_genes + peak_names,
    )

    # --- 构建 AnnData ---
    combined = ad.AnnData(
        X=sp.csr_matrix(all_values),  # 主矩阵存 values
        obs=rna_adata.obs.copy(),
        var=var_df,
    )
    combined.layers["token_ids"] = sp.csr_matrix(all_token_ids)
    combined.layers["values"] = sp.csr_matrix(all_values)

    # 元信息
    combined.uns["rna_gene_token_ids"] = rna_gene_token_ids.tolist()
    combined.uns["peak_token_ids"] = peak_token_ids.tolist()
    combined.uns["n_rna_features"] = n_rna
    combined.uns["n_atac_features"] = n_atac
    combined.uns["n_bins"] = N_BINS

    return combined


def save_gene2idx(
    rna_matched_genes: list[str],
    rna_gene_token_ids: np.ndarray,
    peak_names: list[str],
    peak_token_ids: np.ndarray,
    out_path: Path,
) -> None:
    """将基因和峰的词表映射保存为 JSON。"""
    gene2idx: dict[str, int] = {}
    for gene, tid in zip(rna_matched_genes, rna_gene_token_ids.tolist()):
        gene2idx[gene] = tid
    for peak, tid in zip(peak_names, peak_token_ids.tolist()):
        gene2idx[peak] = tid
    with open(out_path, "w") as f:
        json.dump(gene2idx, f, indent=2)
    print(f"✅ 词表映射已保存：{out_path}  ({len(gene2idx)} 个条目)")


def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 数据格式转换脚本")
    parser.add_argument("--rna_path", type=Path, default=RNA_PATH, help="rna_processed.h5ad 路径")
    parser.add_argument("--atac_path", type=Path, default=ATAC_PATH, help="atac_processed.h5ad 路径")
    parser.add_argument("--vocab_path", type=Path, default=VOCAB_PATH, help="vocab.json 路径")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR, help="输出目录")
    parser.add_argument("--n_bins", type=int, default=N_BINS, help="表达量分箱数（默认 51）")
    args = parser.parse_args()

    # 0. 检查前置文件
    check_prerequisites(args.rna_path, args.atac_path, args.vocab_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n[1/6] 加载预处理数据...")
    rna_adata = ad.read_h5ad(args.rna_path)
    atac_adata = ad.read_h5ad(args.atac_path)
    print(f"  RNA:  {rna_adata.n_obs} 细胞 × {rna_adata.n_vars} 基因")
    print(f"  ATAC: {atac_adata.n_obs} 细胞 × {atac_adata.n_vars} 峰")

    # 确保细胞顺序一致
    common_cells = rna_adata.obs_names.intersection(atac_adata.obs_names)
    if len(common_cells) < rna_adata.n_obs:
        print(f"  ⚠️  细胞 barcode 取交集：{len(common_cells)} 个公共细胞")
    rna_adata = rna_adata[common_cells].copy()
    atac_adata = atac_adata[common_cells].copy()

    # 2. 加载预训练词表
    print("\n[2/6] 加载预训练词表...")
    vocab = load_vocab(args.vocab_path)

    # 3. RNA 表达量分箱
    print("\n[3/6] RNA 表达量分箱（log1p 值 → 离散 token）...")
    if sp.issparse(rna_adata.X):
        expr_dense = np.asarray(rna_adata.X.todense(), dtype=np.float32)
    else:
        expr_dense = np.asarray(rna_adata.X, dtype=np.float32)
    binned_rna = binning_expression(expr_dense, n_bins=args.n_bins)
    print(f"  分箱完成：值域 [{binned_rna.min()}, {binned_rna.max()}]，使用 {args.n_bins} 个区间")

    # 4. 基因词表对齐
    print("\n[4/6] 对齐基因到预训练词表...")
    gene_names = rna_adata.var_names.to_numpy()
    rna_token_ids_matrix, rna_gene_token_ids, rna_matched_genes = align_genes_to_vocab(
        gene_names, vocab, binned_rna
    )

    # 5. 构建 ATAC token
    print("\n[5/6] 构建 ATAC 峰 token...")
    atac_value_matrix, peak_token_ids, peak_names = build_atac_tokens(
        atac_adata, vocab, n_cells=rna_adata.n_obs
    )

    # 6. 合并并保存
    print("\n[6/6] 合并 RNA + ATAC，保存输出...")
    combined_adata = build_combined_adata(
        rna_adata,
        rna_token_ids_matrix,
        rna_gene_token_ids,
        rna_matched_genes,
        atac_value_matrix,
        peak_token_ids,
        peak_names,
    )

    out_adata_path = args.out_dir / "pbmc_scgpt_input.h5ad"
    combined_adata.write_h5ad(out_adata_path)
    print(f"  ✅ 合并 AnnData 已保存：{out_adata_path}")
    print(f"     shape: {combined_adata.n_obs} 细胞 × {combined_adata.n_vars} 特征")
    print(f"     RNA 特征：{combined_adata.uns['n_rna_features']}")
    print(f"     ATAC 特征：{combined_adata.uns['n_atac_features']}")

    out_vocab_path = args.out_dir / "gene2idx.json"
    save_gene2idx(rna_matched_genes, rna_gene_token_ids, peak_names, peak_token_ids, out_vocab_path)

    print("\n🎉 数据准备完成！")
    print(f"   输出目录：{args.out_dir}")
    print(f"   - pbmc_scgpt_input.h5ad  （scGPT 输入数据）")
    print(f"   - gene2idx.json          （基因/峰词表映射）")


if __name__ == "__main__":
    main()
