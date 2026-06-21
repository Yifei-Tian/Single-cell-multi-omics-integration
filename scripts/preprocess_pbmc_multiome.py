from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# 下面这些常量对应本项目当前使用的默认预处理参数。
# 如果后续想调宽或调严过滤阈值，可以优先改这里。
TARGET_SUM = 10_000
MT_THRESHOLD = 0.20
RNA_MIN_CELLS = 3
ATAC_MIN_CELLS = 3
TOP_HVGS = 3000
TOP_HVPS = 5000
DOUBLET_MAD_MULTIPLIER = 4.0


def read_10x_h5(path: Path) -> tuple[sparse.csc_matrix, pd.DataFrame, pd.Index]:
    # 读取 10x Genomics 的 filtered_feature_bc_matrix.h5 文件。
    # 这个文件里 RNA 和 ATAC 共用同一个稀疏矩阵，后续再按 feature_type 拆分。
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]  # 从整个 h5 文件里，取出 matrix 这一层内容，并赋值给变量 group
        # 10x h5 使用 CSC（compressed sparse column）格式存储矩阵三元组。
        matrix = sparse.csc_matrix(  # 创建 csc 格式的稀疏矩阵
            (
                group["data"][:],  # 所有非零元素的值，
                # 因为 group["data"] 是 h5 文件里的一个数据集对象，不是普通 numpy 数组，加上 [:] 才是把它完整读到内存里。
                group["indices"][:],  # 所有非零元素所在的“行号”
                group["indptr"][:],  # 每一列在 data 里的起止位置索引
            ),
            shape=tuple(group["shape"][:]),  # 矩阵的总形状（行数，列数）
        )
        # features 表保存每一行特征的注释信息，比如基因名、峰区间、模态类型等。
        features = pd.DataFrame(
            {
                "feature_id": decode_array(group["features"]["id"][:]),  # 特征的唯一标识符
                "feature_name": decode_array(group["features"]["name"][:]),  # 特征的常规名称
                "feature_type": decode_array(group["features"]["feature_type"][:]),  # 特征的类型
                "genome": decode_array(group["features"]["genome"][:]),  # 特征所在的基因组（通常是 "hg38" 或 "mm10"）
                "interval": decode_array(group["features"]["interval"][:]),  # 对于 ATAC 峰来说是峰的区间信息，对于基因来说通常是空字符串
            }
            # decode_array(...)：这是一个自定义的函数。因为 HDF5 文件中的字符串常常以二进制字节（如 b'ENSG000001'）的形式存储
            # decode_array 的作用就是将这些字节阵列解码为普通的 Python 字符串。
        )
        barcodes = pd.Index(decode_array(group["barcodes"][:]), name="barcode")  # 细胞索引，用作表达量矩阵的行名和列名
    return matrix, features, barcodes


def decode_array(values: np.ndarray) -> np.ndarray:
    # h5 读出来的字符串通常是 bytes，这里统一解码成 Python 字符串。
    return np.array([value.decode("utf-8") if isinstance(value, bytes) else value for value in values])


def subset_rows(matrix: sparse.csc_matrix, mask: np.ndarray) -> sparse.csc_matrix:
    # 原始矩阵的“行”是特征，因此这里按行筛选对应模态的特征。
    return matrix[mask, :]


def make_unique(values: np.ndarray) -> list[str]:
    # 10x 数据里可能存在重复特征名，这会导致 AnnData 给出重复 var_names 警告。
    # 这里用 name, name-1, name-2 的方式保证索引唯一。
    counts: dict[str, int] = {}
    result: list[str] = []
    for value in values:
        key = str(value)
        count = counts.get(key, 0)
        if count == 0:
            result.append(key)
        else:
            result.append(f"{key}-{count}")
        counts[key] = count + 1
    return result


def matrix_to_adata(
    matrix: sparse.csc_matrix,
    var: pd.DataFrame,
    obs_index: pd.Index,
) -> ad.AnnData:
    # AnnData 约定行为细胞、列为特征，因此这里要把 10x 原始矩阵转置。
    obs = pd.DataFrame(index=obs_index.copy())

    # 保留 feature_name 这一列用于注释，但不要把索引名也设成 feature_name，
    # 否则 write_h5ad() 会因为 DataFrame 索引名与列名冲突而报错。
    var = var.copy()
    var.index = pd.Index(make_unique(var["feature_name"].to_numpy()), name="var_name")
    return ad.AnnData(X=matrix.transpose().tocsr(), obs=obs, var=var)


def filter_features_by_cells(adata: ad.AnnData, min_cells: int) -> ad.AnnData:
    # 保留至少在 min_cells 个细胞中出现过的特征。
    # 对 RNA 来说是低表达基因过滤，对 ATAC 来说是低开放峰过滤。
    detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    return adata[:, detected >= min_cells].copy()


def compute_mt_fraction(adata: ad.AnnData) -> np.ndarray:
    # 线粒体比例 = 线粒体基因 counts / 该细胞总 counts。
    # 这里默认用基因名前缀 MT- 识别线粒体基因。
    mt_mask = adata.var["feature_name"].str.upper().str.startswith("MT-").to_numpy()
    total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    mt_counts = np.asarray(adata.X[:, mt_mask].sum(axis=1)).ravel()
    return np.divide(mt_counts, total_counts, out=np.zeros_like(mt_counts, dtype=float), where=total_counts > 0)


def normalize_log1p(adata: ad.AnnData, target_sum: int) -> None:
    # 按细胞归一化到固定文库大小（默认 10,000），再做 log1p 变换。
    # 这是 RNA 预处理中最常见的一步。
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    scale = np.divide(target_sum, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
    adata.X = sparse.diags(scale) @ adata.X
    adata.X = adata.X.log1p()


def highly_variable_by_dispersion(adata: ad.AnnData, n_top: int) -> np.ndarray:
    # 手工计算均值、方差和 dispersion，用于挑选高变基因。
    # 这里没有直接调用 scanpy，而是用较透明的方式实现一版“按离散度筛选”的逻辑。
    mean = np.asarray(adata.X.mean(axis=0)).ravel()
    squared = adata.X.copy()
    squared.data **= 2
    mean_sq = np.asarray(squared.mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    dispersion = np.divide(variance, mean + 1e-12)

    # 按平均表达分箱，再在每个箱内标准化 dispersion，
    # 以减少“高表达特征天然方差更大”带来的偏置。
    valid = np.isfinite(mean) & np.isfinite(dispersion) & (mean > 0)
    bins = pd.qcut(mean[valid], q=min(20, valid.sum()), duplicates="drop")
    disp_series = pd.Series(dispersion[valid], index=np.where(valid)[0])
    grouped = disp_series.groupby(bins, observed=False)
    norm_disp = pd.Series(index=disp_series.index, dtype=float)
    for _, group in grouped:
        std = group.std(ddof=0)
        if std == 0 or np.isnan(std):
            norm_disp.loc[group.index] = 0.0
        else:
            norm_disp.loc[group.index] = (group - group.mean()) / std

    scores = np.full(adata.n_vars, -np.inf, dtype=float)
    scores[norm_disp.index.to_numpy()] = norm_disp.to_numpy()
    selected = np.argsort(scores)[-min(n_top, adata.n_vars) :]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected] = True
    return mask


def binarize_adata(adata: ad.AnnData) -> None:
    # ATAC 常常更关注“峰是否开放”而非原始计数大小，
    # 因此这里把所有非零值都改成 1，得到二值矩阵。
    binary = adata.X.copy()
    binary.data = np.ones_like(binary.data)
    adata.X = binary


def highly_variable_binary_features(adata: ad.AnnData, n_top: int) -> np.ndarray:
    # 对二值 ATAC 矩阵，用 Bernoulli 方差 p(1-p) 作为变异度指标。
    # 在接近 0.5 开放频率时方差最大，更容易被选为高变峰。
    frequency = np.asarray(adata.X.mean(axis=0)).ravel()
    variance = frequency * (1.0 - frequency)
    selected = np.argsort(variance)[-min(n_top, adata.n_vars) :]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[selected] = True
    return mask


def mad(values: np.ndarray) -> float:
    # MAD: median absolute deviation，中位数绝对偏差。
    # 比标准差更稳健，适合做异常值阈值估计。
    return np.median(np.abs(values - np.median(values)))


def detect_doublets(
    rna_raw: ad.AnnData,
    atac_binary: ad.AnnData,
    mad_multiplier: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    # 这里用一个启发式的联合双细胞检测：
    # 如果一个细胞在 RNA 和 ATAC 的多个复杂度指标上都异常偏高，
    # 就认为它更像 doublet。
    metrics = pd.DataFrame(index=rna_raw.obs_names.copy())
    metrics["rna_total_counts"] = np.asarray(rna_raw.X.sum(axis=1)).ravel()
    metrics["rna_detected_genes"] = np.asarray((rna_raw.X > 0).sum(axis=1)).ravel()
    metrics["atac_total_peaks"] = np.asarray(atac_binary.X.sum(axis=1)).ravel()
    metrics["atac_detected_peaks"] = np.asarray((atac_binary.X > 0).sum(axis=1)).ravel()

    flags = []
    for column in metrics.columns:
        # 对每个指标先取 log1p，再用“中位数 + k * MAD”定义高异常阈值。
        logged = np.log1p(metrics[column].to_numpy())
        threshold = np.median(logged) + mad_multiplier * mad(logged)
        flag = logged > threshold
        metrics[f"{column}_outlier"] = flag
        flags.append(flag)

    # 如果一个细胞至少在两个指标上表现为异常高，则标记为 doublet。
    doublet_mask = np.sum(np.column_stack(flags), axis=1) >= 2
    metrics["predicted_doublet"] = doublet_mask
    return doublet_mask, metrics


def main() -> None:
    # 允许从命令行指定输入和输出路径，便于复用到其他样本。
    parser = argparse.ArgumentParser(description="Preprocess 10x multiome PBMC data.")
    parser.add_argument(
        "--input-h5",
        default="data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/pbmc_granulocyte_sorted_10k",
        type=Path,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix, features, barcodes = read_10x_h5(args.input_h5)

    # 利用 feature_type 将同一个 10x h5 中的 RNA 和 ATAC 特征拆开。
    rna_mask = features["feature_type"].eq("Gene Expression").to_numpy()
    atac_mask = features["feature_type"].eq("Peaks").to_numpy()

    rna_raw = matrix_to_adata(subset_rows(matrix, rna_mask), features.loc[rna_mask], barcodes)
    atac_raw = matrix_to_adata(subset_rows(matrix, atac_mask), features.loc[atac_mask], barcodes)

    # 分别做 RNA 基因过滤和 ATAC 峰过滤。
    rna_raw = filter_features_by_cells(rna_raw, RNA_MIN_CELLS)
    atac_raw = filter_features_by_cells(atac_raw, ATAC_MIN_CELLS)

    # 共同质控的第一步：去掉线粒体比例过高的细胞。
    mt_fraction = compute_mt_fraction(rna_raw)
    cell_qc_mask = mt_fraction <= MT_THRESHOLD

    rna_qc = rna_raw[cell_qc_mask].copy()
    atac_qc = atac_raw[cell_qc_mask].copy()
    rna_qc.obs["mt_fraction"] = mt_fraction[cell_qc_mask]

    # 双细胞检测依赖 ATAC 二值矩阵，因此先复制一份再二值化。
    atac_binary = atac_qc.copy()
    binarize_adata(atac_binary)

    doublet_mask, doublet_metrics = detect_doublets(rna_qc, atac_binary, DOUBLET_MAD_MULTIPLIER)

    # 共同质控的第二步：过滤预测为 doublet 的细胞。
    keep_mask = ~doublet_mask
    rna_filtered = rna_qc[keep_mask].copy()
    atac_filtered = atac_binary[keep_mask].copy()

    # RNA 继续做标准化、log1p 和高变基因选择。
    rna_processed = rna_filtered.copy()
    normalize_log1p(rna_processed, TARGET_SUM)
    rna_processed.var["highly_variable"] = highly_variable_by_dispersion(rna_processed, TOP_HVGS)

    # ATAC 保持二值矩阵，并选择高变峰。
    atac_processed = atac_filtered.copy()
    atac_processed.var["highly_variable"] = highly_variable_binary_features(atac_processed, TOP_HVPS)

    # 生成一个简短的质控统计表，方便快速查看过滤前后规模变化。
    qc_summary = pd.DataFrame(
        {
            "metric": [
                "cells_input",
                "cells_after_mt_filter",
                "cells_after_doublet_filter",
                "rna_features_after_min_cells",
                "atac_features_after_min_cells",
                "rna_hvgs",
                "atac_hvps",
            ],
            "value": [
                rna_raw.n_obs,
                rna_qc.n_obs,
                rna_processed.n_obs,
                rna_raw.n_vars,
                atac_raw.n_vars,
                int(rna_processed.var["highly_variable"].sum()),
                int(atac_processed.var["highly_variable"].sum()),
            ],
        }
    )

    # 输出三个层面的结果：
    # 1. 预处理后的 RNA/ATAC AnnData
    # 2. 每个细胞的双细胞判定细节
    # 3. 汇总性质控统计
    rna_processed.write_h5ad(args.output_dir / "rna_processed.h5ad")
    atac_processed.write_h5ad(args.output_dir / "atac_processed.h5ad")
    doublet_metrics.to_csv(args.output_dir / "doublet_metrics.csv")
    qc_summary.to_csv(args.output_dir / "qc_summary.csv", index=False)

    print(qc_summary.to_string(index=False))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
