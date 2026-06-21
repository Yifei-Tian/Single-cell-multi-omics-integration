# Single-cell-multi-omics-integration

## 当前文件结构

```text
Single-cell-multi-omics-integration/
|-- README.md
|-- data/
|   |-- pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
|   `-- pbmc_granulocyte_sorted_10k_atac_peaks.bed
|-- outputs/
|   `-- pbmc_granulocyte_sorted_10k/
|       `-- rna_processed.h5ad
`-- scripts/
    `-- preprocess_pbmc_multiome.py
```

## 目录说明

- `README.md`
  仓库说明文档，目前主要记录项目中的目录结构与文件用途。

- `data/`
  存放从 10x Genomics 下载的原始数据文件。
  - `pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5`：10x Genomics Cell Ranger ARC 输出的过滤后特征矩阵文件。它是一个 `HDF5` 格式的稀疏矩阵，保存了每个细胞 barcode 在每个特征上的信号值，并同时包含两种模态的信息：RNA 的基因表达计数和 ATAC 的染色质开放峰计数。这个文件是当前预处理脚本的主要输入。
  - `pbmc_granulocyte_sorted_10k_atac_peaks.bed`：ATAC 峰集合的基因组区间文件，采用 `BED` 格式保存。每一行通常对应一个开放染色质峰的位置，例如染色体、起始位点和终止位点。它主要用于说明 ATAC 特征在基因组上的坐标含义，方便后续做峰注释、可视化或与基因组区域进行关联分析。

- `scripts/`
  存放预处理脚本。
  - `preprocess_pbmc_multiome.py`：用于读取 10x 的 `.h5` 文件，并执行 RNA / ATAC 预处理与联合质控。

- `outputs/`
  存放预处理后的输出结果。
  - `pbmc_granulocyte_sorted_10k/`：当前样本的输出目录。
  - `rna_processed.h5ad`：已经生成的 RNA 预处理结果文件。

## data 文件夹中的两个数据分别是什么

- `pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5`
  这是“细胞 x 特征”的主体数据矩阵。虽然名字里只写了 `feature matrix`，但在 multiome 数据里，特征既包括 RNA 基因，也包括 ATAC 峰。也就是说，这个文件里真正保存的是后续分析最核心的数值矩阵。

- `pbmc_granulocyte_sorted_10k_atac_peaks.bed`
  这是 ATAC 峰的坐标说明文件。它本身不是每个细胞的计数矩阵，而是告诉我们 `.h5` 文件中的 ATAC 峰分别位于基因组的什么位置。

## 当前状态

目前仓库已经包含：

- 原始下载数据
- 预处理脚本
- 部分预处理输出（当前可见 `rna_processed.h5ad`）

后续如果继续完善流程，可以在 `outputs/pbmc_granulocyte_sorted_10k/` 下继续补充：

- `atac_processed.h5ad`
- `qc_summary.csv`
- `doublet_metrics.csv`

这样可以把 RNA、ATAC 和联合 QC 的结果集中保存在同一个输出目录中。