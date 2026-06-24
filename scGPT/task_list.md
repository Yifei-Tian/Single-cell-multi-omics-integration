# scGPT 多组学整合分析 — 任务清单

> **项目**：Transformer 在单细胞多组学整合中的应用（scGPT 分支）
> **数据**：10x PBMC Multiome（RNA + ATAC 配对数据）
> **目标**：完整运行 scGPT 对 PBMC 数据的多组学整合分析，包括微调、嵌入提取、跨模态翻译和指标评估

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.10 |
| CUDA | >= 11.8 |
| GPU 最低配置 | NVIDIA RTX 3090 (24 GB) 或 V100 (16 GB) |
| 存储空间 | ~20 GB（数据 + 中间结果，不含预训练模型） |

## 预置条件（用户自行准备）

在运行以下任务前，请确认以下文件已就绪：

```
scGPT/pretrained_models/scGPT_human/
    ├── best_model.pt       ← 预训练权重（来自 Hugging Face bowang-lab/scGPT_human）
    ├── vocab.json          ← 基因词表
    └── args.json           ← 模型结构超参数

data/
    └── pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5   ← 10x 原始数据
```

---

## 任务列表

### ✅ Task 1 — 环境安装

**目的**：创建隔离的 Python 环境并安装所有依赖

**输入**：`scGPT/requirements.txt`

**输出**：可用的 `scgpt` conda 环境

**命令**：
```bash
# 创建 conda 环境
conda create -n scgpt python=3.10 -y
conda activate scgpt

# 安装全部依赖
pip install -r scGPT/requirements.txt

# 验证安装成功
python -c "import scgpt; print('scgpt version:', scgpt.__version__)"
python -c "import scanpy, anndata, torch; print('OK')"
```

**预期耗时**：10–20 分钟（取决于网络速度）

**注意**：
- 如需 Flash Attention 加速，额外运行：`pip install flash-attn --no-build-isolation`
- 所有后续命令均需在 `conda activate scgpt` 环境下执行

---

### ✅ Task 2 — 数据预处理（RNA + ATAC）

**目的**：从 10x `.h5` 原始文件生成预处理后的 RNA 和 ATAC AnnData

**输入**：
- `data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5`

**输出**：
- `outputs/pbmc_granulocyte_sorted_10k/rna_processed.h5ad`
- `outputs/pbmc_granulocyte_sorted_10k/atac_processed.h5ad`
- `outputs/pbmc_granulocyte_sorted_10k/qc_summary.csv`
- `outputs/pbmc_granulocyte_sorted_10k/doublet_metrics.csv`

**命令**：
```bash
python scripts/preprocess_pbmc_multiome.py \
    --input-h5 data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
    --output-dir outputs/pbmc_granulocyte_sorted_10k
```

**预处理逻辑**（与 Proposal §5 一致）：
- RNA：过滤低表达基因（min_cells=3）、去线粒体比例 > 20% 细胞、去双细胞、log1p 归一化至 10,000、选择 3,000 高变基因
- ATAC：过滤低开放峰（min_cells=3）、二值化、选择 5,000 高变峰

**预期耗时**：3–10 分钟

---

### ✅ Task 3 — 数据格式转换（scGPT 输入格式）

**目的**：将预处理后的 RNA 和 ATAC 数据转换为 scGPT 所需的 token 序列格式

**输入**：
- `outputs/pbmc_granulocyte_sorted_10k/rna_processed.h5ad`
- `outputs/pbmc_granulocyte_sorted_10k/atac_processed.h5ad`
- `scGPT/pretrained_models/scGPT_human/vocab.json`

**输出**：
- `scGPT/data/pbmc_scgpt_input.h5ad`（合并的多模态 AnnData，含 token_ids 层和 values 层）
- `scGPT/data/gene2idx.json`（基因/峰 → token id 映射）

**命令**：
```bash
python scGPT/prepare_data.py
```

**转换逻辑**：
- RNA 表达量分箱：log1p 值 → 51 个离散区间（与 scGPT 预训练一致）
- 基因词表对齐：仅保留预训练 vocab.json 中存在的基因
- ATAC 峰分配新 token id（从 max(RNA token id) + 1 开始）
- 合并 AnnData 的 `layers["token_ids"]` 和 `layers["values"]` 分别存储 token id 和分箱值
- `var["modality_type"]`：`"rna"` 或 `"atac"` 标识每个特征的模态来源

**预期耗时**：5–15 分钟

---

### ✅ Task 4 — 细胞类型标注

**目的**：通过 leiden 聚类 + PBMC marker 基因自动为细胞赋予类型标签

**输入**：
- `scGPT/data/pbmc_scgpt_input.h5ad`

**输出**：
- `scGPT/data/pbmc_scgpt_input_labeled.h5ad`（含 `obs["cell_type"]` 和 `obs["leiden"]`）

**命令**：
```bash
python scGPT/annotate_cells.py
```

**标注逻辑**：
- 提取 RNA 模态，做 PCA（30 维）+ KNN（15 邻居）+ leiden（分辨率 0.5）
- 对每个簇用 `sc.tl.score_genes` 计算 8 种 PBMC 细胞类型的 marker 基因得分
- 得分最高的细胞类型作为该簇的标签

**PBMC Marker 基因参考**：

| 细胞类型 | Marker 基因 |
|---------|-------------|
| CD4 T cell | CD3D, CD4, IL7R, CCR7 |
| CD8 T cell | CD3D, CD8A, CD8B, GZMK |
| NK cell | GNLY, NKG7, KLRD1 |
| B cell | MS4A1, CD79A, CD79B |
| CD14 Monocyte | CD14, LYZ, S100A8, S100A9 |
| FCGR3A Monocyte | FCGR3A, MS4A7, CX3CR1 |
| Dendritic cell | FCER1A, CST3, CLEC4C |
| Platelet | PPBP, GP1BB, PF4 |

**预期耗时**：5–15 分钟

---

### ✅ Task 5 — scGPT 微调（MLM 预热 + SFT 收敛，两阶段）

**目的**：以预训练权重为起点，采用 **MLM→SFT** 两阶段策略对模型进行多组学整合微调

```
阶段 1 — MLM 预热：随机遮蔽 20% token，让模型适配 RNA+ATAC 联合序列格式（全参数训练）
阶段 2 — SFT 收敛：冻结底层 10 层，解冻顶层 2 层 + 分类头，以 cell_type 标签做联合监督
                    联合损失 = 0.3 × MLM_loss + 0.7 × SFT_loss
```

**输入**：
- `scGPT/data/pbmc_scgpt_input_labeled.h5ad`（含 `obs["cell_type"]` 标签）
- `scGPT/pretrained_models/scGPT_human/best_model.pt`

**输出**：
- `scGPT/result/best_finetuned.pt`（最优模型权重，含 Transformer + 分类头）
- `scGPT/result/mlm_stage_end.pt`（MLM 阶段结束的 checkpoint）
- `scGPT/result/training_log.csv`（各 epoch 损失、SFT 准确率、耗时、显存）

**命令**：
```bash
# 默认两阶段运行（MLM 6 epochs → SFT 4 epochs）
python scGPT/finetune_integration.py

# 显存不足时减小 batch_size
python scGPT/finetune_integration.py --batch_size 32

# 仅运行 MLM 阶段
python scGPT/finetune_integration.py --skip_sft

# 跳过 MLM，直接从已有 checkpoint 做 SFT
python scGPT/finetune_integration.py --skip_mlm --mlm_checkpoint scGPT/result/mlm_stage_end.pt
```

**关键超参数**（参照 Proposal §方法架构详解）：

| 参数 | 值 | 阶段 | 说明 |
|------|-----|------|------|
| embed_dim | 512 | 共用 | Transformer 嵌入维度 |
| n_layers | 12 | 共用 | Transformer 层数 |
| n_heads | 8 | 共用 | 注意力头数 |
| mask_ratio | 0.2 | 共用 | 遮蔽比例（预训练为 0.4） |
| batch_size | 64 | 共用 | 批次大小（显存不足可调为 32） |
| lr_mlm | 1e-4 | MLM | MLM 阶段学习率 |
| n_epochs_mlm | 6 | MLM | MLM 预热轮数 |
| lr_sft | 5e-5 | SFT | SFT 阶段学习率（较小，防止灾难性遗忘） |
| n_epochs_sft | 4 | SFT | SFT 收敛轮数 |
| sft_alpha | 0.3 | SFT | 联合损失中 MLM 权重（SFT 权重 = 0.7） |
| n_frozen_layers | 10 | SFT | SFT 阶段冻结的底层数量（解冻顶层 2 层） |

**训练目标**：
- MLM：交叉熵预测被遮蔽 token 的原始分箱值（51 类分类）
- SFT：交叉熵预测细胞类型标签（[CLS] token → 分类头）
- SFT 阶段联合损失：`0.3 × MLM_loss + 0.7 × SFT_loss`

**预期耗时**：
- RTX 3090: ~2–4 小时
- A100: ~0.5–1 小时

---

### ✅ Task 6 — 嵌入提取

**目的**：加载微调模型，对全部细胞提取三组嵌入向量

**输入**：
- `scGPT/data/pbmc_scgpt_input_labeled.h5ad`
- `scGPT/result/best_finetuned.pt`

**输出**：
- `scGPT/result/joint_embedding.npy`（RNA+ATAC 联合嵌入，shape: n_cells × 512）
- `scGPT/result/rna_embedding.npy`（仅 RNA token 推断，shape: n_cells × 512）
- `scGPT/result/atac_embedding.npy`（仅 ATAC token 推断，shape: n_cells × 512）
- `scGPT/result/adata_with_emb.h5ad`（含 `obsm["X_scgpt"]`、`obsm["X_scgpt_rna"]`、`obsm["X_scgpt_atac"]`）

**命令**：
```bash
python scGPT/extract_embeddings.py

# 显存不足时减小 batch_size
python scGPT/extract_embeddings.py --batch_size 64
```

**嵌入策略**：
- scGPT TransformerModel：提取 `[CLS]` token 输出（d_model=512 维向量）
- 回退模型（scgpt 未安装时）：对所有非填充 token 做 mean pooling

**预期耗时**：10–30 分钟

---

### ✅ Task 7 — 跨模态翻译评估

**目的**：用 RNA 信息预测 ATAC 信号，评估 scGPT 的跨模态生成能力

**输入**：
- `scGPT/data/pbmc_scgpt_input_labeled.h5ad`
- `scGPT/result/best_finetuned.pt`

**输出**：
- `scGPT/result/predicted_atac.npy`（模型预测的 ATAC 信号矩阵）
- `scGPT/result/translation_metrics.csv`（跨模态翻译评估指标）

**命令**：
```bash
python scGPT/cross_modal_translation.py
```

**翻译逻辑**：
1. 推断时将所有 ATAC 模态 token 的值遮蔽为 0（完全遮蔽）
2. 保留 RNA 值不变，让模型从 RNA 上下文推断 ATAC
3. 取模型输出 logits 的 argmax → bin center → 连续预测值

**评估指标**：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| reconstruction_mse | 均方误差（真实 vs 预测 ATAC） | ↓越低越好 |
| pearson_global | 全局 Pearson 相关系数 | ↑越高越好 |
| pearson_per_cell_median | 逐细胞 Pearson 中位数 | ↑越高越好 |
| cosine_similarity | Cell-state cosine similarity | ↑越高越好，上限1 |
| auprc_per_peak_mean | 逐峰 AUPRC 均值 | ↑越高越好，上限1 |

**预期耗时**：15–30 分钟

---

### ✅ Task 8 — 整合质量评估

**目的**：计算 Proposal §6 中定义的全套整合质量和计算效率指标

**输入**：
- `scGPT/result/adata_with_emb.h5ad`

**输出**：
- `scGPT/result/evaluation_metrics.csv`（全套指标）

**命令**：
```bash
python scGPT/evaluate.py
```

**评估指标（参照 Proposal §6.1 & §6.3）**：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| ARI | 聚类 vs 真实标签一致性 | 1.0 |
| NMI | 聚类与标签的互信息 | 1.0 |
| silhouette_score | 细胞类型分离度（余弦距离） | 1.0 |
| cLISI | 细胞类型局部多样性（↓=类型分离好） | 接近 1 |
| iLISI | 批次局部多样性（↑=批次混合好） | 接近批次数 |
| graph_connectivity | 同类型细胞 KNN 图连通性 | 1.0 |
| FOSCTTM | 模态对齐率（↓=对齐越好） | 0.0 |
| training_time_s | 总训练时间（秒） | — |
| inference_time_per_1k_s | 每 1000 细胞推断时间（秒） | — |
| peak_memory_gb | GPU 显存峰值（GB） | — |
| n_params_M | 参数量（百万） | — |

**预期耗时**：10–30 分钟（cLISI/FOSCTTM 计算较耗时）

---

### ✅ Task 9 — 结果可视化

**目的**：生成 UMAP 图和跨模态翻译散点图，用于直观展示整合质量

**输入**：
- `scGPT/result/adata_with_emb.h5ad`
- `scGPT/result/predicted_atac.npy`（可选）
- `scGPT/result/evaluation_metrics.csv`（可选，用于指标条形图）

**输出**：
- `scGPT/result/umap_cell_type.png`：按细胞类型着色的 UMAP（300 dpi）
- `scGPT/result/umap_leiden.png`：按 leiden 簇着色的 UMAP（300 dpi）
- `scGPT/result/umap_modality.png`：RNA 嵌入 vs ATAC 嵌入 UMAP 并排对比（300 dpi）
- `scGPT/result/translation_scatter.png`：真实 ATAC vs 预测 ATAC 散点图（300 dpi）
- `scGPT/result/metrics_summary.png`：整合质量指标条形图（300 dpi）

**命令**：
```bash
python scGPT/visualize.py

# 调整散点图采样量
python scGPT/visualize.py --n_sample_peaks 500 --n_sample_cells 2000
```

**预期耗时**：5–15 分钟

---

## 完整运行顺序

按以下顺序依次执行（每个步骤依赖上一步的输出）：

```bash
conda activate scgpt

# Step 1: 预处理（若 outputs/ 目录不存在）
python scripts/preprocess_pbmc_multiome.py

# Step 2: 数据格式转换
python scGPT/prepare_data.py

# Step 3: 细胞类型标注
python scGPT/annotate_cells.py

# Step 4: 微调训练（最耗时，建议 GPU 环境）
python scGPT/finetune_integration.py

# Step 5: 嵌入提取
python scGPT/extract_embeddings.py

# Step 6 & 7: 跨模态翻译 + 整合质量评估（可并行运行）
python scGPT/cross_modal_translation.py &
python scGPT/evaluate.py &
wait

# Step 8: 可视化
python scGPT/visualize.py
```

---

## 输出文件总览

运行完成后，`scGPT/` 目录结构如下：

```
scGPT/
├── requirements.txt                  ← 依赖清单
├── prepare_data.py                   ← Task 3：数据格式转换
├── annotate_cells.py                 ← Task 4：细胞类型标注
├── finetune_integration.py           ← Task 5：微调训练
├── extract_embeddings.py             ← Task 6：嵌入提取
├── cross_modal_translation.py        ← Task 7：跨模态翻译
├── evaluate.py                       ← Task 8：整合质量评估
├── visualize.py                      ← Task 9：结果可视化
├── task_list.md                      ← 本文件
├── data/
│   ├── pbmc_scgpt_input.h5ad         ← Task 3 输出
│   ├── pbmc_scgpt_input_labeled.h5ad ← Task 4 输出
│   └── gene2idx.json                 ← Task 3 输出（词表映射）
├── pretrained_models/
│   └── scGPT_human/                  ← 【用户自行准备】
│       ├── best_model.pt
│       ├── vocab.json
│       └── args.json
└── result/
    ├── best_finetuned.pt             ← Task 5 输出
    ├── training_log.csv              ← Task 5 输出
    ├── joint_embedding.npy           ← Task 6 输出
    ├── rna_embedding.npy             ← Task 6 输出
    ├── atac_embedding.npy            ← Task 6 输出
    ├── adata_with_emb.h5ad           ← Task 6 输出
    ├── predicted_atac.npy            ← Task 7 输出
    ├── translation_metrics.csv       ← Task 7 输出
    ├── evaluation_metrics.csv        ← Task 8 输出
    ├── umap_cell_type.png            ← Task 9 输出
    ├── umap_leiden.png               ← Task 9 输出
    ├── umap_modality.png             ← Task 9 输出
    ├── translation_scatter.png       ← Task 9 输出
    └── metrics_summary.png           ← Task 9 输出
```

---

## 常见问题

**Q: 显存不足（OOM）**
- 将 `--batch_size` 从 64 调小为 32 或 16
- 安装 flash-attn：`pip install flash-attn --no-build-isolation`

**Q: 预训练模型下载地址**
- Hugging Face：`https://huggingface.co/bowang-lab/scGPT_human`（需自行下载并放置到 `scGPT/pretrained_models/scGPT_human/`）

**Q: scgpt 包导入失败**
- 尝试从源码安装：`pip install git+https://github.com/bowang-lab/scGPT.git`
- 所有脚本均含"回退模式"，在 scgpt 不可用时自动切换到本地轻量实现

**Q: igraph 相关报错**
- 安装：`pip install python-igraph leidenalg`

**Q: 评估结果与论文不一致**
- 检查随机种子（默认 42）
- 检查基因词表对齐比例（`prepare_data.py` 会打印匹配率）
- 参考 scGPT 官方 Tutorial：`https://github.com/bowang-lab/scGPT/tree/main/tutorials`
