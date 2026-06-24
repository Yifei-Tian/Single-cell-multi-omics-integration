# Transformer 在单细胞多组学整合中的应用：系统性对比研究

## 项目 Proposal

**项目类型**：AI × 生物科学实践项目
**项目周期**：8–10 周
**技术栈**：Python, PyTorch, scvi-tools, Scanpy, Muon

---

## 一、项目背景与动机

单细胞测序技术已经能够同时测量同一细胞的多种分子模态（如基因表达 RNA 和染色质可及性 ATAC），但不同模态的数据特征差异巨大：RNA 是稀疏的计数矩阵，ATAC 是二值的峰信号矩阵。如何将这些异质数据整合到统一的低维空间中，同时保留生物学变异（如细胞类型差异），是单细胞计算生物学的核心挑战之一。

近年来，Transformer 架构在单细胞领域展现出巨大潜力。从 scGPT 这样的通用基础模型，到 scMoFormer 这样的专用多组学整合模型，不同技术路线的模型在架构设计、预训练策略和下游应用上各有特色。然而，目前缺乏一个**系统性的、在同一数据集和同一评估框架下**的对比研究，来帮助研究者理解：什么场景下应该选择哪种模型？Transformer 相比传统方法（如 VAE、GNN）是否真的有显著优势？

本项目旨在填补这一空白，通过对比 6 种代表性方法在 RNA+ATAC 配对数据整合任务上的表现，深入理解 Transformer 在单细胞多组学中的适用边界。

---

## 二、研究问题

1. 在 RNA+ATAC 配对数据整合任务中，基于 Transformer 的方法（scGPT、scMoFormer、scmFormer）相比基于 VAE 的方法（MIDAS、MultiVI）和基于 GNN 的方法（GLUE），在整合质量上是否有显著优势？
2. 不同架构的方法在计算效率（训练时间、推理时间、内存占用）上存在多大差异？这种差异是否与数据规模呈线性关系？
3. 各方法对数据稀疏性和批次效应的鲁棒性如何？当测序深度下降或跨批次整合时，性能衰减模式有何不同？
4. 跨模态翻译（从 RNA 预测 ATAC）是否是评估多组学整合质量的可靠代理指标？

---

## 三、任务定义

### 主任务：单细胞 RNA-seq 与 ATAC-seq 配对数据的多组学整合

给定配对的 scRNA-seq 和 scATAC-seq 数据矩阵 $\mathbf{X}^{RNA} \in \mathbb{N}^{n \times g}$ 和 $\mathbf{X}^{ATAC} \in \{0,1\}^{n \times p}$（$n$ 为细胞数，$g$ 为基因数，$p$ 为峰数），目标是学习一个联合嵌入函数 $f: (\mathbf{X}^{RNA}, \mathbf{X}^{ATAC}) \mapsto \mathbf{Z} \in \mathbb{R}^{n \times d}$，使得：

- 同一细胞的不同模态在嵌入空间中彼此接近（模态对齐）
- 不同细胞类型的细胞在嵌入空间中形成分离的簇（生物学变异保留）
- 来自不同批次的同一细胞类型细胞在嵌入空间中混合（批次校正）

### 子任务

**子任务 A：多组学整合质量评估**
- 将配对数据输入模型，获得联合嵌入
- 评估嵌入空间的细胞类型分离度、批次混合度、模态一致性

**子任务 B：跨模态翻译**
- 输入 RNA 模态，预测 ATAC 模态（或反向）
- 评估预测信号与真实信号的相关性

**子任务 C：缺失模态插补**
- 模拟仅有一种模态可用的场景
- 评估模型从单一模态恢复完整细胞状态的能力

---

## 四、对比方法

### 方法选择原则

从论文列表中的 28 种方法中，按照以下原则筛选：
1. **代表性**：覆盖不同架构路线（Transformer、VAE、GNN）
2. **可复现性**：有公开代码和文档，社区活跃度较高
3. **功能匹配**：原生支持或可通过微调支持 RNA+ATAC 整合
4. **可比性**：在同一基准数据集上有已发表结果，便于验证

### 最终选择的 6 种方法

| 方法 | 架构类别 | 核心特点 | 预训练 | 代码可用性 |
|------|---------|---------|--------|-----------|
| **scGPT** | 生成式 Transformer (GPT) | 3300万细胞预训练，支持多模态拼接，生成式掩码预训练 | 是 | GitHub 开源，文档完善 |
| **scMoFormer** | 多编码器 Transformer | 每个模态独立 Transformer 编码器，交叉注意力融合 | 否 | GitHub 开源 |
| **scmFormer** | 多任务 Transformer | 首次整合蛋白质组，多任务联合训练 | 否 | GitHub 开源 |
| **MIDAS** | VAE + 自监督对齐 | 深度概率框架，马赛克整合，知识迁移 | 否 | GitHub 开源 |
| **GLUE** | GNN + 变分推断 | 图连接统一嵌入，建模跨组学调控关系 | 否 | GitHub 开源 |
| **MultiVI** | VAE | 深度生成模型，整合转录组、染色质和蛋白质 | 否 | scvi-tools 内置 |

### 方法架构详解

#### 1. scGPT

**核心架构**：基于 GPT 风格的生成式 Transformer，将基因视为 token，表达量分箱后作为输入。

**为什么适合本任务**：
- scGPT 在预训练阶段已同时学习 RNA 和 ATAC 的表示，通过"模态条件嵌入"区分不同模态
- 支持多模态拼接输入：将 RNA 基因和 ATAC 峰作为统一的 token 序列输入模型
- 在 scGPT 的微调框架中，可以直接进行多组学整合任务

**关键超参数**：
- 嵌入维度：512
- Transformer 层数：12
- 注意力头数：8
- 掩码比例：0.4（预训练），0.2（微调）

**论文**：Cui, H., et al. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21(8), 1470–1480.

#### 2. scMoFormer

**核心架构**：多个独立的 Transformer 编码器分别处理不同模态，通过交叉注意力层进行模态间信息融合。

**为什么适合本任务**：
- 专为多模态设计，每个模态有独立的编码路径，避免了模态特征冲突
- 交叉注意力机制显式建模模态间的对应关系
- 端到端训练，无需预训练

**关键超参数**：
- 每个模态编码器层数：6
- 交叉注意力层数：2
- 隐藏维度：256
- dropout：0.1

**论文**：Tang, W., et al. (2023). Single-cell multimodal prediction via transformers. *CIKM'23*, 2425–2435.

#### 3. scmFormer

**核心架构**：多任务 Transformer，通过共享编码器和任务特定解码器实现多组学联合建模。

**为什么适合本任务**：
- 多任务学习框架天然适合多组学整合
- 在蛋白质组整合上表现优异，架构可迁移至 ATAC
- 支持大规模数据训练

**关键超参数**：
- 共享编码器层数：8
- 任务特定解码器层数：2
- 隐藏维度：512

**论文**：Chen, Y., et al. (2024). scmFormer integrates large-scale single-cell proteomics and transcriptomics data by multi-task transformer. *Advanced Science*, 11(10), 2307835.

#### 4. MIDAS

**核心架构**：变分自编码器（VAE）+ 自监督模态对齐，深度概率框架。

**为什么适合本任务**：
- 虽然核心架构是 VAE 而非 Transformer，但在模态对齐模块中使用了注意力机制
- 支持马赛克整合（部分模态缺失），与 scGPT 形成互补对比
- 在 *Nature Biotechnology* 上发表，方法成熟

**关键超参数**：
- 潜在维度：20
- 编码器隐藏层：[128, 64]
- 学习率：0.001
- 批次大小：256

**论文**：He, Z., et al. (2024). Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS. *Nature Biotechnology*, 42(10), 1594–1605.

#### 5. GLUE

**核心架构**：图神经网络（GNN）+ 变分推断，通过建模跨组学调控关系（如峰-基因关联）实现整合。

**为什么适合本任务**：
- 代表了非 Transformer 路线的主流方法
- 利用先验生物学知识（调控关系图）指导整合
- 与纯数据驱动的方法形成对比

**关键超参数**：
- 图编码器层数：2
- 变分编码器隐藏层：[128, 64]
- 学习率：0.001

**论文**：Cao, Z. J., & Gao, G. (2022). Multi-omics single-cell data integration and regulatory inference with graph-linked unified embedding. *Nature Biotechnology*, 40(10), 1458–1466.

#### 6. MultiVI

**核心架构**：深度生成模型（VAE），通过共享潜在变量整合多模态数据。

**为什么适合本任务**：
- scvi-tools 生态系统内置，代码最成熟稳定
- 作为 VAE 路线的基准方法，社区广泛使用
- 支持配对和非配对数据整合

**关键超参数**：
- 潜在维度：20
- 编码器隐藏层：[128, 128]
- 解码器隐藏层：[128, 128]
- 学习率：0.001

**论文**：Ashuach, T., et al. (2023). MultiVI: deep generative model for the integration of multimodal data. *Nature Methods*, 20(8), 1222–1231.

---

## 五、数据集

### 主数据集：10x PBMC Multiome

**基本信息**：
- 来源：10x Genomics 官方演示数据
- 模态：RNA + ATAC（配对）
- 细胞数：~6,000（经过质控后）
- 组织：人外周血单个核细胞（PBMC）
- 细胞类型：T细胞、B细胞、单核细胞、NK细胞、树突状细胞等
- 批次：1 个批次（便于控制变量）

**为什么选它**：
- 几乎所有多组学整合方法的论文都使用此数据集作为基准
- 细胞类型注释清晰，便于评估生物学保留度
- 数据质量高，技术噪声低
- 规模适中，便于在单 GPU 上快速实验

**下载方式**：
```bash
# 通过 10x Genomics 官网下载
wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5
wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_peaks.bed

# 或通过 Scanpy 示例数据加载
import scanpy as sc
adata = sc.datasets.pbmc3k_processed()  # 参考 RNA 数据
```

**预处理流程**（所有方法统一）：
1. RNA：过滤低表达基因（min_cells=3），归一化至 10,000 counts，log1p 变换，选择高变基因（n_top_genes=3000）
2. ATAC：过滤低开放峰（min_cells=3），二值化，选择高变峰（n_top_peaks=5000）
3. 共同质控：过滤线粒体比例 > 20% 的细胞，过滤双细胞

### 扩展数据集：SHARE-seq Mouse Skin

**基本信息**：
- 来源：Ma et al., *Cell*, 2020
- 模态：RNA + ATAC（配对）
- 细胞数：~35,000
- 组织：小鼠皮肤（毛囊、表皮、真皮）
- 获取方式：GEO GSE140203

**为什么选它**：
- 细胞数更大，适合评估方法的可扩展性
- 细胞类型更复杂（毛囊干细胞、基质细胞、免疫细胞等）
- 跨组织验证，增强结论的泛化性

### 鲁棒性测试数据集

**稀疏性测试**：对 10x PBMC Multiome 进行下采样，分别保留 50%、30%、10% 的 reads，评估方法在不同测序深度下的性能。

**批次效应测试**：使用 NeurIPS 2021 Open Problems 竞赛中的多批次骨髓数据集（~90,000 细胞，3 个批次），评估跨批次整合能力。

---

## 六、评估指标

### 6.1 整合质量指标

| 指标 | 含义 | 理想值 | 计算工具 |
|------|------|--------|---------|
| **ARI** (Adjusted Rand Index) | 聚类结果与真实标签的一致性 | 1.0 | scikit-learn |
| **NMI** (Normalized Mutual Information) | 聚类与标签的互信息 | 1.0 | scikit-learn |
| **cLISI** (cell-type LISI) | 细胞类型局部多样性（高=类型混合差） | 接近细胞类型数 | lisi (scib) |
| **iLISI** (integration LISI) | 批次局部多样性（高=批次混合好） | 接近批次数 | lisi (scib) |
| **kBET** | 批次混合接受率 | 1.0 | kBET (scib) |
| **FOSCTTM** | 模态间最近邻重叠率 | 0.0 | 自定义 |
| **graph connectivity** | 同一细胞类型细胞在 KNN 图中的连通性 | 1.0 | scib |

### 6.2 跨模态翻译指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Peak-gene correlation** | 预测的 ATAC 峰与基因表达的相关性 | 与已知调控关系对比 |
| **Reconstruction MSE** | 重建信号与真实信号的均方误差 | 逐元素计算 |
| **Cell-state cosine similarity** | 翻译后嵌入与真实嵌入的余弦相似度 | 逐细胞计算 |

### 6.3 计算效率指标

| 指标 | 记录方式 |
|------|---------|
| **Training time** | GPU 小时（NVIDIA A100/V100） |
| **Inference time** | 每 1000 细胞秒数 |
| **Peak memory** | 训练过程中 GPU 内存峰值（GB） |
| **Model size** | 参数量（M） |

### 6.4 鲁棒性指标

| 测试场景 | 评估方法 |
|---------|---------|
| 50% reads 下采样 | 与全数据性能的差异 |
| 30% reads 下采样 | 与全数据性能的差异 |
| 10% reads 下采样 | 与全数据性能的差异 |
| 跨批次整合 | 批次校正后的 cLISI 和 iLISI |

---

## 七、实验设计

### 实验 1：基准性能对比（主实验）

**目标**：在 10x PBMC Multiome 上，对比 6 种方法的整合质量、翻译准确性和计算效率。

**流程**：
1. 统一预处理数据
2. 对每个方法：
   - 按照论文推荐超参数训练
   - 记录训练时间、内存占用
   - 获取联合嵌入
   - 计算所有整合质量指标
   - 进行跨模态翻译，计算翻译指标
3. 汇总结果，生成对比表格和可视化

**预期结果**：
- Transformer 方法（scGPT、scMoFormer）在整合质量上可能优于 VAE 方法
- scGPT 由于预训练优势，可能在数据量较小时表现更好
- GLUE 可能利用调控先验知识，在峰-基因关联恢复上有优势
- MultiVI 作为最成熟的 VAE 方法，可能是稳健性基准

### 实验 2：可扩展性测试

**目标**：在 SHARE-seq Mouse Skin（~35,000 细胞）上，测试各方法在大规模数据上的表现。

**流程**：
1. 使用与实验 1 相同的超参数
2. 记录训练时间和内存随细胞数的变化
3. 评估整合质量是否随规模增加而下降

**预期结果**：
- scGPT 由于预训练，可能在大规模数据上训练更快（微调 vs 从头训练）
- GNN 方法（GLUE）的图构建成本可能随细胞数平方增长，成为瓶颈
- VAE 方法通常具有较好的可扩展性

### 实验 3：鲁棒性测试

**目标**：评估各方法对数据质量和批次效应的鲁棒性。

**子实验 3A：稀疏性鲁棒性**
- 对 10x PBMC Multiome 进行 50%、30%、10% 下采样
- 评估各方法在不同稀疏度下的性能衰减曲线

**子实验 3B：批次效应鲁棒性**
- 使用 NeurIPS 2021 多批次骨髓数据集
- 评估各方法在跨批次整合后的批次混合度和生物学保留度

**预期结果**：
- 预训练模型（scGPT）可能对稀疏性更鲁棒，因为预训练数据覆盖了广泛的表达模式
- 基于图的方法（GLUE）可能对批次效应更敏感，因为图结构受批次影响
- 专门设计批次校正的方法（MIDAS、MultiVI）可能在跨批次场景下有优势

### 实验 4：消融分析（以 scGPT 为例）

**目标**：理解 Transformer 架构中各组件对多组学整合的贡献。

**设计**：
- 对比预训练 vs 从头训练 scGPT
- 对比不同掩码比例（0.1, 0.2, 0.4, 0.6）
- 对比不同嵌入维度（128, 256, 512）

**预期结果**：
- 预训练显著优于从头训练，验证基础模型的价值
- 掩码比例存在最优值，过高或过低均影响性能

---

## 八、项目时间线

| 周次 | 任务 | 产出 |
|------|------|------|
| **Week 1** | 环境配置、数据下载与预处理、熟悉 scvi-tools 和 Scanpy | 预处理后的数据文件、环境配置文档 |
| **Week 2** | scGPT 复现：安装、预训练模型下载、微调脚本运行 | scGPT 在 PBMC 上的基准结果 |
| **Week 3** | GLUE 和 MultiVI 复现 | 两个 VAE/GNN 方法的基准结果 |
| **Week 4** | scMoFormer 和 scmFormer 复现 | 两个专用 Transformer 方法的基准结果 |
| **Week 5** | MIDAS 复现、统一评估框架搭建 | 所有方法的基准结果、评估脚本 |
| **Week 6** | 实验 2（可扩展性）：SHARE-seq 数据集实验 | 可扩展性对比结果 |
| **Week 7** | 实验 3（鲁棒性）：下采样和跨批次实验 | 鲁棒性对比结果 |
| **Week 8** | 实验 4（消融分析）：scGPT 组件分析 | 消融分析结果 |
| **Week 9** | 结果汇总、可视化、报告撰写 | 完整对比表格、UMAP 图、性能曲线 |
| **Week 10** | 报告完善、代码整理、文档撰写 | 最终项目报告、GitHub 仓库 |

---

## 九、技术栈与依赖

### 核心库

```
python >= 3.9
pytorch >= 2.0
cuda >= 11.8

# 单细胞分析
scanpy >= 1.9
anndata >= 0.8
muon >= 0.1  # 多模态分析
scvi-tools >= 1.0  # MultiVI, scvi 生态
scib >= 1.1  # 基准评估指标

# 机器学习
numpy, scipy, pandas, scikit-learn
matplotlib, seaborn, plotly  # 可视化

# 各方法特定依赖（见复现步骤）
# scGPT: transformers, flash-attn
# scMoFormer: fairseq 或自定义实现
# scmFormer: 自定义实现
# MIDAS: torch-geometric
# GLUE: PyG, torch-scatter
```

### 硬件需求

- **最低配置**：NVIDIA RTX 3090 (24GB) 或 V100 (16GB)
- **推荐配置**：NVIDIA A100 (40GB) 或 A6000 (48GB)
- **存储**：~100GB（数据集 + 预训练模型 + 实验结果）

---

## 十、风险评估与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|---------|
| 某方法代码无法运行 | 中 | 高 | 优先复现文档最完善的方法（scGPT、MultiVI）；若某方法失败，用列表中备选方法替代（如 scCLIP 替代 scMoFormer） |
| 显存不足 | 中 | 中 | 使用梯度累积、混合精度训练、或减小批次大小；对 scGPT 使用 flash-attention 优化 |
| 预训练模型下载困难 | 低 | 中 | scGPT 预训练模型可从 Hugging Face 下载；提前准备备用下载方案 |
| 评估指标计算耗时 | 中 | 低 | scib 指标计算较耗时，可采样子集计算或使用 GPU 加速 |
| 结果与论文不一致 | 中 | 低 | 记录所有超参数和随机种子；联系作者确认实现细节；在报告中如实报告差异 |

---

## 十一、预期产出

### 直接产出

1. **系统性对比结果**：6 种方法在 3 个数据集、4 个维度上的完整对比表格
2. **可视化分析**：UMAP 嵌入对比图、性能雷达图、效率散点图、鲁棒性衰减曲线
3. **开源代码**：完整的复现脚本、评估框架、预处理流程（GitHub 仓库）
4. **技术报告**：详细记录实验设计、结果分析和结论

### 学术价值

- 为单细胞多组学整合方法的选择提供实证依据
- 揭示 Transformer 架构在单细胞领域的优势场景和局限
- 为后续方法设计提供基准参考

### 个人收获

- 深入理解 6 种前沿单细胞计算方法的原理和实现
- 掌握单细胞多组学数据的处理、分析和可视化流程
- 获得系统性方法对比的科研经验
- 为后续深入研究（如改进方法、发表工作）奠定基础

---

## 十二、附录：方法复现详细步骤

### A. scGPT 复现步骤

**代码仓库**：https://github.com/bowang-lab/scGPT

**步骤**：

```bash
# 1. 克隆仓库
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT

# 2. 创建环境
conda create -n scgpt python=3.10
conda activate scgpt

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 下载预训练模型（可选，推荐）
# 从 Hugging Face 下载全血预训练模型
# https://huggingface.co/bowang-lab/scGPT

# 5. 准备数据
# 将 10x PBMC Multiome 预处理为 scGPT 输入格式
# 参考 tutorials/Tutorial_Integration.ipynb

# 6. 运行整合
python tutorials/Tutorial_Integration.py \
  --data_path data/pbmc_multiome.h5ad \
  --model_path models/scGPT_human \
  --output_path results/scgpt_pbmc/
```

**关键注意事项**：
- scGPT 需要特定格式的输入（基因 token + 表达量分箱），务必参考官方 tutorial
- 预训练模型文件较大（~1GB），提前下载
- 若显存不足，可减小批次大小或使用 flash-attention

### B. scMoFormer 复现步骤

**代码仓库**：https://github.com/xxx/scMoFormer （需确认实际链接）

**步骤**：

```bash
# 1. 克隆仓库
git clone https://github.com/xxx/scMoFormer.git
cd scMoFormer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据
# 将 RNA 和 ATAC 数据分别保存为 .h5ad 格式

# 4. 训练模型
python train.py \
  --rna_data data/pbmc_rna.h5ad \
  --atac_data data/pbmc_atac.h5ad \
  --output_dir results/scmoformer/
```

### C. scmFormer 复现步骤

**代码仓库**：https://github.com/xxx/scmFormer （需确认实际链接）

**步骤**：类似 scMoFormer，参考官方 README。

### D. MIDAS 复现步骤

**代码仓库**：https://github.com/ai4bio-code/MIDAS

**步骤**：

```bash
# 1. 克隆仓库
git clone https://github.com/ai4bio-code/MIDAS.git
cd MIDAS

# 2. 安装
pip install -e .

# 3. 准备数据
# MIDAS 需要特定格式的输入，参考 tutorials/

# 4. 运行整合
python scripts/train.py --config configs/pbmc_multiome.yaml
```

### E. GLUE 复现步骤

**代码仓库**：https://github.com/gao-lab/GLUE

**步骤**：

```bash
# 1. 克隆仓库
git clone https://github.com/gao-lab/GLUE.git
cd GLUE

# 2. 安装
pip install -e ".[r]"

# 3. 准备数据
# 需要构建峰-基因关联图（prior regulatory graph）

# 4. 运行整合
python examples/integration/pbmc_multiome.py
```

**关键注意事项**：
- GLUE 需要预先构建调控关系图，可使用 Signac 或 Cicero 从 ATAC 数据推断
- 若使用先验调控数据库（如 ENCODE），效果可能更好

### F. MultiVI 复现步骤

**代码仓库**：内置在 scvi-tools 中

**步骤**：

```python
import scvi
import muon as mu

# 1. 加载数据
mdata = mu.read("data/pbmc_multiome.h5mu")

# 2. 设置模型
scvi.model.MULTIVI.setup_anndata(mdata, layer="counts")

# 3. 训练模型
model = scvi.model.MULTIVI(mdata)
model.train()

# 4. 获取嵌入
latent = model.get_latent_representation()

# 5. 保存结果
import numpy as np
np.save("results/multivi_latent.npy", latent)
```

**关键注意事项**：
- MultiVI 是 scvi-tools 内置模型，复现最简单
- 需要 MuData 格式的多模态数据对象
- 训练速度较快，适合作为基准对照

---

## 十三、评估脚本框架

### 统一评估流程

```python
# evaluate.py
import scanpy as sc
import scib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

def evaluate_integration(adata, batch_key="batch", label_key="cell_type"):
    """
    评估多组学整合质量
    
    参数:
        adata: AnnData 对象，包含联合嵌入（obsm['X_emb']）
        batch_key: 批次标签的列名
        label_key: 细胞类型标签的列名
    
    返回:
        dict: 各指标值
    """
    metrics = {}
    
    # 1. 聚类与标签一致性
    sc.pp.neighbors(adata, use_rep='X_emb')
    sc.tl.leiden(adata)
    
    metrics['ARI'] = adjusted_rand_score(adata.obs[label_key], adata.obs['leiden'])
    metrics['NMI'] = normalized_mutual_info_score(adata.obs[label_key], adata.obs['leiden'])
    
    # 2. scib 指标
    scib_metrics = scib.metrics.metrics(
        adata, 
        adata_int=adata,  # 已整合数据
        batch_key=batch_key,
        label_key=label_key,
        embed='X_emb',
        ari_=True,
        nmi_=True,
        silhouette_=True,
        pcr_=True,
        graph_conn_=True,
        kBET_=True,
        ilisi_=True,
        clisi_=True
    )
    metrics.update(scib_metrics)
    
    # 3. 模态一致性 (FOSCTTM)
    metrics['FOSCTTM'] = compute_foscttm(adata)
    
    return metrics

def compute_foscttm(adata, mod1_key='X_rna', mod2_key='X_atac'):
    """计算模态间最近邻重叠率"""
    from sklearn.neighbors import NearestNeighbors
    
    # 获取各模态嵌入
    emb1 = adata.obsm[mod1_key]
    emb2 = adata.obsm[mod2_key]
    
    # 计算最近邻
    nn1 = NearestNeighbors(n_neighbors=10).fit(emb1)
    nn2 = NearestNeighbors(n_neighbors=10).fit(emb2)
    
    _, idx1 = nn1.kneighbors(emb1)
    _, idx2 = nn2.kneighbors(emb2)
    
    # 计算重叠率
    overlaps = []
    for i in range(len(adata)):
        overlap = len(set(idx1[i]) & set(idx2[i])) / 10
        overlaps.append(overlap)
    
    return 1 - np.mean(overlaps)  # FOSCTTM 越低越好

def evaluate_translation(y_true, y_pred):
    """评估跨模态翻译质量"""
    from sklearn.metrics import mean_squared_error
    
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['correlation'] = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    return metrics
```

---

## 十四、参考文献

[1] Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., Duan, N., & Wang, B. (2024). scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*, 21(8), 1470–1480.

[2] Tang, W., Wen, H., Dai, X., Wu, Z., Kozareva, V., Regev, A., ... & Li, J. (2023). Single-cell multimodal prediction via transformers. *CIKM'23*, 2425–2435.

[3] Chen, Y., Xu, Z., Wang, Y., & Zhang, X. (2024). scmFormer integrates large-scale single-cell proteomics and transcriptomics data by multi-task transformer. *Advanced Science*, 11(10), 2307835.

[4] He, Z., Hu, S., Chen, Y., An, S., Zhou, J., Liu, R., ... & Ying, X. (2024). Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS. *Nature Biotechnology*, 42(10), 1594–1605.

[5] Cao, Z. J., & Gao, G. (2022). Multi-omics single-cell data integration and regulatory inference with graph-linked unified embedding. *Nature Biotechnology*, 40(10), 1458–1466.

[6] Ashuach, T., Gabitto, M. I., Koodli, R. V., Saldi, G. A., Jordan, M. I., & Yosef, N. (2023). MultiVI: deep generative model for the integration of multimodal data. *Nature Methods*, 20(8), 1222–1231.

[7] Khan, S. A., Lehmann, R., & Theis, F. J. (2024). Transformers in single-cell omics: a review and new perspectives. *Nature Methods*, 21(8), 1379–1392.

[8] Luecken, M. D., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19(1), 41–50.

---

*本 Proposal 为项目规划文档，具体实现细节可能根据实际复现情况调整。*
