"""
finetune_integration.py — scGPT 多组学整合微调脚本（MLM + SFT 两阶段）

训练策略：
  ┌─────────────────────────────────────────────────────────────────┐
  │  阶段 1 — MLM 预热（Masked Language Modeling）                  │
  │    目标：让模型适配多组学 token 格式，学习 RNA+ATAC 联合表示     │
  │    方法：随机遮蔽 mask_ratio 比例的 token 值，交叉熵预测原始分箱值│
  │    epoch：1 ~ n_epochs_mlm                                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  阶段 2 — SFT 收敛（Supervised Fine-Tuning，细胞类型分类）       │
  │    目标：以细胞类型标签为监督信号，优化下游分类任务               │
  │    方法：冻结底层 Transformer，仅训练顶层 2 层 + CLS 分类头      │
  │    损失：α × MLM_loss + (1-α) × SFT_loss（联合损失）            │
  │    epoch：n_epochs_mlm+1 ~ n_epochs_mlm+n_epochs_sft            │
  └─────────────────────────────────────────────────────────────────┘

超参数（参照 Proposal §方法架构详解 scGPT）：
  embed_dim    = 512
  n_layers     = 12
  n_heads      = 8
  mask_ratio   = 0.2   （微调阶段，低于预训练的 0.4）
  batch_size   = 64
  lr           = 1e-4  （MLM 阶段）；5e-5（SFT 阶段）
  n_epochs_mlm = 6
  n_epochs_sft = 4
  sft_alpha    = 0.3   （联合损失中 MLM 权重；SFT 权重 = 1 - sft_alpha）

运行方式：
  python scGPT/finetune_integration.py
  python scGPT/finetune_integration.py --batch_size 32 --n_epochs_mlm 8 --n_epochs_sft 4
  python scGPT/finetune_integration.py --skip_mlm          # 仅运行 SFT 阶段
  python scGPT/finetune_integration.py --skip_sft          # 仅运行 MLM 阶段
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
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "scGPT" / "data" / "pbmc_scgpt_input_labeled.h5ad"
PRETRAINED_DIR = PROJECT_ROOT / "scGPT" / "pretrained_models" / "scGPT_human"
RESULT_DIR = PROJECT_ROOT / "scGPT" / "result"

# ---------------------------------------------------------------------------
# 超参数默认值
# ---------------------------------------------------------------------------
EMBED_DIM = 512
N_LAYERS = 12
N_HEADS = 8
MASK_RATIO = 0.2
BATCH_SIZE = 64
LR_MLM = 1e-4        # MLM 阶段学习率
LR_SFT = 5e-5        # SFT 阶段学习率（较小，避免灾难性遗忘）
N_EPOCHS_MLM = 6     # MLM 预热轮数
N_EPOCHS_SFT = 4     # SFT 收敛轮数
SFT_ALPHA = 0.3      # 联合损失：α*MLM + (1-α)*SFT
N_FROZEN_LAYERS = 10 # SFT 阶段冻结的底层 Transformer 层数（共 12 层，解冻最后 2 层）
MAX_SEQ_LEN = 1200
TRAIN_VAL_RATIO = 0.9
RANDOM_STATE = 42


# ===========================================================================
# Dataset（含细胞类型标签）
# ===========================================================================

class MultiomeDataset(Dataset):
    """
    将 AnnData 的 token_ids 和 values 层转换为 PyTorch Dataset。

    每个样本返回：
      - token_ids:     (seq_len,)  特征 token id
      - values:        (seq_len,)  分箱后的表达/信号值
      - modality_ids:  (seq_len,)  模态标识（0=RNA, 1=ATAC）
      - padding_mask:  (seq_len,)  True 表示填充位置
      - cell_type_id:  int         细胞类型标签（SFT 用）；-1 表示无标签
    """

    def __init__(
        self,
        adata: ad.AnnData,
        max_seq_len: int = MAX_SEQ_LEN,
        cell_type_encoder: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_rna = adata.uns["n_rna_features"]
        self.n_atac = adata.uns["n_atac_features"]

        # 加载 token_ids 层
        if sp.issparse(adata.layers["token_ids"]):
            self.token_ids = adata.layers["token_ids"].toarray().astype(np.int64)
        else:
            self.token_ids = np.asarray(adata.layers["token_ids"], dtype=np.int64)

        # 加载 values 层（分箱值）
        if sp.issparse(adata.layers["values"]):
            self.values = adata.layers["values"].toarray().astype(np.int64)
        else:
            self.values = np.asarray(adata.layers["values"], dtype=np.int64)

        # modality_id：0=RNA，1=ATAC
        self.modality_ids = adata.var["modality_id"].values.astype(np.int64)

        # 细胞类型标签（SFT 监督信号）
        if cell_type_encoder is not None and "cell_type" in adata.obs.columns:
            self.cell_type_ids = np.array(
                [cell_type_encoder.get(str(ct), -1) for ct in adata.obs["cell_type"]],
                dtype=np.int64,
            )
        else:
            self.cell_type_ids = np.full(adata.n_obs, -1, dtype=np.int64)

        self.n_cells = adata.n_obs
        self.n_features = adata.n_vars

        # 若特征数超过 max_seq_len，子采样（保留全部 RNA，ATAC 随机采样）
        self._feature_indices: np.ndarray | None = None
        if self.n_features > max_seq_len:
            rna_idx = np.where(self.modality_ids == 0)[0]
            atac_idx = np.where(self.modality_ids == 1)[0]
            n_atac_keep = max(0, max_seq_len - len(rna_idx))
            atac_sampled = np.random.choice(atac_idx, size=n_atac_keep, replace=False)
            self._feature_indices = np.concatenate([rna_idx, atac_sampled])
            self._feature_indices.sort()
            print(
                f"  ⚠️  特征数 {self.n_features} > max_seq_len {max_seq_len}，"
                f"子采样为 {len(self._feature_indices)} 个特征"
            )

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feat_idx = self._feature_indices if self._feature_indices is not None else np.arange(self.n_features)

        token_ids = self.token_ids[idx, feat_idx]
        values = self.values[idx, feat_idx]
        modality_ids = self.modality_ids[feat_idx]
        seq_len = len(feat_idx)

        # 填充或截断到 max_seq_len
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            token_ids = np.pad(token_ids, (0, pad_len), constant_values=0)
            values = np.pad(values, (0, pad_len), constant_values=0)
            modality_ids = np.pad(modality_ids, (0, pad_len), constant_values=-1)
        else:
            token_ids = token_ids[: self.max_seq_len]
            values = values[: self.max_seq_len]
            modality_ids = modality_ids[: self.max_seq_len]

        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[seq_len:] = True

        return {
            "token_ids":    torch.from_numpy(token_ids).long(),
            "values":       torch.from_numpy(values).long(),
            "modality_ids": torch.from_numpy(modality_ids.astype(np.int64)).long(),
            "padding_mask": padding_mask,
            "cell_type_id": torch.tensor(self.cell_type_ids[idx], dtype=torch.long),
        }


# ===========================================================================
# 分类头（SFT 阶段专用）
# ===========================================================================

class CellTypeClassifier(nn.Module):
    """
    接在 Transformer 输出的 [CLS] token 上的分类头。
    结构：LayerNorm → Linear → GELU → Dropout → Linear
    """

    def __init__(self, embed_dim: int, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """cls_emb: (B, embed_dim) → logits: (B, n_classes)"""
        return self.net(cls_emb)


# ===========================================================================
# 模型构建
# ===========================================================================

def build_model(
    vocab_size: int,
    embed_dim: int = EMBED_DIM,
    n_layers: int = N_LAYERS,
    n_heads: int = N_HEADS,
    n_bins: int = 51,
    n_modalities: int = 2,
    pretrained_path: Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    构建 scGPT Transformer 模型并加载预训练权重。
    优先使用 scgpt 库，不可用时回退到本地实现。
    """
    try:
        from scgpt.model import TransformerModel
        model = TransformerModel(
            ntoken=vocab_size,
            d_model=embed_dim,
            nhead=n_heads,
            d_hid=embed_dim * 4,
            nlayers=n_layers,
            vocab=None,
            dropout=0.1,
            pad_token="<pad>",
            pad_value=0,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=0,
            domain_spec_batchnorm=False,
            n_input_bins=n_bins,
            ecs_threshold=0.3,
            explicit_zero_prob=False,
            use_fast_transformer=False,
            pre_norm=False,
        )
        print("  ✅ 使用 scgpt.model.TransformerModel")
    except Exception as e:
        print(f"  ⚠️  scgpt.model 导入失败（{e}），使用本地替代模型")
        model = _build_fallback_transformer(vocab_size, embed_dim, n_layers, n_heads, n_bins, n_modalities)

    # 加载预训练权重
    if pretrained_path is not None and pretrained_path.exists():
        state_dict = torch.load(pretrained_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ 预训练权重加载：{pretrained_path.name}")
        if missing:
            print(f"     缺失键（共 {len(missing)} 个，通常为新增模态嵌入层，属正常现象）")
        if unexpected:
            print(f"     多余键：{len(unexpected)} 个")
    else:
        print(f"  ⚠️  预训练权重文件未找到：{pretrained_path}，从随机初始化开始训练")

    return model.to(device)


def _build_fallback_transformer(
    vocab_size: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    n_bins: int,
    n_modalities: int,
) -> nn.Module:
    """当 scgpt 库不可用时的轻量 Transformer 替代实现。"""

    class _FallbackTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            self.n_bins = n_bins
            self.token_emb = nn.Embedding(vocab_size + 10, embed_dim, padding_idx=0)
            self.value_emb = nn.Embedding(n_bins + 1, embed_dim, padding_idx=0)
            self.modality_emb = nn.Embedding(n_modalities + 1, embed_dim, padding_idx=-1)
            # CLS token 参数（可学习向量，追加在序列最前面）
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,   # Pre-LN，训练更稳定
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.mlm_head = nn.Linear(embed_dim, n_bins)

        def forward(
            self,
            token_ids: torch.Tensor,
            values: torch.Tensor,
            modality_ids: torch.Tensor,
            padding_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            返回：
              mlm_logits:  (B, L, n_bins)  ← 用于 MLM loss（不含 CLS 位置）
              cls_output:  (B, embed_dim)  ← 用于 SFT loss
            """
            B, L = token_ids.shape
            modality_ids_safe = modality_ids.clone()
            modality_ids_safe[modality_ids_safe < 0] = n_modalities

            x = (
                self.token_emb(token_ids)
                + self.value_emb(values.clamp(0, n_bins))
                + self.modality_emb(modality_ids_safe)
            )  # (B, L, D)

            # 在序列最前插入 CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)                   # (B, L+1, D)
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            full_mask = torch.cat([cls_pad, padding_mask], dim=1)    # (B, L+1)

            hidden = self.transformer(x, src_key_padding_mask=full_mask)  # (B, L+1, D)

            cls_output = hidden[:, 0, :]         # (B, D)   — CLS token 输出
            seq_hidden = hidden[:, 1:, :]        # (B, L, D) — 序列 token 输出
            mlm_logits = self.mlm_head(seq_hidden)  # (B, L, n_bins)

            return mlm_logits, cls_output

    return _FallbackTransformer()


def freeze_lower_layers(model: nn.Module, n_frozen: int) -> None:
    """
    冻结 Transformer 底部 n_frozen 层的参数（SFT 阶段调用）。
    仅解冻顶部 (n_layers - n_frozen) 层 + 分类头，减小 SFT 对预训练表示的破坏。
    """
    frozen_count = 0
    # 回退模型结构
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        for i, layer in enumerate(model.transformer.layers):
            if i < n_frozen:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_count += 1
    # scgpt.TransformerModel 结构（transformer.encoder.layers）
    elif hasattr(model, "transformer") and hasattr(model.transformer, "encoder"):
        for i, layer in enumerate(model.transformer.encoder.layers):
            if i < n_frozen:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen_count += 1

    # 同时冻结嵌入层（词表嵌入不随 SFT 更新）
    for name, param in model.named_parameters():
        if any(key in name for key in ["token_emb", "value_emb", "modality_emb", "embedding"]):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔒 已冻结 {frozen_count} 层，可训练参数：{trainable / 1e6:.1f} M / {total / 1e6:.1f} M")


def unfreeze_all(model: nn.Module) -> None:
    """解冻所有参数（阶段切换时使用）。"""
    for param in model.parameters():
        param.requires_grad = True


# ===========================================================================
# 损失函数
# ===========================================================================

def apply_masking(
    values: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_ratio: float = MASK_RATIO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机遮蔽 token 值，返回 masked_values 和遮蔽位置 mask。"""
    rand_mat = torch.rand_like(values, dtype=torch.float)
    mask_positions = (rand_mat < mask_ratio) & (~padding_mask)
    masked_values = values.clone()
    masked_values[mask_positions] = 0
    return masked_values, mask_positions


def compute_mlm_loss(
    mlm_logits: torch.Tensor,
    target_values: torch.Tensor,
    mask_positions: torch.Tensor,
) -> torch.Tensor:
    """仅在被遮蔽位置计算交叉熵损失（MLM 目标）。"""
    if not mask_positions.any():
        return torch.tensor(0.0, device=mlm_logits.device, requires_grad=True)
    logits_masked = mlm_logits[mask_positions]
    targets_masked = target_values[mask_positions]
    return nn.functional.cross_entropy(logits_masked, targets_masked)


def compute_sft_loss(
    cls_logits: torch.Tensor,
    cell_type_ids: torch.Tensor,
) -> torch.Tensor:
    """
    细胞类型分类交叉熵损失（SFT 目标）。
    忽略标签为 -1 的样本（无标注细胞）。
    """
    valid = cell_type_ids >= 0
    if not valid.any():
        return torch.tensor(0.0, device=cls_logits.device, requires_grad=True)
    return nn.functional.cross_entropy(cls_logits[valid], cell_type_ids[valid])


def compute_sft_accuracy(cls_logits: torch.Tensor, cell_type_ids: torch.Tensor) -> float:
    """计算 SFT 分类准确率（仅统计有效标签）。"""
    valid = cell_type_ids >= 0
    if not valid.any():
        return 0.0
    preds = cls_logits[valid].argmax(dim=-1)
    return float((preds == cell_type_ids[valid]).float().mean().item())


# ===========================================================================
# 前向传播辅助（兼容 scgpt 官方接口和本地回退模型）
# ===========================================================================

def forward_pass(
    model: nn.Module,
    classifier: CellTypeClassifier | None,
    token_ids: torch.Tensor,
    masked_values: torch.Tensor,
    modality_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    need_cls: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    统一的前向传播接口，返回 (mlm_logits, cls_output)。

    mlm_logits: (B, L, n_bins) — 用于 MLM loss
    cls_output: (B, n_classes) — 用于 SFT loss（仅 need_cls=True 时不为 None）
    """
    try:
        # scgpt.TransformerModel 接口
        output = model(
            src=token_ids,
            values=masked_values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
            CLS=need_cls,
            CCE=False,
            MVC=True,
            ECS=False,
        )
        mlm_logits = output["mvc_output"]    # (B, L, n_bins)
        raw_cls = output.get("cls_output")   # (B, embed_dim) or None
        cls_output = classifier(raw_cls) if (need_cls and raw_cls is not None and classifier is not None) else None

    except Exception:
        # 本地回退模型接口
        mlm_logits, raw_cls = model(token_ids, masked_values, modality_ids, padding_mask)
        cls_output = classifier(raw_cls) if (need_cls and raw_cls is not None and classifier is not None) else None

    return mlm_logits, cls_output


# ===========================================================================
# 训练循环（MLM 阶段）
# ===========================================================================

def train_epoch_mlm(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """MLM 阶段单轮训练，返回 {'loss': ...}。"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        token_ids    = batch["token_ids"].to(device)
        values       = batch["values"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        masked_values, mask_positions = apply_masking(values, padding_mask, mask_ratio)

        optimizer.zero_grad()
        mlm_logits, _ = forward_pass(model, None, token_ids, masked_values,
                                     modality_ids, padding_mask, need_cls=False)
        loss = compute_mlm_loss(mlm_logits, values, mask_positions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def eval_epoch_mlm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
) -> dict[str, float]:
    """MLM 阶段验证，返回 {'loss': ...}。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        token_ids    = batch["token_ids"].to(device)
        values       = batch["values"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        masked_values, mask_positions = apply_masking(values, padding_mask, mask_ratio)
        mlm_logits, _ = forward_pass(model, None, token_ids, masked_values,
                                     modality_ids, padding_mask, need_cls=False)
        loss = compute_mlm_loss(mlm_logits, values, mask_positions)
        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


# ===========================================================================
# 训练循环（SFT 阶段，联合损失）
# ===========================================================================

def train_epoch_sft(
    model: nn.Module,
    classifier: CellTypeClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mask_ratio: float,
    alpha: float,
) -> dict[str, float]:
    """
    SFT 阶段单轮训练。
    联合损失 = alpha * MLM_loss + (1 - alpha) * SFT_loss
    返回 {'loss', 'mlm_loss', 'sft_loss', 'sft_acc'}。
    """
    model.train()
    classifier.train()
    total_loss = total_mlm = total_sft = total_acc = 0.0
    n_batches = 0

    for batch in loader:
        token_ids     = batch["token_ids"].to(device)
        values        = batch["values"].to(device)
        modality_ids  = batch["modality_ids"].to(device)
        padding_mask  = batch["padding_mask"].to(device)
        cell_type_ids = batch["cell_type_id"].to(device)

        masked_values, mask_positions = apply_masking(values, padding_mask, mask_ratio)

        optimizer.zero_grad()
        mlm_logits, cls_logits = forward_pass(
            model, classifier, token_ids, masked_values,
            modality_ids, padding_mask, need_cls=True,
        )

        mlm_loss = compute_mlm_loss(mlm_logits, values, mask_positions)
        sft_loss = compute_sft_loss(cls_logits, cell_type_ids) if cls_logits is not None \
                   else torch.tensor(0.0, device=device)

        loss = alpha * mlm_loss + (1.0 - alpha) * sft_loss
        loss.backward()
        # 对模型主体和分类头分别裁剪
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(classifier.parameters()), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        acc = compute_sft_accuracy(cls_logits, cell_type_ids) if cls_logits is not None else 0.0
        total_loss += loss.item()
        total_mlm  += mlm_loss.item()
        total_sft  += sft_loss.item()
        total_acc  += acc
        n_batches  += 1

    nb = max(n_batches, 1)
    return {
        "loss":     total_loss / nb,
        "mlm_loss": total_mlm  / nb,
        "sft_loss": total_sft  / nb,
        "sft_acc":  total_acc  / nb,
    }


@torch.no_grad()
def eval_epoch_sft(
    model: nn.Module,
    classifier: CellTypeClassifier,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    alpha: float,
) -> dict[str, float]:
    """SFT 阶段验证，返回联合损失和分类准确率。"""
    model.eval()
    classifier.eval()
    total_loss = total_mlm = total_sft = total_acc = 0.0
    n_batches = 0

    for batch in loader:
        token_ids     = batch["token_ids"].to(device)
        values        = batch["values"].to(device)
        modality_ids  = batch["modality_ids"].to(device)
        padding_mask  = batch["padding_mask"].to(device)
        cell_type_ids = batch["cell_type_id"].to(device)

        masked_values, mask_positions = apply_masking(values, padding_mask, mask_ratio)
        mlm_logits, cls_logits = forward_pass(
            model, classifier, token_ids, masked_values,
            modality_ids, padding_mask, need_cls=True,
        )

        mlm_loss = compute_mlm_loss(mlm_logits, values, mask_positions)
        sft_loss = compute_sft_loss(cls_logits, cell_type_ids) if cls_logits is not None \
                   else torch.tensor(0.0, device=device)
        loss = alpha * mlm_loss + (1.0 - alpha) * sft_loss
        acc  = compute_sft_accuracy(cls_logits, cell_type_ids) if cls_logits is not None else 0.0

        total_loss += loss.item()
        total_mlm  += mlm_loss.item()
        total_sft  += sft_loss.item()
        total_acc  += acc
        n_batches  += 1

    nb = max(n_batches, 1)
    return {
        "loss":     total_loss / nb,
        "mlm_loss": total_mlm  / nb,
        "sft_loss": total_sft  / nb,
        "sft_acc":  total_acc  / nb,
    }


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 多组学整合微调脚本（MLM + SFT 两阶段）")
    parser.add_argument("--data_path",      type=Path,  default=DATA_PATH)
    parser.add_argument("--pretrained_dir", type=Path,  default=PRETRAINED_DIR)
    parser.add_argument("--result_dir",     type=Path,  default=RESULT_DIR)
    parser.add_argument("--embed_dim",      type=int,   default=EMBED_DIM)
    parser.add_argument("--n_layers",       type=int,   default=N_LAYERS)
    parser.add_argument("--n_heads",        type=int,   default=N_HEADS)
    parser.add_argument("--mask_ratio",     type=float, default=MASK_RATIO)
    parser.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr_mlm",         type=float, default=LR_MLM,
                        help="MLM 阶段学习率（默认 1e-4）")
    parser.add_argument("--lr_sft",         type=float, default=LR_SFT,
                        help="SFT 阶段学习率（默认 5e-5）")
    parser.add_argument("--n_epochs_mlm",   type=int,   default=N_EPOCHS_MLM,
                        help="MLM 预热轮数（默认 6）")
    parser.add_argument("--n_epochs_sft",   type=int,   default=N_EPOCHS_SFT,
                        help="SFT 收敛轮数（默认 4）")
    parser.add_argument("--sft_alpha",      type=float, default=SFT_ALPHA,
                        help="SFT 阶段联合损失中 MLM 的权重 α（默认 0.3）")
    parser.add_argument("--n_frozen_layers",type=int,   default=N_FROZEN_LAYERS,
                        help="SFT 阶段冻结的底层数量（默认 10，共 12 层）")
    parser.add_argument("--max_seq_len",    type=int,   default=MAX_SEQ_LEN)
    parser.add_argument("--random_state",   type=int,   default=RANDOM_STATE)
    parser.add_argument("--skip_mlm",       action="store_true",
                        help="跳过 MLM 阶段，直接进行 SFT")
    parser.add_argument("--skip_sft",       action="store_true",
                        help="跳过 SFT 阶段，仅进行 MLM")
    parser.add_argument("--mlm_checkpoint", type=Path,  default=None,
                        help="已有 MLM 阶段 checkpoint 路径（配合 --skip_mlm 使用）")
    args = parser.parse_args()

    # 检查前置文件
    if not args.data_path.exists():
        print(f"❌ 输入数据不存在：{args.data_path}")
        print("   请先运行：python scGPT/annotate_cells.py")
        sys.exit(1)

    pretrained_model_path = args.pretrained_dir / "best_model.pt"
    args.result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 使用设备：{device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # -----------------------------------------------------------------------
    # 1. 加载数据，构建细胞类型编码器
    # -----------------------------------------------------------------------
    print("\n[1/6] 加载数据...")
    adata = ad.read_h5ad(args.data_path)
    n_bins = adata.uns.get("n_bins", 51)
    rna_token_ids = adata.uns["rna_gene_token_ids"]
    peak_token_ids = adata.uns["peak_token_ids"]
    vocab_size = max(max(rna_token_ids), max(peak_token_ids)) + 100

    # 构建细胞类型 → 整数 id 映射（SFT 监督信号）
    if "cell_type" in adata.obs.columns:
        unique_types = sorted(adata.obs["cell_type"].dropna().unique().tolist())
        cell_type_encoder = {ct: i for i, ct in enumerate(unique_types)}
        n_cell_types = len(unique_types)
        print(f"  细胞类型：{n_cell_types} 种 → {unique_types}")
    else:
        print("  ⚠️  obs['cell_type'] 不存在，SFT 阶段将跳过分类损失")
        cell_type_encoder = None
        n_cell_types = 1

    print(f"  词表大小（估算）：{vocab_size}")

    # -----------------------------------------------------------------------
    # 2. 构建 Dataset & DataLoader
    # -----------------------------------------------------------------------
    print("\n[2/6] 构建 Dataset & DataLoader...")
    dataset = MultiomeDataset(adata, max_seq_len=args.max_seq_len,
                              cell_type_encoder=cell_type_encoder)
    n_train = int(len(dataset) * TRAIN_VAL_RATIO)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.random_state),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"  训练集：{n_train} 细胞，验证集：{n_val} 细胞")

    # -----------------------------------------------------------------------
    # 3. 构建模型 + 分类头
    # -----------------------------------------------------------------------
    print("\n[3/6] 构建 scGPT 模型...")
    model = build_model(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_bins=n_bins,
        pretrained_path=pretrained_model_path if pretrained_model_path.exists() else None,
        device=device,
    )
    classifier = CellTypeClassifier(args.embed_dim, n_cell_types).to(device)
    n_params_backbone = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_cls      = sum(p.numel() for p in classifier.parameters())
    print(f"  Transformer 参数量：{n_params_backbone / 1e6:.1f} M")
    print(f"  分类头参数量：{n_params_cls / 1e3:.1f} K")

    log_rows: list[dict] = []
    training_start = time.time()
    best_val_loss = float("inf")

    # -----------------------------------------------------------------------
    # 阶段 1：MLM 预热
    # -----------------------------------------------------------------------
    if not args.skip_mlm:
        print(f"\n{'='*60}")
        print(f"  阶段 1 — MLM 预热（{args.n_epochs_mlm} epochs，mask_ratio={args.mask_ratio}）")
        print(f"{'='*60}")

        # MLM 阶段：全参数训练
        unfreeze_all(model)
        optimizer_mlm = torch.optim.AdamW(model.parameters(), lr=args.lr_mlm, weight_decay=0.01)
        scheduler_mlm = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_mlm, T_max=len(train_loader) * args.n_epochs_mlm
        )

        for epoch in range(1, args.n_epochs_mlm + 1):
            t0 = time.time()
            train_metrics = train_epoch_mlm(
                model, train_loader, optimizer_mlm, scheduler_mlm, device, args.mask_ratio
            )
            val_metrics = eval_epoch_mlm(model, val_loader, device, args.mask_ratio)
            epoch_time = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

            print(
                f"  [MLM] Epoch {epoch:>2d}/{args.n_epochs_mlm}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"time={epoch_time:.1f}s  mem={peak_mem:.2f}GB"
            )
            row = {
                "phase": "MLM", "epoch": epoch,
                "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
                "train_mlm_loss": train_metrics["loss"], "val_mlm_loss": val_metrics["loss"],
                "train_sft_loss": "", "val_sft_loss": "",
                "train_sft_acc": "", "val_sft_acc": "",
                "epoch_time_s": epoch_time, "peak_mem_gb": peak_mem,
            }
            log_rows.append(row)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {"epoch": epoch, "phase": "MLM",
                     "model_state_dict": model.state_dict(),
                     "classifier_state_dict": classifier.state_dict(),
                     "val_loss": val_metrics["loss"],
                     "args": vars(args), "n_bins": n_bins,
                     "vocab_size": vocab_size, "n_cell_types": n_cell_types,
                     "cell_type_encoder": cell_type_encoder},
                    args.result_dir / "best_finetuned.pt",
                )
                print(f"    ↑ [MLM] 新最优 checkpoint（val_loss={val_metrics['loss']:.4f}）")

        # 保存 MLM 阶段结束 checkpoint（供 SFT 阶段或断点续训使用）
        torch.save(
            {"model_state_dict": model.state_dict(),
             "classifier_state_dict": classifier.state_dict(),
             "phase_end": "MLM", "args": vars(args),
             "n_bins": n_bins, "vocab_size": vocab_size,
             "n_cell_types": n_cell_types, "cell_type_encoder": cell_type_encoder},
            args.result_dir / "mlm_stage_end.pt",
        )
        print(f"\n  ✅ MLM 阶段结束，checkpoint 已保存：{args.result_dir / 'mlm_stage_end.pt'}")

    else:
        # 若跳过 MLM，从指定 checkpoint 加载
        ckpt_path = args.mlm_checkpoint or (args.result_dir / "mlm_stage_end.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            n_cell_types = ckpt.get("n_cell_types", n_cell_types)
            cell_type_encoder = ckpt.get("cell_type_encoder", cell_type_encoder)
            print(f"\n  ✅ 跳过 MLM，加载 checkpoint：{ckpt_path}")
        else:
            print(f"\n  ⚠️  跳过 MLM 且未找到 checkpoint：{ckpt_path}，使用当前权重继续 SFT")

    # -----------------------------------------------------------------------
    # 阶段 2：SFT 收敛
    # -----------------------------------------------------------------------
    if not args.skip_sft:
        print(f"\n{'='*60}")
        print(f"  阶段 2 — SFT 收敛（{args.n_epochs_sft} epochs，"
              f"α={args.sft_alpha}×MLM + {1-args.sft_alpha:.1f}×SFT）")
        print(f"{'='*60}")

        # 冻结底层，解冻顶层
        freeze_lower_layers(model, args.n_frozen_layers)

        # SFT 优化器只更新可训练的 Transformer 层 + 分类头
        sft_params = [p for p in model.parameters() if p.requires_grad] + \
                     list(classifier.parameters())
        optimizer_sft = torch.optim.AdamW(sft_params, lr=args.lr_sft, weight_decay=0.01)
        scheduler_sft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sft, T_max=len(train_loader) * args.n_epochs_sft
        )

        best_sft_val_loss = float("inf")

        for epoch in range(1, args.n_epochs_sft + 1):
            t0 = time.time()
            train_metrics = train_epoch_sft(
                model, classifier, train_loader, optimizer_sft, scheduler_sft,
                device, args.mask_ratio, args.sft_alpha,
            )
            val_metrics = eval_epoch_sft(
                model, classifier, val_loader, device, args.mask_ratio, args.sft_alpha,
            )
            epoch_time = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

            print(
                f"  [SFT] Epoch {epoch:>2d}/{args.n_epochs_sft}  "
                f"loss={val_metrics['loss']:.4f}  "
                f"mlm={val_metrics['mlm_loss']:.4f}  "
                f"sft={val_metrics['sft_loss']:.4f}  "
                f"acc={val_metrics['sft_acc']:.3f}  "
                f"time={epoch_time:.1f}s  mem={peak_mem:.2f}GB"
            )
            row = {
                "phase": "SFT", "epoch": epoch,
                "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
                "train_mlm_loss": train_metrics["mlm_loss"], "val_mlm_loss": val_metrics["mlm_loss"],
                "train_sft_loss": train_metrics["sft_loss"], "val_sft_loss": val_metrics["sft_loss"],
                "train_sft_acc":  train_metrics["sft_acc"],  "val_sft_acc":  val_metrics["sft_acc"],
                "epoch_time_s": epoch_time, "peak_mem_gb": peak_mem,
            }
            log_rows.append(row)

            if val_metrics["loss"] < best_sft_val_loss:
                best_sft_val_loss = val_metrics["loss"]
                torch.save(
                    {"epoch": epoch, "phase": "SFT",
                     "model_state_dict": model.state_dict(),
                     "classifier_state_dict": classifier.state_dict(),
                     "val_loss": val_metrics["loss"],
                     "val_sft_acc": val_metrics["sft_acc"],
                     "args": vars(args), "n_bins": n_bins,
                     "vocab_size": vocab_size, "n_cell_types": n_cell_types,
                     "cell_type_encoder": cell_type_encoder},
                    args.result_dir / "best_finetuned.pt",
                )
                print(
                    f"    ↑ [SFT] 新最优 checkpoint"
                    f"（val_loss={val_metrics['loss']:.4f}，val_acc={val_metrics['sft_acc']:.3f}）"
                )

    # -----------------------------------------------------------------------
    # 5. 保存训练日志
    # -----------------------------------------------------------------------
    total_time = time.time() - training_start
    print(f"\n[6/6] 保存训练日志...")
    log_path = args.result_dir / "training_log.csv"
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        with open(log_path, "a") as f:
            f.write(f"\n# 总训练时间：{total_time:.1f} s\n")
            if device.type == "cuda":
                f.write(f"# GPU 显存峰值：{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB\n")

    print(f"  ✅ 训练日志已保存：{log_path}")
    print(f"\n🎉 微调完成！")
    print(f"   训练策略：{'MLM' if not args.skip_mlm else ''}{'→' if not args.skip_mlm and not args.skip_sft else ''}{'SFT' if not args.skip_sft else ''}")
    print(f"   总训练时间：{total_time / 3600:.2f} 小时")
    print(f"   最优模型权重：{args.result_dir / 'best_finetuned.pt'}")


if __name__ == "__main__":
    main()
