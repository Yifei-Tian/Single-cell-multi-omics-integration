"""
extract_embeddings.py — scGPT 细胞嵌入提取脚本

功能：
  1. 加载微调后的模型权重（scGPT/result/best_finetuned.pt）
  2. 对全部细胞进行前向推断，提取三组嵌入：
     - 联合嵌入（RNA + ATAC token 共同推断）→ joint_embedding.npy
     - RNA 嵌入（仅使用 RNA 模态 token）    → rna_embedding.npy
     - ATAC 嵌入（仅使用 ATAC 模态 token）  → atac_embedding.npy
  3. 将联合嵌入写入 AnnData 的 obsm["X_scgpt"]
  4. 保存带嵌入的 AnnData：scGPT/result/adata_with_emb.h5ad

嵌入提取策略：
  - scGPT TransformerModel：提取 [CLS] token 输出（d_model 维向量）
  - 回退模型：对所有 token 做 mean pooling（排除填充位置）

运行方式：
  python scGPT/extract_embeddings.py
  python scGPT/extract_embeddings.py --batch_size 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 复用 finetune_integration 中的 Dataset 和模型构建函数
sys.path.insert(0, str(Path(__file__).parent))
from finetune_integration import MultiomeDataset, build_model

# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "scGPT" / "data" / "pbmc_scgpt_input_labeled.h5ad"
CHECKPOINT_PATH = PROJECT_ROOT / "scGPT" / "result" / "best_finetuned.pt"
RESULT_DIR = PROJECT_ROOT / "scGPT" / "result"

BATCH_SIZE = 128
MAX_SEQ_LEN = 1200


# ===========================================================================
# 嵌入提取
# ===========================================================================

@torch.no_grad()
def extract_embeddings_scgpt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    embed_dim: int,
    mode: str = "joint",  # "joint" | "rna_only" | "atac_only"
) -> np.ndarray:
    """
    从 scGPT TransformerModel 中提取嵌入。

    mode 控制使用哪些 token 参与推断：
      - "joint"     ：使用全部 token（RNA + ATAC）
      - "rna_only"  ：遮蔽 ATAC token 值为 0，仅让 RNA token 参与
      - "atac_only" ：遮蔽 RNA token 值为 0，仅让 ATAC token 参与
    """
    model.eval()
    all_embeddings = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        values = batch["values"].to(device)
        modality_ids = batch["modality_ids"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        # 根据 mode 遮蔽特定模态的值
        if mode == "rna_only":
            # ATAC 位置（modality_id == 1）的值设为 0（未检测）
            atac_mask = modality_ids == 1
            values = values.clone()
            values[atac_mask] = 0
            # 同时将 ATAC 位置也加入 padding_mask，使 attention 忽略它们
            padding_mask = padding_mask | atac_mask
        elif mode == "atac_only":
            rna_mask = modality_ids == 0
            values = values.clone()
            values[rna_mask] = 0
            padding_mask = padding_mask | rna_mask

        # 前向推断
        try:
            # scgpt.TransformerModel 接口
            output = model(
                src=token_ids,
                values=values,
                src_key_padding_mask=padding_mask,
                batch_labels=None,
                CLS=True,   # 请求 CLS 嵌入
                CCE=False,
                MVC=False,
                ECS=False,
            )
            # 优先使用 CLS token 嵌入
            if "cls_output" in output and output["cls_output"] is not None:
                emb = output["cls_output"]  # (B, d_model)
            else:
                # 回退：对非填充 token 做 mean pooling
                hidden = output.get("hidden_states", output.get("transformer_output", None))
                if hidden is None:
                    raise KeyError("无法获取 hidden states，尝试回退模式")
                emb = _mean_pool(hidden, padding_mask)
        except Exception:
            # 回退到本地模型接口
            logits = model(token_ids, values, modality_ids, padding_mask)
            # 回退模型没有直接暴露 hidden states，从 token_emb + transformer 手动提取
            emb = _extract_from_fallback(model, token_ids, values, modality_ids, padding_mask, device)

        all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _mean_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    对 hidden states 做 mean pooling，排除填充位置。

    hidden:       (B, L, D)
    padding_mask: (B, L) bool，True=填充
    返回：        (B, D)
    """
    mask_float = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
    summed = (hidden * mask_float).sum(dim=1)           # (B, D)
    count = mask_float.sum(dim=1).clamp(min=1.0)        # (B, 1)
    return summed / count


@torch.no_grad()
def _extract_from_fallback(
    model: nn.Module,
    token_ids: torch.Tensor,
    values: torch.Tensor,
    modality_ids: torch.Tensor,
    padding_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    从回退模型中提取 transformer 编码后的 mean pool 嵌入。
    通过注册 forward hook 捕获 Transformer 最后一层输出。
    """
    hidden_output: list[torch.Tensor] = []

    def hook_fn(module, input, output):
        hidden_output.append(output)

    # 注册 hook 到最后一个 TransformerEncoderLayer
    transformer = model.transformer
    layers = list(transformer.layers)
    hook = layers[-1].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            n_modalities = 2
            modality_ids_safe = modality_ids.clone()
            modality_ids_safe[modality_ids_safe < 0] = n_modalities

            x = (
                model.token_emb(token_ids)
                + model.value_emb(values.clamp(0, model.mlm_head.in_features))
                + model.modality_emb(modality_ids_safe)
            )
            _ = transformer(x, src_key_padding_mask=padding_mask)
    finally:
        hook.remove()

    hidden = hidden_output[0]  # (B, L, D)
    return _mean_pool(hidden, padding_mask)


# ===========================================================================
# 主函数
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="scGPT 嵌入提取脚本")
    parser.add_argument("--data_path", type=Path, default=DATA_PATH)
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

    # 1. 加载数据和 checkpoint
    print("\n[1/5] 加载数据和模型 checkpoint...")
    adata = ad.read_h5ad(args.data_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    n_bins = checkpoint.get("n_bins", adata.uns.get("n_bins", 51))
    vocab_size = checkpoint.get("vocab_size", 60000)
    saved_args = checkpoint.get("args", {})
    embed_dim = saved_args.get("embed_dim", 512)
    n_layers = saved_args.get("n_layers", 12)
    n_heads = saved_args.get("n_heads", 8)

    print(f"  模型参数：embed_dim={embed_dim}, n_layers={n_layers}, n_heads={n_heads}")

    # 2. 构建模型并加载权重
    print("\n[2/5] 构建模型并加载微调权重...")
    model = build_model(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_bins=n_bins,
        pretrained_path=None,  # 直接加载微调后的权重
        device=device,
    )
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("  ✅ 微调权重加载完成")

    # 3. 构建 Dataset & DataLoader（不需要 shuffle）
    dataset = MultiomeDataset(adata, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. 提取三组嵌入
    n_cells = adata.n_obs
    print(f"\n[3/5] 提取嵌入（{n_cells} 个细胞）...")

    t0 = time.time()
    for mode, fname in [
        ("joint",     "joint_embedding.npy"),
        ("rna_only",  "rna_embedding.npy"),
        ("atac_only", "atac_embedding.npy"),
    ]:
        print(f"  → {mode} 模式...")
        emb = extract_embeddings_scgpt(model, loader, device, embed_dim, mode=mode)
        out_path = args.result_dir / fname
        np.save(out_path, emb)
        print(f"    ✅ 已保存：{out_path}  shape={emb.shape}")

    inference_time_per_1k = (time.time() - t0) / n_cells * 1000
    print(f"\n  推断速度：{inference_time_per_1k:.2f} 秒/1000 细胞")

    # 5. 将联合嵌入写入 AnnData
    print("\n[4/5] 将联合嵌入写入 AnnData...")
    joint_emb = np.load(args.result_dir / "joint_embedding.npy")
    adata.obsm["X_scgpt"] = joint_emb
    adata.obsm["X_scgpt_rna"] = np.load(args.result_dir / "rna_embedding.npy")
    adata.obsm["X_scgpt_atac"] = np.load(args.result_dir / "atac_embedding.npy")

    # 保存推断效率指标到 uns
    adata.uns["inference_time_per_1k_s"] = inference_time_per_1k
    if device.type == "cuda":
        adata.uns["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    adata.uns["embed_dim"] = embed_dim
    adata.uns["n_params_M"] = checkpoint.get("n_params_M", 0.0)

    out_adata_path = args.result_dir / "adata_with_emb.h5ad"
    adata.write_h5ad(out_adata_path)
    print(f"  ✅ 带嵌入的 AnnData 已保存：{out_adata_path}")

    print(f"\n🎉 嵌入提取完成！")
    print(f"   joint_embedding.npy  : {joint_emb.shape}")
    print(f"   rna_embedding.npy    : {adata.obsm['X_scgpt_rna'].shape}")
    print(f"   atac_embedding.npy   : {adata.obsm['X_scgpt_atac'].shape}")
    print(f"   推断速度              : {inference_time_per_1k:.2f} s / 1000 cells")


if __name__ == "__main__":
    main()
