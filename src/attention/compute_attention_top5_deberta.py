"""
compute_attention_top5_deberta.py
Author: Zakaria JOUILIL

Description:
    Compute per-layer attention sums and top-k flags for all DeBERTa-v3 tokens
    in all sentences. The exported file contains the top-k flags only; full
    attention-sum columns remain disabled to limit CSV size.

Inputs:
    --model pretrained|untrained
    --seed (required for untrained)
    --input_csv: DeBERTa final feature CSV
    --top_k: number of highest-attention tokens per layer (default: 5)
    --device: optional torch device
    --dataset: snli, agnews, or arxiv

Outputs:
    outputs/deberta/deberta_attention_top5_<dataset>_pretrained.csv
    outputs/deberta/deberta_attention_top5_<dataset>_untrained_seed_XXX.csv

Usage (pretrained):
    python -m src.deberta.compute_attention_top5_deberta \
        --model pretrained \
        --input_csv outputs/deberta/deberta_snli_final_features.csv \
        --dataset snli

Usage (untrained):
    python -m src.deberta.compute_attention_top5_deberta \
        --model untrained --seed 10 \
        --input_csv outputs/deberta/deberta_snli_final_features.csv \
        --dataset snli
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_NAME = "microsoft/deberta-v3-small"
EXPECTED_LAYERS = 6


def fix_seed(seed: int) -> None:
    """Set deterministic random seeds for an untrained model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_deberta_model(pretrained: bool, seed: int = None):
    """Load pretrained DeBERTa-v3-small or initialize it from its config."""
    if pretrained:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            output_attentions=True,
        )
    else:
        if seed is None:
            raise ValueError("An untrained model requires --seed.")
        fix_seed(seed)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.output_attentions = True
        model = AutoModel.from_config(config)

    model.config.output_attentions = True
    model.eval()

    actual_layers = getattr(model.config, "num_hidden_layers", None)
    if actual_layers != EXPECTED_LAYERS:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYERS} DeBERTa layers, but model config reports {actual_layers}."
        )
    return model


def compute_attention(model, tokenizer, deberta_tokens):
    """Compute attention received by each supplied token at every layer.

    Tokens are converted directly to IDs; they are not re-tokenized. This
    preserves the token sequence already stored in the DeBERTa feature CSV.
    """
    input_ids_list = tokenizer.convert_tokens_to_ids(deberta_tokens)
    if isinstance(input_ids_list, int):
        input_ids_list = [input_ids_list]
    if len(input_ids_list) != len(deberta_tokens):
        raise RuntimeError("Tokenizer returned an unexpected number of token IDs.")
    if any(token_id is None for token_id in input_ids_list):
        raise RuntimeError("Tokenizer returned None for at least one DeBERTa token.")

    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if getattr(outputs, "attentions", None) is None:
        raise RuntimeError(
            "DeBERTa did not return attentions. Check output_attentions=True "
            "and the Transformers version/model implementation."
        )

    # Expected per layer: (batch=1, heads, sequence, sequence).
    attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
    if attentions.ndim != 4:
        raise RuntimeError(
            f"Unexpected attention tensor shape: {tuple(attentions.shape)}"
        )

    num_layers, num_heads, seq_len, source_len = attentions.shape
    if seq_len != len(deberta_tokens) or source_len != len(deberta_tokens):
        raise RuntimeError(
            "Attention sequence dimensions do not match the supplied DeBERTa tokens: "
            f"attention={tuple(attentions.shape)}, tokens={len(deberta_tokens)}."
        )

    layer_sums = [
        attentions[layer].sum(dim=0).sum(dim=0)
        for layer in range(num_layers)
    ]

    return {
        "tokens": deberta_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
    }


def topk_boolean_flags(arr, k=5):
    """Return a 0/1 vector marking the k largest values."""
    if k < 1:
        raise ValueError("--top_k must be at least 1.")
    seq_len = arr.shape[0]
    kk = min(k, seq_len)
    topk_idx = np.argsort(-arr, kind="stable")[:kk]
    flags = np.zeros(seq_len, dtype=int)
    flags[topk_idx] = 1
    return flags


def main():
    parser = argparse.ArgumentParser(
        description="Compute DeBERTa-v3 attention top-k labels for all sentences."
    )
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed required for an untrained model.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["snli", "agnews", "arxiv"])
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1.")
    if args.model == "untrained" and args.seed is None:
        parser.error("--seed is required when --model untrained.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading DeBERTa-v3-small ({args.model}) ...")

    model = load_deberta_model(
        pretrained=(args.model == "pretrained"),
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model.to(device)
    model.eval()

    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    required_cols = ("sentence_id", "deberta_index", "deberta_token")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")
    if df.empty:
        raise ValueError("Input CSV is empty.")

    # Validate token order/index before starting the expensive model pass.
    df["deberta_index"] = pd.to_numeric(df["deberta_index"], errors="raise").astype(int)
    if df[list(required_cols)].isnull().any().any():
        raise ValueError("Required DeBERTa columns contain missing values.")

    out_dir = "outputs/deberta"
    os.makedirs(out_dir, exist_ok=True)
    if args.model == "pretrained":
        out_name = f"deberta_attention_top5_{args.dataset}_pretrained.csv"
    else:
        out_name = (
            f"deberta_attention_top5_{args.dataset}_untrained_seed_{args.seed:03d}.csv"
        )
    out_path = os.path.join(out_dir, out_name)

    # DeBERTa-v3-small has 6 encoder layers.
    n_layers = EXPECTED_LAYERS
    for layer in range(1, n_layers + 1):
        df[f"top5_l{layer}"] = 0

    grouped = df.groupby("sentence_id", sort=False)
    print(f"[INFO] Processing {len(grouped)} sentences ...")

    for sid, group in tqdm(grouped, total=len(grouped), desc="Sentences"):
        group = group.sort_values("deberta_index", kind="stable")
        deberta_tokens = group["deberta_token"].astype(str).tolist()
        df_indices = group.index.tolist()

        att = compute_attention(model, tokenizer, deberta_tokens)
        layer_sums = att["layer_sums"]
        seq_len = len(deberta_tokens)

        if not isinstance(layer_sums, (list, tuple)):
            raise RuntimeError(f"Invalid type for layer_sums in sentence {sid}: {type(layer_sums)}")
        if len(layer_sums) != n_layers:
            raise RuntimeError(
                f"Sentence {sid}: expected {n_layers} layers, got {len(layer_sums)}."
            )

        for layer_idx, tensor in enumerate(layer_sums):
            if tensor is None:
                raise RuntimeError(f"Sentence {sid}, layer {layer_idx + 1}: attention sum is None.")
            if tensor.shape[0] != seq_len:
                raise RuntimeError(
                    f"Sentence {sid}, layer {layer_idx + 1}: expected vector length "
                    f"{seq_len}, got {tensor.shape[0]}."
                )
            layer_sum_np = tensor.detach().float().cpu().numpy()
            flags = topk_boolean_flags(layer_sum_np, k=args.top_k)
            df.loc[df_indices, f"top5_l{layer_idx + 1}"] = flags

    for layer in range(1, n_layers + 1):
        df[f"top5_l{layer}"] = df[f"top5_l{layer}"].astype(int)

    print(f"[INFO] Saving final file: {out_path}")
    df.to_csv(out_path, sep=";", index=False)
    print("[OK] DONE — DeBERTa attention top-k computation complete.")


if __name__ == "__main__":
    main()
