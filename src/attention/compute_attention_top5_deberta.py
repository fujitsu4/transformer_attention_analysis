"""
compute_attention_top5_deberta.py
Author: Zakaria JOUILIL

DeBERTa-v3-small adaptation of compute_attention_top5.py.
The attention calculations and top-k selection are preserved.
Stored tokens are converted directly to IDs; no retokenization is performed.
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
N_LAYERS = 6


# ---------------------------------------------------------
# Fix random seed (for untrained models)
# ---------------------------------------------------------
def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Load model: pretrained or random
# ---------------------------------------------------------
def load_model(pretrained: bool, seed: int = None):
    if pretrained:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            output_attentions=True
        )
    else:
        if seed is None:
            raise ValueError("Seed must be provided for untrained model.")
        fix_seed(seed)
        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            output_attentions=True
        )
        model = AutoModel.from_config(config)

    model.config.output_attentions = True
    model.eval()
    return model


# ---------------------------------------------------------
# Extract attention matrices and compute summaries
# ---------------------------------------------------------
def compute_attention(model, tokenizer, deberta_tokens):
    """Compute attention summaries from stored DeBERTa tokens."""
    # Convert stored tokens directly without retokenization.
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(deberta_tokens)],
        dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        if getattr(outputs, "attentions", None) is None:
            raise RuntimeError(
                "Model did not return attentions. "
                "Ensure model.config.output_attentions=True."
            )

        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)

    num_layers, num_heads, seq_len, _ = attentions.shape

    # Same layer-sum calculation as the original BERT core.
    layer_sums = []
    for L in range(num_layers):
        layer_sum = attentions[L].sum(dim=0).sum(dim=0)
        layer_sums.append(layer_sum)

    # Same head-wise sum calculation as the original BERT core.
    head_sums = []
    for L in range(num_layers):
        per_layer = []
        for H in range(num_heads):
            head_sum = attentions[L][H].sum(dim=0)
            per_layer.append(head_sum)
        head_sums.append(per_layer)

    return {
        "tokens": deberta_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
        "head_sums": head_sums
    }


# ---------------------------------------------------------
# Utility: select top-k for 1D numpy array
# ---------------------------------------------------------
def topk_boolean_flags(arr, k=5):
    seq_len = arr.shape[0]
    kk = min(k, seq_len)
    idx_desc = np.argsort(-arr)
    topk_idx = idx_desc[:kk]
    flags = np.zeros(seq_len, dtype=int)
    flags[topk_idx] = 1
    return flags


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="Seed required for untrained")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["snli", "agnews", "arxiv"],
        help="Dataset being processed"
    )
    args = parser.parse_args()

    # Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model selection
    if args.model == "pretrained":
        pretrained_flag = True
        print("[INFO] Loading PRETRAINED DeBERTa-v3-small model ...")
    else:
        if args.seed is None:
            raise ValueError("Untrained model requires --seed")
        pretrained_flag = False
        print(
            f"[INFO] Loading RANDOMLY INITIALIZED DeBERTa-v3-small "
            f"model with seed {args.seed} ..."
        )
        fix_seed(args.seed)

    model = load_model(pretrained_flag, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model.to(device)
    model.eval()

    # Load input CSV
    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    for col in ("sentence_id", "deberta_index", "deberta_token"):
        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # Output directory and filename pattern are unchanged from the original.
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:
        out_name = f"attention_top5_{args.dataset}_pretrained.csv"
    else:
        out_name = f"attention_top5_{args.dataset}_untrained_seed_{args.seed:03d}.csv"

    out_path = os.path.join(out_dir, out_name)

    # DeBERTa-v3-small has 6 layers.
    n_layers = N_LAYERS
    for L in range(1, n_layers + 1):
        df[f"top5_l{L}"] = 0

    # Process all sentences.
    grouped = df.groupby("sentence_id", sort=False)
    print(f"[INFO] Processing {len(grouped)} sentences ...")

    for sid, group in tqdm(grouped, total=len(grouped), desc="Sentences"):
        group = group.sort_values("deberta_index")
        token_list = group["deberta_token"].astype(str).tolist()
        df_indices = group.index.tolist()

        att = compute_attention(model, tokenizer, token_list)
        layer_sums = att["layer_sums"]
        seq_len = len(token_list)

        # Sanity checks (same checks as the original).
        if not isinstance(layer_sums, (list, tuple)):
            raise RuntimeError(f"Invalid type for layer_sums: {type(layer_sums)}")

        if len(layer_sums) != n_layers:
            raise RuntimeError(f"Expected {n_layers} layers but got {len(layer_sums)}")

        for li, tens in enumerate(layer_sums):
            if tens is None:
                raise RuntimeError(f"Layer {li+1}: attention sum is None")
            if tens.shape[0] != seq_len:
                raise RuntimeError(
                    f"Shape mismatch at layer {li+1}: "
                    f"tensor length={tens.shape[0]} vs tokens={seq_len}"
                )

        for li in range(n_layers):
            layer_sum_np = layer_sums[li].detach().cpu().numpy()
            col_top = f"top5_l{li+1}"
            flags = topk_boolean_flags(layer_sum_np, k=args.top_k)
            df.loc[df_indices, col_top] = flags

    # Enforce types.
    for L in range(1, n_layers + 1):
        df[f"top5_l{L}"] = df[f"top5_l{L}"].astype(int)

    # Save output to the unchanged path.
    print(f"[INFO] Saving final file: {out_path}")
    df.to_csv(out_path, sep=";", index=False)
    print("[OK] DONE — DeBERTa attention top5 computation complete.")


if __name__ == "__main__":
    main()
