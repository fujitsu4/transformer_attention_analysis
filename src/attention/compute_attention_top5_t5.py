"""
compute_attention_top5_t5.py
Author: Zakaria JOUILIL

Description:
    Compute per-layer attention sums and top-5 flags for ALL tokens of ALL sentences.
    Uses ONLY compute_attention() (tested and validated before), and processes
    the entire dataset.

Inputs and paramters : 
    --model : choose pretrained or untrained configuration
    --seed : choose the fixed seed (for untrained model)
    --input_csv : the csv containg t5 tokens (with features)
    --top_k : choose top_k token (default : top 5)
    --device : choose device (GPU)

Outputs:
    - outputs/attention/attention_top5_pretrained.csv
    - outputs/attention/attention_top5_untrained_seed_XXX.csv   (10 seeds)

Usage : 
    For pretained model :
        python -m src.attention.compute_attention_top5 \
        --model pretrained \
        --input_csv outputs/t5/t5_snli_final_features.csv --dataset snli

    For untrained model (example with seed 10) : 
        python -m src.attention.compute_attention_top5 \
        --model untrained --seed 10 \
        --input_csv outputs/t5/t5_snli_final_features.csv --dataset snli

"""
"""
UPDATE 2025-12-03:
Due to very large output file sizes (≈105 MB) we reduced the exported CSV by removing:
- the full attention SUM columns for all 12 layers,

These fields are now DISABLED BY DEFAULT to keep results under 50 MB
and compatible with GitHub storage limits.

To reproduce the full original output (archival version), simply
re-enable the commented block in lines 144, 189, 193 and 203 where SUM columns
are computed and added to the dataframe before saving.

This change does NOT affect the top-5 attention labels or any analysis
used in the thesis.
"""
"""
UPDATE 2025-12-04:
Finally, all intermediate results will no longer be sent to GitHub to reduce the
size of the repository. Only the datasets (intermediate and final) and the logs will
be kept in their entirety. For the other files, samples with 5 lines will be shared
(read the readme file if needed).
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

MODEL_NAME = "google-t5/t5-small"


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.attention.compute_attention_core_t5 import (
    load_model,
    compute_attention,
    fix_seed,
)


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
    parser.add_argument("--dataset", type=str, required=True,
                    choices=["snli", "agnews", "arxiv"],
                    help="Dataset being processed")
    args = parser.parse_args()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    if args.model == "pretrained":
        pretrained_flag = True
        print("[INFO] Loading PRETRAINED model ...")
    else:
        if args.seed is None:
            raise ValueError("Untrained model requires --seed")
        pretrained_flag = False
        print(f"[INFO] Loading RANDOMLY INITIALIZED model with seed {args.seed} ...")
        fix_seed(args.seed)

    model = load_model(pretrained_flag, seed=args.seed)
    n_layers = model.config.num_layers

    print(f"[INFO] T5 encoder layers: {n_layers}")
    print(f"[INFO] T5 attention heads: {model.config.num_heads}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )
    
    # -----------------------------------------------------
    # Load input CSV (with ALL 10k sentences, ALL tokens)
    # -----------------------------------------------------
    print(f"[INFO] Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    # minimal columns required
    for col in ("sentence_id", "t5_index", "t5_token"):
        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # -----------------------------------------------------
    # Prepare output directory
    # -----------------------------------------------------
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:
        out_name = f"attention_top5_{args.dataset}_pretrained.csv"
    else:
        out_name = f"attention_top5_{args.dataset}_untrained_seed_{args.seed:03d}.csv"

    out_path = os.path.join(out_dir, out_name)

    # -----------------------------------------------------
    # Add 24 output columns:
    # sum_l1 .. sum_l12  +  top5_l1 .. top5_l12
    # -----------------------------------------------------
    for L in range(1, n_layers + 1):
        #df[f"sum_l{L}"] = np.nan # Disabled Code (too large .csv to upload to GitHub)
        df[f"top5_l{L}"] = 0

    # -----------------------------------------------------
    # Process ALL sentences
    # -----------------------------------------------------
    grouped = df.groupby("sentence_id", sort=False)
    print(f"[INFO] Processing {len(grouped)} sentences ...")

    for sid, group in tqdm(grouped, total=len(grouped), desc="Sentences"):
        # Sort tokens by t5_index
        group = group.sort_values("t5_index")
        token_list = group["t5_token"].astype(str).tolist()
        df_indices = group.index.tolist()

        # ------------------------
        # ATTENTION computation
        # ------------------------
        # feed EXACT tokens → no retokenization ever
        att = compute_attention(model, tokenizer, token_list)

        layer_sums = att["layer_sums"]   # list of 12 tensors (seq_len,)
        seq_len = len(token_list)

        # ---- SANITY CHECKS ----
        if not isinstance(layer_sums, (list, tuple)):
            raise RuntimeError(f"Invalid type for layer_sums: {type(layer_sums)}")

        if len(layer_sums) != n_layers:
            raise RuntimeError(f"Expected {n_layers} layers but got {len(layer_sums)}")

        for li, tens in enumerate(layer_sums):
            if tens is None:
                raise RuntimeError(f"Layer {li+1}: attention sum is None")

            if tens.shape[0] != seq_len:
                raise RuntimeError(f"Shape mismatch at layer {li+1}: tensor length={tens.shape[0]} vs tokens={seq_len}")

        # ------------------------
        # For each layer :
        #   1) save attention sum
        #   2) compute top-5 flags
        # ------------------------
        for li in range(n_layers):
            layer_sum_np = layer_sums[li].detach().cpu().numpy()  # (seq_len,)
            #col_sum = f"sum_l{li+1}" # Disabled Code (too large .csv to upload to GitHub)
            col_top = f"top5_l{li+1}"

            # Vectorized: assign all sums at once
            #df.loc[df_indices, col_sum] = layer_sum_np # Disabled Code (too large .csv to upload to GitHub)

            # Vectorized: assign top-k flags at once
            flags = topk_boolean_flags(layer_sum_np, k=args.top_k)
            df.loc[df_indices, col_top] = flags

    # -----------------------------------------------------
    # Enforce types
    # -----------------------------------------------------
    for L in range(1, n_layers + 1):
    #    df[f"sum_l{L}"] = df[f"sum_l{L}"].astype(float) ## Disabled Code (too large .csv to upload to GitHub)
        df[f"top5_l{L}"] = df[f"top5_l{L}"].astype(int)

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    print(f"[INFO] Saving final file: {out_path}")
    df.to_csv(out_path, sep=";", index=False)
    print("[OK] DONE — attention top5 computation complete.")


if __name__ == "__main__":
    main()