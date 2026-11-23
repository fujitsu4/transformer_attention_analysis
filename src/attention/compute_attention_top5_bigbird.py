
"""
compute_attention_top5_bigbird.py
Author: Zakaria JOUILIL

BigBird attention calculations and top-k selection.
Stored tokens are converted directly to IDs; no retokenization.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.attention.compute_attention_core_bigbird import (
    load_model,
    compute_attention,
    fix_seed
)

MODEL_NAME = "google/bigbird-roberta-base"
N_LAYERS = 12


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

    parser.add_argument(
        "--model",
        choices=["pretrained", "untrained"],
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed required for untrained"
    )

    parser.add_argument("--input_csv", required=True)

    parser.add_argument(
        "--top_k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"]
    )

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
    pretrained_flag = args.model == "pretrained"

    if pretrained_flag:

        print("[INFO] Loading PRETRAINED BigBird-RoBERTa-base ...")

    else:

        if args.seed is None:
            raise ValueError("Untrained model requires --seed")

        print(
            f"[INFO] Loading RANDOMLY INITIALIZED BigBird-RoBERTa-base "
            f"with seed {args.seed} ..."
        )

        fix_seed(args.seed)

    model = load_model(
        pretrained=pretrained_flag,
        seed=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required.")

    if tokenizer.pad_token_id is None:
        raise RuntimeError("BigBird tokenizer has no pad_token_id.")

    model.to(device)
    model.eval()

    # -----------------------------------------------------
    # Load input CSV
    # -----------------------------------------------------
    print(f"[INFO] Loading input CSV: {args.input_csv}")

    df = pd.read_csv(
        args.input_csv,
        sep=";",
        keep_default_na=False
    )

    required_columns = (
        "sentence_id",
        "bigbird_index",
        "bigbird_token"
    )

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # -----------------------------------------------------
    # Output directory and filename
    # -----------------------------------------------------
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:

        out_name = (
            f"attention_top5_{args.dataset}_bigbird_pretrained.csv"
        )

    else:

        out_name = (
            f"attention_top5_{args.dataset}_bigbird_"
            f"untrained_seed_{args.seed:03d}.csv"
        )

    out_path = os.path.join(out_dir, out_name)

    for L in range(1, N_LAYERS + 1):
        df[f"top5_l{L}"] = 0

    # -----------------------------------------------------
    # Process all sentences
    # -----------------------------------------------------
    grouped = df.groupby("sentence_id", sort=False)

    print(f"[INFO] Processing {len(grouped)} sentences ...")

    for sid, group in tqdm(
        grouped,
        total=len(grouped),
        desc="Sentences"
    ):

        group = group.sort_values("bigbird_index")

        token_list = (
            group["bigbird_token"]
            .astype(str)
            .tolist()
        )

        df_indices = group.index.tolist()

        att = compute_attention(
            model,
            tokenizer,
            token_list
        )

        layer_sums = att["layer_sums"]
        seq_len = len(token_list)

        # -------------------------------------------------
        # Sanity checks
        # -------------------------------------------------
        if not isinstance(layer_sums, (list, tuple)):
            raise RuntimeError(
                f"Invalid type for layer_sums: {type(layer_sums)}"
            )

        if len(layer_sums) != N_LAYERS:
            raise RuntimeError(
                f"Expected {N_LAYERS} layers but got {len(layer_sums)}"
            )

        for li, tens in enumerate(layer_sums):

            if tens is None:
                raise RuntimeError(
                    f"Layer {li+1}: attention sum is None"
                )

            if tens.shape[0] != seq_len:
                raise RuntimeError(
                    f"Shape mismatch at layer {li+1}: "
                    f"tensor length={tens.shape[0]} "
                    f"vs tokens={seq_len}"
                )

        # -------------------------------------------------
        # Top-k selection
        # -------------------------------------------------
        for li in range(N_LAYERS):

            layer_sum_np = (
                layer_sums[li]
                .detach()
                .cpu()
                .numpy()
            )

            col_top = f"top5_l{li+1}"

            flags = topk_boolean_flags(
                layer_sum_np,
                k=args.top_k
            )

            df.loc[df_indices, col_top] = flags

    # -----------------------------------------------------
    # Enforce types
    # -----------------------------------------------------
    for L in range(1, N_LAYERS + 1):
        df[f"top5_l{L}"] = df[f"top5_l{L}"].astype(int)

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    print(f"[INFO] Saving final file: {out_path}")

    df.to_csv(
        out_path,
        sep=";",
        index=False
    )

    print("[OK] DONE — BigBird attention top5 computation complete.")


if __name__ == "__main__":
    main()
