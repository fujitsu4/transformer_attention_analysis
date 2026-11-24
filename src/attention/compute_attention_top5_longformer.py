
"""
compute_attention_top5_longformer.py
Author: Zakaria JOUILIL

Longformer attention calculations and top-k selection.
Stored tokens are converted directly to IDs; no retokenization.

PROTOCOL:
    Native Longformer sliding-window local attention.
    No global_attention_mask is used.
    Attention received by each token is reconstructed from
    the relative positions in the sliding window.

Optimizations:
    - Attention sums computed in one block per forward pass.
    - One GPU->CPU transfer per batch.
    - Preallocated output array.
    - No full attention matrices kept in memory.
    - Sentences grouped by padded length.

Usage:
    python -m src.attention.compute_attention_top5_longformer \
        --model pretrained --dataset snli \
        --input_csv outputs/longformer/longformer_snli_final_features.csv \
        [--batch_size 1]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.attention.compute_attention_core_longformer import (
    load_model,
    layer_sums_local_batch,
    padded_length_for,
    fix_seed
)


MODEL_NAME = "allenai/longformer-base-4096"
N_LAYERS = 12


# ---------------------------------------------------------
# Utility: select top-k for 1D numpy array
# ---------------------------------------------------------

def topk_boolean_flags(arr, k=5):

    seq_len = arr.shape[0]
    kk = min(k, seq_len)

    idx_desc = np.argsort(-arr, kind="stable")
    topk_idx = idx_desc[:kk]

    flags = np.zeros(seq_len, dtype=int)
    flags[topk_idx] = 1

    return flags


# ---------------------------------------------------------
# Build (sentence_id, positional row indices, tokens) once
# ---------------------------------------------------------

def build_sentences(df):

    tokens_all = (
        df["longformer_token"]
        .astype(str)
        .to_numpy()
    )

    index_all = df["longformer_index"].to_numpy()

    sentences = []

    for sid, pos in df.groupby(
        "sentence_id",
        sort=False
    ).indices.items():

        pos = pos[
            np.argsort(index_all[pos], kind="stable")
        ]

        sentences.append(
            (sid, pos, tokens_all[pos].tolist())
        )

    return sentences


# ---------------------------------------------------------
# Validate attention sums
# ---------------------------------------------------------

def check_sums(sums_2d, seq_len, sid):

    if sums_2d.shape != (N_LAYERS, seq_len):
        raise RuntimeError(
            f"Sentence {sid}: expected shape "
            f"{(N_LAYERS, seq_len)}, got {sums_2d.shape}"
        )

    if not np.isfinite(sums_2d).all():
        raise RuntimeError(
            f"Sentence {sid}: non-finite attention sums."
        )

    if (sums_2d < -1e-7).any():
        raise RuntimeError(
            f"Sentence {sid}: negative attention sums detected."
        )


# ---------------------------------------------------------
# Write top-k flags
# ---------------------------------------------------------

def write_flags(top, pos, sums_2d, k):

    for layer_index in range(N_LAYERS):

        top[pos, layer_index] = topk_boolean_flags(
            sums_2d[layer_index],
            k=k
        )


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
        help="Seed required for untrained model."
    )

    parser.add_argument(
        "--input_csv",
        required=True
    )

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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sentences per Longformer forward pass."
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = args.device

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(f"[INFO] Device: {device}")

    if device == "cpu":
        print(
            "[WARN] Running on CPU: Longformer attention "
            "computation may be slow."
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------

    pretrained_flag = args.model == "pretrained"

    if pretrained_flag:

        print(
            "[INFO] Loading PRETRAINED Longformer-base-4096 ..."
        )

    else:

        if args.seed is None:
            raise ValueError(
                "Untrained model requires --seed."
            )

        print(
            "[INFO] Loading RANDOMLY INITIALIZED "
            f"Longformer-base-4096 with seed {args.seed} ..."
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
        raise RuntimeError(
            "A fast tokenizer is required."
        )

    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "Longformer tokenizer has no pad_token_id."
        )

    model.to(device)
    model.eval()

    if model.config.num_hidden_layers != N_LAYERS:
        raise RuntimeError(
            f"Expected {N_LAYERS} layers, "
            f"got {model.config.num_hidden_layers}."
        )

    # -----------------------------------------------------
    # Load input CSV
    # -----------------------------------------------------

    print(f"[INFO] Loading input CSV: {args.input_csv}")

    df = pd.read_csv(
        args.input_csv,
        sep=";",
        keep_default_na=False
    )

    required_cols = (
        "sentence_id",
        "longformer_index",
        "longformer_token"
    )

    for col in required_cols:

        if col not in df.columns:
            raise ValueError(
                f"Missing column in input CSV: {col}"
            )

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df["sentence_id"] = pd.to_numeric(
        df["sentence_id"],
        errors="raise"
    ).astype(int)

    df["longformer_index"] = pd.to_numeric(
        df["longformer_index"],
        errors="raise"
    ).astype(int)

    if df["longformer_token"].isna().any():
        raise ValueError(
            "longformer_token contains missing values."
        )

    # -----------------------------------------------------
    # Output path
    # -----------------------------------------------------

    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:

        out_name = (
            f"attention_top5_{args.dataset}_"
            "longformer_pretrained.csv"
        )

    else:

        out_name = (
            f"attention_top5_{args.dataset}_"
            f"longformer_untrained_seed_{args.seed:03d}.csv"
        )

    out_path = os.path.join(
        out_dir,
        out_name
    )

    # -----------------------------------------------------
    # Split sentences
    # -----------------------------------------------------

    sentences = build_sentences(df)

    if not sentences:
        raise ValueError("No sentences found in input CSV.")

    lens = np.array(
        [len(sentence[2]) for sentence in sentences]
    )

    print(
        f"[INFO] {len(sentences)} sentences | "
        f"length min/median/p95/max = "
        f"{lens.min()}/{int(np.median(lens))}/"
        f"{int(np.percentile(lens, 95))}/{lens.max()}"
    )

    print(
        f"[INFO] Attention window: "
        f"{model.config.attention_window}"
    )

    # -----------------------------------------------------
    # Validate sentence lengths and prepare batches
    # -----------------------------------------------------

    print("[INFO] Checking padded sequence lengths...")

    for sid, pos, tokens in sentences:

        padded_length_for(
            model,
            len(tokens)
        )

    # Group sentences by padded length.
    # This avoids unnecessary padding differences within a batch.

    sentences = sorted(
        sentences,
        key=lambda s: (
            padded_length_for(model, len(s[2])),
            len(s[2])
        )
    )

    batches = []
    current_batch = []
    current_padded_length = None

    for sentence in sentences:

        padded_length = padded_length_for(
            model,
            len(sentence[2])
        )

        if current_batch and (
            len(current_batch) >= args.batch_size
            or padded_length != current_padded_length
        ):

            batches.append(current_batch)
            current_batch = []

        if not current_batch:
            current_padded_length = padded_length

        current_batch.append(sentence)

    if current_batch:
        batches.append(current_batch)

    print(
        f"[INFO] Prepared {len(batches)} batches "
        f"(batch_size={args.batch_size})."
    )

    # -----------------------------------------------------
    # Preallocated result
    # -----------------------------------------------------

    top = np.zeros(
        (len(df), N_LAYERS),
        dtype=np.int64
    )

    # -----------------------------------------------------
    # Longformer local attention
    # -----------------------------------------------------

    t0 = time.time()

    for batch in tqdm(
        batches,
        desc="Longformer local attention"
    ):

        token_lists = [
            sentence[2]
            for sentence in batch
        ]

        sums = layer_sums_local_batch(
            model,
            tokenizer,
            token_lists
        )

        if sums.shape[0] != N_LAYERS:
            raise RuntimeError(
                f"Expected {N_LAYERS} layers, got {sums.shape[0]}."
            )

        for batch_index, (sid, pos, tokens) in enumerate(batch):

            n = len(tokens)

            sums_j = sums[
                :,
                batch_index,
                :n
            ]

            check_sums(
                sums_j,
                n,
                sid
            )

            write_flags(
                top,
                pos,
                sums_j,
                args.top_k
            )

    elapsed = time.time() - t0

    print(
        f"[INFO] Longformer attention computation "
        f"completed in {elapsed:.1f}s."
    )

    # -----------------------------------------------------
    # Add top-k columns
    # -----------------------------------------------------

    for layer_index in range(1, N_LAYERS + 1):

        df[f"top5_l{layer_index}"] = (
            top[:, layer_index - 1].astype(int)
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print(f"[INFO] Saving final file: {out_path}")

    df.to_csv(
        out_path,
        sep=";",
        index=False
    )

    print(
        "[OK] DONE — Longformer attention top5 computation complete."
    )


if __name__ == "__main__":
    main()
