"""
compute_attention_top5_bigbird.py
Author: Zakaria JOUILIL

BigBird attention calculations and top-k selection.
Stored tokens are converted directly to IDs; no retokenization.

REFERENCE PROTOCOL (default):
    true BigBird block_sparse attention, input padded to >= 1024 tokens,
    for ALL sentences. Output: attention_top5_<dataset>_bigbird_*.csv

Optimizations that do NOT change the mechanism:
    - attention sums computed in one block per forward pass
    - ONE GPU->CPU transfer per sentence (or per batch)
    - preallocated output array, no pandas .loc inside the loop
    - no full attention matrices / head sums kept in memory
    - optional --sparse_batch_size: several sentences per block_sparse forward

OPTIONAL VARIANT (--full_for_short):
    sentences with <= block_size tokens use 'original_full' attention on
    unpadded inputs (equivalent in theory, much faster). It is a SEPARATE
    experiment: the output file gets the suffix '_fullshort' so it can never
    overwrite or be mixed with the reference.

Usage:
    python -m src.attention.compute_attention_top5_bigbird \
        --model pretrained --dataset snli \
        --input_csv outputs/bigbird/bigbird_snli_final_features.csv \
        [--sparse_batch_size 4] [--full_for_short]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.attention.compute_attention_core_bigbird import (
    load_model,
    layer_sums_full_batch,
    layer_sums_sparse_batch,
    padded_length_for,
    set_attention_mode,
    fix_seed
)

MODEL_NAME = "google/bigbird-roberta-base"
N_LAYERS = 12


# ---------------------------------------------------------
# Utility: select top-k for 1D numpy array (unchanged)
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
# Build (sentence_id, positional row indices, tokens) once
# ---------------------------------------------------------
def build_sentences(df):

    tokens_all = df["bigbird_token"].astype(str).to_numpy()
    bidx_all = df["bigbird_index"].to_numpy()

    sentences = []

    # .indices -> {sentence_id: array of POSITIONAL row numbers}
    for sid, pos in df.groupby("sentence_id", sort=False).indices.items():

        pos = pos[np.argsort(bidx_all[pos], kind="stable")]
        sentences.append((sid, pos, tokens_all[pos].tolist()))

    return sentences


def check_sums(sums_2d, seq_len, sid):
    """sums_2d: (N_LAYERS, seq_len) numpy array."""

    if sums_2d.shape != (N_LAYERS, seq_len):
        raise RuntimeError(
            f"Sentence {sid}: expected shape {(N_LAYERS, seq_len)}, "
            f"got {sums_2d.shape}"
        )

    if not np.isfinite(sums_2d).all():
        raise RuntimeError(f"Sentence {sid}: non-finite attention sums")


def write_flags(top, pos, sums_2d, k):
    for li in range(N_LAYERS):
        top[pos, li] = topk_boolean_flags(sums_2d[li], k=k)


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
    parser.add_argument("--dataset", type=str, required=True, choices=["snli", "agnews", "arxiv"])

    parser.add_argument(
        "--sparse_batch_size", type=int, default=1,
        help="Sentences per block_sparse forward pass (reference path). "
             "1 = original behaviour. Each sentence costs ~0.6 GB of attention "
             "matrices at 1024 tokens, so keep it small."
    )
    parser.add_argument(
        "--full_for_short", action="store_true",
        help="VARIANT: use original_full on sentences <= block_size tokens. "
             "Writes a separate '_fullshort' file."
    )
    parser.add_argument(
        "--full_batch_size", type=int, default=32,
        help="Batch size of the original_full variant"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device = args.device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")

    if device == "cpu":
        print("[WARN] Running on CPU: BigBird on 1024 tokens will be very slow. "
              "In Colab: Runtime > Change runtime type > GPU.")

    # -----------------------------------------------------
    # Model selection (unchanged -> same weights as before)
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

    model = load_model(pretrained=pretrained_flag, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required.")

    if tokenizer.pad_token_id is None:
        raise RuntimeError("BigBird tokenizer has no pad_token_id.")

    model.to(device)
    model.eval()

    block_size = model.config.block_size

    # -----------------------------------------------------
    # Load input CSV
    # -----------------------------------------------------
    print(f"[INFO] Loading input CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    for col in ("sentence_id", "bigbird_index", "bigbird_token"):
        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # -----------------------------------------------------
    # Output path (variant never shares the reference file name)
    # -----------------------------------------------------
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    variant = "_fullshort" if args.full_for_short else ""

    if pretrained_flag:
        out_name = f"attention_top5_{args.dataset}_bigbird{variant}_pretrained.csv"
    else:
        out_name = (
            f"attention_top5_{args.dataset}_bigbird{variant}_"
            f"untrained_seed_{args.seed:03d}.csv"
        )

    out_path = os.path.join(out_dir, out_name)

    # -----------------------------------------------------
    # Split sentences
    # -----------------------------------------------------
    sentences = build_sentences(df)
    lens = np.array([len(s[2]) for s in sentences])

    print(
        f"[INFO] {len(sentences)} sentences | length min/median/p95/max = "
        f"{lens.min()}/{int(np.median(lens))}/{int(np.percentile(lens, 95))}/{lens.max()}"
    )
    print(f"[INFO] Sentences with <= {block_size} tokens: {(lens <= block_size).sum()}")

    if args.full_for_short:
        short = sorted(
            (s for s in sentences if len(s[2]) <= block_size),
            key=lambda s: len(s[2])
        )
        sparse = [s for s in sentences if len(s[2]) > block_size]
        print("[INFO] VARIANT ACTIVE: original_full for short sentences "
              f"({len(short)}), block_sparse for the others ({len(sparse)}).")
    else:
        short = []
        sparse = sentences
        print(f"[INFO] REFERENCE PROTOCOL: block_sparse, padded >= 1024, "
              f"for all {len(sparse)} sentences.")

    # Preallocated result
    top = np.zeros((len(df), N_LAYERS), dtype=np.int64)

    # -----------------------------------------------------
    # VARIANT path: full attention, unpadded, batched
    # -----------------------------------------------------
    if short:

        t0 = time.time()
        set_attention_mode(model, "original_full")

        for i in tqdm(range(0, len(short), args.full_batch_size), desc="original_full (short)"):

            batch = short[i:i + args.full_batch_size]
            sums = layer_sums_full_batch(model, tokenizer, [b[2] for b in batch])

            for j, (sid, pos, toks) in enumerate(batch):
                n = len(toks)
                sums_j = sums[:, j, :n]
                check_sums(sums_j, n, sid)
                write_flags(top, pos, sums_j, args.top_k)

        print(f"[INFO] original_full part done in {time.time() - t0:.1f}s")

    # -----------------------------------------------------
    # REFERENCE path: block_sparse, padded >= 1024
    # -----------------------------------------------------
    if sparse:

        t1 = time.time()
        set_attention_mode(model, "block_sparse")

        # Group by padded length (all 1024 unless a sentence is > 1024 tokens),
        # similar lengths together, so batches never change a sentence's padding.
        sparse = sorted(
            sparse, key=lambda s: (padded_length_for(model, len(s[2])), len(s[2]))
        )

        batches = []
        cur = []
        for s in sparse:
            if cur and (
                len(cur) >= args.sparse_batch_size
                or padded_length_for(model, len(s[2])) != padded_length_for(model, len(cur[0][2]))
            ):
                batches.append(cur)
                cur = []
            cur.append(s)
        if cur:
            batches.append(cur)

        for batch in tqdm(batches, desc="block_sparse (reference)"):

            sums = layer_sums_sparse_batch(model, tokenizer, [b[2] for b in batch])

            for j, (sid, pos, toks) in enumerate(batch):
                n = len(toks)
                sums_j = sums[:, j, :n]
                check_sums(sums_j, n, sid)
                write_flags(top, pos, sums_j, args.top_k)

        print(f"[INFO] block_sparse part done in {time.time() - t1:.1f}s")

    # -----------------------------------------------------
    # Write columns once, in the same order as before
    # -----------------------------------------------------
    for L in range(1, N_LAYERS + 1):
        df[f"top5_l{L}"] = top[:, L - 1].astype(int)

    print(f"[INFO] Saving final file: {out_path}")

    df.to_csv(out_path, sep=";", index=False)

    print("[OK] DONE — BigBird attention top5 computation complete.")


if __name__ == "__main__":
    main()