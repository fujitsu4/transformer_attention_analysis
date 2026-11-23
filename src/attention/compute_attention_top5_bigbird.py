"""
compute_attention_top5_bigbird.py
Author: Zakaria JOUILIL

BigBird attention calculations and top-k selection (OPTIMIZED VERSION).
Stored tokens are converted directly to IDs; no retokenization.

Two execution paths, same results:

  FAST path  (real length <= block_size = 64 tokens)
      Real tokens all live in block 0, which is a GLOBAL block in BigBird:
      it attends to every key, and padded keys are masked. The result is
      therefore identical to full attention over the real tokens, so we run
      'original_full' on UNPADDED inputs, in BATCHES.

  LEGACY path (real length > 64 tokens)
      Original behaviour: pad to 1024 and run true block_sparse attention,
      one sentence at a time.

Use --max_full_len 0 to disable the fast path (= exactly the old behaviour).

Usage (unchanged + 2 optional args):
    python -m src.attention.compute_attention_top5_bigbird \
        --model pretrained --dataset snli \
        --input_csv outputs/bigbird/bigbird_snli_final_features.csv \
        [--batch_size 32] [--max_full_len 64]
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
    compute_attention,
    layer_sums_full_batch,
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
        "--batch_size", type=int, default=32,
        help="Batch size for the fast path (short sentences)"
    )
    parser.add_argument(
        "--max_full_len", type=int, default=None,
        help="Max real length for the fast path. Default = block_size (64). "
             "0 disables the fast path (legacy behaviour for everything)."
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
    max_full_len = block_size if args.max_full_len is None else args.max_full_len

    if max_full_len > block_size:
        raise ValueError(
            f"--max_full_len ({max_full_len}) must be <= block_size ({block_size}): "
            f"beyond one block the block-sparse pattern is no longer equal "
            f"to full attention."
        )

    # -----------------------------------------------------
    # Load input CSV
    # -----------------------------------------------------
    print(f"[INFO] Loading input CSV: {args.input_csv}")

    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)

    for col in ("sentence_id", "bigbird_index", "bigbird_token"):
        if col not in df.columns:
            raise ValueError(f"Missing column in input CSV: {col}")

    # -----------------------------------------------------
    # Output path
    # -----------------------------------------------------
    out_dir = "outputs/attention"
    os.makedirs(out_dir, exist_ok=True)

    if pretrained_flag:
        out_name = f"attention_top5_{args.dataset}_bigbird_pretrained.csv"
    else:
        out_name = (
            f"attention_top5_{args.dataset}_bigbird_"
            f"untrained_seed_{args.seed:03d}.csv"
        )

    out_path = os.path.join(out_dir, out_name)

    # -----------------------------------------------------
    # Split sentences: fast path vs legacy path
    # -----------------------------------------------------
    sentences = build_sentences(df)

    short = sorted(
        (s for s in sentences if len(s[2]) <= max_full_len),
        key=lambda s: len(s[2])          # similar lengths -> little padding
    )
    long_ = [s for s in sentences if len(s[2]) > max_full_len]

    lens = np.array([len(s[2]) for s in sentences])

    print(
        f"[INFO] {len(sentences)} sentences | "
        f"length min/median/p95/max = "
        f"{lens.min()}/{int(np.median(lens))}/{int(np.percentile(lens, 95))}/{lens.max()}"
    )
    print(f"[INFO] Fast path   (<= {max_full_len} tokens): {len(short)} sentences")
    print(f"[INFO] Legacy path (>  {max_full_len} tokens): {len(long_)} sentences")

    # Preallocated result: one write per sentence/layer, no pandas .loc in the loop.
    top = np.zeros((len(df), N_LAYERS), dtype=np.int64)

    t0 = time.time()

    # -----------------------------------------------------
    # FAST path: full attention, unpadded, batched
    # -----------------------------------------------------
    if short:

        set_attention_mode(model, "original_full")

        for i in tqdm(range(0, len(short), args.batch_size), desc="Fast path (batches)"):

            batch = short[i:i + args.batch_size]

            sums = layer_sums_full_batch(
                model, tokenizer, [b[2] for b in batch]
            )  # (N_LAYERS, B, max_len)

            if sums.shape[0] != N_LAYERS:
                raise RuntimeError(f"Expected {N_LAYERS} layers, got {sums.shape[0]}")

            for j, (sid, pos, toks) in enumerate(batch):

                n = len(toks)
                sums_j = sums[:, j, :n]
                check_sums(sums_j, n, sid)

                for li in range(N_LAYERS):
                    top[pos, li] = topk_boolean_flags(sums_j[li], k=args.top_k)

    print(f"[INFO] Fast path done in {time.time() - t0:.1f}s")
    t1 = time.time()

    # -----------------------------------------------------
    # LEGACY path: padded 1024, true block-sparse, 1 by 1
    # -----------------------------------------------------
    if long_:

        set_attention_mode(model, "block_sparse")

        for sid, pos, toks in tqdm(long_, desc="Legacy path (block_sparse)"):

            att = compute_attention(
                model,
                tokenizer,
                toks,
                return_attentions=False,   # not used here -> don't keep them
                return_head_sums=False     # not used here -> skip 144 slices
            )

            layer_sums = att["layer_sums"]

            if len(layer_sums) != N_LAYERS:
                raise RuntimeError(
                    f"Expected {N_LAYERS} layers but got {len(layer_sums)}"
                )

            # ONE transfer instead of 12 syncs
            sums_np = torch.stack(layer_sums, dim=0).detach().cpu().numpy()
            check_sums(sums_np, len(toks), sid)

            for li in range(N_LAYERS):
                top[pos, li] = topk_boolean_flags(sums_np[li], k=args.top_k)

        print(f"[INFO] Legacy path done in {time.time() - t1:.1f}s")

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