"""
benchmark_bigbird_paths.py
Author: Zakaria JOUILIL

Times and compares the execution paths on the SAME sample of sentences.

  REF      : compute_attention() one sentence at a time (original code,
             block_sparse, padded >= 1024)  -> the reference
  SPARSE-B : layer_sums_sparse_batch(), block_sparse, padded >= 1024,
             B sentences per forward pass (same mechanism)
  FULL     : layer_sums_full_batch(), original_full, unpadded, batched
             (variant, short sentences only)

For each path: time, extrapolated time for the whole corpus, peak GPU memory,
max |sum difference| and number of top-5 flag mismatches versus REF.

Usage:
    python -m src.attention.benchmark_bigbird_paths \
        --input_csv outputs/bigbird/bigbird_snli_final_features.csv \
        --n 64 --sparse_batches 1,4,8
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.attention.compute_attention_core_bigbird import (
    load_model,
    compute_attention,
    layer_sums_full_batch,
    layer_sums_sparse_batch,
    set_attention_mode,
    fix_seed
)
from src.attention.compute_attention_top5_bigbird import (
    build_sentences,
    topk_boolean_flags,
    N_LAYERS,
    MODEL_NAME
)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def compare(ref, other, k=5):
    max_abs, mism = 0.0, 0
    for a, b in zip(ref, other):
        max_abs = max(max_abs, float(np.abs(a - b).max()))
        for li in range(N_LAYERS):
            if not np.array_equal(topk_boolean_flags(a[li], k), topk_boolean_flags(b[li], k)):
                mism += 1
    return max_abs, mism


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--sparse_batches", default="1,4,8")
    ap.add_argument("--full_batch", type=int, default=32)
    ap.add_argument("--model", choices=["pretrained", "untrained"], default="pretrained")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = args.model == "pretrained"

    if not pretrained:
        if args.seed is None:
            raise ValueError("--seed required for untrained")
        fix_seed(args.seed)

    model = load_model(pretrained=pretrained, seed=args.seed).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)
    sentences = build_sentences(df)
    total = len(sentences)
    block = model.config.block_size

    rng = np.random.default_rng(args.sample_seed)
    idx = rng.choice(total, size=min(args.n, total), replace=False)
    sample = [sentences[i] for i in idx]
    toks_list = [s[2] for s in sample]
    n = len(sample)

    print(f"[INFO] device={device}  sample={n} sentences  corpus={total} sentences")

    rows = []

    def record(name, secs, ref_sums, sums, peak):
        d, m = (0.0, 0) if sums is ref_sums else compare(ref_sums, sums)
        per = secs / n
        rows.append((name, secs, per, per * total / 60.0, peak, d, m))

    def peak_mem():
        return torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else float("nan")

    # ---------------- warm-up ----------------
    set_attention_mode(model, "block_sparse")
    for t in toks_list[:2]:
        compute_attention(model, tok, t, return_attentions=False, return_head_sums=False)
    sync(device)

    # ---------------- REF ----------------
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ref = []
    for t in toks_list:
        att = compute_attention(model, tok, t, return_attentions=False, return_head_sums=False)
        ref.append(torch.stack(att["layer_sums"]).cpu().numpy())
    sync(device)
    record("REF  (1 by 1, original code)", time.time() - t0, ref, ref, peak_mem())

    # ---------------- SPARSE-B ----------------
    for B in [int(x) for x in args.sparse_batches.split(",")]:

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        out = []
        try:
            t0 = time.time()
            for i in range(0, n, B):
                chunk = toks_list[i:i + B]
                sums = layer_sums_sparse_batch(model, tok, chunk)
                for j, t in enumerate(chunk):
                    out.append(sums[:, j, :len(t)])
            sync(device)
            record(f"SPARSE-B={B}", time.time() - t0, ref, out, peak_mem())
        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] SPARSE-B={B}: out of memory, skipped")
            torch.cuda.empty_cache()

    # ---------------- FULL (short sentences only) ----------------
    short_idx = [i for i, t in enumerate(toks_list) if len(t) <= block]

    if short_idx:

        set_attention_mode(model, "original_full")
        short_toks = [toks_list[i] for i in short_idx]

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        out = []
        t0 = time.time()
        for i in range(0, len(short_toks), args.full_batch):
            chunk = short_toks[i:i + args.full_batch]
            sums = layer_sums_full_batch(model, tok, chunk)
            for j, t in enumerate(chunk):
                out.append(sums[:, j, :len(t)])
        sync(device)
        secs = time.time() - t0

        d, m = compare([ref[i] for i in short_idx], out)
        per = secs / len(short_idx)
        rows.append((f"FULL (variant, {len(short_idx)} short sent.)", secs, per,
                     per * total / 60.0, peak_mem(), d, m))

    # ---------------- report ----------------
    print(f"\n{'path':<38}{'time(s)':>9}{'s/sent':>9}{'est.min':>9}{'peakGB':>8}{'max|diff|':>12}{'top5 mism.':>12}")
    for name, secs, per, est, peak, d, m in rows:
        print(f"{name:<38}{secs:>9.2f}{per:>9.3f}{est:>9.1f}{peak:>8.2f}{d:>12.2e}{m:>12d}")

    print(f"\n(est.min = extrapolation to the {total} sentences of the corpus; "
          f"top5 mism. counts (sentence, layer) pairs out of {n * N_LAYERS})")


if __name__ == "__main__":
    main()