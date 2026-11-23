"""
verify_bigbird_fastpath.py
Author: Zakaria JOUILIL

Checks that the FAST path (original_full, unpadded, batched) gives the same
attention sums / top-5 flags as the LEGACY path (padded 1024, block_sparse)
on short sentences (<= block_size tokens).

Also (optional) inspects the row sums of the block-sparse attention matrices
on a sentence longer than one block, to see whether HF's reconstructed
attention rows still sum to 1 in eval mode.

Usage:
    python -m src.attention.verify_bigbird_fastpath \
        --input_csv outputs/bigbird/bigbird_snli_final_features.csv \
        --n 100 [--model untrained --seed 10] [--rowsums]
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.attention.compute_attention_core_bigbird import (
    load_model,
    compute_attention,
    layer_sums_full_batch,
    set_attention_mode,
    fix_seed
)
from src.attention.compute_attention_top5_bigbird import (
    build_sentences,
    topk_boolean_flags,
    N_LAYERS,
    MODEL_NAME
)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--model", choices=["pretrained", "untrained"], default="pretrained")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rowsums", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = args.model == "pretrained"

    if not pretrained:
        if args.seed is None:
            raise ValueError("Untrained model requires --seed")
        fix_seed(args.seed)

    model = load_model(pretrained=pretrained, seed=args.seed).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    df = pd.read_csv(args.input_csv, sep=";", keep_default_na=False)
    sentences = build_sentences(df)

    block = model.config.block_size
    short = [s for s in sentences if len(s[2]) <= block]
    long_ = [s for s in sentences if len(s[2]) > block]

    print(f"[INFO] {len(short)} short / {len(long_)} long sentences (block_size={block})")

    rng = np.random.default_rng(args.sample_seed)
    pick = rng.choice(len(short), size=min(args.n, len(short)), replace=False)
    sample = [short[i] for i in pick]

    # ---------------- legacy ----------------
    set_attention_mode(model, "block_sparse")
    legacy = []
    for sid, pos, toks in sample:
        att = compute_attention(model, tokenizer, toks,
                                return_attentions=False, return_head_sums=False)
        legacy.append(torch.stack(att["layer_sums"]).cpu().numpy())   # (12, n)

    # ---------------- fast ----------------
    set_attention_mode(model, "original_full")
    fast = []
    bs = 16
    for i in range(0, len(sample), bs):
        chunk = sample[i:i + bs]
        sums = layer_sums_full_batch(model, tokenizer, [c[2] for c in chunk])
        for j, (_, _, toks) in enumerate(chunk):
            fast.append(sums[:, j, :len(toks)])

    # ---------------- compare ----------------
    max_abs, max_rel = 0.0, 0.0
    n_flag_mismatch = 0
    risky = []

    for (sid, _, toks), a, b in zip(sample, legacy, fast):

        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max()))
        max_rel = max(max_rel, float((d / (np.abs(a) + 1e-12)).max()))

        for li in range(N_LAYERS):
            fa = topk_boolean_flags(a[li], 5)
            fb = topk_boolean_flags(b[li], 5)
            if not np.array_equal(fa, fb):
                n_flag_mismatch += 1
                srt = np.sort(a[li])[::-1]
                gap = float(srt[4] - srt[5]) if len(srt) > 5 else float("nan")
                risky.append((sid, li + 1, gap))

    total = len(sample) * N_LAYERS
    print(f"\n[RESULT] sentences compared      : {len(sample)}")
    print(f"[RESULT] max |sum_legacy-sum_fast|: {max_abs:.3e}")
    print(f"[RESULT] max relative difference  : {max_rel:.3e}")
    print(f"[RESULT] top-5 flag mismatches    : {n_flag_mismatch} / {total} (sentence, layer) pairs")

    for sid, layer, gap in risky[:10]:
        print(f"    sid={sid} layer={layer}  gap(5th-6th) in legacy = {gap:.2e}")

    if n_flag_mismatch == 0:
        print("[OK] Fast path reproduces the legacy top-5 exactly on this sample.")
    else:
        print("[WARN] Mismatches: if the 5th/6th gap is ~1e-6 they are float-tie flips; "
              "otherwise investigate.")

    # ---------------- optional: row sums on a long sentence ----------------
    if args.rowsums:

        if not long_:
            print("[INFO] No sentence longer than one block: nothing to inspect.")
            return

        set_attention_mode(model, "block_sparse")
        sid, pos, toks = max(long_, key=lambda s: len(s[2]))

        att = compute_attention(model, tokenizer, toks,
                                return_attentions=True, return_head_sums=False)
        A = att["attentions"]                    # (layers, heads, n, n)
        rows = A.sum(dim=-1)                     # (layers, heads, n)

        print(f"\n[ROWSUMS] sentence {sid}, {len(toks)} tokens (block-sparse, eval)")
        print(f"[ROWSUMS] first block rows  : min={rows[..., :block].min():.4f} "
              f"max={rows[..., :block].max():.4f}")
        if len(toks) > block:
            print(f"[ROWSUMS] later blocks rows : min={rows[..., block:].min():.4f} "
                  f"max={rows[..., block:].max():.4f}")
        print("[ROWSUMS] a row of a true attention distribution restricted to real "
              "keys should be ~1.0; values clearly below 1 mean mass is lost in the "
              "dense reconstruction.")


if __name__ == "__main__":
    main()