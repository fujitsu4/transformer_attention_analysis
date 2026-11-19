"""
verify_t5_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Verifies outputs of compute_t5_statistical_features.py.

Input:
    --features_csv : outputs/t5/t5_snli_statistical_features.csv

Output:
    --log_file : logs/verify_t5_statistical_features.log

Usage:
    python -m src.t5.verify_t5_statistical_features \
        --features_csv outputs/t5/t5_snli_statistical_features.csv \
        --log_file logs/verify_t5_statistical_features.log

Checks:
    - Required columns and numeric data types
    - Valid binary flags
    - token_relative_frequency calculation (sampled sentences)
    - Special-token conventions (zero statistical values)
    - word_sentence_frequency propagation and recomputation
    - token_rank dense-rank recomputation
    - word_burstiness recomputation and bounds
    - Summary statistics

Note:
    This verifier does NOT enforce num_subtokens == -1 for special tokens,
    because the provided T5 statistical-feature script does not impose that rule.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logger(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", encoding="utf-8")

    def log(msg="", end="\n"):
        s = str(msg)
        print(s, end=end)
        f.write(s + end)
        f.flush()

    return log, f


def fatal(log, fhandle, msg):
    log("[ERROR] " + msg)
    log("[ERROR] Verification FAILED.")
    fhandle.close()
    sys.exit(1)


def approx_equal(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def clean_t5_token(token):
    token = str(token)
    return token[1:] if token.startswith("▁") else token


def compute_burstiness(counts):
    counts = np.asarray(list(counts), dtype=float)
    if len(counts) == 0:
        return 0.0
    mu = counts.mean()
    sigma = counts.std()  # population standard deviation, matching compute script
    if mu + sigma == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def parse_word_id(value):
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return np.nan
    try:
        v = float(s)
        if np.isfinite(v) and v.is_integer():
            return int(v)
    except (TypeError, ValueError):
        pass
    return np.nan


def main():
    parser = argparse.ArgumentParser(description="Verify T5 statistical features output")
    parser.add_argument("--features_csv", required=True,
                       help="CSV produced by compute_t5_statistical_features.py")
    parser.add_argument("--log_file", required=True,
                       help="Log file to write verification logs")
    args = parser.parse_args()

    log, fhandle = setup_logger(args.log_file)
    log(f"[INFO] Logging -> {args.log_file}")
    log("[INFO] Starting verification of T5 statistical features")

    try:
        df = pd.read_csv(args.features_csv, sep=";", keep_default_na=False, na_values=[""])
    except Exception as e:
        fatal(log, fhandle, f"Cannot read CSV: {e}")

    log(f"[INFO] Rows loaded: {len(df)}")
    expected = [
        "sentence_id", "t5_index", "t5_token", "word_id", "is_special_token",
        "token_length", "is_subword", "position_in_sentence", "relative_position",
        "num_subtokens", "token_relative_frequency", "word_sentence_frequency",
        "token_rank", "word_burstiness", "dataset"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        fatal(log, fhandle, f"Missing expected columns: {missing}")
    log(f"[OK] All required columns found ({len(expected)})")

    def numeric(col, dtype="float"):
        s = df[col].replace("", np.nan)
        converted = pd.to_numeric(s, errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            sample = df.loc[invalid, ["sentence_id", "t5_index", "t5_token", col]].head(10)
            fatal(log, fhandle, f"Column {col} has missing/non-numeric values. Examples:\n{sample.to_string(index=False)}")
        if dtype == "int":
            non_integer = (converted % 1 != 0)
            if non_integer.any():
                fatal(log, fhandle, f"Column {col} contains non-integer values")
            return converted.astype(int)
        return converted.astype(float)

    # Numeric columns
    sentence_id = numeric("sentence_id", "int")
    t5_index = numeric("t5_index", "int")
    position = numeric("position_in_sentence", "int")
    token_length = numeric("token_length", "int")
    num_subtokens = numeric("num_subtokens", "int")
    relative_position = numeric("relative_position")
    token_rel_freq = numeric("token_relative_frequency")
    word_sent_freq = numeric("word_sentence_frequency", "int")
    token_rank = numeric("token_rank", "int")
    word_burst = numeric("word_burstiness")

    # Flags: accept 0/1 and boolean-like strings
    def parse_flag(col):
        s = df[col].astype(str).str.strip().str.lower()
        mapping = {"0": 0, "1": 1, "false": 0, "true": 1}
        bad = ~s.isin(mapping)
        if bad.any():
            sample = df.loc[bad, ["sentence_id", "t5_index", "t5_token", col]].head(10)
            fatal(log, fhandle, f"{col} has invalid flag values. Examples:\n{sample.to_string(index=False)}")
        return s.map(mapping).astype(int)

    is_special = parse_flag("is_special_token")
    is_subword = parse_flag("is_subword")
    tokens = df["t5_token"].astype(str)

    # 1. IDs/positions and numeric validity
    if (t5_index < 0).any() or (position < 0).any():
        fatal(log, fhandle, "Negative t5_index or position_in_sentence detected")
    if not np.isfinite(relative_position).all():
        fatal(log, fhandle, "relative_position contains non-finite values")
    if not np.isfinite(token_rel_freq).all() or not np.isfinite(word_burst).all():
        fatal(log, fhandle, "Statistical features contain NaN or infinite values")
    log("[OK] IDs, positions and numeric values are valid")

    # 2. Token indexing should be unique within each sentence
    duplicated = df.duplicated(subset=["sentence_id", "t5_index"], keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, ["sentence_id", "t5_index", "t5_token"]].head(10)
        fatal(log, fhandle, f"Duplicate (sentence_id, t5_index) pairs found:\n{sample.to_string(index=False)}")
    log("[OK] (sentence_id, t5_index) pairs are unique")

    # 3. Statistical-feature range checks
    if ((token_rel_freq < -1e-9) | (token_rel_freq > 1 + 1e-9)).any():
        fatal(log, fhandle, "token_relative_frequency outside [0,1]")
    if (word_sent_freq < 0).any() or (token_rank < 0).any():
        fatal(log, fhandle, "Negative word_sentence_frequency or token_rank values")
    log("[OK] Statistical feature ranges are valid")

    # 4. Special-token conventions (matching the T5 calculation script)
    specials = df[is_special == 1]
    if not specials.empty:
        idx = specials.index
        if (token_rel_freq.loc[idx] != 0).any():
            fatal(log, fhandle, "Special tokens have token_relative_frequency != 0")
        if (token_rank.loc[idx] != 0).any():
            fatal(log, fhandle, "Special tokens have token_rank != 0")
        if (word_sent_freq.loc[idx] != 0).any():
            fatal(log, fhandle, "Special tokens have word_sentence_frequency != 0")
    log("[OK] Special tokens have zero token-relative frequency, rank and word sentence frequency")

    # 5. token_relative_frequency recomputation, first 200 sentences
    log("[INFO] Recomputing token_relative_frequency for up to the first 200 sentences")
    bad = []
    sample_sents = df["sentence_id"].drop_duplicates().head(200).tolist()
    for sid in tqdm(sample_sents, desc="checking T5 token relative frequency"):
        g = df[df["sentence_id"] == sid]
        non_special = g[is_special.loc[g.index] == 0]
        total = len(non_special)
        if total == 0:
            continue
        counts = non_special["t5_token"].value_counts()
        expected_freq = counts / total
        for idx, row in non_special.iterrows():
            exp = float(expected_freq[row["t5_token"]])
            act = float(token_rel_freq.loc[idx])
            if not approx_equal(act, exp, tol=1e-6):
                bad.append((sid, row["t5_token"], exp, act))
                if len(bad) >= 10:
                    break
        if len(bad) >= 10:
            break
    if bad:
        fatal(log, fhandle, f"token_relative_frequency mismatches (examples): {bad}")
    log("[OK] token_relative_frequency verified on sampled sentences")

    # 6. Reconstruct spaCy words using word_id and SentencePiece ▁ marker
    log("[INFO] Reconstructing words to verify word-level statistics")
    work = df.copy()
    work["_word_id"] = work["word_id"].apply(parse_word_id)
    work["_clean_piece"] = work["t5_token"].apply(clean_t5_token)
    word_rows = work[work["_word_id"].notna()].copy()
    word_rows["_word_id"] = word_rows["_word_id"].astype(int)
    reconstructed = (word_rows.groupby(["sentence_id", "_word_id"], sort=False)["_clean_piece"]
                    .apply(lambda pieces: "".join(pieces)).reset_index(name="word_form"))

    # 6a. word_sentence_frequency: distinct sentences containing each word form
    expected_wsf = reconstructed.groupby("word_form")["sentence_id"].nunique()
    reconstructed["_expected_wsf"] = reconstructed["word_form"].map(expected_wsf)
    wsf_map = reconstructed.set_index(["sentence_id", "_word_id"])["_expected_wsf"]
    actual_word_groups = work[work["_word_id"].notna()].copy()
    actual_word_groups["_word_id"] = actual_word_groups["_word_id"].astype(int)
    inconsistent = []
    mismatched_wsf = []
    for (sid, wid), g in actual_word_groups.groupby(["sentence_id", "_word_id"]):
        vals = g["word_sentence_frequency"].astype(int).unique()
        if len(vals) != 1:
            inconsistent.append((sid, wid, vals.tolist()))
        key = (sid, wid)
        exp = wsf_map.get(key, 0)
        if int(is_special.loc[g.index[0]]) == 1:
            exp = 0
        if len(vals) == 1 and int(vals[0]) != int(exp):
            mismatched_wsf.append((sid, wid, int(vals[0]), int(exp)))
    if inconsistent:
        fatal(log, fhandle, f"word_sentence_frequency inconsistent within word groups: {inconsistent[:10]}")
    if mismatched_wsf:
        fatal(log, fhandle, f"word_sentence_frequency does not match reconstructed-word computation: {mismatched_wsf[:10]}")
    log("[OK] word_sentence_frequency propagation and recomputation verified")

    # 7. token_rank recomputation using dense rank, excluding special tokens
    log("[INFO] Recomputing token_rank (dense rank, global corpus)")
    non_special_mask = is_special == 0
    counts = df.loc[non_special_mask].groupby("t5_token").size()
    expected_ranks = counts.rank(method="dense", ascending=False).astype(int)
    rank_mismatches = []
    for tok, expected_rank in expected_ranks.items():
        rows = df.index[(df["t5_token"] == tok) & non_special_mask]
        recorded = token_rank.loc[rows].unique()
        if len(recorded) != 1 or int(recorded[0]) != int(expected_rank):
            rank_mismatches.append((tok, recorded.tolist(), int(expected_rank)))
            if len(rank_mismatches) >= 20:
                break
    if rank_mismatches:
        fatal(log, fhandle, f"token_rank mismatches: {rank_mismatches}")
    log("[OK] token_rank matches dense rank for all non-special T5 tokens")

    # 8. word_burstiness recomputation (matching computation script)
    log("[INFO] Recomputing word_burstiness")
    # word occurrences per word-form per sentence
    word_sentence_counts = reconstructed.groupby(["word_form", "sentence_id"]).size()
    burst_map = {}
    for word, subdf in word_sentence_counts.groupby(level=0):
        counts_for_word = subdf.droplevel(0).values
        burst_map[word] = compute_burstiness(counts_for_word)
    reconstructed["_expected_burst"] = reconstructed["word_form"].map(burst_map)
    burst_map_by_group = reconstructed.set_index(["sentence_id", "_word_id"])["_expected_burst"]
    burst_mismatches = []
    for (sid, wid), g in actual_word_groups.groupby(["sentence_id", "_word_id"]):
        expected_b = float(burst_map_by_group.get((sid, wid), 0.0))
        recorded = word_burst.loc[g.index].astype(float).values
        if not np.allclose(recorded, expected_b, atol=1e-6, rtol=0):
            burst_mismatches.append((sid, wid, recorded[:5].tolist(), expected_b))
            if len(burst_mismatches) >= 10:
                break
    if burst_mismatches:
        fatal(log, fhandle, f"word_burstiness recomputation mismatches: {burst_mismatches}")
    if ((word_burst < -1 - 1e-9) | (word_burst > 1 + 1e-9)).any():
        fatal(log, fhandle, "word_burstiness outside theoretical [-1,1] bounds")
    log("[OK] word_burstiness matches recomputation and lies within [-1,1]")

    # 9. Summary
    log("[INFO] Summary statistics:")
    log(f"  unique sentences: {df['sentence_id'].nunique()}")
    log(f"  unique T5 tokens (surface token strings): {df['t5_token'].nunique()}")
    log(f"  special-token rows: {int(is_special.sum())}")
    log(f"  non-special-token rows: {int((is_special == 0).sum())}")
    log(f"  rows: {len(df)}")
    log(f"  word_id-associated rows: {int(work['_word_id'].notna().sum())}")
    log("[INFO] Verification completed successfully.")
    fhandle.close()


if __name__ == "__main__":
    main()
