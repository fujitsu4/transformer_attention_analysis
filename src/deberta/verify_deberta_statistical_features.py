
"""
verify_deberta_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Verifies outputs of compute_deberta_statistical_features.py

Inputs:
    --features_csv : outputs/deberta/deberta_statistical_features.csv

Outputs:
    --log_file : outputs/logs/deberta_statistical_features_logs.txt

Usage:
    python -m src.deberta.verify_deberta_statistical_features \
        --features_csv outputs/deberta/deberta_statistical_features.csv \
        --log_file outputs/logs/deberta_statistical_features_logs.txt
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
def setup_logger(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    f = open(path, "w", encoding="utf-8")

    def log(msg="", end="\n"):
        s = str(msg)
        print(s, end=end)
        f.write(s + end)
        f.flush()

    return log, f


# -------------------------
def fatal(log, fhandle, msg):
    log("[ERROR] " + msg)
    log("[ERROR] Verification FAILED.")
    fhandle.close()
    sys.exit(1)


# -------------------------
def approx_equal(a, b, tol=1e-9):
    return abs(a - b) <= tol


# -------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Verify DeBERTa statistical features output"
    )

    parser.add_argument(
        "--features_csv",
        required=True,
        help="CSV produced by compute_deberta_statistical_features.py"
    )

    parser.add_argument(
        "--log_file",
        required=True,
        help="Log file to write verification logs"
    )

    args = parser.parse_args()

    log, fhandle = setup_logger(args.log_file)

    log(f"[INFO] Logging -> {args.log_file}")
    log("[INFO] Starting verification of DeBERTa statistical features")

    # ------------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------------

    try:
        df = pd.read_csv(
            args.features_csv,
            sep=";",
            keep_default_na=False,
            na_values=[""]
        )
    except Exception as e:
        fatal(log, fhandle, f"Cannot read CSV: {e}")

    log(f"[INFO] Rows loaded: {len(df)}")

    # ------------------------------------------------------------
    # Expected columns
    # ------------------------------------------------------------

    expected = [
        "sentence_id",
        "deberta_index",
        "deberta_token",
        "word_id",
        "is_special_token",
        "token_length",
        "is_subword",
        "position_in_sentence",
        "relative_position",
        "num_subtokens",
        "token_relative_frequency",
        "word_sentence_frequency",
        "token_rank",
        "word_burstiness",
        "dataset"
    ]

    missing = [c for c in expected if c not in df.columns]

    if missing:
        fatal(
            log,
            fhandle,
            f"Missing expected columns: {missing}"
        )

    log(
        f"[OK] Columns presence: all required columns found ({len(expected)})"
    )

    # ------------------------------------------------------------
    # Numeric conversions
    # ------------------------------------------------------------

    def to_int_series(col, name, allow_na=False):
        try:
            s = df[col].replace({"": np.nan})
            return pd.to_numeric(
                s,
                errors="coerce"
            ).astype("Int64")
        except Exception as e:
            fatal(
                log,
                fhandle,
                f"Column {name} cannot be converted to integer-like: {e}"
            )

    def to_float_series(col, name):
        try:
            s = df[col].replace({"": np.nan})
            return pd.to_numeric(
                s,
                errors="coerce"
            ).astype(float)
        except Exception as e:
            fatal(
                log,
                fhandle,
                f"Column {name} cannot be converted to float: {e}"
            )

    sent_id = to_int_series("sentence_id", "sentence_id")
    deberta_idx = to_int_series("deberta_index", "deberta_index")
    pos_in_sent = to_int_series(
        "position_in_sentence",
        "position_in_sentence"
    )
    token_len = to_int_series("token_length", "token_length")
    num_subtokens = to_int_series("num_subtokens", "num_subtokens")

    # Floats
    rel_pos = to_float_series("relative_position", "relative_position")
    token_rel_freq = to_float_series(
        "token_relative_frequency",
        "token_relative_frequency"
    )
    word_sent_freq = to_int_series(
        "word_sentence_frequency",
        "word_sentence_frequency"
    )
    token_rank = to_int_series("token_rank", "token_rank")
    word_burst = to_float_series(
        "word_burstiness",
        "word_burstiness"
    )

    # Boolean-like fields
    is_special = df["is_special_token"].astype(str).str.strip()
    is_subw = df["is_subword"].astype(str).str.strip()

    # ------------------------------------------------------------
    # 1) sentence_id, deberta_index, position_in_sentence
    # ------------------------------------------------------------

    n = len(df)

    if sent_id.isna().any():
        fatal(log, fhandle, "Some sentence_id values are missing or non-numeric")

    if deberta_idx.isna().any():
        fatal(log, fhandle, "Some deberta_index values are missing or non-numeric")

    if (deberta_idx < 0).any():
        fatal(log, fhandle, "Some deberta_index values are negative")

    if pos_in_sent.isna().any():
        fatal(
            log,
            fhandle,
            "Some position_in_sentence values are missing or non-numeric"
        )

    log(
        "[OK] sentence_id / deberta_index / position_in_sentence "
        "convertible to integers; deberta_index is non-negative"
    )

    # ------------------------------------------------------------
    # 2) is_special_token and is_subword
    # ------------------------------------------------------------

    valid_special = is_special.isin(
        {"0", "1", "True", "False", "true", "false"}
    )

    valid_subword = is_subw.isin(
        {"0", "1", "True", "False", "true", "false"}
    )

    if not valid_special.all():
        sample = df.loc[
            ~valid_special,
            [
                "sentence_id",
                "deberta_index",
                "deberta_token",
                "is_special_token"
            ]
        ].head(10)

        fatal(
            log,
            fhandle,
            "is_special_token contains invalid values.\n"
            + sample.to_string(index=False)
        )

    if not valid_subword.all():
        sample = df.loc[
            ~valid_subword,
            [
                "sentence_id",
                "deberta_index",
                "deberta_token",
                "is_subword"
            ]
        ].head(10)

        fatal(
            log,
            fhandle,
            "is_subword contains invalid values.\n"
            + sample.to_string(index=False)
        )

    is_special_int = (
        is_special
        .replace({
            "True": "1",
            "true": "1",
            "False": "0",
            "false": "0"
        })
        .astype(int)
    )

    is_subw_int = (
        is_subw
        .replace({
            "True": "1",
            "true": "1",
            "False": "0",
            "false": "0"
        })
        .astype(int)
    )

    log(
        "[OK] is_special_token and is_subword contain only 0/1-like values"
    )

    # ------------------------------------------------------------
    # 3) token_relative_frequency in [0,1]
    # ------------------------------------------------------------

    if token_rel_freq.isna().any():
        fatal(
            log,
            fhandle,
            "token_relative_frequency contains NaN values"
        )

    if (
        (token_rel_freq < -1e-9).any()
        or (token_rel_freq > 1.0 + 1e-9).any()
    ):
        bad_mask = (
            (token_rel_freq < -1e-9)
            | (token_rel_freq > 1.0 + 1e-9)
        )

        bad = df.loc[
            bad_mask,
            [
                "sentence_id",
                "deberta_index",
                "deberta_token",
                "token_relative_frequency"
            ]
        ].head(10)

        fatal(
            log,
            fhandle,
            "token_relative_frequency out of [0,1] range.\n"
            + bad.to_string(index=False)
        )

    log("[OK] token_relative_frequency in [0,1] for all rows")

    # ------------------------------------------------------------
    # 4) Special tokens conventions
    # ------------------------------------------------------------

    specials = df[is_special_int == 1]

    if not specials.empty:

        special_indices = specials.index

        bad_rt = specials[
            token_rel_freq.loc[special_indices].astype(float) != 0.0
        ]

        if not bad_rt.empty:
            fatal(
                log,
                fhandle,
                "Some special tokens have token_relative_frequency != 0"
            )

        bad_tr = specials[
            token_rank.loc[special_indices].astype(float) != 0
        ]

        if not bad_tr.empty:
            fatal(
                log,
                fhandle,
                "Some special tokens have token_rank != 0"
            )

        bad_wsf = specials[
            word_sent_freq.loc[special_indices].astype(float) != 0
        ]

        if not bad_wsf.empty:
            fatal(
                log,
                fhandle,
                "Some special tokens have word_sentence_frequency != 0"
            )

        bad_ns = specials[
            num_subtokens.loc[special_indices].astype(float) != -1
        ]

        if not bad_ns.empty:
            fatal(
                log,
                fhandle,
                "Some special tokens do not have num_subtokens == -1"
            )

    log(
        "[OK] Specials respect conventions "
        "(relative_freq=0, token_rank=0, "
        "word_sentence_frequency=0, num_subtokens=-1)"
    )

    # ------------------------------------------------------------
    # 5) token_relative_frequency correctness per sentence
    # ------------------------------------------------------------

    log(
        "[INFO] Verifying token_relative_frequency equals "
        "count(token)/n_tokens per sentence (sampled checks)..."
    )

    bad_count = 0

    sample_sents = list(df["sentence_id"].unique())[:200]

    for sid in tqdm(
        sample_sents,
        desc="checking token_relative_frequency (sample)"
    ):

        g = df[df["sentence_id"] == sid]

        if g.empty:
            continue

        mask_non_special = (
            g["is_special_token"]
            .astype(str)
            .isin(["0", "False", "false"])
        )

        total_non_special = mask_non_special.sum()

        if total_non_special == 0:
            continue

        cnts = (
            g[mask_non_special]["deberta_token"]
            .value_counts()
            .to_dict()
        )

        for tok, expected_count in cnts.items():

            row = g[
                (g["deberta_token"] == tok)
                & mask_non_special
            ].iloc[0]

            expected_rel = expected_count / total_non_special
            actual_rel = float(row["token_relative_frequency"])

            if not approx_equal(
                actual_rel,
                expected_rel,
                tol=1e-6
            ):
                bad_count += 1

                if bad_count <= 5:
                    log(
                        f"[ERROR] token_relative_frequency mismatch "
                        f"in sentence {sid}, token='{tok}' "
                        f"expected={expected_rel:.6f} "
                        f"actual={actual_rel:.6f}"
                    )

        if bad_count:
            break

    if bad_count:
        fatal(
            log,
            fhandle,
            f"token_relative_frequency mismatches found "
            f"(example count {bad_count})"
        )

    log(
        "[OK] token_relative_frequency verified on sampled sentences "
        "(first 200)"
    )

    # ------------------------------------------------------------
    # 6) word_sentence_frequency propagation & consistency
    # ------------------------------------------------------------

    log(
        "[INFO] Verifying word_sentence_frequency propagation "
        "to sub-tokens and integer-ness"
    )

    def parse_wid(v):
        s = str(v).strip()

        if s == "" or s.lower() in ("nan", "none", "null"):
            return None

        try:
            return int(float(s))
        except Exception:
            return None

    wids = df["word_id"].apply(parse_wid)

    df["_word_id_parsed"] = wids

    grouped = (
        df[df["_word_id_parsed"].notna()]
        .groupby(["sentence_id", "_word_id_parsed"])
    )

    inconsistent = []

    for key, g in grouped:

        vals = set(
            g["word_sentence_frequency"].astype(str).tolist()
        )

        if len(vals) > 1:
            inconsistent.append((key, list(vals)[:5]))

    if inconsistent:
        fatal(
            log,
            fhandle,
            "word_sentence_frequency inconsistent within "
            "(sentence_id, word_id). Examples: "
            f"{inconsistent[:10]}"
        )

    log(
        "[OK] word_sentence_frequency is propagated consistently "
        "to sub-tokens"
    )

    # ------------------------------------------------------------
    # 7) token_rank monotonicity check with global counts
    # ------------------------------------------------------------

    log(
        "[INFO] Verifying token_rank mapping using DENSE RANK "
        "(ties = same rank)"
    )

    token_counts = (
        df[df["is_special_token"] == 0]
        .groupby("deberta_token")
        .size()
    )

    dense_rank = (
        token_counts
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    expected_map = dense_rank.to_dict()

    mismatches = []

    for tok, expected in expected_map.items():

        recorded = df.loc[
            (df["deberta_token"] == tok)
            & (df["is_special_token"] == 0),
            "token_rank"
        ].iloc[0]

        if recorded != expected:
            mismatches.append((tok, recorded, expected))

            if len(mismatches) >= 20:
                break

    if mismatches:

        log(
            "[ERROR] token_rank mismatches detected "
            "(dense ranking used for verification):"
        )

        for tok, rec, exp in mismatches:
            log(
                f"  token='{tok}' recorded={rec} expected={exp}"
            )

        fatal(
            log,
            fhandle,
            "token_rank does not match expected dense ranking"
        )

    else:
        log("[OK] token_rank matches dense rank for all tokens.")

    # ------------------------------------------------------------
    # 8) word_burstiness values numeric & within [-1,1]
    # ------------------------------------------------------------

    log(
        "[INFO] Checking word_burstiness numeric and within [-1,1]"
    )

    if word_burst.isna().any():
        fatal(
            log,
            fhandle,
            "word_burstiness contains NaN values"
        )

    if not np.isfinite(word_burst.to_numpy()).all():
        fatal(
            log,
            fhandle,
            "word_burstiness contains infinite values"
        )

    if (
        (word_burst < -1.0 - 1e-9).any()
        or (word_burst > 1.0 + 1e-9).any()
    ):

        bad_mask = (
            (word_burst < -1.0 - 1e-9)
            | (word_burst > 1.0 + 1e-9)
        )

        bad = df.loc[
            bad_mask,
            [
                "sentence_id",
                "deberta_token",
                "word_burstiness"
            ]
        ].head(10)

        fatal(
            log,
            fhandle,
            "Some word_burstiness values are outside [-1,1].\n"
            + bad.to_string(index=False)
        )

    log(
        "[OK] word_burstiness is numeric, finite and within [-1,1]"
    )

    # ------------------------------------------------------------
    # 9) Basic statistics & summary
    # ------------------------------------------------------------

    log("[INFO] Summary statistics:")
    log(f"  unique sentences: {df['sentence_id'].nunique()}")
    log(f"  unique tokens (deberta_token): {df['deberta_token'].nunique()}")
    log(f"  tokens flagged special: {int(is_special_int.sum())}")
    log(f"  num rows: {len(df)}")

    log("[INFO] Verification completed successfully.")

    fhandle.close()


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
