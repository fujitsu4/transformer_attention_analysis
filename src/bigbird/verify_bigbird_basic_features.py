
"""
verify_bigbird_basic_features.py
Author: Zakaria JOUILIL

Description:
    Exhaustive verification of the output of
    compute_bigbird_basic_features.py.

Checks performed:
    - Expected columns and types
    - is_special_token matches token string
    - is_subword matches SentencePiece prefix convention
    - token_length excludes the SentencePiece marker ▁
    - position_in_sentence equals bigbird_index
    - relative_position consistency
    - num_subtokens consistency per (sentence_id, word_id)
    - Special tokens have num_subtokens = -1
    - Dataset column present and last
    - Basic statistics and warnings for suspicious values

Usage:
    python -m src.bigbird.verify_bigbird_basic_features \
        --features_csv outputs/bigbird/bigbird_snli_basic_features.csv \
        --log_file outputs/logs/bigbird_basic_features_logs.txt
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np


# ----------------------------------------------------------
# LOGGER
# ----------------------------------------------------------

def setup_logger(log_file):

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout


# ----------------------------------------------------------
# FATAL ERROR
# ----------------------------------------------------------

def fatal(msg):
    print(f"[ERROR] {msg}")
    print("[ERROR] Verification FAILED.")
    sys.exit(1)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Verify bigbird basic features CSV"
    )

    parser.add_argument(
        "--features_csv",
        required=True,
        help="CSV produced by compute_bigbird_basic_features.py"
    )

    parser.add_argument(
        "--log_file",
        required=True,
        help="Log file path"
    )

    parser.add_argument(
        "--max_subtoken_warn",
        type=int,
        default=50,
        help="Warn if num_subtokens exceeds this threshold"
    )

    parser.add_argument(
        "--relative_tol",
        type=float,
        default=1e-6,
        help="Tolerance for relative position verification"
    )

    args = parser.parse_args()

    setup_logger(args.log_file)

    print(f"[INFO] Logging enabled -> {args.log_file}")
    print("[INFO] Starting verification of bigbird basic features")
    print(f"[INFO] Input CSV: {args.features_csv}")


    # ------------------------------------------------------
    # LOAD CSV
    # ------------------------------------------------------

    try:
        df = pd.read_csv(
            args.features_csv,
            sep=";",
            keep_default_na=False,
            na_values=[""],
            dtype=str
        )
    except Exception as e:
        fatal(f"Cannot read CSV: {e}")

    print(f"[INFO] Rows loaded: {len(df)}")

    if len(df) == 0:
        fatal("Empty CSV")

    df.columns = [c.strip() for c in df.columns]


    # ------------------------------------------------------
    # EXPECTED COLUMNS
    # ------------------------------------------------------

    expected_cols = [
        "sentence_id",
        "bigbird_index",
        "bigbird_token",
        "word_id",
        "is_special_token",
        "token_length",
        "is_subword",
        "position_in_sentence",
        "relative_position",
        "num_subtokens",
        "dataset"
    ]

    missing = [c for c in expected_cols if c not in df.columns]

    if missing:
        fatal(f"Missing required columns: {missing}")

    print("[OK] Required columns present")

    if df.columns[-1] != "dataset":
        print("[WARN] dataset column is not the last column")


    # ------------------------------------------------------
    # TYPE CONVERSIONS
    # ------------------------------------------------------

    int_cols = [
        "sentence_id",
        "bigbird_index",
        "position_in_sentence"
    ]

    converted = {}

    for col in int_cols:
        converted[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).astype("Int64")

        if converted[col].isna().any():
            fatal(f"Invalid integer values in {col}")

    tokens = df["bigbird_token"].fillna("").astype(str)

    for col in ["is_special_token", "is_subword"]:

        vals = df[col].fillna("").astype(str).str.strip()

        invalid = vals[
            ~vals.isin({"0", "1", "True", "False", "true", "false"})
        ]

        if len(invalid) > 0:
            fatal(
                f"Invalid values in {col}: {invalid.unique()[:10]}"
            )

    is_special = (
        df["is_special_token"]
        .astype(str).str.strip()
        .replace({
            "True": "1", "true": "1",
            "False": "0", "false": "0"
        })
        .astype(int)
    )

    is_subword = (
        df["is_subword"]
        .astype(str).str.strip()
        .replace({
            "True": "1", "true": "1",
            "False": "0", "false": "0"
        })
        .astype(int)
    )

    try:
        token_length_col = pd.to_numeric(
            df["token_length"], errors="coerce"
        ).astype("Int64")

        rel_pos = pd.to_numeric(
            df["relative_position"], errors="coerce"
        )

        num_subtokens = pd.to_numeric(
            df["num_subtokens"], errors="coerce"
        )

    except Exception as e:
        fatal(f"Numeric conversion error: {e}")

    if token_length_col.isna().any():
        fatal("Invalid token_length values")

    if rel_pos.isna().any():
        fatal("Invalid relative_position values")

    if num_subtokens.isna().any():
        fatal("Invalid num_subtokens values")

    df["word_id"] = pd.to_numeric(
        df["word_id"], errors="coerce"
    ).astype("Int64")


    # ------------------------------------------------------
    # 1) SPECIAL TOKEN CHECK
    # ------------------------------------------------------

    special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}

    mask_special = tokens.isin(special_tokens).astype(int)

    mismatch_special = mask_special != is_special

    if mismatch_special.any():

        print(
            f"[ERROR] is_special_token mismatch: "
            f"{mismatch_special.sum()} rows"
        )

        for i in df.index[mismatch_special][:5]:
            print(
                f"idx={i}, token={tokens.iat[i]}, "
                f"is_special_token={is_special.iat[i]}"
            )

        fatal("is_special_token misalignment")

    print("[OK] is_special_token matches token strings")


    # ------------------------------------------------------
    # 2) IS_SUBWORD CHECK
    # ------------------------------------------------------

    # SentencePiece convention:
    # ▁ indicates the beginning of a word.
    # Tokens without ▁ are subsequent subword pieces.
    # Special tokens are never considered subwords.

    mask_subword = (
        ~tokens.str.startswith("▁")
        & (is_special == 0)
    ).astype(int)

    mismatch_sub = mask_subword != is_subword

    if mismatch_sub.any():

        print(
            f"[ERROR] is_subword mismatch: "
            f"{mismatch_sub.sum()} rows"
        )

        for i in df.index[mismatch_sub][:10]:
            print(
                f"idx={i}, token={tokens.iat[i]}, "
                f"is_subword={is_subword.iat[i]}"
            )

        fatal("is_subword misalignment")

    print("[OK] is_subword consistent with SentencePiece convention")


    # ------------------------------------------------------
    # 3) TOKEN LENGTH CHECK
    # ------------------------------------------------------

    # Exclude ▁ before calculating token length.

    clean_tokens = tokens.str.replace(
        r"^▁", "", regex=True
    )

    calc_len = clean_tokens.str.len().astype(int)

    mismatch_len = (
        calc_len != token_length_col.astype(int)
    )

    if mismatch_len.any():

        print(
            f"[ERROR] token_length mismatch: "
            f"{mismatch_len.sum()} rows"
        )

        for i in df.index[mismatch_len][:10]:
            print(
                f"idx={i}, token={tokens.iat[i]}, "
                f"expected={calc_len.iat[i]}, "
                f"reported={token_length_col.iat[i]}"
            )

        fatal("token_length mismatch")

    print("[OK] token_length matches cleaned token length")


    # ------------------------------------------------------
    # 4) POSITION CHECK
    # ------------------------------------------------------

    pos_mismatch = (
        converted["bigbird_index"]
        != converted["position_in_sentence"]
    )

    if pos_mismatch.any():
        fatal("position_in_sentence != bigbird_index")

    print("[OK] position_in_sentence equals bigbird_index")


    # ------------------------------------------------------
    # 5) RELATIVE POSITION CHECK
    # ------------------------------------------------------

    print("[INFO] Verifying relative positions...")

    df["_bigbird_index_int"] = (
        converted["bigbird_index"].astype(int)
    )

    bad_rel = 0

    for sid, g in df.groupby("sentence_id"):

        idxs = g.index
        max_idx = g["_bigbird_index_int"].max()

        if max_idx <= 0:
            continue

        expected = (
            g["_bigbird_index_int"] / float(max_idx)
        )

        actual = rel_pos.loc[idxs]

        diff = (expected - actual).abs()

        if (diff > args.relative_tol).any():

            bad = idxs[diff > args.relative_tol].tolist()[:5]

            print(
                f"[ERROR] Relative position mismatch "
                f"in sentence {sid}: {bad}"
            )

            bad_rel += len(bad)
            break

    if bad_rel:
        fatal(f"{bad_rel} relative_position mismatches found.")

    print("[OK] relative_position consistent with bigbird_index / max_index")


    # ------------------------------------------------------
    # 6) NUM_SUBTOKENS CHECK
    # ------------------------------------------------------

    print("[INFO] Checking num_subtokens consistency...")

    grouped = df[df["word_id"].notna()].copy()

    if grouped.empty:

        print("[WARN] No valid word_id found.")

    else:

        grp_counts = (
            grouped
            .groupby(["sentence_id", "word_id"])
            .size()
            .rename("count")
            .reset_index()
        )

        lookup = {
            (int(r["sentence_id"]), int(r["word_id"])): int(r["count"])
            for _, r in grp_counts.iterrows()
        }

        inconsistencies = []
        big_values = []

        for idx, row in grouped.iterrows():

            sid = int(row["sentence_id"])
            wid = int(row["word_id"])

            reported = int(float(row["num_subtokens"]))
            expected = lookup.get((sid, wid))

            if expected is None or reported != expected:
                inconsistencies.append(
                    (idx, sid, wid, reported, expected)
                )

            if expected is not None and expected > args.max_subtoken_warn:
                big_values.append((sid, wid, expected))

        if inconsistencies:

            print(
                f"[ERROR] num_subtokens mismatch: "
                f"{len(inconsistencies)} rows"
            )

            for item in inconsistencies[:10]:
                print(item)

            fatal("num_subtokens grouping inconsistent")

        print("[OK] num_subtokens matches word-level token counts")

        if big_values:

            print("[WARN] Unusually large num_subtokens:")

            for item in big_values[:10]:
                print(item)

            print(
                f"[WARN] Threshold: {args.max_subtoken_warn}"
            )


    # ------------------------------------------------------
    # 7) SPECIAL TOKENS NUM_SUBTOKENS CHECK
    # ------------------------------------------------------

    specials_df = df[is_special == 1]

    if not specials_df.empty:

        bad_specials = specials_df[
            specials_df["num_subtokens"].astype(float) != -1.0
        ]

        if not bad_specials.empty:

            print("[ERROR] Special tokens with num_subtokens != -1")

            for _, row in bad_specials.head(10).iterrows():
                print(
                    row["bigbird_token"],
                    row["num_subtokens"]
                )

            fatal("Special tokens must have num_subtokens = -1")

    print("[OK] Special tokens have num_subtokens = -1")


    # ------------------------------------------------------
    # 8) NUM_SUBTOKENS BOUNDS
    # ------------------------------------------------------

    if (num_subtokens < -1).any():
        fatal("num_subtokens contains values < -1")

    above_reasonable = num_subtokens[
        num_subtokens > args.max_subtoken_warn
    ]

    if len(above_reasonable):

        print(
            f"[WARN] {len(above_reasonable)} tokens have "
            f"num_subtokens > {args.max_subtoken_warn}"
        )

    else:
        print("[OK] num_subtokens values within reasonable bounds")


    # ------------------------------------------------------
    # 9) EMPTY TOKEN CHECK
    # ------------------------------------------------------

    empty_tokens = tokens.str.strip() == ""

    if empty_tokens.any():

        idxs = df.index[empty_tokens][:10].tolist()

        print("[ERROR] Empty tokens:", idxs)

        fatal("Empty bigbird_token strings found")

    print("[OK] No empty bigbird_token values")


    # ------------------------------------------------------
    # 10) CONSISTENCY WITHIN WORD GROUPS
    # ------------------------------------------------------

    pairs = df[df["word_id"].notna()][
        ["sentence_id", "word_id", "num_subtokens"]
    ].copy()

    if not pairs.empty:

        coll = (
            pairs
            .groupby(["sentence_id", "word_id"])["num_subtokens"]
            .nunique()
        )

        bad = coll[coll > 1]

        if not bad.empty:

            print(
                "[ERROR] Inconsistent num_subtokens "
                "within the same word group"
            )

            for (sid, wid), nvals in bad.head(10).items():
                print(
                    f"sentence={sid}, word_id={wid}, "
                    f"distinct values={nvals}"
                )

            fatal("Inconsistent num_subtokens within word groups")

    print("[OK] num_subtokens consistent within word groups")


    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n[INFO] === VERIFICATION COMPLETE ===")
    print("[INFO] All checks passed (or warnings reported).")
    print(f"[INFO] Detailed log written to: {args.log_file}")

    return 0


if __name__ == "__main__":
    main()
