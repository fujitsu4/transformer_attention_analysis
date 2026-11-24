
"""
verify_longformer_final_features.py
Author: Zakaria JOUILIL

Description:
    Exhaustive verification script for the final Longformer feature file
    (longformer_snli_final_features.csv), used as input for the Decision Tree.

    It checks:
      - Exact column order and presence
      - Types and ranges for each column
      - Per-sentence invariants
      - Cross-feature consistency
      - Special-token conventions
      - num_subtokens grouping
      - Propagation of word-level statistics to sub-tokens
      - prev/next POS neighbor correctness
      - Statistical sanity

Input:
    outputs/longformer/longformer_snli_final_features.csv

Output:
    A log file containing the verification report.

Usage:
    python -m src.longformer.verify_longformer_final_features \
        --input outputs/longformer/longformer_snli_final_features.csv \
        --log_file outputs/logs/longformer_final_logs.txt
"""

import argparse
import sys
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from tqdm import tqdm


# -------------------------------------------------------------------
# Logger helper
# -------------------------------------------------------------------

def setup_logger(log_file_path):
    """Redirect stdout/stderr to both console and a log file."""
    os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
    log_f = open(log_file_path, "w", encoding="utf-8")

    class Logger:
        def __init__(self, terminal, logfile):
            self.terminal = terminal
            self.log = logfile

        def write(self, msg):
            self.terminal.write(msg)
            self.log.write(msg)
            self.log.flush()

        def flush(self):
            try:
                self.terminal.flush()
            except Exception:
                pass
            try:
                self.log.flush()
            except Exception:
                pass

    sys.stdout = Logger(sys.stdout, log_f)
    sys.stderr = sys.stdout


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def info(msg):
    print(f"[{ts()}] [INFO] {msg}")


def ok(msg):
    print(f"[{ts()}] [OK] {msg}")


def warn(msg):
    print(f"[{ts()}] [WARN] {msg}")


def error(msg):
    print(f"[{ts()}] [ERROR] {msg}")


def fatal(msg, code=1):
    error(msg)
    error("Verification FAILED (fatal). Exiting.")
    sys.exit(code)


# -------------------------------------------------------------------
# Main verification
# -------------------------------------------------------------------

def verify(input_csv, log_file, relative_tol=1e-6, sample_sentences=200):

    setup_logger(log_file)

    info(f"Logging -> {log_file}")
    info("=== VERIFYING LONGFORMER FINAL FEATURES ===")
    info(f"Reading CSV: {input_csv}")

    try:
        df = pd.read_csv(
            input_csv,
            sep=";",
            keep_default_na=False,
            na_values=[""]
        )
    except Exception as e:
        fatal(f"Cannot read input CSV '{input_csv}': {e}")

    # ----------------------------------------------------------------
    # Expected exact column order
    # ----------------------------------------------------------------

    expected_cols = [
        "sentence_id", "longformer_index", "longformer_token", "word_id",
        "is_numeric", "is_punctuation", "is_stopword", "is_pronoun",
        "pos_id", "dep_id", "ent_id", "depth_in_tree", "num_dependents",
        "is_special_token", "token_length", "is_subword",
        "position_in_sentence", "relative_position", "num_subtokens",
        "token_relative_frequency", "word_sentence_frequency",
        "token_rank", "word_burstiness", "prev_pos_id", "next_pos_id",
        "dataset"
    ]

    info("1) Checking columns presence and exact order")

    got_cols = list(df.columns)

    if got_cols != expected_cols:
        error("Column mismatch or wrong order.")
        error(f"Expected: {expected_cols}")
        error(f"Found:    {got_cols}")

        missing = [c for c in expected_cols if c not in got_cols]
        extra = [c for c in got_cols if c not in expected_cols]

        if missing:
            error(f"Missing columns: {missing}")

        if extra:
            error(f"Unexpected extra columns: {extra}")

        fatal("Columns are not exactly as expected.")

    ok("Column names and order are EXACTLY correct.")

    n = len(df)
    info(f"Total rows: {n}")

    # ----------------------------------------------------------------
    # Basic type normalization
    # ----------------------------------------------------------------

    info("2) Basic type normalization and null checks")

    int_cols = [
        "sentence_id",
        "longformer_index",
        "position_in_sentence"
    ]

    for col in int_cols:
        try:
            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            ).astype(int)
        except Exception as e:
            fatal(f"Column {col} must be integer-like. Error: {e}")

    # Longformer uses RoBERTa-style byte-level BPE.
    # Remove a possible leading Ġ marker only for surface-form checks.
    # Preserve original tokens in the dataframe.
    df["longformer_token"] = df["longformer_token"].astype(str)

    clean_tokens = df["longformer_token"].str.replace(
        r"^Ġ",
        "",
        regex=True
    )

    df["dataset"] = df["dataset"].astype(str)

    # ----------------------------------------------------------------
    # Boolean-like fields
    # ----------------------------------------------------------------

    bin_cols = [
        "is_numeric",
        "is_punctuation",
        "is_stopword",
        "is_pronoun",
        "is_special_token",
        "is_subword"
    ]

    for col in bin_cols:

        s = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({
                "True": "1",
                "true": "1",
                "False": "0",
                "false": "0"
            })
        )

        nums = pd.to_numeric(s, errors="coerce")

        if nums.isna().any():
            sample_bad = s[nums.isna()].unique()[:10].tolist()
            fatal(
                f"Column {col} contains values that cannot be parsed "
                f"as 0/1. Examples: {sample_bad}"
            )

        nums = nums.round().astype(int)

        unique_vals = set(nums.unique().tolist())

        if not unique_vals.issubset({0, 1}):
            fatal(
                f"Column {col} contains values outside {{0,1}}. "
                f"Found: {unique_vals}"
            )

        df[col] = nums

    ok("Boolean-like columns are 0/1 and normalized.")

    # ----------------------------------------------------------------
    # Float fields
    # ----------------------------------------------------------------

    float_cols = [
        "relative_position",
        "token_relative_frequency",
        "word_burstiness"
    ]

    for col in float_cols:
        try:
            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            ).astype(float)
        except Exception as e:
            fatal(f"Column {col} must be numeric. Error: {e}")

    # ----------------------------------------------------------------
    # Integer-like fields
    # ----------------------------------------------------------------

    int_like_cols = [
        "word_id",
        "pos_id",
        "dep_id",
        "ent_id",
        "depth_in_tree",
        "num_dependents",
        "num_subtokens",
        "token_rank",
        "word_sentence_frequency",
        "prev_pos_id",
        "next_pos_id"
    ]

    for col in int_like_cols:

        if col == "word_id":
            df[col] = df[col].replace("", np.nan)

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
        else:
            try:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="raise"
                ).astype(int)
            except Exception as e:
                fatal(
                    f"Column {col} must be integer-like. Error: {e}"
                )

    ok("Numeric columns converted and validated.")

    # ----------------------------------------------------------------
    # 3) Primary invariants
    # ----------------------------------------------------------------

    info("3) Checking primary invariants")

    # a) Longformer index equals position_in_sentence

    mism = (
        df["longformer_index"]
        != df["position_in_sentence"]
    )

    if mism.any():
        sample = df.loc[
            mism,
            [
                "sentence_id",
                "longformer_index",
                "position_in_sentence",
                "longformer_token"
            ]
        ].head(10)

        error("Mismatch: longformer_index != position_in_sentence.")
        print(sample.to_string(index=False))

        fatal(
            "longformer_index and position_in_sentence "
            "must be identical."
        )

    ok("longformer_index == position_in_sentence.")

    # b) token_relative_frequency in [0,1]

    if (
        df["token_relative_frequency"].lt(0).any()
        or df["token_relative_frequency"].gt(1).any()
    ):
        fatal("token_relative_frequency must be within [0,1].")

    ok("token_relative_frequency values are within [0,1].")

    # c) word_burstiness finite and in [-1,1]

    if not np.isfinite(df["word_burstiness"]).all():
        fatal("word_burstiness contains non-finite values.")

    if (
        (df["word_burstiness"] < -1.0 - 1e-9).any()
        or (df["word_burstiness"] > 1.0 + 1e-9).any()
    ):
        fatal("word_burstiness must be within [-1,1].")

    ok("word_burstiness values are finite and within [-1,1].")

    # d) token_rank conventions

    non_special_ranks = df.loc[
        df["is_special_token"] == 0,
        "token_rank"
    ]

    if (non_special_ranks < 1).any():

        bad = df.loc[
            (df["is_special_token"] == 0)
            & (df["token_rank"] < 1)
        ].head(10)

        error("Non-special tokens must have token_rank >= 1.")

        print(
            bad[
                ["sentence_id", "longformer_token", "token_rank"]
            ].to_string(index=False)
        )

        fatal("token_rank invalid for some non-special tokens.")

    specials_with_nonzero_rank = df.loc[
        (df["is_special_token"] == 1)
        & (df["token_rank"] != 0)
    ]

    if not specials_with_nonzero_rank.empty:

        warn(
            f"{len(specials_with_nonzero_rank)} special tokens "
            "have token_rank != 0 (expected convention: 0)."
        )

        print(
            specials_with_nonzero_rank.head(10)[
                ["sentence_id", "longformer_token", "token_rank"]
            ].to_string(index=False)
        )

    ok("token_rank convention checked.")

    # e) Special tokens must have neighbor POS values equal to -1

    specials = df[df["is_special_token"] == 1]

    if not specials.empty:

        bad_prev = specials.loc[
            specials["prev_pos_id"] != -1
        ]

        bad_next = specials.loc[
            specials["next_pos_id"] != -1
        ]

        if not bad_prev.empty or not bad_next.empty:

            error(
                "Special tokens must have prev_pos_id and "
                "next_pos_id == -1."
            )

            fatal("Special neighbors convention violated.")

    ok("Special tokens have prev_pos_id and next_pos_id == -1.")

    # f) Non-special spaCy fields must not be missing

    non_specials = df[df["is_special_token"] == 0]

    required_non_special_fields = [
        "pos_id",
        "dep_id",
        "ent_id",
        "depth_in_tree",
        "num_dependents",
        "num_subtokens",
        "word_sentence_frequency",
        "prev_pos_id",
        "next_pos_id"
    ]

    for col in required_non_special_fields:

        if non_specials[col].isna().any():
            fatal(
                f"Non-special tokens contain missing values in {col}."
            )

    ok("Required non-special feature fields contain no missing values.")

    # g) word_id must be valid for all non-special tokens

    ns = df[df["is_special_token"] == 0]

    if ns["word_id"].isna().any():

        bad = ns[ns["word_id"].isna()].head(10)

        error("Some non-special tokens have missing word_id.")

        print(
            bad[
                ["sentence_id", "longformer_index", "longformer_token"]
            ].to_string(index=False)
        )

        fatal("word_id missing for non-special tokens.")

    if (ns["word_id"] < 0).any():
        fatal("Non-special tokens have negative word_id.")

    ok("word_id is present and non-negative for all non-special tokens.")

    # ----------------------------------------------------------------
    # 4) Per-sentence checks
    # ----------------------------------------------------------------

    info("4) Per-sentence checks")

    sent_ids = sorted(df["sentence_id"].unique())

    info(f"Unique sentences: {len(sent_ids)}")

    # a) token_relative_frequency consistency

    for sid, group in df.groupby("sentence_id"):

        non_special = group[group["is_special_token"] == 0]

        if non_special.empty:
            continue

        N = len(non_special)

        counts = non_special["longformer_token"].value_counts()

        for tok, expected_freq in (counts / N).items():

            actual = non_special.loc[
                non_special["longformer_token"] == tok,
                "token_relative_frequency"
            ]

            if not np.allclose(
                actual,
                expected_freq,
                atol=relative_tol
            ):
                fatal(
                    f"token_relative_frequency mismatch in sentence "
                    f"{sid} for token '{tok}': expected "
                    f"{expected_freq}, got {actual.unique()}"
                )

    ok("token_relative_frequency is consistent per sentence.")

    # b) num_subtokens grouping consistency

    info("4.b) Checking num_subtokens grouping consistency.")

    grouped = (
        df[df["is_special_token"] == 0]
        .groupby(["sentence_id", "word_id"])
    )

    inconsistent = []

    for (sid, wid), group in grouped:

        reported_vals = pd.unique(
            group["num_subtokens"].astype(int)
        )

        if len(reported_vals) != 1:

            inconsistent.append(
                (sid, wid, reported_vals.tolist(), len(group))
            )

        else:

            reported = int(reported_vals[0])
            expected = len(group)

            if reported != expected:
                inconsistent.append(
                    (sid, wid, reported, expected)
                )

        if len(inconsistent) > 10:
            break

    if inconsistent:

        error(
            "Inconsistent num_subtokens for some "
            "(sentence_id, word_id)."
        )

        for example in inconsistent[:10]:
            print("  ", example)

        fatal("num_subtokens inconsistency detected.")

    ok("num_subtokens equals group size for each word.")

    # c) word_sentence_frequency propagation

    info("4.c) Checking word_sentence_frequency propagation.")

    grouped_wsf = (
        df[df["is_special_token"] == 0]
        .groupby(["sentence_id", "word_id"])[
            "word_sentence_frequency"
        ]
        .nunique()
    )

    bad = grouped_wsf[grouped_wsf > 1]

    if not bad.empty:

        error(
            "word_sentence_frequency is not consistently propagated "
            "to all subtokens of a word."
        )

        print(bad.head(10).to_string())

        fatal("word_sentence_frequency propagation inconsistency.")

    ok("word_sentence_frequency is consistent across subtokens.")

    # d) prev_pos_id / next_pos_id correctness

    info("4.d) Checking neighbor POS correctness on sampled sentences.")

    neighbor_errors = []

    for sid in tqdm(
        sent_ids[:sample_sentences],
        desc="checking neighbors"
    ):

        sub = df[df["sentence_id"] == sid]

        nonsp = sub[sub["is_special_token"] == 0]

        if nonsp.empty:
            continue

        grouped_word = (
            nonsp.groupby("word_id")["pos_id"].nunique()
        )

        if (grouped_word > 1).any():

            neighbor_errors.append(
                (sid, "pos_id inconsistent across subtokens")
            )

            continue

        rep = (
            nonsp.groupby("word_id")["pos_id"]
            .first()
            .to_dict()
        )

        for _, row in nonsp.iterrows():

            wid = int(row["word_id"])

            expected_prev = rep.get(wid - 1, -1)
            expected_next = rep.get(wid + 1, -1)

            if int(row["prev_pos_id"]) != int(expected_prev):

                neighbor_errors.append(
                    (
                        sid,
                        row["longformer_index"],
                        "prev",
                        wid,
                        int(row["prev_pos_id"]),
                        int(expected_prev)
                    )
                )

            if int(row["next_pos_id"]) != int(expected_next):

                neighbor_errors.append(
                    (
                        sid,
                        row["longformer_index"],
                        "next",
                        wid,
                        int(row["next_pos_id"]),
                        int(expected_next)
                    )
                )

            if len(neighbor_errors) > 10:
                break

        if neighbor_errors:
            break

    if neighbor_errors:

        error("Neighbor POS mismatches detected.")

        for example in neighbor_errors[:10]:
            print("  ", example)

        fatal("prev_pos_id / next_pos_id verification failed.")

    ok("Neighbor POS features are consistent on sampled sentences.")

    # e) Contiguous word_id ranges

    info("4.e) Checking contiguous word_id ranges per sentence.")

    gaps = []

    for sid in sent_ids[:sample_sentences]:

        sub = df[
            (df["sentence_id"] == sid)
            & (df["is_special_token"] == 0)
        ]

        if sub.empty:
            continue

        wids = sorted(
            sub["word_id"].dropna().astype(int).unique().tolist()
        )

        if not wids:
            continue

        expected_wids = list(
            range(wids[0], wids[-1] + 1)
        )

        if wids[0] != 0 or wids != expected_wids:

            gaps.append(
                (sid, wids[:5], wids[-5:])
            )

        if len(gaps) >= 5:
            break

    if gaps:

        warn(
            "Found sentences with non-contiguous or non-zero-starting "
            "word_id ranges. Examples:"
        )

        for example in gaps[:5]:
            print("  ", example)

        warn(
            "Investigate if the pipeline expects contiguous "
            "0..N word indexing."
        )

    else:

        ok("word_id ranges appear contiguous and start at 0.")

    # ----------------------------------------------------------------
    # 5) Cross-feature logical checks
    # ----------------------------------------------------------------

    info("5) Cross-feature logical checks")

    # Rule A: is_pronoun implies valid pos_id

    pronoun_viol = df[
        (df["is_pronoun"] == 1)
        & (df["pos_id"] < 0)
    ]

    if not pronoun_viol.empty:

        error("Some pronouns have invalid pos_id.")

        print(
            pronoun_viol.head(10)[
                [
                    "sentence_id",
                    "longformer_index",
                    "longformer_token",
                    "is_pronoun",
                    "pos_id"
                ]
            ].to_string(index=False)
        )

        fatal("is_pronoun implies invalid pos_id.")

    ok("is_pronoun implies valid pos_id.")

    # Rule B: punctuation flags vs token content

    punc_viol = df[
        (df["is_punctuation"] == 1)
        & (~clean_tokens.str.match(r"^\W+$", na=False))
    ]

    if len(punc_viol) > 0:

        warn(
            f"{len(punc_viol)} tokens flagged as punctuation "
            "but not consisting exclusively of punctuation."
        )

        print(
            punc_viol.head(10)[
                [
                    "sentence_id",
                    "longformer_index",
                    "longformer_token"
                ]
            ].to_string(index=False)
        )

    else:

        ok("Punctuation flags are consistent with token strings.")

    # Rule C: numeric flags vs token content

    numeric_content = (
        clean_tokens
        .str.replace(r"[^\d\.]", "", regex=True)
        .str.len()
        .gt(0)
    )

    num_viol = df[
        (df["is_numeric"] == 1)
        & (~numeric_content)
    ]

    if len(num_viol) > 0:

        warn(
            f"{len(num_viol)} tokens flagged as numeric "
            "but their token text does not look numeric."
        )

        print(
            num_viol.head(10)[
                [
                    "sentence_id",
                    "longformer_index",
                    "longformer_token"
                ]
            ].to_string(index=False)
        )

    else:

        ok("Numeric flags are consistent with token strings.")

    # ----------------------------------------------------------------
    # 6) Summary diagnostics
    # ----------------------------------------------------------------

    info("6) Summary diagnostics")

    unique_sent = df["sentence_id"].nunique()
    unique_tokens = df["longformer_token"].nunique()
    n_specials = int(df["is_special_token"].sum())

    info(f"Unique sentences: {unique_sent}")
    info(f"Unique Longformer token types: {unique_tokens}")
    info(f"Total rows: {n}")
    info(f"Special tokens (rows): {n_specials}")

    ok("Basic summary computed.")

    info("=== ALL CHECKS COMPLETED ===")
    info(
        "If no [ERROR]/fatal occurred above, the file is verified. "
        "Warnings may still require investigation."
    )

    return 0


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Verify Longformer final features"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV (final Longformer features)"
    )

    parser.add_argument(
        "--log_file",
        required=True,
        help="Path to save verbose verification log"
    )

    parser.add_argument(
        "--relative_tol",
        type=float,
        default=1e-6,
        help="Tolerance for relative-frequency comparisons"
    )

    parser.add_argument(
        "--sample_sentences",
        type=int,
        default=200,
        help="Number of sentences sampled for expensive checks"
    )

    args = parser.parse_args()

    verify(
        args.input,
        args.log_file,
        relative_tol=args.relative_tol,
        sample_sentences=args.sample_sentences
    )
