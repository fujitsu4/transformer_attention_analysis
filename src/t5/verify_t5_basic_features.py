"""
verify_t5_basic_features.py
Author: Zakaria JOUILIL

Verifies the output of compute_t5_basic_features.py.

Usage:
python -m src.t5.verify_t5_basic_features --features_csv outputs/t5/t5_snli_basic_features.csv --log_file logs/t5/t5_snli_basic_features_logs.txt
"""

import argparse
import os
import sys
import pandas as pd

SPECIAL_TOKENS = {"</s>", "<pad>", "<unk>"}


def setup_logger(log_file):
    parent = os.path.dirname(log_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

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


def fatal(message):
    print(f"[ERROR] {message}")
    print("[ERROR] Verification FAILED.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Verify T5 basic features CSV")
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--max_subtoken_warn", type=int, default=50)
    parser.add_argument("--relative_tol", type=float, default=1e-6)
    args = parser.parse_args()

    setup_logger(args.log_file)
    print(f"[INFO] Starting T5 basic-feature verification: {args.features_csv}")

    try:
        df = pd.read_csv(args.features_csv, sep=";", keep_default_na=False,
                         na_values=[], dtype=str)
    except Exception as exc:
        fatal(f"Cannot read CSV: {exc}")

    if df.empty:
        fatal("Empty CSV")
    df.columns = [c.strip() for c in df.columns]

    expected_cols = [
        "sentence_id", "t5_index", "t5_token", "word_id",
        "is_special_token", "token_length", "is_subword",
        "position_in_sentence", "relative_position", "num_subtokens", "dataset"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        fatal(f"Missing required columns: {missing}")
    print("[OK] Required columns present")

    if df.columns[-1] != "dataset":
        print("[WARN] dataset is not the last column (cosmetic issue)")

    int_cols = [
        "sentence_id", "t5_index", "position_in_sentence",
        "is_special_token", "token_length", "is_subword", "num_subtokens"
    ]
    nums = {}
    for col in int_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or ((vals % 1) != 0).any():
            fatal(f"{col} contains missing, non-numeric, or non-integer values")
        nums[col] = vals.astype(int)

    rel = pd.to_numeric(df["relative_position"], errors="coerce")
    if rel.isna().any():
        fatal("relative_position contains missing or non-numeric values")

    tokens = df["t5_token"].astype(str)
    missing_wid = df["word_id"].astype(str).str.strip().eq("")
    word_ids = pd.to_numeric(df["word_id"].replace("", pd.NA), errors="coerce")
    if ((~missing_wid) & word_ids.isna()).any():
        fatal("Some non-empty word_id values are not integers")

    for col in ("is_special_token", "is_subword"):
        if not nums[col].isin([0, 1]).all():
            fatal(f"{col} must contain only 0 and 1")

    # Sequential indices within each sentence.
    for sid, group in df.groupby(nums["sentence_id"], sort=False):
        got = nums["t5_index"].loc[group.index].tolist()
        if got != list(range(len(group))):
            fatal(f"t5_index is not sequential in sentence {sid}: {got[:15]}")
    print("[OK] t5_index is sequential within each sentence")

    # Special tokens.
    expected_special = tokens.isin(SPECIAL_TOKENS).astype(int)
    mismatch = expected_special != nums["is_special_token"]
    if mismatch.any():
        fatal(f"is_special_token mismatch in {int(mismatch.sum())} rows")
    print("[OK] is_special_token matches </s>, <pad>, and <unk>")

    # Structural specials should have no word_id; ordinary lexical pieces should.
    if ((expected_special == 0) & missing_wid).any():
        fatal("A non-special token has no word_id")
    # <unk> may have a word_id because it can represent an original word.
    print("[OK] word_id presence is consistent with token roles")

    # is_subword: first piece for a word_id is 0; subsequent pieces are 1.
    expected_subword = pd.Series(0, index=df.index, dtype=int)
    valid = df.loc[~missing_wid].copy()
    valid["_wid"] = word_ids.loc[valid.index].astype(int)
    valid["_idx"] = nums["t5_index"].loc[valid.index]
    valid = valid.sort_values(["sentence_id", "_idx"], kind="stable")
    continuation = valid.groupby(["sentence_id", "_wid"], sort=False).cumcount().gt(0)
    expected_subword.loc[continuation.index] = continuation.astype(int)
    mismatch = expected_subword != nums["is_subword"]
    if mismatch.any():
        fatal(f"is_subword mismatch in {int(mismatch.sum())} rows")
    print("[OK] is_subword follows within-word T5 piece order")

    # token_length: remove the leading SentencePiece marker ▁.
    lexical_tokens = tokens.str.replace(r"^▁", "", regex=True)
    expected_length = lexical_tokens.str.len().astype(int)
    mismatch = expected_length != nums["token_length"]
    if mismatch.any():
        fatal(f"token_length mismatch in {int(mismatch.sum())} rows")
    print("[OK] token_length excludes the leading ▁ marker")

    # Position.
    mismatch = nums["t5_index"] != nums["position_in_sentence"]
    if mismatch.any():
        fatal("position_in_sentence does not equal t5_index")
    print("[OK] position_in_sentence equals t5_index")

    # Relative position.
    for sid, group in df.groupby(nums["sentence_id"], sort=False):
        idx = group.index
        max_idx = nums["t5_index"].loc[idx].max()
        expected = (nums["t5_index"].loc[idx] / float(max_idx)
                    if max_idx > 0 else pd.Series(0.0, index=idx))
        if ((expected - rel.loc[idx]).abs() > args.relative_tol).any():
            fatal(f"relative_position mismatch in sentence {sid}")
    print("[OK] relative_position is t5_index / max(t5_index)")

    # num_subtokens: count all pieces carrying a word_id, including <unk>
    # if it is mapped to an original word. Structural specials have no word_id.
    valid = df.loc[~missing_wid].copy()
    valid["_wid"] = word_ids.loc[valid.index].astype(int)

    valid["sentence_id"] = valid["sentence_id"].astype(int)
    valid["_wid"] = valid["_wid"].astype(int)

    counts = valid.groupby(["sentence_id", "_wid"]).size().to_dict()


    for idx, row in valid.iterrows():
      key = (int(nums["sentence_id"].loc[idx]), int(row["_wid"]))

      if key not in counts:
          fatal(f"Cannot find word group {key} in num_subtokens counts")

      expected_count = int(counts[key])
      actual_count = int(df.loc[idx, "num_subtokens"])

      if actual_count != expected_count:
          fatal(
              f"Incorrect num_subtokens at row {idx}: "
              f"expected {expected_count}, got {actual_count}"
          )
    print("[OK] num_subtokens matches the number of pieces per word")

    structural_special = (expected_special == 1) & missing_wid
    if (nums["num_subtokens"].loc[structural_special] != -1).any():
        fatal("Special tokens without word_id must have num_subtokens=-1")
    print("[OK] Structural special tokens have num_subtokens=-1")

    if (nums["num_subtokens"] < -1).any():
        fatal("num_subtokens contains values below -1")
    large = nums["num_subtokens"] > args.max_subtoken_warn
    if large.any():
        print(f"[WARN] {int(large.sum())} rows exceed max_subtoken_warn={args.max_subtoken_warn}")
    else:
        print("[OK] num_subtokens values are within the warning threshold")

    # ▁ alone is valid (length 0); only an actually empty string is invalid.
    if tokens.eq("").any():
        fatal("Empty t5_token strings found")
    print("[OK] No empty t5_token strings; standalone ▁ is accepted")

    if df["dataset"].astype(str).str.strip().eq("").any():
        fatal("dataset contains empty values")
    print("[OK] dataset values are present")

    print("\\n[INFO] === VERIFICATION COMPLETE ===")
    print("[INFO] All checks passed (or warnings reported).")
    print(f"[INFO] Detailed log written to: {args.log_file}")
    return 0


if __name__ == "__main__":
    main()
