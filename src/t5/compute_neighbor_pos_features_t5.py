"""
compute_neighbor_pos_features.py
Author: Zakaria JOUILIL

Description:
    Compute spaCy neighbor POS features for merged T5+spaCy tokens.
    Specifically fills prev_pos_id and next_pos_id using spaCy pos_id.

    HOW IT WORKS:
      - For each sentence_id, retrieve valid spaCy word-level POS features.
      - For each T5 token mapped to word_id:
            prev_pos_id = pos_id of word_id - 1
            next_pos_id = pos_id of word_id + 1
      - Tokens with irrelevant word_id (including special tokens):
            prev_pos_id = -1
            next_pos_id = -1

    Multiple T5 subtokens may share the same word_id. They receive the same
    previous/next POS values, based on the neighboring spaCy words.

Input:
    --input    : merged CSV (output of merge_spacy_and_t5.py)
Output:
    --output   : same CSV with filled prev_pos_id and next_pos_id

Usage:
    python -m src.t5.compute_neighbor_pos_features \
        --input outputs/t5/spacy_t5_merged.csv \
        --output outputs/t5/t5_final_features.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute neighbor POS features for T5")
parser.add_argument("--input", required=True, help="Merged input CSV")
parser.add_argument("--output", required=True, help="Output CSV path")
args = parser.parse_args()


# -------------------------------------------------------------------
# Load input
# -------------------------------------------------------------------
print("[INFO] Loading merged file:", args.input)

df = pd.read_csv(
    args.input,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# -------------------------------------------------------------------
# Sanity checks and normalization
# -------------------------------------------------------------------
required_cols = {
    "sentence_id",
    "t5_index",
    "word_id",
    "pos_id",
    "prev_pos_id",
    "next_pos_id",
    "is_special_token",
    "dataset"
}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"[ERROR] Missing required columns in input: {missing}")

df["sentence_id"] = pd.to_numeric(
    df["sentence_id"], errors="raise"
).astype(int)

df["word_id"] = pd.to_numeric(
    df["word_id"], errors="coerce"
)

df["pos_id"] = pd.to_numeric(
    df["pos_id"], errors="coerce"
)

df["is_special_token"] = pd.to_numeric(
    df["is_special_token"], errors="raise"
).astype(int)

if not df["is_special_token"].isin([0, 1]).all():
    raise ValueError("[ERROR] is_special_token must contain only 0 or 1.")

if df["t5_index"].duplicated().any():
    # t5_index is expected to be unique within a sentence, not necessarily corpus-wide.
    duplicate_mask = df.duplicated(subset=["sentence_id", "t5_index"], keep=False)
    if duplicate_mask.any():
        examples = df.loc[
            duplicate_mask, ["sentence_id", "t5_index", "t5_token"]
        ].head(10)
        raise ValueError(
            "[ERROR] Duplicate (sentence_id, t5_index) pairs detected:\n"
            f"{examples.to_string(index=False)}"
        )


# -------------------------------------------------------------------
# Initialize outputs
# -------------------------------------------------------------------
print("[INFO] Computing neighbor POS features...")

prev_pos = np.full(len(df), -1, dtype=int)
next_pos = np.full(len(df), -1, dtype=int)


# -------------------------------------------------------------------
# Process per sentence and dataset
# -------------------------------------------------------------------
# The spaCy features are word-level. Multiple T5 subtokens can map to one
# word_id, so first build a unique word_id -> pos_id mapping for each sentence.
# Include dataset in the grouping key to avoid mixing different datasets.
for (sid, dataset), group in df.groupby(["sentence_id", "dataset"]):
    valid_rows = group[
        (group["is_special_token"] == 0)
        & group["word_id"].notna()
        & group["pos_id"].notna()
    ]

    if valid_rows.empty:
        continue

    # Collapse multiple T5 subtokens to one spaCy word-level record.
    word_pos = (
        valid_rows[["word_id", "pos_id"]]
        .drop_duplicates()
        .sort_values("word_id")
    )

    # A word should not map to multiple POS IDs.
    inconsistent = word_pos.groupby("word_id")["pos_id"].nunique()
    if (inconsistent > 1).any():
        bad_word_ids = inconsistent[inconsistent > 1].index.tolist()
        raise ValueError(
            f"[ERROR] Conflicting POS IDs for sentence_id={sid}, "
            f"dataset={dataset}, word_id(s)={bad_word_ids}"
        )

    mapping = {
        int(wid): int(pos)
        for wid, pos in word_pos.drop_duplicates("word_id").itertuples(index=False)
    }

    # Every T5 token mapped to a valid word gets neighbor POS values.
    for row_index in group.index:
        if df.loc[row_index, "is_special_token"] != 0:
            continue

        wid = df.loc[row_index, "word_id"]

        if pd.isna(wid):
            continue

        wid = int(wid)

        prev_pos[row_index] = mapping.get(wid - 1, -1)
        next_pos[row_index] = mapping.get(wid + 1, -1)


# -------------------------------------------------------------------
# Final overwrite for special tokens
# -------------------------------------------------------------------
df.loc[df["is_special_token"] == 1, ["prev_pos_id", "next_pos_id"]] = -1

df["prev_pos_id"] = prev_pos
df["next_pos_id"] = next_pos


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("[INFO] Saving with neighbor features to:", output_path)

df.to_csv(
    output_path,
    sep=";",
    index=False
)

print("[OK] Neighbor POS computation completed successfully.")
print("[INFO] Output shape:", df.shape)
print("[INFO] Rows with previous POS feature != -1:", int((df["prev_pos_id"] != -1).sum()))
print("[INFO] Rows with next POS feature != -1:", int((df["next_pos_id"] != -1).sum()))
