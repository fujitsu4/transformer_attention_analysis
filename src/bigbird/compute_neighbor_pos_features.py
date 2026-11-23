
"""
compute_neighbor_pos_features.py
Author: Zakaria JOUILIL

Description:
    Compute spaCy neighbor POS features for merged i+spaCy tokens.
    Specifically fills prev_pos_id and next_pos_id using the spaCy pos_id.

    HOW IT WORKS:
      - For each sentence_id, we sort tokens by word_id.
      - For each token, prev_pos_id = pos_id of word_id - 1
                           next_pos_id = pos_id of word_id + 1
      - Tokens with irrelevant word_id (including specials):
            prev_pos_id = -1
            next_pos_id = -1

Input:
    --input : merged CSV (output of merge_spacy_and_bigbird.py)

Output:
    --output : same CSV + filled prev_pos_id, next_pos_id

Usage:
    python -m src.bigbird.compute_neighbor_pos_features \
        --input outputs/bigbird/bigbird_snli_merged.csv \
        --output outputs/bigbird/bigbird_snli_final_features.csv
"""

import argparse
import pandas as pd
import numpy as np


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute bigbird neighbor POS features"
)

parser.add_argument(
    "--input",
    required=True,
    help="Merged bigbird+spaCy input CSV"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output CSV path"
)

args = parser.parse_args()


# -------------------------------------------------------------------
# Load input
# -------------------------------------------------------------------

print("[INFO] Loading merged bigbird file:", args.input)

df = pd.read_csv(
    args.input,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# -------------------------------------------------------------------
# Sanity checks
# -------------------------------------------------------------------

required_cols = {
    "sentence_id",
    "word_id",
    "pos_id",
    "prev_pos_id",
    "next_pos_id",
    "is_special_token"
}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        f"[ERROR] Missing required columns in input: {missing}"
    )


# -------------------------------------------------------------------
# Normalize numeric fields
# -------------------------------------------------------------------

df["sentence_id"] = pd.to_numeric(
    df["sentence_id"],
    errors="raise"
).astype(int)

df["word_id"] = pd.to_numeric(
    df["word_id"],
    errors="coerce"
)

df["pos_id"] = pd.to_numeric(
    df["pos_id"],
    errors="coerce"
)

df["is_special_token"] = pd.to_numeric(
    df["is_special_token"],
    errors="raise"
).astype(int)


# -------------------------------------------------------------------
# Initialize outputs
# -------------------------------------------------------------------

print("[INFO] Computing neighbor POS features...")

prev_pos = np.full(len(df), -1, dtype=int)
next_pos = np.full(len(df), -1, dtype=int)


# -------------------------------------------------------------------
# Process per sentence
# -------------------------------------------------------------------

for sid, group in df.groupby("sentence_id"):

    # Keep ONLY real words with valid word_id
    valid_rows = group[
        (group["is_special_token"] == 0)
        & (group["word_id"].notna())
        & (group["word_id"] >= 0)
    ]

    if valid_rows.empty:
        continue

    # Sort tokens by spaCy word_id
    valid_sorted = valid_rows.sort_values(
        ["word_id"]
    )

    word_ids = valid_sorted["word_id"].astype(int).values

    pos_ids = (
        valid_sorted["pos_id"]
        .fillna(-1)
        .astype(int)
        .values
    )

    original_idx = valid_sorted.index.values

    # Build word_id -> pos_id mapping
    mapping = dict(zip(word_ids, pos_ids))

    for wid, row_index in zip(word_ids, original_idx):

        prev_pos[row_index] = mapping.get(wid - 1, -1)
        next_pos[row_index] = mapping.get(wid + 1, -1)


# -------------------------------------------------------------------
# Final overwrite for special tokens
# -------------------------------------------------------------------

special_mask = df["is_special_token"] == 1

prev_pos[special_mask.to_numpy()] = -1
next_pos[special_mask.to_numpy()] = -1

df["prev_pos_id"] = prev_pos
df["next_pos_id"] = next_pos


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

print(
    "[INFO] Saving bigbird final features with neighbor POS to:",
    args.output
)

import os

os.makedirs(
    os.path.dirname(args.output) or ".",
    exist_ok=True
)

df.to_csv(
    args.output,
    sep=";",
    index=False
)

print("[INFO] Neighbor POS computation completed successfully.")
print("[INFO] Output shape:", df.shape)
