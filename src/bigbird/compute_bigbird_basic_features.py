
"""
compute_bigbird_basic_features.py
Author: Zakaria JOUILIL

Description:
    Compute basic bigbird token features:
        - is_special_token
        - token_length
        - is_subword
        - position_in_sentence
        - relative_position
        - num_subtokens

Inputs:
    --bigbird_csv: outputs/bigbird/bigbird_tokens.csv

Outputs:
    outputs/bigbird/bigbird_basic_features.csv

Usage:
    python -m src.bigbird.compute_bigbird_basic_features \
        --bigbird_csv outputs/bigbird/bigbird_tokens.csv \
        --output outputs/bigbird/bigbird_basic_features.csv
"""

import argparse
import pandas as pd
from src.utils.paths import get_project_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute basic bigbird token features"
)

parser.add_argument(
    "--bigbird_csv",
    type=str,
    required=True,
    help="Input bigbird_tokens.csv file"
)

parser.add_argument(
    "--output",
    type=str,
    default="outputs/bigbird/bigbird_basic_features.csv",
    help="Output CSV file"
)

args = parser.parse_args()

bigbird_csv = get_project_path(*args.bigbird_csv.split("/"))
output_csv = get_project_path(*args.output.split("/"))

print("[INFO] Loading bigbird tokens from:", bigbird_csv)

df = pd.read_csv(
    bigbird_csv,
    sep=";",
    keep_default_na=False,
    na_values=[],
    dtype={
        "word_id": str,
        "bigbird_token": str
    }
)

df["sentence_id"] = df["sentence_id"].astype(int)
df["bigbird_index"] = df["bigbird_index"].astype(int)
df["bigbird_token"] = df["bigbird_token"].fillna("").astype(str)


# ----------------------------------------------------------
# 1) is_special_token
# ----------------------------------------------------------
df["is_special_token"] = (
    df["bigbird_token"]
    .isin(["[CLS]", "[SEP]", "[PAD]", "[UNK]"])
    .astype(int)
)


# ----------------------------------------------------------
# 2) token_length
# ----------------------------------------------------------
# Exclude the SentencePiece word-boundary marker ▁

df["token_length"] = (
    df["bigbird_token"]
    .str.replace(r"^▁", "", regex=True)
    .str.len()
)


# ----------------------------------------------------------
# 3) is_subword
# ----------------------------------------------------------
# SentencePiece:
# ▁ indicates the beginning of a word.
# Tokens without ▁ are subsequent subword pieces.

df["is_subword"] = (
    ~df["bigbird_token"].str.startswith("▁")
    & (df["is_special_token"] == 0)
).astype(int)


# ----------------------------------------------------------
# 4) position_in_sentence
# ----------------------------------------------------------
df["position_in_sentence"] = df["bigbird_index"]


# ----------------------------------------------------------
# 5) relative_position = bigbird_index / max_index
# ----------------------------------------------------------
print("[INFO] Computing relative positions...")

df["relative_position"] = df.groupby("sentence_id")[
    "bigbird_index"
].transform(
    lambda col: col / col.max()
)


# ----------------------------------------------------------
# 6) num_subtokens
# ----------------------------------------------------------

def normalize_word_id(x):
    s = str(x).strip()

    if s == "":
        return pd.NA

    try:
        return int(float(s))
    except (ValueError, TypeError):
        return pd.NA


df["word_id"] = df["word_id"].apply(normalize_word_id)

print("[INFO] Computing num_subtokens per word...")

valid = df[df["is_special_token"] == 0]

counts = (
    valid
    .groupby(["sentence_id", "word_id"])
    .size()
    .rename("num_subtokens")
    .reset_index()
)

df = df.merge(
    counts,
    how="left",
    on=["sentence_id", "word_id"]
)

df["num_subtokens"] = (
    df["num_subtokens"]
    .fillna(-1)
    .astype(int)
)


# ----------------------------------------------------------
# 7) Reorder dataset column
# ----------------------------------------------------------
if "dataset" in df.columns:
    df = df.reindex(
        columns=[
            c for c in df.columns if c != "dataset"
        ] + ["dataset"]
    )


# ----------------------------------------------------------
# 8) Save output
# ----------------------------------------------------------
df.to_csv(
    output_csv,
    sep=";",
    index=False
)

print("[INFO] Saved bigbird features to:", output_csv)
