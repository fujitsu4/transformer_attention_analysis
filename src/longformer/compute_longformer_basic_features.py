
"""
compute_longformer_basic_features.py
Author: Zakaria JOUILIL

Description:
    Compute basic Longformer token features:
        - is_special_token
        - token_length
        - is_subword
        - position_in_sentence
        - relative_position
        - num_subtokens

Inputs:
    --longformer_csv: outputs/longformer/longformer_tokens.csv

Outputs:
    outputs/longformer/longformer_basic_features.csv

Usage:
    python -m src.longformer.compute_longformer_basic_features \
        --longformer_csv outputs/longformer/longformer_tokens.csv \
        --output outputs/longformer/longformer_basic_features.csv
"""

import argparse
import pandas as pd
from src.utils.paths import get_project_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute basic Longformer token features"
)

parser.add_argument(
    "--longformer_csv",
    type=str,
    required=True,
    help="Input longformer_tokens.csv file"
)

parser.add_argument(
    "--output",
    type=str,
    default="outputs/longformer/longformer_basic_features.csv",
    help="Output CSV file"
)

args = parser.parse_args()

longformer_csv = get_project_path(*args.longformer_csv.split("/"))
output_csv = get_project_path(*args.output.split("/"))

print("[INFO] Loading Longformer tokens from:", longformer_csv)

df = pd.read_csv(
    longformer_csv,
    sep=";",
    keep_default_na=False,
    na_values=[],
    dtype={
        "word_id": str,
        "longformer_token": str
    }
)

df["sentence_id"] = df["sentence_id"].astype(int)
df["longformer_index"] = df["longformer_index"].astype(int)
df["longformer_token"] = df["longformer_token"].fillna("").astype(str)


# ----------------------------------------------------------
# 1) is_special_token
# ----------------------------------------------------------
# Longformer uses RoBERTa special tokens.

SPECIAL_TOKENS = [
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "<mask>"
]

df["is_special_token"] = (
    df["longformer_token"]
    .isin(SPECIAL_TOKENS)
    .astype(int)
)


# ----------------------------------------------------------
# 2) token_length
# ----------------------------------------------------------
# RoBERTa byte-level BPE may use Ġ as a word-boundary marker.
# Remove it when present.

df["token_length"] = (
    df["longformer_token"]
    .str.replace(r"^Ġ", "", regex=True)
    .str.len()
)


# ----------------------------------------------------------
# 3) is_subword
# ----------------------------------------------------------
# Use word_id instead of a tokenizer-specific marker.
#
# First subtoken of a spaCy word: is_subword = 0
# Subsequent subtokens:          is_subword = 1
# Special tokens:                is_subword = 0

def normalize_word_id(x):
    s = str(x).strip()

    if s == "":
        return pd.NA

    try:
        return int(float(s))
    except (ValueError, TypeError):
        return pd.NA


df["word_id"] = df["word_id"].apply(normalize_word_id)

df["is_subword"] = 0

valid_mask = df["is_special_token"] == 0

df.loc[valid_mask, "is_subword"] = (
    df.loc[valid_mask]
    .groupby(["sentence_id", "word_id"])
    .cumcount()
    .gt(0)
    .astype(int)
)


# ----------------------------------------------------------
# 4) position_in_sentence
# ----------------------------------------------------------

df["position_in_sentence"] = df["longformer_index"]


# ----------------------------------------------------------
# 5) relative_position
# ----------------------------------------------------------

print("[INFO] Computing relative positions...")

df["relative_position"] = df.groupby("sentence_id")[
    "longformer_index"
].transform(
    lambda col: col / col.max() if col.max() > 0 else 0
)


# ----------------------------------------------------------
# 6) num_subtokens
# ----------------------------------------------------------

print("[INFO] Computing num_subtokens per word...")

valid = df[
    (df["is_special_token"] == 0)
    & (df["word_id"].notna())
]

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

print("[INFO] Saved Longformer features to:", output_csv)
