
"""
compute_t5_basic_features.py
Author: Zakaria JOUILIL

Description:
    Compute basic T5 token features:
        - is_special_token
        - token_length
        - is_subword
        - position_in_sentence
        - relative_position
        - num_subtokens

Inputs:
    --t5_csv: T5 tokenization CSV

Outputs:
    CSV file containing all original T5 tokens + new feature columns.

Usage:
    python -m src.t5.compute_t5_basic_features \
        --t5_csv outputs/t5/t5_snli_tokens.csv \
        --output outputs/t5/t5_snli_basic_features.csv
"""

import argparse
import pandas as pd
from src.utils.paths import get_project_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute basic T5 token features"
)

parser.add_argument(
    "--t5_csv",
    type=str,
    required=True,
    help="Input T5 tokens CSV file"
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output CSV file"
)

args = parser.parse_args()

t5_csv = get_project_path(*args.t5_csv.split("/"))
output_csv = get_project_path(*args.output.split("/"))

print("[INFO] Loading T5 tokens from:", t5_csv)

df = pd.read_csv(
    t5_csv,
    sep=";",
    keep_default_na=False,
    na_values=[],
    dtype={"word_id": str, "t5_token": str}
)

df["sentence_id"] = df["sentence_id"].astype(int)
df["t5_index"] = df["t5_index"].astype(int)
df["t5_token"] = df["t5_token"].fillna("").astype(str)


# ----------------------------------------------------------
# 1) is_special_token
# ----------------------------------------------------------
df["is_special_token"] = (
    df["t5_token"].isin(["</s>", "<pad>", "<unk>"])
).astype(int)


# ----------------------------------------------------------
# 2) token_length
# ----------------------------------------------------------
# Remove the initial SentencePiece word-boundary marker ▁
df["token_length"] = (
    df["t5_token"]
    .str.replace(r"^▁", "", regex=True)
    .str.len()
)


# ----------------------------------------------------------
# 3) is_subword
# ----------------------------------------------------------
# A subword is any token occurring after the first token
# associated with the same original spaCy word.

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

valid_word_tokens = df["word_id"].notna()

df.loc[valid_word_tokens, "is_subword"] = (
    df.loc[valid_word_tokens]
    .groupby(["sentence_id", "word_id"])
    .cumcount()
    .gt(0)
    .astype(int)
)


# ----------------------------------------------------------
# 4) position_in_sentence
# ----------------------------------------------------------
df["position_in_sentence"] = df["t5_index"]


# ----------------------------------------------------------
# 5) relative_position
# ----------------------------------------------------------
print("[INFO] Computing relative positions...")

df["relative_position"] = df.groupby("sentence_id")["t5_index"].transform(
    lambda col: col / col.max() if col.max() > 0 else 0
)


# ----------------------------------------------------------
# 6) num_subtokens
# ----------------------------------------------------------
print("[INFO] Computing num_subtokens per word...")

# Count all tokens associated with an original word_id.
# This includes <unk> when it has a valid word_id.
valid = df[df["word_id"].notna()]

counts = (
    valid
    .groupby(["sentence_id", "word_id"])
    .size()
    .rename("num_subtokens")
    .reset_index()
)

# Merge counts back into the original dataframe
df = df.merge(
    counts,
    how="left",
    on=["sentence_id", "word_id"]
)

# Special tokens without a word_id receive -1
df["num_subtokens"] = (
    df["num_subtokens"]
    .fillna(-1)
    .astype(int)
)


# ----------------------------------------------------------
# 7) Preserve dataset column at the end
# ----------------------------------------------------------
df = df.reindex(
    columns=[c for c in df.columns if c != "dataset"] + ["dataset"]
)


# ----------------------------------------------------------
# 8) Save output
# ----------------------------------------------------------
df.to_csv(
    output_csv,
    sep=";",
    index=False
)

print("[INFO] Saved T5 features to:", output_csv)