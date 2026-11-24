
"""
compute_longformer_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Compute statistical features for Longformer tokens based on the enriched
    longformer_basic_features.csv file.

    Added features:
        - token_relative_frequency  (Longformer-token-level, per sentence)
        - word_sentence_frequency   (spaCy-word-level, global corpus, propagated)
        - token_rank                (Longformer-token-level, global corpus)
        - word_burstiness           (spaCy-word-level, global corpus, propagated)

Input:
    --longformer_features : outputs/longformer/longformer_basic_features.csv

Output:
    A CSV containing all original columns + the 4 new statistical features.
"""

import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def compute_burstiness(counts_per_sentence_dict):
    """
    Given a dict: sentence_id -> count_in_sentence
    Returns burstiness = (sigma - mu) / (sigma + mu)
    """
    counts = np.array(list(counts_per_sentence_dict.values()), dtype=float)

    if len(counts) == 0:
        return 0.0

    mu = counts.mean()
    sigma = counts.std()

    if mu + sigma == 0:
        return 0.0

    return float((sigma - mu) / (sigma + mu))


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute statistical Longformer features"
)

parser.add_argument(
    "--longformer_features",
    required=True,
    help="Input longformer_basic_features.csv"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output CSV path"
)

parser.add_argument(
    "--model_name",
    default="allenai/longformer-base-4096",
    help="Longformer tokenizer name or path"
)

args = parser.parse_args()


# ------------------------------------------------------------------
# Load tokenizer and data
# ------------------------------------------------------------------

print("[INFO] Loading Longformer tokenizer:", args.model_name)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    use_fast=True
)

print("[INFO] Loading Longformer basic features from:", args.longformer_features)

df = pd.read_csv(
    args.longformer_features,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# ------------------------------------------------------------------
# Normalize numeric fields
# ------------------------------------------------------------------

df["sentence_id"] = df["sentence_id"].astype(int)
df["longformer_index"] = df["longformer_index"].astype(int)
df["word_id"] = pd.to_numeric(
    df["word_id"], errors="coerce"
).astype("Int64")

df["num_subtokens"] = df["num_subtokens"].astype(int)
df["is_special_token"] = df["is_special_token"].astype(int)

df = df.sort_values(
    ["sentence_id", "longformer_index"]
).reset_index(drop=True)


# ------------------------------------------------------------------
# 1) token_relative_frequency
#    Token count in sentence / total non-special tokens in sentence
# ------------------------------------------------------------------

print("[INFO] Computing token_relative_frequency...")

df["token_relative_frequency"] = 0.0

for sid, group in df.groupby("sentence_id"):

    non_special = group[group["is_special_token"] == 0]

    if len(non_special) == 0:
        continue

    N = len(non_special)

    freqs = non_special["longformer_token"].value_counts() / N

    df.loc[
        non_special.index,
        "token_relative_frequency"
    ] = non_special["longformer_token"].map(freqs)

df.loc[
    df["is_special_token"] == 1,
    "token_relative_frequency"
] = 0.0


# ------------------------------------------------------------------
# 2) Reconstruct spaCy words from Longformer subtokens
# ------------------------------------------------------------------

print("[INFO] Reconstructing spaCy words from Longformer subtokens...")

df_words = df[
    (df["is_special_token"] == 0) &
    (df["word_id"].notna())
].copy()


def reconstruct_word(tokens):
    """
    Reconstruct a spaCy word from its ordered Longformer subtokens.
    Uses the tokenizer's byte-level BPE decoding.
    """
    return tokenizer.convert_tokens_to_string(
        tokens.tolist()
    ).strip()


reconstructed = (
    df_words
    .sort_values(["sentence_id", "longformer_index"])
    .groupby(["sentence_id", "word_id"])["longformer_token"]
    .apply(reconstruct_word)
    .reset_index(name="word_form")
)


# ------------------------------------------------------------------
# 3) word_sentence_frequency
#    Number of distinct sentences containing each reconstructed word
# ------------------------------------------------------------------

print("[INFO] Computing word_sentence_frequency...")

word_sentfreq = (
    reconstructed
    .groupby("word_form")["sentence_id"]
    .nunique()
    .rename("word_sentence_frequency")
)

reconstructed = reconstructed.merge(
    word_sentfreq,
    on="word_form",
    how="left"
)

df = df.merge(
    reconstructed[
        ["sentence_id", "word_id", "word_sentence_frequency"]
    ],
    on=["sentence_id", "word_id"],
    how="left"
)

df["word_sentence_frequency"] = (
    df["word_sentence_frequency"]
    .fillna(0)
    .astype(int)
)

df.loc[
    df["is_special_token"] == 1,
    "word_sentence_frequency"
] = 0


# ------------------------------------------------------------------
# 4) token_rank
#    Global frequency rank of Longformer tokens
# ------------------------------------------------------------------

print("[INFO] Computing token_rank...")

token_counts = (
    df[df["is_special_token"] == 0]
    .groupby("longformer_token")
    .size()
)

token_ranks = (
    token_counts
    .rank(method="dense", ascending=False)
    .astype(int)
)

df["token_rank"] = df["longformer_token"].map(token_ranks)

df["token_rank"] = df["token_rank"].fillna(0)

df.loc[
    df["is_special_token"] == 1,
    "token_rank"
] = 0

df["token_rank"] = df["token_rank"].astype(int)


# ------------------------------------------------------------------
# 5) word_burstiness
#    spaCy-word-level burstiness across the corpus
# ------------------------------------------------------------------

print("[INFO] Computing word_burstiness...")

reconstructed2 = reconstructed[
    ["sentence_id", "word_id", "word_form"]
].copy()

word_sentence_counts = (
    reconstructed2
    .groupby(["word_form", "sentence_id"])
    .size()
    .rename("count")
)

burstiness_values = {}

for word, subdf in word_sentence_counts.groupby(level=0):

    counts_dict = {
        sid: cnt
        for (_, sid), cnt in subdf.items()
    }

    burstiness_values[word] = compute_burstiness(counts_dict)


burstiness_df = pd.DataFrame(
    [
        (word, burstiness_values[word])
        for word in burstiness_values
    ],
    columns=["word_form", "word_burstiness"]
)

reconstructed2 = reconstructed2.merge(
    burstiness_df,
    on="word_form",
    how="left"
)

df = df.merge(
    reconstructed2[
        ["sentence_id", "word_id", "word_burstiness"]
    ],
    on=["sentence_id", "word_id"],
    how="left"
)

df["word_burstiness"] = (
    df["word_burstiness"].fillna(0.0)
)

df.loc[
    df["is_special_token"] == 1,
    "word_burstiness"
] = 0.0


# ------------------------------------------------------------------
# Final column order
# ------------------------------------------------------------------

final_cols = [
    "sentence_id",
    "longformer_index",
    "longformer_token",
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

df = df[final_cols]


# ------------------------------------------------------------------
# Save output
# ------------------------------------------------------------------

print("[INFO] Saving output to:", args.output)

df.to_csv(
    args.output,
    sep=";",
    index=False
)

print("[INFO] Statistical Longformer features successfully computed.")
print("[INFO] Output shape:", df.shape)
