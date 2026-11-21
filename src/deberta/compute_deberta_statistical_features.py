
"""
compute_deberta_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Compute statistical features for DeBERTa tokens based on the enriched
    deberta_basic_features.csv file.

    Added features:
        - token_relative_frequency  (DeBERTa-token-level, per sentence)
        - word_sentence_frequency   (spaCy-word-level, global corpus, propagated)
        - token_rank                (DeBERTa-token-level, global corpus)
        - word_burstiness           (spaCy-word-level, global corpus, propagated)

Input:
    --deberta_features : outputs/deberta/deberta_basic_features.csv

Output:
    A CSV containing all original columns + the 4 new statistical features.

Usage:
    python -m src.deberta.compute_deberta_statistical_features \
        --deberta_features outputs/deberta/deberta_basic_features.csv \
        --output outputs/deberta/deberta_statistical_features.csv
"""

import argparse
import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def clean_tok(token):
    """
    Remove the leading SentencePiece word-boundary marker.
    DeBERTa uses '▁' rather than BERT's '##' convention.
    """
    return token[1:] if token.startswith("▁") else token


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute statistical DeBERTa features"
)

parser.add_argument(
    "--deberta_features",
    required=True,
    help="Input deberta_basic_features.csv"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output CSV path"
)

args = parser.parse_args()


print("[INFO] Loading DeBERTa basic features from:", args.deberta_features)

df = pd.read_csv(
    args.deberta_features,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# ------------------------------------------------------------
# Normalization of numeric fields
# ------------------------------------------------------------

df["sentence_id"] = df["sentence_id"].astype(int)
df["deberta_index"] = df["deberta_index"].astype(int)
df["word_id"] = pd.to_numeric(df["word_id"], errors="coerce").astype("Int64")
df["num_subtokens"] = df["num_subtokens"].astype(int)
df["is_special_token"] = df["is_special_token"].astype(int)

# Ensure consistent token ordering
df = df.sort_values(
    ["sentence_id", "deberta_index"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 1) token_relative_frequency
#    (token count in sentence / total tokens in sentence)
# ------------------------------------------------------------

print(
    "[INFO] Computing token_relative_frequency "
    "(per sentence, excluding specials)..."
)

df["token_relative_frequency"] = 0.0

for sid, group in df.groupby("sentence_id"):

    non_special = group[group["is_special_token"] == 0]

    if len(non_special) == 0:
        continue

    N = len(non_special)

    freqs = non_special["deberta_token"].value_counts() / N

    df.loc[
        non_special.index,
        "token_relative_frequency"
    ] = non_special["deberta_token"].map(freqs)

# Special tokens must ALWAYS have local frequency = 0
df.loc[
    df["is_special_token"] == 1,
    "token_relative_frequency"
] = 0.0


# ------------------------------------------------------------
# 2) word_sentence_frequency
#    (spaCy word level, global corpus)
#
#    Count number of DISTINCT sentences in which each word_id
#    appears. Propagate to all DeBERTa subtokens with that word_id.
# ------------------------------------------------------------

print(
    "[INFO] Computing word_sentence_frequency "
    "(global, spaCy-word-level)..."
)

# Filter valid words (no specials)
df_words = df[df["word_id"].notna()].copy()

print("[INFO] Reconstructing spaCy words from DeBERTa subtokens...")


# Remove SentencePiece marker and reconstruct each spaCy word
df_words_copy = df_words.copy()

df_words_copy["clean_subtoken"] = (
    df_words_copy["deberta_token"].apply(clean_tok)
)

reconstructed = (
    df_words_copy
    .sort_values("deberta_index")
    .groupby(["sentence_id", "word_id"])["clean_subtoken"]
    .apply(lambda toks: "_".join(toks))
    .reset_index(name="word_form")
)

# Sentence frequency per unique reconstructed word
word_sentfreq = (
    reconstructed
    .groupby("word_form")["sentence_id"]
    .nunique()
    .rename("word_sentence_frequency")
)

# Merge frequencies back
reconstructed = reconstructed.merge(
    word_sentfreq,
    on="word_form",
    how="left"
)

# Attach frequencies to the full dataframe
df = df.merge(
    reconstructed[
        ["sentence_id", "word_id", "word_sentence_frequency"]
    ],
    on=["sentence_id", "word_id"],
    how="left"
)

# Specials get 0
df["word_sentence_frequency"] = (
    df["word_sentence_frequency"]
    .fillna(0)
    .astype(int)
)

df.loc[
    df["is_special_token"] == 1,
    "word_sentence_frequency"
] = 0


# ------------------------------------------------------------
# 3) token_rank
#    (global frequency rank of DeBERTa tokens)
# ------------------------------------------------------------

print("[INFO] Computing token_rank (DeBERTa-token-level)...")

# Compute token frequencies, excluding special tokens
token_counts = (
    df[df["is_special_token"] == 0]
    .groupby("deberta_token")
    .size()
)

# Assign tied ranks (same frequency -> same rank)
token_ranks = (
    token_counts
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Map ranks to dataframe
df["token_rank"] = df["deberta_token"].map(token_ranks)

# Replace missing ranks with 0
df["token_rank"] = df["token_rank"].fillna(0)

# Specials = 0
df.loc[
    df["is_special_token"] == 1,
    "token_rank"
] = 0

df["token_rank"] = df["token_rank"].astype(int)


# ------------------------------------------------------------
# 4) word_burstiness
#    (spaCy-word-level)
# ------------------------------------------------------------

print("[INFO] Computing word_burstiness (spaCy-word-level)...")

# Reconstruct words again for burstiness calculation
df_words_copy2 = df_words_copy.copy()

reconstructed2 = (
    df_words_copy2
    .sort_values("deberta_index")
    .groupby(["sentence_id", "word_id"])["clean_subtoken"]
    .apply(lambda toks: "_".join(toks))
    .reset_index(name="word_form")
)

# Count occurrences per sentence
word_sentence_counts = (
    reconstructed2
    .groupby(["word_form", "sentence_id"])
    .size()
    .rename("count")
)

# Compute burstiness for each word form
burstiness_values = {}

for word, subdf in word_sentence_counts.groupby(level=0):

    # Build dict: sentence_id -> count
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

# Merge burstiness back to reconstructed words
reconstructed2 = reconstructed2.merge(
    burstiness_df,
    on="word_form",
    how="left"
)

# Merge burstiness into dataframe
df = df.merge(
    reconstructed2[
        ["sentence_id", "word_id", "word_burstiness"]
    ],
    on=["sentence_id", "word_id"],
    how="left"
)

# Specials get 0
df["word_burstiness"] = df["word_burstiness"].fillna(0.0)

df.loc[
    df["is_special_token"] == 1,
    "word_burstiness"
] = 0.0


# ------------------------------------------------------------
# Final column order
# ------------------------------------------------------------

final_cols = [
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

df = df[final_cols]


# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------

print("[INFO] Saving output to:", args.output)

df.to_csv(
    args.output,
    sep=";",
    index=False
)

print("[INFO] Statistical DeBERTa features successfully computed.")
print("[INFO] Output shape:", df.shape)
