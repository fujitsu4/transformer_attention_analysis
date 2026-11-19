
"""
compute_t5_statistical_features.py
Author: Zakaria JOUILIL

Description:
    Compute statistical features for T5 tokens based on the enriched
    t5_basic_features.csv file.

    Added features:
        - token_relative_frequency  (T5-token-level, per sentence)
        - word_sentence_frequency   (spaCy-word-level, global corpus, propagated)
        - token_rank                (T5-token-level, global corpus)
        - word_burstiness           (spaCy-word-level, global corpus, propagated)

Input:
    --t5_features : outputs/t5/t5_snli_basic_features.csv

Output:
    A CSV containing all original columns + the 4 new statistical features.

Usage:
    python -m src.t5.compute_t5_statistical_features \
        --t5_features outputs/t5/t5_snli_basic_features.csv \
        --output outputs/t5/t5_snli_statistical_features.csv
"""

import argparse
import pandas as pd
import numpy as np

from src.utils.paths import get_project_path


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


def clean_t5_token(token):
    """
    Remove the leading SentencePiece word-boundary marker ▁.

    Examples:
        ▁hello -> hello
        ▁      -> ""
        world  -> world
        </s>   -> </s>
    """
    token = str(token)

    if token.startswith("▁"):
        return token[1:]

    return token


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Compute statistical T5 features"
)

parser.add_argument(
    "--t5_features",
    required=True,
    help="Input t5_basic_features.csv"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output CSV path"
)

args = parser.parse_args()

t5_features_path = get_project_path(*args.t5_features.split("/"))
output_path = get_project_path(*args.output.split("/"))

print("[INFO] Loading T5 basic features from:", t5_features_path)

df = pd.read_csv(
    t5_features_path,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# ------------------------------------------------------------
# Normalization of numeric fields
# ------------------------------------------------------------

df["sentence_id"] = df["sentence_id"].astype(int)
df["t5_index"] = df["t5_index"].astype(int)

df["word_id"] = (
    df["word_id"]
    .replace("", np.nan)
    .astype(float)
)

df["num_subtokens"] = df["num_subtokens"].astype(int)
df["is_special_token"] = df["is_special_token"].astype(int)

df["t5_token"] = df["t5_token"].astype(str)


# ------------------------------------------------------------
# 1) token_relative_frequency
#    T5-token count in sentence / total non-special T5 tokens
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

    freqs = non_special["t5_token"].value_counts() / N

    df.loc[non_special.index, "token_relative_frequency"] = (
        non_special["t5_token"].map(freqs)
    )

# Special tokens must ALWAYS have local frequency = 0
df.loc[
    df["is_special_token"] == 1,
    "token_relative_frequency"
] = 0.0


# ------------------------------------------------------------
# 2) word_sentence_frequency
#    spaCy-word-level, global corpus
#
#    Count number of DISTINCT sentences in which each reconstructed
#    word appears.
#
#    Propagate the value to all T5 pieces belonging to that word_id.
# ------------------------------------------------------------

print(
    "[INFO] Computing word_sentence_frequency "
    "(global, spaCy-word-level)..."
)

# Keep only tokens associated with actual words
df_words = df[df["word_id"].notna()].copy()

print("[INFO] Reconstructing spaCy words from T5 subtokens...")

# Remove the SentencePiece marker ▁
df_words["clean_subtoken"] = (
    df_words["t5_token"].apply(clean_t5_token)
)

# Reconstruct each word by concatenating its T5 pieces
reconstructed = (
    df_words
    .groupby(["sentence_id", "word_id"])["clean_subtoken"]
    .apply(lambda toks: "".join(toks))
    .reset_index(name="word_form")
)

# Count distinct sentences containing each reconstructed word
word_sentfreq = (
    reconstructed
    .groupby("word_form")["sentence_id"]
    .nunique()
    .rename("word_sentence_frequency")
)

# Attach sentence frequencies to reconstructed words
reconstructed = reconstructed.merge(
    word_sentfreq,
    on="word_form",
    how="left"
)

# Propagate frequencies to all T5 subtokens
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
#    Global frequency rank of T5 tokens
# ------------------------------------------------------------

print("[INFO] Computing token_rank (T5-token-level)...")

# Compute global frequencies, excluding special tokens
token_counts = (
    df[df["is_special_token"] == 0]
    .groupby("t5_token")
    .size()
)

# Assign tied ranks:
# same frequency -> same rank
token_ranks = (
    token_counts
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Map ranks to the complete dataframe
df["token_rank"] = df["t5_token"].map(token_ranks)

# Replace missing ranks with 0
df["token_rank"] = df["token_rank"].fillna(0)

# Special tokens = 0
df.loc[
    df["is_special_token"] == 1,
    "token_rank"
] = 0

df["token_rank"] = df["token_rank"].astype(int)


# ------------------------------------------------------------
# 4) word_burstiness
#    spaCy-word-level, global corpus
# ------------------------------------------------------------

print("[INFO] Computing word_burstiness (spaCy-word-level)...")

# Reconstruct words again for burstiness computation
df_words_copy = df_words.copy()

df_words_copy["clean_subtoken"] = (
    df_words_copy["t5_token"].apply(clean_t5_token)
)

reconstructed2 = (
    df_words_copy
    .groupby(["sentence_id", "word_id"])["clean_subtoken"]
    .apply(lambda toks: "".join(toks))
    .reset_index(name="word_form")
)

# Count occurrences of each word in each sentence
word_sentence_counts = (
    reconstructed2
    .groupby(["word_form", "sentence_id"])
    .size()
    .rename("count")
)

# Compute burstiness for each reconstructed word
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

# Attach burstiness to reconstructed words
reconstructed2 = reconstructed2.merge(
    burstiness_df,
    on="word_form",
    how="left"
)

# Propagate burstiness to all T5 subtokens
df = df.merge(
    reconstructed2[
        ["sentence_id", "word_id", "word_burstiness"]
    ],
    on=["sentence_id", "word_id"],
    how="left"
)

# Specials get 0
df["word_burstiness"] = (
    df["word_burstiness"].fillna(0.0)
)


# ------------------------------------------------------------
# Final column order
# ------------------------------------------------------------

final_cols = [
    "sentence_id",
    "t5_index",
    "t5_token",
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

from pathlib import Path

output_path = Path(output_path)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("[INFO] Saving output to:", output_path)

df.to_csv(
    output_path,
    sep=";",
    index=False
)

print("[OK] T5 statistical features successfully computed.")
print("[INFO] Output shape:", df.shape)
