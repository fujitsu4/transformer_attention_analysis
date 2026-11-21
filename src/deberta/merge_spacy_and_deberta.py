
"""
merge_spacy_and_deberta.py
Author: Zakaria JOUILIL

Description:
    Merge spaCy features (word-level) with DeBERTa basic features
    (token-level) and statistical features (token or word level)
    into a unified dataframe that will later be enriched with
    neighbor POS features.

Final column order:
    sentence_id
    deberta_index
    deberta_token
    word_id
    is_numeric
    is_punctuation
    is_stopword
    is_pronoun
    pos_id
    dep_id
    ent_id
    depth_in_tree
    num_dependents
    is_special_token
    token_length
    is_subword
    position_in_sentence
    relative_position
    num_subtokens
    token_relative_frequency
    word_sentence_frequency
    token_rank
    word_burstiness
    prev_pos_id
    next_pos_id
    dataset

Inputs:
    --spacy_csv : outputs/spacy/spacy_features.csv
    --deberta_csv : outputs/deberta/deberta_statistical_features.csv

Output:
    outputs/deberta/spacy_deberta_merged.csv

Usage:
    python -m src.deberta.merge_spacy_and_deberta \
        --spacy_csv outputs/spacy/spacy_snli_features.csv \
        --deberta_csv outputs/deberta/deberta_snli_statistical_features.csv \
        --output outputs/deberta/deberta_snli_merged.csv

Note:
    Only numeric columns that can be directly used as inputs for
    the decision tree model are retained.

    The following spaCy string columns are dropped:
        - pos
        - dep_label
        - ent_label
"""

import argparse
import os
import pandas as pd


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Merge spaCy and DeBERTa features"
)

parser.add_argument(
    "--spacy_csv",
    required=True,
    help="Path to spacy_features.csv"
)

parser.add_argument(
    "--deberta_csv",
    required=True,
    help="Path to deberta_statistical_features.csv"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output merged CSV path"
)

args = parser.parse_args()


# -------------------------------------------------------------------
# Load CSV files
# -------------------------------------------------------------------

print("[INFO] Loading spaCy features:", args.spacy_csv)

spacy_df = pd.read_csv(
    args.spacy_csv,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)

print(
    "[INFO] Loading DeBERTa basic AND statistical features:",
    args.deberta_csv
)

deberta_df = pd.read_csv(
    args.deberta_csv,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# -------------------------------------------------------------------
# 1. Normalize column names for merging
# -------------------------------------------------------------------

spacy_df = spacy_df.rename(
    columns={"word_index": "word_id"}
)


# -------------------------------------------------------------------
# 2. Drop unused STRING columns (only keep IDs for ML)
# -------------------------------------------------------------------

cols_to_drop = [
    "pos",
    "dep_label",
    "ent_label"
]

for col in cols_to_drop:
    if col in spacy_df.columns:
        spacy_df = spacy_df.drop(columns=[col])


# -------------------------------------------------------------------
# 3. Merge on (sentence_id, word_id, dataset)
# -------------------------------------------------------------------

print("[INFO] Merging on (sentence_id, word_id, dataset)...")

merged = deberta_df.merge(
    spacy_df,
    on=["sentence_id", "word_id", "dataset"],
    how="left"
)


# -------------------------------------------------------------------
# 4. Initialize prev/next POS IDs
#    (will be computed in the next script)
# -------------------------------------------------------------------

merged["prev_pos_id"] = -1
merged["next_pos_id"] = -1


# -------------------------------------------------------------------
# 5. Reorder columns
# -------------------------------------------------------------------

final_cols = [
    "sentence_id",
    "deberta_index",
    "deberta_token",
    "word_id",

    "is_numeric",
    "is_punctuation",
    "is_stopword",
    "is_pronoun",
    "pos_id",
    "dep_id",
    "ent_id",
    "depth_in_tree",
    "num_dependents",

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

    "prev_pos_id",
    "next_pos_id",

    "dataset"
]

print("[INFO] Reordering columns...")

merged = merged[final_cols]


# -------------------------------------------------------------------
# 6. Assign correct default values for special tokens
# -------------------------------------------------------------------

print("[INFO] Assigning default values for special tokens...")

merged["word_id"] = (
    pd.to_numeric(merged["word_id"], errors="coerce")
    .fillna(-1)
    .astype(int)
)

special_mask = (
    pd.to_numeric(
        merged["is_special_token"],
        errors="coerce"
    ).fillna(0).astype(int) == 1
)


# 1) Boolean spaCy features -> 0

bool_cols = [
    "is_numeric",
    "is_punctuation",
    "is_stopword",
    "is_pronoun"
]

for col in bool_cols:
    if col in merged.columns:
        merged.loc[special_mask, col] = 0


# 2) Syntactic or contextual features -> -1
#    (the value 0 already has a real meaning)

id_cols_minus1 = [
    "pos_id",
    "dep_id",
    "ent_id",
    "depth_in_tree",
    "num_dependents"
]

for col in id_cols_minus1:
    if col in merged.columns:
        merged.loc[special_mask, col] = -1


# -------------------------------------------------------------------
# 7. Save output
# -------------------------------------------------------------------

print("[INFO] Saving merged output to:", args.output)

os.makedirs(
    os.path.dirname(args.output) or ".",
    exist_ok=True
)

merged.to_csv(
    args.output,
    sep=";",
    index=False
)

print("[INFO] Merge completed successfully.")
print("[INFO] Output shape:", merged.shape)
