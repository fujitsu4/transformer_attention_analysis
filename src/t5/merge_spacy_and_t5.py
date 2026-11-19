"""
merge_spacy_and_t5.py
Author: Zakaria JOUILIL

Description:
    Merge spaCy features (word-level) with T5 basic features (token-level) and
    statistical features (token or word level) into a unified dataframe that will
    later be enriched with neighbor POS features.

Final column order:
    sentence_id
    t5_index
    t5_token
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
    prev_pos_id       (placeholder = -1)
    next_pos_id       (placeholder = -1)
    dataset

Inputs:
    --spacy_csv : outputs/spacy/spacy_features.csv
    --t5_csv    : outputs/t5/t5_statistical_features.csv

Output:
    --output    : outputs/merged/spacy_t5_merged.csv

Usage:
    python -m src.t5.merge_spacy_and_t5 \
        --spacy_csv outputs/spacy/spacy_features.csv \
        --t5_csv outputs/t5/t5_statistical_features.csv \
        --output outputs/t5/spacy_t5_merged.csv

Note:
    String columns used only for debugging (pos, dep_label, ent_label) are
    dropped from the spaCy dataframe. Numeric features are retained.
"""

import argparse
from pathlib import Path
import pandas as pd


# -------------------------------------------------------------------
# Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge spaCy and T5 features")

parser.add_argument(
    "--spacy_csv",
    required=True,
    help="Path to spacy_features.csv"
)

parser.add_argument(
    "--t5_csv",
    required=True,
    help="Path to t5_statistical_features.csv"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output merged CSV path"
)

args = parser.parse_args()


# -------------------------------------------------------------------
# Load input files
# -------------------------------------------------------------------
print("[INFO] Loading spaCy features:", args.spacy_csv)

spacy_df = pd.read_csv(
    args.spacy_csv,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)

print("[INFO] Loading T5 basic AND statistical features:", args.t5_csv)

t5_df = pd.read_csv(
    args.t5_csv,
    sep=";",
    keep_default_na=False,
    na_values=[""]
)


# -------------------------------------------------------------------
# 1. Normalize column names for merging
# -------------------------------------------------------------------
if "word_index" not in spacy_df.columns:
    raise ValueError("spaCy CSV is missing required column: word_index")

spacy_df = spacy_df.rename(columns={"word_index": "word_id"})


# -------------------------------------------------------------------
# 2. Drop unused string columns (only keep IDs for ML)
# -------------------------------------------------------------------
cols_to_drop = ["pos", "dep_label", "ent_label"]

for col in cols_to_drop:
    if col in spacy_df.columns:
        spacy_df = spacy_df.drop(columns=[col])


# -------------------------------------------------------------------
# 3. Validate required columns and normalize merge keys
# -------------------------------------------------------------------
required_spacy = ["sentence_id", "word_id", "dataset"]
required_t5 = [
    "sentence_id", "t5_index", "t5_token", "word_id", "dataset",
    "is_special_token"
]

missing_spacy = [c for c in required_spacy if c not in spacy_df.columns]
missing_t5 = [c for c in required_t5 if c not in t5_df.columns]

if missing_spacy:
    raise ValueError(f"Missing required spaCy columns: {missing_spacy}")

if missing_t5:
    raise ValueError(f"Missing required T5 columns: {missing_t5}")

# Normalize sentence IDs and word IDs so the merge keys have compatible types.
spacy_df["sentence_id"] = pd.to_numeric(
    spacy_df["sentence_id"], errors="raise"
).astype(int)

t5_df["sentence_id"] = pd.to_numeric(
    t5_df["sentence_id"], errors="raise"
).astype(int)

spacy_df["word_id"] = pd.to_numeric(
    spacy_df["word_id"], errors="coerce"
)

t5_df["word_id"] = pd.to_numeric(
    t5_df["word_id"].replace("", pd.NA), errors="coerce"
)

# SpaCy word indices should be valid word IDs.
if spacy_df["word_id"].isna().any():
    raise ValueError("spaCy word_id contains missing or non-numeric values")

spacy_df["word_id"] = spacy_df["word_id"].astype(int)

# Check uniqueness of spaCy word-level records for the merge keys.
duplicate_spacy = spacy_df.duplicated(
    subset=["sentence_id", "word_id", "dataset"],
    keep=False
)

if duplicate_spacy.any():
    sample = spacy_df.loc[
        duplicate_spacy, ["sentence_id", "word_id", "dataset"]
    ].head(10)

    raise ValueError(
        "Duplicate spaCy rows found for (sentence_id, word_id, dataset). "
        "This could multiply T5 rows during merging.\n"
        f"{sample.to_string(index=False)}"
)


# -------------------------------------------------------------------
# 4. Merge on (sentence_id, word_id, dataset)
# -------------------------------------------------------------------
print("[INFO] Merging on (sentence_id, word_id, dataset)...")

merged = t5_df.merge(
    spacy_df,
    on=["sentence_id", "word_id", "dataset"],
    how="left",
    validate="many_to_one"
)

print("[INFO] T5 rows before merge:", len(t5_df))
print("[INFO] Rows after merge:", len(merged))

if len(merged) != len(t5_df):
    raise RuntimeError(
        "Merge changed the number of T5 rows. "
        "Unexpected row multiplication or row loss detected."
    )


# -------------------------------------------------------------------
# 5. Initialize previous/next POS IDs (computed in a later script)
# -------------------------------------------------------------------
merged["prev_pos_id"] = -1
merged["next_pos_id"] = -1


# -------------------------------------------------------------------
# 6. Reorder columns
# -------------------------------------------------------------------
final_cols = [
    "sentence_id",
    "t5_index",
    "t5_token",
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

missing_final = [c for c in final_cols if c not in merged.columns]

if missing_final:
    raise ValueError(
        "Some expected columns are missing after the merge: "
        f"{missing_final}"
    )

print("[INFO] Reordering columns...")
merged = merged[final_cols]


# -------------------------------------------------------------------
# 7. Assign default values for special tokens
# -------------------------------------------------------------------
print("[INFO] Assigning default values for special tokens...")

merged["word_id"] = merged["word_id"].fillna(-1).astype(int)

special_mask = pd.to_numeric(
    merged["is_special_token"], errors="raise"
).astype(int) == 1

# 1) Boolean spaCy features -> 0
bool_cols = [
    "is_numeric",
    "is_punctuation",
    "is_stopword",
    "is_pronoun"
]

for col in bool_cols:
    merged.loc[special_mask, col] = 0

# 2) Syntactic/contextual features -> -1
#    Zero is a valid value for these features, so -1 is used as a placeholder.
id_cols_minus1 = [
    "pos_id",
    "dep_id",
    "ent_id",
    "depth_in_tree",
    "num_dependents"
]

for col in id_cols_minus1:
    merged.loc[special_mask, col] = -1


# -------------------------------------------------------------------
# 8. Report merge coverage
# -------------------------------------------------------------------
non_special_mask = ~special_mask

# Non-special T5 tokens should normally be mapped to a spaCy word.
unmatched_non_special = merged.loc[
    non_special_mask & merged["pos_id"].isna()
]

if len(unmatched_non_special) > 0:
    print(
        "[WARN] Non-special T5 tokens without matching spaCy features:",
        len(unmatched_non_special)
    )
    print(
        unmatched_non_special[
            ["sentence_id", "t5_index", "t5_token", "word_id", "dataset"]
        ].head(10).to_string(index=False)
    )
else:
    print("[OK] All non-special T5 tokens have matching spaCy features.")

print("[INFO] Special-token rows:", int(special_mask.sum()))
print("[INFO] Non-special T5 rows:", int(non_special_mask.sum()))


# -------------------------------------------------------------------
# 9. Save output
# -------------------------------------------------------------------
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

print("[INFO] Saving merged output to:", output_path)

merged.to_csv(
    output_path,
    sep=";",
    index=False
)

print("[OK] Merge completed successfully.")
print("[INFO] Output shape:", merged.shape)
