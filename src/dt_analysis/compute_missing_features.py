
"""
compute_missing_features.py
Author: Zakaria JOUILIL

Description:
    Compare features analyzed vs features never selected
    in decision tree results for pretrained models.

    Architecture-independent: columns ending in "_index"
    or "_token" are always excluded.

Inputs:
    --attention_csv: the attention CSV file containing all features
    --pretrained_csv: aggregated features by depth file

Output:
    --log_file: log file containing missing features

NOTE:
    The attention file is required to retrieve the exact names
    of all features.
    The attention file is preferred over the final features file
    because the latter contains duplicate features
    (both string and ID) used for debugging.

Usage:
    python -m src.dt_analysis.compute_missing_features
        --dataset snli
"""

import argparse
import pandas as pd
from datetime import datetime
import os


ID_COLUMNS = {
    "sentence_id", "word_id", "dataset"
}

ATTENTION_PREFIX = "top5_"

EXCLUDED_SUFFIXES = ("_index", "_token")


def is_excluded_column(col):
    """
    Return True for metadata columns that must always be excluded.
    """
    return (
        col in ID_COLUMNS
        or col.startswith(ATTENTION_PREFIX)
        or col.endswith(EXCLUDED_SUFFIXES)
    )


def extract_all_features(attention_csv):

    df = pd.read_csv(attention_csv, sep=";")

    features = []

    for col in df.columns:

        if is_excluded_column(col):
            continue

        features.append(col)

    return sorted(features)


def extract_used_features(dt_csv):

    df = pd.read_csv(dt_csv, sep=";")

    used_features = set(df["Feature"].dropna().unique())

    # Exclude architecture-specific index/token columns
    used_features = {
        feature
        for feature in used_features
        if not str(feature).endswith(EXCLUDED_SUFFIXES)
    }

    return used_features


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"],
        help="Dataset being processed"
    )

    args = parser.parse_args()

    dataset = args.dataset

    attention_csv = (
        f"outputs/attention/attention_top5_{dataset}_pretrained.csv"
    )

    pretrained_csv = (
        f"outputs/dt_analysis/{dataset}/features_by_depth_{dataset}_pretrained.csv"
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = (
        f"{log_dir}/missing_features_comparison_{dataset}.txt"
    )

    # ---------------------------------------------------------
    # EXTRACT FEATURES
    # ---------------------------------------------------------
    all_features = extract_all_features(attention_csv)

    pretrained_used = extract_used_features(pretrained_csv)

    # ---------------------------------------------------------
    # COMPUTE MISSING FEATURES
    # ---------------------------------------------------------
    pretrained_missing = sorted(
        set(all_features) - pretrained_used
    )

    # ---------------------------------------------------------
    # WRITE LOG
    # ---------------------------------------------------------
    with open(log_file, "w", encoding="utf8") as f:

        f.write("=== Feature Comparison Log ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write(
            f"Total features analyzed: {len(all_features)}\n"
        )
        f.write(", ".join(all_features) + "\n\n")

        f.write(
            "=== Pretrained Model: Features NEVER selected ===\n"
        )
        f.write(
            f"Total Missing Features: {len(pretrained_missing)}\n"
        )
        f.write(", ".join(pretrained_missing) + "\n\n")

    print(
        f"[OK] Dataset: {dataset} | "
        f"Comparison log written to {log_file}"
    )


if __name__ == "__main__":
    main()
