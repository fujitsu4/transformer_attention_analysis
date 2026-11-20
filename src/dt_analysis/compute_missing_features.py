"""
compute_missing_features.py
Author: Zakaria JOUILIL

Description:
    Compare features analyzed vs features never selected
    in decision tree results for pretrained BERT.

Inputs :
    --attention_csv : the attention csv file containing all features
    --pretrained_csv : aggregated features by depth file (pretrained model)

Output :
    --log_file : log file containing missing features for both pretrained model

NOTE:
    The attention file is required as an argument to retrieve the exact names of all features.
    The attention file is preferred over the bert final features file because the latter contains
    duplicate features (both string and ID) used for debug.

Usage:
    python -m src.dt_analysis.compute_missing_features \
        --attention_csv outputs/attention/attention_top5_pretrained_sample.csv \
        --pretrained_csv outputs/dt_analysis/features_by_depth_pretrained.csv \
        --log_file logs/missing_features_comparison.txt
"""

import argparse
import pandas as pd
from datetime import datetime
import os


ID_COLUMNS = {
    "sentence_id", "bert_index", "bert_token", "word_id", "dataset"
}

ATTENTION_PREFIX = "top5_"


def extract_all_features(attention_csv):
    df = pd.read_csv(attention_csv, sep = ";")
    features = []

    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        if col.startswith(ATTENTION_PREFIX):
            continue
        features.append(col)

    return sorted(features)


def extract_used_features(dt_csv):
    df = pd.read_csv(dt_csv, sep = ";")
    return set(df["Feature"].unique())


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

    log_file = f"{log_dir}/missing_features_comparison_{dataset}.txt"

    all_features = extract_all_features(attention_csv)
    pretrained_used = extract_used_features(pretrained_csv)

    pretrained_missing = sorted(set(all_features) - pretrained_used)

    with open(log_file, "w", encoding="utf8") as f:
        f.write(f"=== Feature Comparison Log ===\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write(f"Total features analyzed: {len(all_features)}\n")
        f.write(", ".join(all_features) + "\n\n")

        f.write("=== Pretrained BERT: Features NEVER selected ===\n")
        f.write(f"Total Missing Features: {len(pretrained_missing)}\n")
        f.write(", ".join(pretrained_missing) + "\n\n")

    print(f"[OK] Dataset: {dataset} | Comparison log written to {log_file}")


if __name__ == "__main__":
    main()