
"""
compute_decision_trees_bigbird.py
Author: Zakaria JOUILIL

Description:
    Main pipeline for training decision trees on BigBird token features,
    extracting rules, and saving them for all 12 layers.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.decision_tree.decision_tree_core import (
    train_and_extract_rules_from_df,
    DEFAULT_FEATURE_COLS,
)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["pretrained", "untrained"],
        required=True
    )

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"]
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # INPUT FILE
    # --------------------------------------------------
    if args.model == "pretrained":
        input_csv = (
            f"outputs/attention/"
            f"attention_top5_{args.dataset}_bigbird_pretrained.csv"
        )
    else:
        if args.seed is None:
            raise ValueError("Untrained mode requires --seed.")

        input_csv = (
            f"outputs/attention/"
            f"attention_top5_{args.dataset}_bigbird_untrained_seed_{args.seed:03d}.csv"
        )

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------
    if args.model == "pretrained":
        out_dir = f"outputs/decision_tree/pretrained/bigbird/{args.dataset}"

        print(
            f"[INFO] BIGBIRD PRETRAINED | Dataset: {args.dataset}"
            f" -> saving to: {out_dir}"
        )

    else:
        out_dir = (
            f"outputs/decision_tree/untrained/bigbird/"
            f"{args.dataset}/seed_{args.seed:06d}"
        )

        print(
            f"[INFO] BIGBIRD UNTRAINED | Dataset: {args.dataset}"
            f" | Seed: {args.seed} -> saving to: {out_dir}"
        )

    ensure_dir(out_dir)

    # --------------------------------------------------
    # LOAD DATAFRAME
    # --------------------------------------------------
    print(f"[INFO] Loading input CSV: {input_csv}")

    df = pd.read_csv(
        input_csv,
        sep=";",
        keep_default_na=False
    )

    print(f"[INFO] Loaded {len(df):,} token rows.")

    # --------------------------------------------------
    # FEATURE CHECKS
    # --------------------------------------------------
    print("\n=== EMPTY VALUES IN FEATURES ===")

    for col in DEFAULT_FEATURE_COLS:
        values = df[col].astype(str).str.strip()
        empty_count = (values == "").sum()

        if empty_count > 0:
            print(f"{col}: {empty_count} empty values")

    print("\n=== NON NUMERIC VALUES IN FEATURES ===")

    for col in DEFAULT_FEATURE_COLS:
        values = df[col].astype(str).str.strip()
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = numeric.isna() & (values != "")

        if invalid.any():
            print(f"\n{col}:")
            print(values[invalid].value_counts().head(10))

    # --------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------
    required_columns = [
        "sentence_id",
        "bigbird_index",
        "bigbird_token",
    ] + DEFAULT_FEATURE_COLS

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column in input CSV: {col}"
            )

    # --------------------------------------------------
    # TRAIN + EXTRACT RULES FOR 12 LAYERS
    # --------------------------------------------------
    print("\n[INFO] Starting Decision Tree extraction for 12 layers ...")

    NUM_LAYERS = 12

    for layer in range(1, NUM_LAYERS + 1):

        print(f"\n[INFO] Processing layer {layer}/12 ...")

        rules = train_and_extract_rules_from_df(
            df,
            layer=layer,
            feature_cols=DEFAULT_FEATURE_COLS,
            max_depth=4,
            min_samples_leaf=20
        )

        # --------------------------------------------------
        # SAVE THREE RULESETS
        # --------------------------------------------------
        layer_tag = f"layer_{layer:02d}"

        filenames = {
            "rules_full": f"{layer_tag}_rules_full.txt",
            "rules_pos_only": f"{layer_tag}_rules_pos.txt",
            "rules_pos_simplified": f"{layer_tag}_rules_simp_pos.txt",
        }

        for key, fname in filenames.items():

            out_path = os.path.join(out_dir, fname)

            print(f"[SAVE] -> {out_path}")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(rules[key])

        print(
            f"[OK] Layer {layer}: "
            f"{rules['n_samples']:,} samples, "
            f"{rules['n_pos']:,} positive samples."
        )

    print("\n[OK] DONE — All 12 BigBird layers processed successfully.")


if __name__ == "__main__":
    main()
