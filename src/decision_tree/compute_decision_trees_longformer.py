
"""
compute_decision_trees_longformer.py
Author: Zakaria JOUILIL

Description:
    Main pipeline for training decision trees on per-token features
    (pretrained Longformer), extracting rules, and saving them
    in the required directory structure.

Output structure:

    PRETRAINED:
        outputs/decision_tree/pretrained/<dataset>/
            layer_01_rules_full.txt
            layer_01_rules_pos.txt
            layer_01_rules_simp_pos.txt
            ...

    UNTRAINED:
        outputs/decision_tree/untrained/<dataset>/seed_XXXXXX/
            layer_01_rules_full.txt
            layer_01_rules_pos.txt
            layer_01_rules_simp_pos.txt
            ...

Usage:
    Pretrained Longformer:
        python -m src.decision_tree.compute_decision_trees_longformer \
            --model pretrained --dataset snli

    Untrained Longformer:
        python -m src.decision_tree.compute_decision_trees_longformer \
            --model untrained --seed 42 --dataset snli
"""

import argparse
import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

from src.decision_tree.decision_tree_core import (
    train_and_extract_rules_from_df,
    DEFAULT_FEATURE_COLS
)


# ---------------------------------------------------------
# Utility: ensure directory
# ---------------------------------------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["pretrained", "untrained"],
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed required for untrained model."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["snli", "agnews", "arxiv"],
        help="Dataset being processed"
    )

    args = parser.parse_args()

    # ----------------------------------------------
    # Validate model configuration
    # ----------------------------------------------
    if args.model == "untrained" and args.seed is None:
        raise ValueError("Untrained mode requires --seed.")

    # ----------------------------------------------
    # INPUT CSV
    # ----------------------------------------------
    if args.model == "pretrained":
        input_csv = (
            f"outputs/attention/"
            f"attention_top5_{args.dataset}_longformer_pretrained.csv"
        )
    else:
        input_csv = (
            f"outputs/attention/"
            f"attention_top5_{args.dataset}_longformer_untrained_seed_"
            f"{args.seed:03d}.csv"
        )

    # ----------------------------------------------
    # MODEL type and output directory
    # ----------------------------------------------
    if args.model == "pretrained":

        out_dir = (
            f"outputs/decision_tree/pretrained/{args.dataset}"
        )

        ensure_dir(out_dir)

        print(
            f"[INFO] PRETRAINED Longformer mode | "
            f"Dataset: {args.dataset} → saving to: {out_dir}"
        )

    else:

        out_dir = (
            f"outputs/decision_tree/untrained/{args.dataset}/"
            f"seed_{args.seed:06d}"
        )

        ensure_dir(out_dir)

        print(
            f"[INFO] UNTRAINED Longformer mode | "
            f"Dataset: {args.dataset} | "
            f"Seed: {args.seed} → saving to: {out_dir}"
        )

    # ----------------------------------------------
    # LOAD DATAFRAME
    # ----------------------------------------------
    print(f"[INFO] Loading input CSV: {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    df = pd.read_csv(
        input_csv,
        sep=";",
        keep_default_na=False
    )

    # ----------------------------------------------
    # CHECK EMPTY VALUES IN FEATURES
    # ----------------------------------------------
    print("\n=== EMPTY VALUES IN FEATURES ===")

    for col in DEFAULT_FEATURE_COLS:

        if col not in df.columns:
            raise ValueError(
                f"Missing required feature column: {col}"
            )

        values = df[col].astype(str).str.strip()
        empty_count = (values == "").sum()

        if empty_count > 0:
            print(f"{col}: {empty_count} empty values")

    # ----------------------------------------------
    # CHECK NON NUMERIC VALUES IN FEATURES
    # ----------------------------------------------
    print("\n=== NON NUMERIC VALUES IN FEATURES ===")

    for col in DEFAULT_FEATURE_COLS:

        values = df[col].astype(str).str.strip()
        numeric = pd.to_numeric(values, errors="coerce")

        invalid = numeric.isna() & (values != "")

        if invalid.any():

            print(f"\n{col}:")
            print(
                values[invalid].value_counts().head(10)
            )

    # ----------------------------------------------
    # SANITY CHECK FOR REQUIRED COLUMNS
    # ----------------------------------------------
    required_columns = [
        "sentence_id",
        "longformer_index",
        "longformer_token",
    ]

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(
                f"Missing required column in input CSV: {col}"
            )

    # Check the 12 attention labels
    for layer in range(1, 13):

        label_col = f"top5_l{layer}"

        if label_col not in df.columns:
            raise ValueError(
                f"Missing required attention label: {label_col}"
            )

    # ----------------------------------------------
    # TRAIN + EXTRACT RULES FOR 12 LAYERS
    # ----------------------------------------------
    print(
        "[INFO] Starting Decision Tree extraction "
        "for 12 Longformer layers ..."
    )

    NUM_LAYERS = 12

    for layer in range(1, NUM_LAYERS + 1):

        print(
            f"\n[INFO] Processing layer {layer}/12 ..."
        )

        # -------------------------------------------------
        # TRAIN MODEL AND EXTRACT 3 RULESETS
        # -------------------------------------------------
        rules = train_and_extract_rules_from_df(
            df,
            layer=layer,
            feature_cols=DEFAULT_FEATURE_COLS,
            max_depth=4,
            min_samples_leaf=20
        )

        # -------------------------------------------------
        # SAVE 3 OUTPUT FILES
        # -------------------------------------------------
        layer_tag = f"layer_{layer:02d}"

        filenames = {
            "rules_full": f"{layer_tag}_rules_full.txt",
            "rules_pos_only": f"{layer_tag}_rules_pos.txt",
            "rules_pos_simplified": (
                f"{layer_tag}_rules_simp_pos.txt"
            ),
        }

        for key, fname in filenames.items():

            out_path = os.path.join(out_dir, fname)

            print(f"[SAVE] → {out_path}")

            with open(
                out_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(rules[key])

    print(
        "\n[OK] DONE — All 12 Longformer layers "
        "processed successfully."
    )


if __name__ == "__main__":
    main()
