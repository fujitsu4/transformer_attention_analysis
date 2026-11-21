"""
compute_decision_trees_deberta.py
Author: Zakaria JOUILIL

Description:
    Train decision trees on DeBERTa-v3 per-token features and attention top-5
    labels, extract the three rule representations, and save them by dataset,
    model type, seed, and layer.

    The feature definitions and decision-tree methodology are unchanged from
    the shared decision_tree_core module. Only architecture-specific paths,
    token-column validation, and the number of layers are adapted here.

Output structure:
    PRETRAINED:
        outputs/deberta/decision_tree/pretrained/<dataset>/
            deberta_layer_01_rules_full.txt
            deberta_layer_01_rules_pos.txt
            deberta_layer_01_rules_simp_pos.txt
            ... through layer 06

    UNTRAINED:
        outputs/deberta/decision_tree/untrained/<dataset>/seed_XXXXXX/
            deberta_layer_01_rules_full.txt
            deberta_layer_01_rules_pos.txt
            deberta_layer_01_rules_simp_pos.txt
            ... through layer 06

Usage:
    Pretrained:
        python -m src.deberta.compute_decision_trees_deberta \
            --model pretrained --dataset snli

    Untrained (example seed 10):
        python -m src.deberta.compute_decision_trees_deberta \
            --model untrained --seed 10 --dataset snli
"""

import argparse
import os
from pathlib import Path

import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.decision_tree.decision_tree_core import (
    train_and_extract_rules_from_df,
    DEFAULT_FEATURE_COLS,
)


NUM_LAYERS = 6  # microsoft/deberta-v3-small


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train DeBERTa-v3 decision trees and extract rules for all six layers."
    )
    parser.add_argument("--model", choices=["pretrained", "untrained"], required=True)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed required for an untrained model.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["snli", "agnews", "arxiv"],
        help="Dataset being processed.",
    )
    args = parser.parse_args()

    if args.model == "untrained" and args.seed is None:
        parser.error("--seed is required when --model untrained.")

    # Match the exact output filename produced by compute_attention_top5_deberta.py.
    if args.model == "pretrained":
        input_csv = (
            f"outputs/deberta/deberta_attention_top5_{args.dataset}_pretrained.csv"
        )
        out_dir = f"outputs/deberta/decision_tree/pretrained/{args.dataset}"
    else:
        input_csv = (
            f"outputs/deberta/deberta_attention_top5_{args.dataset}_"
            f"untrained_seed_{args.seed:03d}.csv"
        )
        out_dir = (
            f"outputs/deberta/decision_tree/untrained/{args.dataset}/"
            f"seed_{args.seed:06d}"
        )

    ensure_dir(out_dir)

    mode_label = args.model.upper()
    if args.model == "pretrained":
        print(
            f"[INFO] {mode_label} DeBERTa-v3 | Dataset: {args.dataset} "
            f"-> saving to: {out_dir}"
        )
    else:
        print(
            f"[INFO] {mode_label} DeBERTa-v3 | Dataset: {args.dataset} "
            f"| Seed: {args.seed} -> saving to: {out_dir}"
        )

    # ---------------------------------------------------------
    # Load attention labels + final DeBERTa features
    # ---------------------------------------------------------
    print(f"[INFO] Loading input CSV: {input_csv}")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"DeBERTa attention input file not found: {input_csv}. "
            "Run compute_attention_top5_deberta.py first."
        )

    df = pd.read_csv(input_csv, sep=";", keep_default_na=False)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    # Check architecture-specific token columns and all feature/label columns.
    required_columns = ["sentence_id", "deberta_index", "deberta_token"]
    required_columns.extend(DEFAULT_FEATURE_COLS)
    required_columns.extend([f"top5_l{layer}" for layer in range(1, NUM_LAYERS + 1)])

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required DeBERTa columns: {missing}")

    # Basic token-column validation.
    try:
        df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="raise").astype(int)
        df["deberta_index"] = pd.to_numeric(df["deberta_index"], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(f"sentence_id and deberta_index must be integer-like: {exc}") from exc

    if (df["deberta_index"] < 0).any():
        raise ValueError("deberta_index contains negative values.")

    if df["deberta_token"].astype(str).str.strip().eq("").any():
        raise ValueError("deberta_token contains empty token strings.")

    # Feature diagnostics retained from the original pipeline.
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

    # ---------------------------------------------------------
    # Train + extract rules for all six DeBERTa layers
    # ---------------------------------------------------------
    print(f"[INFO] Starting Decision Tree extraction for {NUM_LAYERS} DeBERTa layers ...")

    for layer in range(1, NUM_LAYERS + 1):
        print(f"\n[INFO] Processing DeBERTa layer {layer}/{NUM_LAYERS} ...")

        rules = train_and_extract_rules_from_df(
            df,
            layer=layer,
            feature_cols=DEFAULT_FEATURE_COLS,
            max_depth=4,
            min_samples_leaf=20,
        )

        # Include 'deberta' in every output filename as requested.
        layer_tag = f"deberta_layer_{layer:02d}"
        filenames = {
            "rules_full": f"{layer_tag}_rules_full.txt",
            "rules_pos_only": f"{layer_tag}_rules_pos.txt",
            "rules_pos_simplified": f"{layer_tag}_rules_simp_pos.txt",
        }

        for key, filename in filenames.items():
            out_path = os.path.join(out_dir, filename)
            print(f"[SAVE] -> {out_path}")
            with open(out_path, "w", encoding="utf-8") as file:
                file.write(rules[key])

    print(
        f"\n[OK] DONE — All {NUM_LAYERS} DeBERTa layers processed successfully."
    )


if __name__ == "__main__":
    main()
