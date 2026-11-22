
"""
tokenize_with_deberta.py
Author: Zakaria JOUILIL

Description:
    Tokenize sentences using DeBERTa-v3-base, preserving the exact
    correspondence between DeBERTa tokens and original words (word_id).

    Uses the Fast tokenizer from microsoft/deberta-v3-base.
    Special tokens are preserved with an empty word_id.

Inputs:
    --sentences_csv : data/cleaned/snli_filtered.csv
    --spacy_csv : outputs/spacy/spacy_snli_features.csv
    --output : output CSV path depending on the dataset

Outputs:
    Example: outputs/deberta/deberta_snli_tokens.csv

Usage:
    !python -m src.deberta.tokenize_with_deberta \
    --sentences_csv data/cleaned/snli_filtered.csv \
    --spacy_csv outputs/spacy/spacy_snli_features.csv \
    --output outputs/deberta/deberta_snli_tokens.csv
"""

import argparse
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


MODEL_NAME = "microsoft/deberta-v3-base"


def load_word_sequences(spacy_csv):
    df = pd.read_csv(
        spacy_csv,
        sep=";",
        keep_default_na=False,
        na_values=[]
    )

    word_sequences = {}

    for sid, group in df.groupby("sentence_id"):
        group_sorted = group.sort_values("word_index")
        words = group_sorted["word"].tolist()
        word_sequences[sid] = words

    return word_sequences


def tokenize_sentences(sentences_csv, spacy_csv, output_csv):

    sentences_df = pd.read_csv(
        sentences_csv,
        sep=";",
        keep_default_na=False,
        na_values=[]
    )

    word_sequences = load_word_sequences(spacy_csv)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "[FATAL] A fast tokenizer is required to retrieve word_ids()."
        )

    print(f"[INFO] Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"[INFO] Fast tokenizer: {tokenizer.is_fast}")

    output_rows = []

    for _, row in tqdm(
        sentences_df.iterrows(),
        total=len(sentences_df)
    ):

        sid = row["sentence_id"]
        dataset = row.get("dataset", "")

        if sid not in word_sequences:
            raise ValueError(
                f"[FATAL] sentence_id {sid} not in spacy csv!"
            )

        words = word_sequences[sid]

        encoded = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        word_ids = encoded.word_ids()

        for deberta_index, (tok, wid) in enumerate(
            zip(tokens, word_ids)
        ):

            if wid is None:
                wid_val = ""
            else:
                wid_val = int(wid)

            ds_val = dataset if dataset is not None else ""

            output_rows.append({
                "sentence_id": int(sid),
                "deberta_index": int(deberta_index),
                "deberta_token": tok,
                "word_id": wid_val,
                "dataset": ds_val
            })

    out_df = pd.DataFrame(output_rows)

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True
    )

    out_df.to_csv(
        output_csv,
        sep=";",
        index=False,
        na_rep=""
    )

    print(f"[OK] Tokenization saved in: {output_csv}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentences_csv",
        required=True,
        help="CSV containing the input sentences"
    )

    parser.add_argument(
        "--spacy_csv",
        required=True,
        help="CSV containing spaCy features"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for DeBERTa tokens"
    )

    args = parser.parse_args()

    tokenize_sentences(
        args.sentences_csv,
        args.spacy_csv,
        args.output
    )


if __name__ == "__main__":
    main()
