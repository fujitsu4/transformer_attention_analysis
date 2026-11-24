
"""
tokenize_with_longformer.py
Author: Zakaria JOUILIL

Description:
    Tokenize sentences using Longformer-base-4096, preserving the exact
    correspondence between Longformer tokens and original words (word_id).

    Uses the Fast tokenizer from allenai/longformer-base-4096.
    Special tokens are preserved with an empty word_id.

Inputs:
    --sentences_csv : data/cleaned/snli_filtered.csv
    --spacy_csv : outputs/spacy/spacy_snli_features.csv
    --output : output CSV path depending on the dataset

Outputs:
    Example: outputs/longformer/longformer_snli_tokens.csv

Usage:
    !python -m src.longformer.tokenize_with_longformer \
    --sentences_csv data/cleaned/snli_filtered.csv \
    --spacy_csv outputs/spacy/spacy_snli_features.csv \
    --output outputs/longformer/longformer_snli_tokens.csv
"""

import argparse
import os
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


MODEL_NAME = "allenai/longformer-base-4096"


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

    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"[INFO] Fast tokenizer: {tokenizer.is_fast}")
    print(f"[INFO] Special tokens: {tokenizer.special_tokens_map}")

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

        if len(tokens) != len(word_ids):
            raise ValueError(
                f"[FATAL] Token/word_id length mismatch for sentence {sid}."
            )

        for longformer_index, (tok, wid) in enumerate(
            zip(tokens, word_ids)
        ):

            if wid is None:
                wid_val = ""
            else:
                wid_val = int(wid)

                if wid_val < 0 or wid_val >= len(words):
                    raise ValueError(
                        f"[FATAL] Invalid word_id {wid_val} "
                        f"for sentence {sid}."
                    )

            ds_val = dataset if dataset is not None else ""

            output_rows.append({
                "sentence_id": int(sid),
                "longformer_index": int(longformer_index),
                "longformer_token": tok,
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
    print(f"[INFO] Total tokens: {len(out_df)}")
    print(f"[INFO] Total sentences: {out_df['sentence_id'].nunique()}")


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
        help="Output CSV path for Longformer tokens"
    )

    args = parser.parse_args()

    tokenize_sentences(
        args.sentences_csv,
        args.spacy_csv,
        args.output
    )


if __name__ == "__main__":
    main()
