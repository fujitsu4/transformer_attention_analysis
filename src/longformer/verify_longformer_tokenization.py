
"""
verify_longformer_tokenization.py
Author: Zakaria JOUILIL

Description:
    Verifies Longformer tokenization while preserving the exact
    alignment between Longformer subword tokens and original spaCy words.

    Uses the Fast tokenizer from allenai/longformer-base-4096.

    Checks performed:
        1. Global reconstruction of spaCy words.
        2. Sentence ID consistency.
        3. Completeness of sentence coverage.
        4. Sequential Longformer token indices.
        5. Correct special tokens (<s>, </s>).
        6. Validity of word_id.
        7. No spaCy word lost.
        8. Word reconstruction fidelity.
        9. No empty tokens.

Inputs:
    --longformer_tokens_csv : Longformer tokenization CSV
    --spacy_csv             : spaCy features CSV

Output:
    --log_file              : Verification log file

Usage:
    python -m src.longformer.verify_longformer_tokenization \
    --longformer_tokens_csv outputs/longformer/longformer_snli_tokens.csv \
    --spacy_csv outputs/spacy/spacy_snli_features.csv \
    --log_file logs/longformer_tokenization_logs.txt
"""

import pandas as pd
from tqdm import tqdm
import sys
import os
from transformers import AutoTokenizer


MODEL_NAME = "allenai/longformer-base-4096"


def setup_logger(log_file):
    """Redirect all prints to both console and a .log file."""

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout


def normalize_word_id(x):
    """Normalize word_id, treating empty values as None."""

    if pd.isna(x) or str(x).strip() == "":
        return None

    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


def reconstruct_word(tokens, tokenizer):
    """
    Reconstruct a spaCy word from its Longformer subword tokens.

    Longformer uses RoBERTa byte-level BPE.
    convert_tokens_to_string handles the token representation.
    """

    return tokenizer.convert_tokens_to_string(tokens)


def verify_longformer_tokenization(
    longformer_csv,
    spacy_csv,
    log_file
):

    print("\n[INFO] === VERIFYING LONGFORMER TOKENIZATION ===")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "[FATAL] A fast tokenizer is required."
        )

    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Tokenizer: {tokenizer.__class__.__name__}")
    print(f"[INFO] Special tokens: {tokenizer.special_tokens_map}")

    longformer = pd.read_csv(
        longformer_csv,
        sep=";",
        keep_default_na=False
    )

    spacy = pd.read_csv(
        spacy_csv,
        sep=";",
        keep_default_na=False
    )

    # ---- Check required columns ----

    required_longformer = {
        "sentence_id",
        "longformer_index",
        "longformer_token",
        "word_id"
    }

    required_spacy = {
        "sentence_id",
        "word_index",
        "word"
    }

    if not required_longformer.issubset(longformer.columns):
        raise ValueError(
            "[FATAL] Missing required columns in Longformer CSV."
        )

    if not required_spacy.issubset(spacy.columns):
        raise ValueError(
            "[FATAL] Missing required columns in spaCy CSV."
        )

    # ---- Convert indices ----

    longformer["sentence_id"] = (
        longformer["sentence_id"].astype(int)
    )

    longformer["longformer_index"] = (
        longformer["longformer_index"].astype(int)
    )

    longformer["word_id"] = pd.Series(
    [
        normalize_word_id(x)
        for x in longformer["word_id"]
    ],
    dtype="object"
)

    spacy["sentence_id"] = spacy["sentence_id"].astype(int)
    spacy["word_index"] = spacy["word_index"].astype(int)

    # ---- spaCy words grouped by sentence ----

    spacy_by_sent = {
        sid: group.sort_values("word_index")["word"].tolist()
        for sid, group in spacy.groupby("sentence_id")
    }

    # ---------- TEST 1: GLOBAL RECONSTRUCTION ----------

    for sid, group in longformer.groupby("sentence_id"):

        group = group.sort_values("longformer_index")

        words_subtokens = {}

        for tok, wid in zip(
            group["longformer_token"],
            group["word_id"]
        ):

            if wid is None:
                continue

            words_subtokens.setdefault(wid, []).append(tok)

        if sid not in spacy_by_sent:
            raise ValueError(
                f"[FATAL] Sentence {sid} missing from spaCy CSV."
            )

        spacy_words = spacy_by_sent[sid]

        reconstructed = [
            reconstruct_word(
                words_subtokens.get(i, []),
                tokenizer
            )
            for i in range(len(spacy_words))
        ]

        if reconstructed != spacy_words:

            print(f"\n[ERROR] Global reconstruction mismatch: {sid}")

            for original, recon in zip(
                spacy_words,
                reconstructed
            ):
                if original != recon:
                    print(f"spaCy       : {repr(original)}")
                    print(f"Longformer  : {repr(recon)}")

            sys.exit(1)

    print(
        "[INFO] ✓ Test 1 passed: global reconstruction matches spaCy."
    )

    print(f"[INFO] Loaded Longformer tokens: {len(longformer)}")
    print(f"[INFO] Loaded spaCy words: {len(spacy)}")

    # ---------- TEST 2 & 3: SENTENCE CONSISTENCY ----------

    sent_ids_longformer = set(longformer["sentence_id"])
    sent_ids_spacy = set(spacy["sentence_id"])

    if sent_ids_longformer != sent_ids_spacy:

        missing_in_longformer = (
            sent_ids_spacy - sent_ids_longformer
        )

        missing_in_spacy = (
            sent_ids_longformer - sent_ids_spacy
        )

        print("[ERROR] Sentence ID mismatch!")
        print("[ERROR] Missing in Longformer:", missing_in_longformer)
        print("[ERROR] Missing in spaCy:", missing_in_spacy)

        sys.exit(1)

    print(
        "[INFO] ✓ Test 2 & 3 passed: sentence_id consistency verified."
    )

    # ---------- SENTENCE-LEVEL TESTS ----------

    for sid, group in tqdm(
        longformer.groupby("sentence_id"),
        total=len(sent_ids_longformer)
    ):

        group = group.sort_values("longformer_index")

        longformer_tokens = group["longformer_token"].tolist()
        word_ids = group["word_id"].tolist()

        spacy_words = spacy_by_sent[sid]
        n_words = len(spacy_words)

        # TEST 4: Sequential indices

        if list(group["longformer_index"]) != list(range(len(group))):

            print(
                f"\n[ERROR] Non-sequential longformer_index "
                f"in sentence {sid}"
            )

            sys.exit(1)

        # TEST 5: Special tokens

        if (
            len(longformer_tokens) < 2
            or longformer_tokens[0] != "<s>"
            or longformer_tokens[-1] != "</s>"
        ):

            print(
                f"\n[ERROR] Incorrect special tokens "
                f"in sentence {sid}"
            )

            print("[ERROR] First token:", longformer_tokens[0])
            print("[ERROR] Last token:", longformer_tokens[-1])

            sys.exit(1)

        # Every special token must have a null word_id.

        for tok, wid in zip(longformer_tokens, word_ids):

            if tok in tokenizer.all_special_tokens:
                
                if wid is not None:
                    print(
                f"\n[ERROR] Special token '{tok}' has "
                f"non-null word_id in sentence {sid}"
            )

                    sys.exit(1)

        if word_ids[0] is not None or word_ids[-1] is not None:

            print(
                f"\n[ERROR] Boundary special tokens have "
                f"non-null word_id in sentence {sid}"
            )

            sys.exit(1)

        # TEST 6: Validity of word_id

        for wi in word_ids:

            if wi is None:
                continue

            if not (0 <= wi < n_words):

                print(
                    f"\n[ERROR] Invalid word_id={wi} in sentence {sid}"
                )

                print(
                    f"[ERROR] Expected 0..{n_words - 1}"
                )

                sys.exit(1)

        # TEST 7: No spaCy word lost

        counted = {
            wi for wi in word_ids
            if wi is not None
        }

        expected = set(range(n_words))

        if expected - counted:

            print(
                f"\n[ERROR] Some spaCy words were not tokenized "
                f"in sentence {sid}"
            )

            print("[ERROR] Missing:", expected - counted)

            sys.exit(1)

        # TEST 8: Word reconstruction fidelity

        reconstructed = [[] for _ in range(n_words)]

        for tok, wi in zip(longformer_tokens, word_ids):

            if wi is None:
                continue

            reconstructed[wi].append(tok)

        for idx in range(n_words):

            recon = reconstruct_word(
                reconstructed[idx],
                tokenizer
            )

            original = spacy_words[idx]

            if original != recon:

                print(
                    f"\n[ERROR] Word reconstruction mismatch "
                    f"in sentence {sid}"
                )

                print(f"[ERROR] Word index   : {idx}")
                print(f"[ERROR] Original     : {repr(original)}")
                print(f"[ERROR] Reconstructed: {repr(recon)}")

                sys.exit(1)

        # TEST 9: No empty tokens

        for tok in longformer_tokens:

            if tok.strip() == "":

                print(
                    f"\n[ERROR] Empty token in sentence {sid}"
                )

                sys.exit(1)

    print("\n[INFO] ✓ Tests 4 to 9 passed: sentence-level tests")

    print("\n[INFO] === ALL TESTS PASSED SUCCESSFULLY ===")

    print(
        "[INFO] Longformer tokenization is correct "
        "and perfectly aligned with spaCy."
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--longformer_tokens_csv",
        required=True
    )

    parser.add_argument(
        "--spacy_csv",
        required=True
    )

    parser.add_argument(
        "--log_file",
        required=True,
        help="Log file to save all verification outputs"
    )

    args = parser.parse_args()

    print(f"[INFO] Logging enabled -> {args.log_file}")

    setup_logger(args.log_file)

    verify_longformer_tokenization(
        args.longformer_tokens_csv,
        args.spacy_csv,
        args.log_file
    )
