
"""
verify_t5_tokenization.py
Author: Zakaria JOUILIL

Description:
    Performs deterministic T5 tokenization verification while preserving
    the exact alignment between T5 subword tokens and the original spaCy words.

Checks performed:
    - Ensures that spaCy segmentation is reconstructed using word_index.
    - Verifies sentence_id consistency between T5 and spaCy files.
    - Checks sequential t5_index values.
    - Verifies the presence and position of the T5 special token </s>.
    - Checks word_id validity and completeness.
    - Reconstructs original words from T5 subword tokens.
    - Checks that no invalid empty tokens are present.

Inputs:
    --t5_tokens_csv : T5 tokenization CSV
    --spacy_csv      : spaCy features CSV

Outputs:
    --log_file       : Log file containing verification reports

Usage:
    python -m src.t5.verify_t5_tokenization \
    --t5_tokens_csv outputs/t5/t5_snli_tokens.csv \
    --spacy_csv outputs/spacy/spacy_snli_features.csv \
    --log_file logs/t5_tokenization_logs.txt
"""

import pandas as pd
from tqdm import tqdm
import sys
import re
import os


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


def clean_subtoken(tok):
    """
    Remove the SentencePiece word-start marker.

    Examples:
        ▁person -> person
        ▁       -> ""
        er      -> er
    """
    return tok[1:] if tok.startswith("▁") else tok


def normalize_word(text):
    """Normalize words for reconstruction comparison."""
    return re.sub(r"\W+", "", str(text).lower(), flags=re.UNICODE)


def verify_t5_tokenization(t5_csv, spacy_csv, log_file):

    print("\n[INFO] === VERIFYING T5 TOKENIZATION ===")

    t5 = pd.read_csv(t5_csv, sep=";", keep_default_na=False)
    spacy = pd.read_csv(spacy_csv, sep=";", keep_default_na=False)

    # ---- Check required columns ----
    required_t5 = {
        "sentence_id", "t5_index", "t5_token", "word_id", "dataset"
    }
    required_spacy = {
        "sentence_id", "word_index", "word"
    }

    if not required_t5.issubset(t5.columns):
        print("[ERROR] Missing columns in T5 CSV:",
              required_t5 - set(t5.columns))
        sys.exit(1)

    if not required_spacy.issubset(spacy.columns):
        print("[ERROR] Missing columns in spaCy CSV:",
              required_spacy - set(spacy.columns))
        sys.exit(1)

    # ---- Convert indexes ----
    t5["sentence_id"] = pd.to_numeric(
        t5["sentence_id"], errors="raise"
    ).astype(int)

    t5["t5_index"] = pd.to_numeric(
        t5["t5_index"], errors="raise"
    ).astype(int)

    spacy["sentence_id"] = pd.to_numeric(
        spacy["sentence_id"], errors="raise"
    ).astype(int)

    spacy["word_index"] = pd.to_numeric(
        spacy["word_index"], errors="raise"
    ).astype(int)

    # ---- Normalize word_id ----
    def normalize_word_id(x):
        if x is None or pd.isna(x) or str(x).strip() == "":
            return None

        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None

    t5["word_id"] = [
        normalize_word_id(x) for x in t5["word_id"]
    ]

    # ---------- TEST 1 : Sentence reconstruction ----------
    spacy_by_sent = {
        sid: g.sort_values("word_index")["word"].tolist()
        for sid, g in spacy.groupby("sentence_id")
    }

    problems = 0

    for sid, group in t5.groupby("sentence_id"):

        group = group.sort_values("t5_index")

        words_subtoks = {}

        for tok, wid in zip(group["t5_token"], group["word_id"]):

            if wid is None:
                continue

            words_subtoks.setdefault(wid, [])
            words_subtoks[wid].append(clean_subtoken(tok))

        if sid not in spacy_by_sent:
            print(f"[ERROR] Sentence {sid} missing from spaCy.")
            sys.exit(1)

        spacy_words = spacy_by_sent[sid]

        reconstructed = [
            "".join(words_subtoks.get(i, []))
            for i in range(len(spacy_words))
        ]

        for idx, (original, recon) in enumerate(
            zip(spacy_words, reconstructed)
        ):

            original_norm = normalize_word(original)
            recon_norm = normalize_word(recon)

            if original_norm != recon_norm:
                problems += 1
                print(
                    f"[ERROR] Word reconstruction mismatch "
                    f"in sentence {sid}, word_id {idx}"
                )
                print("[ERROR] Original     :", original)
                print("[ERROR] Reconstructed:", recon)
                sys.exit(1)

    if problems == 0:
        print(
            "[INFO] ✓ Test 1 passed: "
            "word reconstruction matches spaCy."
        )

    print(f"[INFO] Loaded T5 tokens : {len(t5)}")
    print(f"[INFO] Loaded spaCy words : {len(spacy)}")

    # ---------- TEST 2 & 3 : Sentence ID consistency ----------
    sent_ids_t5 = set(t5["sentence_id"])
    sent_ids_spacy = set(spacy["sentence_id"])

    if sent_ids_t5 != sent_ids_spacy:

        missing_in_t5 = sent_ids_spacy - sent_ids_t5
        missing_in_spacy = sent_ids_t5 - sent_ids_spacy

        print("[ERROR] Sentence ID mismatch!")
        print("[ERROR] Missing in T5 :", missing_in_t5)
        print("[ERROR] Missing in spaCy:", missing_in_spacy)

        sys.exit(1)

    print(
        "[OK] ✓ Test 2 & 3 passed: "
        "sentence_id consistency verified."
    )

    # ---------- Sentence-level tests ----------
    for sid, group in tqdm(
        t5.groupby("sentence_id"),
        total=len(sent_ids_t5)
    ):

        group = group.sort_values("t5_index")

        t5_tokens = group["t5_token"].tolist()
        word_ids = group["word_id"].tolist()

        spacy_words = spacy_by_sent[sid]
        n_words = len(spacy_words)

        # TEST 4 : t5_index strictly sequential
        if list(group["t5_index"]) != list(range(len(group))):
            print(
                f"[ERROR] Non-sequential t5_index "
                f"in sentence {sid}"
            )
            sys.exit(1)

        # TEST 5 : T5 special token
        if not t5_tokens or t5_tokens[-1] != "</s>":
            print(
                f"[ERROR] Missing </s> at end of sentence {sid}"
            )
            sys.exit(1)

        if t5_tokens.count("</s>") != 1:
            print(
                f"[ERROR] Unexpected number of </s> tokens "
                f"in sentence {sid}"
            )
            sys.exit(1)

        if word_ids[-1] is not None:
            print(
                f"[ERROR] Special token </s> has a non-null "
                f"word_id in sentence {sid}"
            )
            sys.exit(1)

        # TEST 6 : validity of word_id
        for wi in word_ids:

            if wi is None:
                continue

            if not (0 <= wi < n_words):
                print(
                    f"[ERROR] Invalid word_id={wi} "
                    f"in sentence {sid}"
                )
                print(f"[ERROR] Expected 0..{n_words - 1}")
                sys.exit(1)

        # TEST 7 : no spaCy word lost
        counted = {
            wi for wi in word_ids if wi is not None
        }

        expected = set(range(n_words))

        if expected - counted:
            print(
                f"[ERROR] Some spaCy words were not tokenized "
                f"in sentence {sid}"
            )
            print("[ERROR] Missing:", expected - counted)
            sys.exit(1)

        # TEST 8 : reconstruction fidelity
        reconstructed = [""] * n_words

        for tok, wi in zip(t5_tokens, word_ids):

            if wi is None:
                continue

            reconstructed[wi] += clean_subtoken(tok)

        for idx in range(n_words):

            original = normalize_word(spacy_words[idx])
            recon = normalize_word(reconstructed[idx])

            if original != recon:

                print(
                    f"[ERROR] Word reconstruction mismatch "
                    f"in sentence {sid}"
                )
                print("[ERROR] word_id      :", idx)
                print("[ERROR] Original     :", spacy_words[idx])
                print("[ERROR] Reconstructed:", reconstructed[idx])

                sys.exit(1)

        # TEST 9 : invalid tokens
        for tok in t5_tokens:

            if not isinstance(tok, str) or tok == "":
                print(
                    f"[ERROR] Invalid empty token "
                    f"in sentence {sid}"
                )
                sys.exit(1)

    print("\n[INFO] ✓ Test 4 to 9 passed: sentence-level tests")
    print("\n[INFO] === ALL TESTS PASSED SUCCESSFULLY ===")
    print(
        "[INFO] T5 tokenization is correct "
        "and perfectly aligned with spaCy."
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--t5_tokens_csv",
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

    verify_t5_tokenization(
        args.t5_tokens_csv,
        args.spacy_csv,
        args.log_file
    )
