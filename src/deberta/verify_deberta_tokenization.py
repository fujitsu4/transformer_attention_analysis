"""
verify_deberta_tokenization.py
Author: Zakaria JOUILIL

Description:
    Performs deterministic deberta tokenization while preserving the exact
    alignment between deberta subword tokens and the original spaCy words.
    This script produces a minimal, reusable tokenization file that is
    used by subsequent analysis modules (feature extraction, merging,
    attention studies, etc.).
    
Checks performed:
    - Ensures that the spaCy segmentation is reconstructed exactly
      (based on word_index) before tokenization.
    - deberta tokenization is applied with `is_split_into_words=True`,
      ensuring stable and error-free mapping via `word_id`.
    - For every token, stores:
          • sentence_id
          • deberta_index (sequential index in the deberta sequence)
          • deberta_token (string representation)
          • word_id (index of the originating spaCy word; None for specials)

Inputs:
    --deberta_tokens.csv  : The deberta tokenization CSV
    --spacy_csv : The spaCy features CSV (required to recover the exact list of words per sentence)

Outputs :
    --log_file      : A log file where all reports and progress messages are stored

    Usage:
    python -m src.deberta.verify_deberta_tokenization \
    --deberta_tokens_csv outputs/deberta/deberta_tokens.csv \
    --spacy_csv outputs/spacy/spacy_features.csv \
    --log_file logs/deberta_tokenization_logs.txt
"""

import pandas as pd
from tqdm import tqdm
import sys
import re

import sys

def setup_logger(log_file):
    """Redirect all prints to both console and a .log file."""
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
    sys.stderr = sys.stdout  # redirect errors too

def clean_subtoken(tok):
    """Remove SentencePiece word-boundary marker."""
    return tok[1:] if tok.startswith("▁") else tok

def verify_deberta_tokenization(deberta_csv, spacy_csv, log_file):

    print("\n[INFO] === VERIFYING deberta TOKENIZATION ===")

    deberta = pd.read_csv(deberta_csv, sep=";", keep_default_na=False)
    spacy = pd.read_csv(spacy_csv, sep=";", keep_default_na=False)

    # ---- Convert sentence_id and deberta_index ----
    deberta["sentence_id"] = deberta["sentence_id"].astype(int)
    deberta["deberta_index"] = deberta["deberta_index"].astype(int)

    # ---- Normalize word_id even if float64 ----
    def normalize_word_id(x, tok):
        # special tokens
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            return None

        if pd.isna(x):
            return None

        try:
            return int(round(float(x)))
        except:
            return None

    deberta["word_id"] = [
        normalize_word_id(x, tok) for x, tok in zip(deberta["word_id"], deberta["deberta_token"])
    ]

    # ---- Test 1 : Check reconstruction ----
    problems = 0

    for sid, group in deberta.groupby("sentence_id"):
        # Extract tokens (ignore special tokens)
        words_subtoks = {}
        
        for tok, wid in zip(group["deberta_token"], group["word_id"]):
            if wid is None or pd.isna(wid):
                continue

            # Force integer index
            wid = int(wid)

            if wid not in words_subtoks:
                words_subtoks[wid] = []

            words_subtoks[wid].append(clean_subtoken(tok))

        # Compare with spaCy
        spacy_words = list(
            spacy[spacy["sentence_id"]==sid].sort_values("word_index")["word"]
        )

        reconstructed = ["".join(words_subtoks[i]) for i in range(len(spacy_words))]

        if reconstructed != spacy_words:
            problems += 1
            print(f"[ERROR] Mismatch in sentence {sid}")
            print("spaCy :", spacy_words)
            print("deberta  :", reconstructed)
            sys.exit(1)

    if problems == 0:
        print("[INFO] ✓ Test 1 passed: global reconstruction matches spaCy.")

    print(f"[INFO] Loaded deberta tokens : {len(deberta)}")
    print(f"[INFO] Loaded spaCy words : {len(spacy)}")

    # group spacy words per sentence
    spacy_by_sent = {
        sid: g.sort_values("word_index")["word"].tolist()
        for sid, g in spacy.groupby("sentence_id")
    }

    # ---------- TEST 2 & 3 : completeness & uniqueness ----------
    sent_ids_deberta = set(deberta["sentence_id"])
    sent_ids_spacy = set(spacy["sentence_id"])

    if sent_ids_deberta != sent_ids_spacy:
        missing_in_deberta = sent_ids_spacy - sent_ids_deberta
        missing_in_spacy = sent_ids_deberta - sent_ids_spacy
        print("[ERROR] Sentence ID mismatch!")
        print("[ERROR] Missing in deberta :", missing_in_deberta)
        print("[ERROR] Missing in spaCy:", missing_in_spacy)
        sys.exit(1)

    print("[OK] ✓ Test 2 & 3 passed: sentence_id consistency verified.")

    # ---------- begin sentence-level tests ----------
    for sid, group in tqdm(deberta.groupby("sentence_id"), total=len(sent_ids_deberta)):

        group = group.sort_values("deberta_index")
        deberta_tokens = group["deberta_token"].tolist()
        word_ids = group["word_id"].tolist()

        spacy_words = spacy_by_sent[sid]
        n_words = len(spacy_words)

        # TEST 4 : deberta_index strictly sequential
        if list(group["deberta_index"]) != list(range(len(group))):
            print(f"[ERROR] Non-sequential deberta_index in sentence {sid}")
            sys.exit(1)

        # TEST 5 : special tokens
        if deberta_tokens[0] != "[CLS]":
            print(f"[ERROR] Missing [CLS] at start of sentence {sid}")
            sys.exit(1)

        if deberta_tokens[-1] != "[SEP]":
            print(f"[ERROR] Missing [SEP] at end of sentence {sid}")
            sys.exit(1)

        if pd.notna(word_ids[0]) or pd.notna(word_ids[-1]):
          print(type(word_ids[0]), repr(word_ids[0]))
          print(f"[ERROR] Special tokens have non-null word_id in sentence {sid}")
          sys.exit(1)

        # TEST 5 : validity of word_id
        for wi in word_ids:
            if wi is None or pd.isna(wi):
                continue
            if not (0 <= wi < n_words):
                print(f"[ERROR] Invalid word_id={wi} in sentence {sid}")
                print(f"[ERROR] Expected 0..{n_words-1}")
                sys.exit(1)

        # TEST 6 : no spaCy word lost
        counted = set([wi for wi in word_ids if wi is not None])
        expected = set(range(n_words))
        if expected - counted:
          print(f"[ERROR] Some spaCy words were not tokenized in sentence {sid}")
          print("[ERROR] Missing:", expected - counted)
          sys.exit(1)

        # TEST 7 & 8 : reconstruction fidelity
        reconstructed = [""] * n_words
        for tok, wi in zip(deberta_tokens, word_ids):
          if wi is None or pd.isna(wi):
            continue

          wi = int(wi)
          
          reconstructed[wi] += clean_subtoken(tok)

        for idx in range(n_words):
            original = re.sub(r"\W+", "", spacy_words[idx].lower())
            recon = re.sub(r"\W+", "", reconstructed[idx].lower())

            if original != recon:
                print(f"[ERROR] Word reconstruction mismatch in sentence {sid}")
                print(f"[ERROR] Original     : {spacy_words[idx]}")
                print(f"[ERROR] Reconstructed: {reconstructed[idx]}")
                sys.exit(1)

        # TEST 9 : invalid tokens
        for tok in deberta_tokens:
            if tok.strip() == "":
                print(f"[ERROR] Empty token in sentence {sid}")
                sys.exit(1)

    print("\n[INFO] ✓ Test 4 to 9 passed: sentence-level tests")
    print("\n[INFO] === ALL TESTS PASSED SUCCESSFULLY ===")
    print("[INFO] deberta tokenization is correct and perfectly aligned with spaCy.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deberta_tokens_csv", required=True)
    parser.add_argument("--spacy_csv", required=True)
    parser.add_argument("--log_file", required=True,
                    help="Log file to save all verification outputs")

    args = parser.parse_args()
    print(f"[INFO] Logging enabled -> {args.log_file}")
    setup_logger(args.log_file)

    verify_deberta_tokenization(args.deberta_tokens_csv, args.spacy_csv, args.log_file)
