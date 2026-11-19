
"""
compute_attention_core_t5.py
Author: Zakaria JOUILIL

Description:
    Core module for extracting encoder attention matrices
    from a T5 model (pretrained or randomly initialized).

    This module does NOT save files.

Output format:
    {
        "tokens": list of T5 tokens,
        "attentions": tensor
                      (num_layers, num_heads, seq_len, seq_len),
        "layer_sums": list of tensors (seq_len,),
        "head_sums": list of lists of tensors (seq_len,)
    }

    Attention extraction is performed on the T5 ENCODER only.
"""

import torch
from transformers import T5Config, T5Model


MODEL_NAME = "google-t5/t5-small"


# ---------------------------------------------------------
# Fix random seed
# ---------------------------------------------------------
def fix_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Load T5 model: pretrained or random
# ---------------------------------------------------------
def load_model(pretrained: bool, seed: int = None):

    if pretrained:

        print("[INFO] Loading PRETRAINED T5-small model ...")

        config = T5Config.from_pretrained(MODEL_NAME)
        config.output_attentions = True

        model = T5Model.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager"
        )

    else:

        if seed is None:
            raise ValueError(
                "Seed must be provided for untrained model."
            )

        print(
            f"[INFO] Loading RANDOMLY INITIALIZED T5-small "
            f"model with seed {seed} ..."
        )

        fix_seed(seed)

        config = T5Config.from_pretrained(MODEL_NAME)
        config.output_attentions = True

        model = T5Model(config)

    model.config.output_attentions = True
    model.eval()

    return model


# ---------------------------------------------------------
# Extract T5 ENCODER attention matrices
# ---------------------------------------------------------
def compute_attention(model, tokenizer, t5_tokens):

    """
    Input:
        model       : T5Model
        tokenizer   : T5 tokenizer
        t5_tokens   : list of T5 tokens in correct order

    Returns:
        Dict containing:

            tokens:
                Original T5 tokens.

            attentions:
                (num_layers, num_heads, seq_len, seq_len)

            layer_sums:
                List of (seq_len,) tensors.
                Sum over all heads and all query positions.

            head_sums:
                [layer][head] -> (seq_len,)
                Sum over all query positions.
    """

    # -----------------------------------------------------
    # 1. Validate input
    # -----------------------------------------------------

    if len(t5_tokens) == 0:
        raise ValueError("Empty token sequence.")

    # -----------------------------------------------------
    # 2. Convert existing T5 tokens to token IDs
    #    NO RETOKENIZATION
    # -----------------------------------------------------

    input_ids_list = tokenizer.convert_tokens_to_ids(t5_tokens)

    if isinstance(input_ids_list, int):
        input_ids_list = [input_ids_list]

    vocab_size = model.shared.num_embeddings

    invalid_ids = [
        (token, token_id)
        for token, token_id in zip(t5_tokens, input_ids_list)
        if token_id is None or token_id < 0 or token_id >= vocab_size
    ]

    if invalid_ids:
        raise ValueError(
            f"Invalid T5 token IDs. "
            f"Model vocabulary size: {vocab_size}. "
            f"Examples: {invalid_ids[:5]}"
        )

    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long
    )

    attention_mask = torch.ones_like(input_ids)

    # -----------------------------------------------------
    # 3. Move inputs to model device
    # -----------------------------------------------------

    device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # -----------------------------------------------------
    # 4. Forward pass through T5 ENCODER ONLY
    # -----------------------------------------------------

    with torch.no_grad():

        outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        if getattr(outputs, "attentions", None) is None:
            raise RuntimeError(
                "T5 encoder did not return attentions. "
                "Check output_attentions=True."
            )

        # Tuple of num_layers tensors:
        # (batch, num_heads, seq_len, seq_len)

        attentions = torch.stack(
            outputs.attentions,
            dim=0
        ).squeeze(1)

    # -----------------------------------------------------
    # 5. Validate attention dimensions
    # -----------------------------------------------------

    num_layers, num_heads, seq_len, seq_len_2 = attentions.shape

    if seq_len != len(t5_tokens):
        raise RuntimeError(
            f"Sequence mismatch: "
            f"attention={seq_len}, tokens={len(t5_tokens)}"
        )

    if seq_len != seq_len_2:
        raise RuntimeError("Attention matrix is not square.")

    # -----------------------------------------------------
    # 6. Compute layer-wise attention sums
    # -----------------------------------------------------

    layer_sums = []

    for L in range(num_layers):

        layer_sum = (
            attentions[L]
            .sum(dim=0)
            .sum(dim=0)
        )

        layer_sums.append(layer_sum)

    # -----------------------------------------------------
    # 7. Compute head-wise attention sums
    # -----------------------------------------------------

    head_sums = []

    for L in range(num_layers):

        per_layer = []

        for H in range(num_heads):

            head_sum = attentions[L][H].sum(dim=0)

            per_layer.append(head_sum)

        head_sums.append(per_layer)

    # -----------------------------------------------------
    # 8. Return results
    # -----------------------------------------------------

    return {
        "tokens": t5_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
        "head_sums": head_sums
    }
