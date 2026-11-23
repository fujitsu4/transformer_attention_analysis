"""
compute_attention_core_bigbird.py
Author: Zakaria JOUILIL

Optimized BigBird block-sparse attention core.
Stored tokens are converted directly to IDs; no retokenization.
Attention scores are computed only over real tokens.
"""

import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModel

MODEL_NAME = "google/bigbird-roberta-base"
MIN_SEQUENCE_LENGTH = 1024


# ---------------------------------------------------------
# Fix random seed (for untrained models)
# ---------------------------------------------------------
def fix_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Load model: pretrained or random
# ---------------------------------------------------------
def load_model(pretrained: bool, seed: int = None):

    if not pretrained:
        if seed is None:
            raise ValueError("Seed must be provided for untrained model.")
        fix_seed(seed)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        output_attentions=True
    )

    config.attention_type = "block_sparse"

    if pretrained:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config
        )
    else:
        model = AutoModel.from_config(config)

    model.config.output_attentions = True
    model.eval()

    return model



# ---------------------------------------------------------
# Switch between full and block-sparse attention (same weights)
# ---------------------------------------------------------
def set_attention_mode(model, mode: str):
    """
    Switch the attention implementation WITHOUT touching the weights
    (q/k/v Linear layers are shared by reference by HF).
    Cheap if the model is already in the requested mode.
    """
    if mode not in ("original_full", "block_sparse"):
        raise ValueError(f"Unknown attention mode: {mode}")

    if model.attention_type != mode:
        model.set_attention_type(mode)


# ---------------------------------------------------------
# FAST PATH: batched, exact for sentences <= block_size tokens
# ---------------------------------------------------------
def layer_sums_full_batch(model, tokenizer, token_lists):
    """
    Batched attention sums for SHORT sequences (len <= block_size).

    Why this is equivalent to the padded-1024 block-sparse run:
    all real tokens sit in block 0, which is a GLOBAL block in BigBird
    (it attends to every key). Padded keys are masked. So the attention
    rows of real tokens are exactly full attention over the real tokens.

    The model must be in 'original_full' mode (see set_attention_mode).

    Returns: np.ndarray (num_layers, batch, max_len) with
             sum over heads and over REAL query positions, per key.
             Positions >= len(tokens_i) are garbage/zero: slice them off.
    """
    if model.attention_type != "original_full":
        raise RuntimeError("Call set_attention_mode(model, 'original_full') first.")

    device = next(model.parameters()).device
    lens = [len(t) for t in token_lists]

    if min(lens) == 0:
        raise ValueError("Empty token sequence.")

    batch = len(token_lists)
    max_len = max(lens)

    input_ids = torch.full(
        (batch, max_len), tokenizer.pad_token_id,
        dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (batch, max_len), dtype=torch.long, device=device
    )

    for i, toks in enumerate(token_lists):
        ids = tokenizer.convert_tokens_to_ids(toks)
        input_ids[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, :len(ids)] = 1

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        if len(outputs.attentions) != model.config.num_hidden_layers:
            raise RuntimeError("Unexpected number of attention layers.")

        m = attention_mask.to(outputs.attentions[0].dtype)
        q_mask = m[:, None, :, None]   # real query rows
        k_mask = m[:, None, None, :]   # real key columns

        sums = []
        for layer_att in outputs.attentions:          # (B, H, L, L)
            masked = layer_att * q_mask * k_mask
            sums.append(masked.sum(dim=(1, 2)))       # heads, then queries -> (B, L)

        sums = torch.stack(sums, dim=0)               # (layers, B, L)

    # ONE device->host transfer for the whole batch.
    return sums.cpu().numpy()


# ---------------------------------------------------------
# Extract attention matrices and compute summaries
# ---------------------------------------------------------
def compute_attention(
    model,
    tokenizer,
    bigbird_tokens,
    return_attentions=True,
    return_head_sums=True
):

    """
    Input:
        model              : BigBird model
        tokenizer          : corresponding BigBird tokenizer
        bigbird_tokens     : stored BigBird tokens in correct order
        return_attentions  : whether to return full attention matrices
        return_head_sums   : whether to return head-wise sums

    Returns:
        Dict containing:
            tokens        : original BigBird tokens
            attentions    : full matrices if requested, otherwise None
            layer_sums    : list of (real_len,) tensors
            head_sums     : [layer][head] if requested, otherwise None
    """

    real_length = len(bigbird_tokens)

    if real_length == 0:
        raise ValueError("Empty token sequence.")

    # The legacy path must always run the true block-sparse attention.
    set_attention_mode(model, "block_sparse")

    block_size = model.config.block_size

    # BigBird block_sparse requires at least 1024 tokens.
    padded_length = max(
        MIN_SEQUENCE_LENGTH,
        ((real_length + block_size - 1) // block_size) * block_size
    )

    if padded_length > model.config.max_position_embeddings:
        raise ValueError(
            f"Sequence too long: padded length={padded_length}, "
            f"maximum={model.config.max_position_embeddings}"
        )

    # Convert stored tokens directly without retokenization.
    token_ids = tokenizer.convert_tokens_to_ids(bigbird_tokens)

    if len(token_ids) != real_length:
        raise RuntimeError(
            "Token/ID length mismatch. "
            "Stored tokens were not converted correctly."
        )

    device = next(model.parameters()).device

    # Build padded input directly on the target device.
    input_ids = torch.full(
        (1, padded_length),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )

    input_ids[0, :real_length] = torch.as_tensor(
        token_ids,
        dtype=torch.long,
        device=device
    )

    # Real tokens = 1; padding = 0.
    attention_mask = torch.zeros(
        (1, padded_length),
        dtype=torch.long,
        device=device
    )

    attention_mask[0, :real_length] = 1

    layer_sums = []
    head_sums = [] if return_head_sums else None
    attention_matrices = [] if return_attentions else None

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        if getattr(outputs, "attentions", None) is None:
            raise RuntimeError(
                "Model did not return attentions. "
                "Ensure model.config.output_attentions=True."
            )

        if len(outputs.attentions) != model.config.num_hidden_layers:
            raise RuntimeError(
                f"Expected {model.config.num_hidden_layers} layers, "
                f"got {len(outputs.attentions)}."
            )

        for layer_attention in outputs.attentions:

            # Shape: (1, num_heads, padded_len, padded_len)
            # Restrict both query and key dimensions to real tokens.
            real_attention = layer_attention[
                0, :, :real_length, :real_length
            ]

            # Same calculation as the original implementation:
            # sum over heads, then over query positions.
            layer_sum = (
                real_attention
                .sum(dim=0)
                .sum(dim=0)
            )

            layer_sums.append(layer_sum)

            # Optional head-wise sums.
            if return_head_sums:

                per_head = real_attention.sum(dim=1)
                head_sums.append([
                    per_head[h]
                    for h in range(per_head.shape[0])
                ])

            # Optional full attention matrices.
            if return_attentions:
                attention_matrices.append(real_attention)

        if return_attentions:
            attentions = torch.stack(attention_matrices, dim=0)
        else:
            attentions = None

    # Release references to model outputs.
    del outputs
    del input_ids
    del attention_mask

    return {
        "tokens": bigbird_tokens,
        "attentions": attentions,
        "layer_sums": layer_sums,
        "head_sums": head_sums
    }