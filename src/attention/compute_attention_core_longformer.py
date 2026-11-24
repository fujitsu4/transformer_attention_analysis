
"""
compute_attention_core_longformer.py
Author: Zakaria JOUILIL

Longformer sliding-window attention core.
Stored tokens are converted directly to IDs; no retokenization.

Attention received by each token is reconstructed from the local
sliding-window attention probabilities.

No global_attention_mask is used:
all tokens, including <s>, use local attention only.
"""

import random
import numpy as np
import torch
from transformers import AutoConfig, AutoModel


MODEL_NAME = "allenai/longformer-base-4096"


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
# Attention-window helpers
# ---------------------------------------------------------

def get_attention_windows(model):

    windows = model.config.attention_window

    if isinstance(windows, int):
        windows = [windows] * model.config.num_hidden_layers

    if len(windows) != model.config.num_hidden_layers:
        raise RuntimeError(
            "attention_window configuration does not match "
            "the number of layers."
        )

    return [int(w) for w in windows]


def padded_length_for(model, real_length: int) -> int:
    """
    Longformer requires sequence lengths to be multiples
    of the attention window.
    """

    if real_length <= 0:
        raise ValueError("Empty token sequence.")

    windows = get_attention_windows(model)

    if len(set(windows)) != 1:
        raise RuntimeError(
            "This implementation expects the same attention window "
            "for every Longformer layer."
        )

    window = windows[0]

    padded_length = (
        (real_length + window - 1) // window
    ) * window

    max_length = model.config.max_position_embeddings

    # RoBERTa-style position embeddings reserve positions
    # for padding_idx.
    pad_token_id = model.config.pad_token_id

    if pad_token_id is not None:
        max_length = min(
            max_length,
            model.config.max_position_embeddings - pad_token_id - 1
        )

    if padded_length > max_length:
        raise ValueError(
            f"Sequence too long: padded length={padded_length}, "
            f"maximum usable length={max_length}."
        )

    return padded_length


# ---------------------------------------------------------
# Load model: pretrained or random
# ---------------------------------------------------------

def load_model(pretrained: bool, seed: int = None):

    if not pretrained:
        if seed is None:
            raise ValueError(
                "Seed must be provided for untrained model."
            )
        fix_seed(seed)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        output_attentions=True
    )

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
# Reconstruct attention received by each key token
# ---------------------------------------------------------


def _local_attention_sums(
    layer_attention,
    attention_mask,
    attention_window,
    layer_index
):
    """
    Convert Longformer's local attention probabilities into
    attention received by each absolute token position.

    Expected input shape:
        (batch, num_heads, sequence_length, attention_window + 1)
    """

    if layer_attention.ndim != 4:
        raise RuntimeError(
            f"Layer {layer_index}: expected a 4D attention tensor, "
            f"got shape {tuple(layer_attention.shape)}."
        )

    batch_size, num_heads, seq_len, window_slots = layer_attention.shape

    expected_slots = attention_window + 1

    if window_slots != expected_slots:
        raise RuntimeError(
            f"Layer {layer_index}: expected {expected_slots} local slots, "
            f"got {window_slots}."
        )

    if attention_mask.shape != (batch_size, seq_len):
        raise RuntimeError(
            f"Layer {layer_index}: attention mask shape "
            f"{tuple(attention_mask.shape)} does not match "
            f"batch/sequence shape {(batch_size, seq_len)}."
        )

    # Sum attention over all heads.
    # Result: (batch, sequence_length, window_slots)
    attention = layer_attention.sum(dim=1)

    half_window = attention_window // 2

    # Relative positions represented by the local attention slots.
    offsets = (
        torch.arange(
            window_slots,
            device=layer_attention.device
        ) - half_window
    )

    query_positions = torch.arange(
        seq_len,
        device=layer_attention.device
    )

    key_positions = (
        query_positions[:, None] + offsets[None, :]
    )

    valid_keys = (
        (key_positions >= 0) &
        (key_positions < seq_len)
    )

    safe_key_positions = key_positions.clamp(
        min=0,
        max=seq_len - 1
    )

    key_mask = attention_mask.bool()
    valid_keys = valid_keys.unsqueeze(0) & key_mask[:, safe_key_positions]

    # Ignore padded query positions.
    valid_queries = key_mask.unsqueeze(-1)

    valid = valid_keys & valid_queries

    attention = attention * valid.to(attention.dtype)

    # Accumulate local attention probabilities into absolute key positions.
    result = torch.zeros(
        (batch_size, seq_len),
        dtype=attention.dtype,
        device=attention.device
    )

    indices = safe_key_positions.unsqueeze(0).expand(
        batch_size, -1, -1
    )

    result.scatter_add_(
        dim=1,
        index=indices.reshape(batch_size, -1),
        src=attention.reshape(batch_size, -1)
    )

    return result



# ---------------------------------------------------------
# Batched Longformer attention sums
# ---------------------------------------------------------

def layer_sums_local_batch(model, tokenizer, token_lists):
    """
    Compute attention received by each token for a batch of sentences.

    Input:
        token_lists: list of stored Longformer token lists.

    Returns:
        np.ndarray of shape:
            (num_layers, batch_size, max_real_length)

        For sentence i, only:
            sums[:, i, :len(token_lists[i])]
        is meaningful.
    """

    device = next(model.parameters()).device

    if not token_lists:
        raise ValueError("Empty batch.")

    lens = [len(tokens) for tokens in token_lists]

    if min(lens) <= 0:
        raise ValueError("Empty token sequence.")

    padded_lengths = [
        padded_length_for(model, length)
        for length in lens
    ]

    if len(set(padded_lengths)) != 1:
        raise ValueError(
            "All sequences in a batch must have the same padded length."
        )

    padded_length = padded_lengths[0]
    batch_size = len(token_lists)
    max_real_length = max(lens)

    input_ids = torch.full(
        (batch_size, padded_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.zeros(
        (batch_size, padded_length),
        dtype=torch.long,
        device=device
    )

    for i, tokens in enumerate(token_lists):

        ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(ids) != len(tokens):
            raise RuntimeError(
                f"Sentence {i}: token/ID length mismatch."
            )

        input_ids[i, :len(ids)] = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=device
        )

        attention_mask[i, :len(ids)] = 1

    attention_windows = get_attention_windows(model)

    if len(set(attention_windows)) != 1:
        raise RuntimeError(
            "Different attention windows across layers are not supported."
        )

    attention_window = attention_windows[0]

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        attentions = getattr(outputs, "attentions", None)

        n_layers = model.config.num_hidden_layers

        if attentions is None or len(attentions) != n_layers:
            raise RuntimeError(
                f"Expected {n_layers} attention tensors, "
                f"got {None if attentions is None else len(attentions)}."
            )

        # Restrict the mask to the returned sequence length.
        real_mask = attention_mask[:, :max_real_length]

        layer_sums = []

        for layer_index, layer_attention in enumerate(attentions):

            # Longformer returns local sliding-window probabilities,
            # not a dense (sequence_length x sequence_length) matrix.
            sums = _local_attention_sums(
                layer_attention=layer_attention,
                attention_mask=attention_mask,
                attention_window=attention_window,
                layer_index=layer_index
            )

            # Keep only positions corresponding to real tokens.
            layer_sums.append(
                sums[:, :max_real_length]
            )

        sums = torch.stack(
            layer_sums,
            dim=0
        )

    # One GPU -> CPU transfer for the whole batch.
    return sums.cpu().numpy()


# ---------------------------------------------------------
# Single-sentence compatibility function
# ---------------------------------------------------------

def compute_attention(
    model,
    tokenizer,
    longformer_tokens
):
    """
    Compute attention received by each token for one sentence.

    Returns:
        {
            "tokens": original tokens,
            "layer_sums": list of numpy arrays, one per layer
        }
    """

    sums = layer_sums_local_batch(
        model,
        tokenizer,
        [longformer_tokens]
    )

    return {
        "tokens": longformer_tokens,
        "layer_sums": [
            sums[layer_index, 0, :len(longformer_tokens)]
            for layer_index in range(sums.shape[0])
        ]
    }
