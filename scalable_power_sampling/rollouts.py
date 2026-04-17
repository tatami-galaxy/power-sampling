"""Batched rollout generation with log-likelihood tracking."""

import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache


@torch.no_grad()
def generate_rollouts(
    model: PreTrainedModel,
    prefix_ids: Tensor,
    candidate_token_ids: Tensor,
    num_rollouts: int,
    lookahead: int,
    past_key_values=None,
) -> tuple[Tensor, Tensor]:
    """Generate rollouts from each candidate token and track log-likelihoods.

    For each candidate token, generates num_rollouts completions of length
    lookahead, collecting the sum of log-probabilities along each rollout.

    Args:
        model: HuggingFace causal LM.
        prefix_ids: Token IDs for the shared prefix, shape (1, seq_len).
        candidate_token_ids: Top-K candidate token IDs, shape (K,).
        num_rollouts: Number of rollouts (M) per candidate.
        lookahead: Number of tokens (H) to generate per rollout.
        past_key_values: Optional cached key-values for the prefix.

    Returns:
        rollout_log_likelihoods: Sum of log-probs for each rollout,
            shape (K, M). Each entry is sum_{s=1}^{H} log p(x_s | x_{<s}).
        past_key_values: KV-cache for the prefix (reusable for next position).
    """
    device = prefix_ids.device
    K = candidate_token_ids.shape[0]
    M = num_rollouts
    eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    # Step 1: Compute prefix KV-cache if not provided
    if past_key_values is None:
        prefix_out = model(prefix_ids, use_cache=True)
        past_key_values = prefix_out.past_key_values

    # Step 2: Process all K candidates in one forward pass to get their KV-caches
    # and the logits for the first rollout position.
    # Shape: (K, 1)
    candidate_input = candidate_token_ids.unsqueeze(-1)

    # Expand prefix cache for K candidates
    candidate_past = _expand_kv_cache(past_key_values, K)

    candidate_out = model(
        candidate_input,
        past_key_values=candidate_past,
        use_cache=True,
    )
    # candidate_out.logits: (K, 1, vocab_size)
    # candidate_out.past_key_values: cache including the candidate token
    candidate_cache = candidate_out.past_key_values
    first_logits = candidate_out.logits[:, -1, :]  # (K, vocab_size)

    # Step 3: For each candidate, generate M rollouts
    # We process one candidate at a time to manage memory.
    rollout_log_likelihoods = torch.zeros(K, M, device=device)

    for k in range(K):
        # Extract KV-cache for this candidate and expand for M rollouts
        single_cache = _select_kv_cache(candidate_cache, k)
        rollout_cache = _expand_kv_cache(single_cache, M)

        # First token logits for this candidate, expanded for M rollouts
        logits = first_logits[k].unsqueeze(0).expand(M, -1)  # (M, vocab_size)

        # Track cumulative log-likelihood and active mask
        cum_log_prob = torch.zeros(M, device=device)
        active = torch.ones(M, dtype=torch.bool, device=device)

        for h in range(lookahead):
            # Sample next tokens
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (M, 1)

            # Collect log-prob of sampled tokens
            log_probs = torch.log_softmax(logits, dim=-1)
            sampled_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)  # (M,)

            # Only accumulate for active (non-EOS) sequences
            cum_log_prob += sampled_log_probs * active.float()

            # Check for EOS
            if eos_token_id is not None:
                active = active & (next_tokens.squeeze(-1) != eos_token_id)
                if not active.any():
                    break

            # Forward pass for next step
            out = model(next_tokens, past_key_values=rollout_cache, use_cache=True)
            rollout_cache = out.past_key_values
            logits = out.logits[:, -1, :]  # (M, vocab_size)

        rollout_log_likelihoods[k] = cum_log_prob

    return rollout_log_likelihoods, past_key_values


def _transform_cache(past_key_values, fn):
    """Apply `fn` to each layer's (keys, values) and rebuild a DynamicCache.

    Iteration over a transformers>=5 `Cache` yields
    `(keys, values, sliding_window_tensor_or_None)` per layer; earlier versions
    yielded `(keys, values)`. The `DynamicCache(ddp_cache_data=...)` constructor
    accepts either arity, so we forward the sliding-window tensor through
    unchanged when present.
    """
    if isinstance(past_key_values, Cache):
        data = []
        for entry in past_key_values:
            keys, values = entry[0], entry[1]
            new_keys, new_values = fn(keys), fn(values)
            if len(entry) == 3 and entry[2] is not None:
                data.append((new_keys, new_values, entry[2]))
            else:
                data.append((new_keys, new_values))
        return DynamicCache(ddp_cache_data=data)

    # Legacy tuple-of-tuples format
    transformed = [tuple(fn(t) for t in layer_past) for layer_past in past_key_values]
    if isinstance(past_key_values, list):
        return transformed
    return type(past_key_values)(transformed)


def _expand_kv_cache(past_key_values, n: int):
    """Repeat each KV-cache tensor n times along the batch dimension."""
    return _transform_cache(past_key_values, lambda t: t.repeat_interleave(n, dim=0))


def _select_kv_cache(past_key_values, index: int):
    """Select a single batch element from a KV-cache."""
    return _transform_cache(past_key_values, lambda t: t[index : index + 1].clone())


def _batch_select_kv_cache(past_key_values, indices: Tensor):
    """Select multiple batch elements from a KV-cache by index tensor."""
    return _transform_cache(past_key_values, lambda t: t[indices])
