"""Log-space math utilities and helper functions."""

import torch
import torch.nn.functional as F
from torch import Tensor


def log_softmax_top_k(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Select top-K tokens and return their log-probabilities and indices.

    Args:
        logits: Raw logits of shape (batch, vocab_size).
        k: Number of top candidates to keep.

    Returns:
        top_k_log_probs: Log-probabilities of top-K tokens, shape (batch, k).
        top_k_indices: Token indices, shape (batch, k).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    top_k_log_probs, top_k_indices = torch.topk(log_probs, k, dim=-1)
    return top_k_log_probs, top_k_indices


def log_sum_exp(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-sum-exp.

    Args:
        x: Input tensor.
        dim: Dimension to reduce.

    Returns:
        log(sum(exp(x))) along dim.
    """
    return torch.logsumexp(x, dim=dim)


def log_mean_exp(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-mean-exp.

    Computes log((1/N) * sum(exp(x))) = logsumexp(x) - log(N).

    Args:
        x: Input tensor.
        dim: Dimension to reduce.

    Returns:
        log(mean(exp(x))) along dim.
    """
    n = x.shape[dim]
    return torch.logsumexp(x, dim=dim) - torch.log(torch.tensor(n, dtype=x.dtype, device=x.device))
