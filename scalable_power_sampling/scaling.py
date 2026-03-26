"""Scaling factor (zeta) computation and jackknife bias correction."""

import torch
from torch import Tensor

from scalable_power_sampling.utils import log_mean_exp, log_sum_exp


def compute_log_scaling_factors(
    rollout_log_likelihoods: Tensor,
    alpha: float,
) -> Tensor:
    """Compute log scaling factors from rollout log-likelihoods.

    The scaling factor for candidate k is:
        zeta_k = (1/M) * sum_r exp((alpha-1) * L_r)
    where L_r is the sum of log-probs for rollout r.

    In log-space:
        log zeta_k = log_mean_exp((alpha-1) * L)

    Args:
        rollout_log_likelihoods: Shape (K, M). Each entry is the sum of
            log-probs along a rollout.
        alpha: Power exponent.

    Returns:
        log_zeta: Log scaling factors, shape (K,).
    """
    # (alpha - 1) * sum_log_probs for each rollout
    weighted = (alpha - 1.0) * rollout_log_likelihoods  # (K, M)
    log_zeta = log_mean_exp(weighted, dim=-1)  # (K,)
    return log_zeta


def compute_power_distribution(
    top_k_log_probs: Tensor,
    log_zeta: Tensor,
    alpha: float,
) -> Tensor:
    """Compute the power distribution over top-K candidates.

    p_pow(x_t) proportional to p^alpha(x_t) * zeta(x_t)

    In log-space:
        log p_pow(x_t) = alpha * log p(x_t) + log zeta(x_t) - log Z

    Args:
        top_k_log_probs: Log-probabilities of top-K candidates, shape (batch, K).
        log_zeta: Log scaling factors, shape (batch, K) or (K,).
        alpha: Power exponent.

    Returns:
        power_probs: Normalized power distribution, shape (batch, K).
    """
    log_unnorm = alpha * top_k_log_probs + log_zeta  # (batch, K)
    log_Z = log_sum_exp(log_unnorm, dim=-1, )  # (batch,)
    log_power_probs = log_unnorm - log_Z.unsqueeze(-1)
    return torch.exp(log_power_probs)


def jackknife_power_distribution(
    top_k_log_probs: Tensor,
    rollout_log_likelihoods: Tensor,
    alpha: float,
) -> Tensor:
    """Compute jackknife-corrected power distribution.

    Reduces bias from O(1/M) to O(1/M^2) using leave-one-out estimates.

    The jackknife estimate is:
        p_JK = M * p_full - ((M-1)/M) * sum_s p_LOO_s

    where p_LOO_s is the power distribution computed without rollout s.

    Args:
        top_k_log_probs: Log-probs of top-K candidates, shape (batch, K).
        rollout_log_likelihoods: Shape (K, M) or (batch, K, M).
        alpha: Power exponent.

    Returns:
        power_probs_jk: Jackknife-corrected distribution, shape (batch, K).
    """
    if rollout_log_likelihoods.dim() == 2:
        rollout_log_likelihoods = rollout_log_likelihoods.unsqueeze(0)  # (1, K, M)
    if top_k_log_probs.dim() == 1:
        top_k_log_probs = top_k_log_probs.unsqueeze(0)  # (1, K)

    batch, K, M = rollout_log_likelihoods.shape

    # Full estimate
    log_zeta_full = compute_log_scaling_factors(
        rollout_log_likelihoods.view(batch * K, M), alpha
    ).view(batch, K)
    p_full = compute_power_distribution(top_k_log_probs, log_zeta_full, alpha)

    # Leave-one-out estimates
    # For each s in [0, M), compute zeta excluding rollout s
    # Efficient: use the sum-minus-one trick in log-space
    weighted = (alpha - 1.0) * rollout_log_likelihoods  # (batch, K, M)

    p_loo_sum = torch.zeros(batch, K, device=top_k_log_probs.device)

    for s in range(M):
        # Exclude rollout s: take all indices except s
        mask = torch.ones(M, dtype=torch.bool, device=weighted.device)
        mask[s] = False
        weighted_loo = weighted[:, :, mask]  # (batch, K, M-1)

        log_zeta_loo = log_mean_exp(weighted_loo.reshape(batch * K, M - 1), dim=-1)
        log_zeta_loo = log_zeta_loo.view(batch, K)

        p_loo = compute_power_distribution(top_k_log_probs, log_zeta_loo, alpha)
        p_loo_sum += p_loo

    # Jackknife combination
    p_jk = M * p_full - ((M - 1.0) / M) * p_loo_sum

    # Clamp and renormalize (jackknife can produce negative values)
    p_jk = p_jk.clamp(min=0.0)
    p_jk = p_jk / p_jk.sum(dim=-1, keepdim=True)

    return p_jk.squeeze(0)
