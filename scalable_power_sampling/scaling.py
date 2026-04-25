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

    # Leave-one-out estimates — vectorized over all M exclusions at once.
    weighted = (alpha - 1.0) * rollout_log_likelihoods  # (batch, K, M)

    # Build LOO index: for each s in [0,M), the M-1 indices excluding s
    idx = torch.arange(M, device=weighted.device)
    loo_idx = (idx.unsqueeze(0).expand(M, M))[idx.unsqueeze(1) != idx.unsqueeze(0)].view(M, M - 1)

    # Gather all LOO sets at once: (batch, K, M) -> (batch, K, M, M-1)
    weighted_loo = weighted[:, :, loo_idx]

    # log_mean_exp over the M-1 rollouts for each LOO set
    log_M1 = torch.log(torch.tensor(M - 1, dtype=weighted.dtype, device=weighted.device))
    log_zeta_loo = torch.logsumexp(weighted_loo, dim=-1) - log_M1  # (batch, K, M)

    # Compute power distribution for all M LOO sets in parallel
    # Treat the M LOO sets as an extra batch dimension
    log_zeta_loo = log_zeta_loo.permute(0, 2, 1)                    # (batch, M, K)
    top_k_expanded = top_k_log_probs.unsqueeze(1).expand(batch, M, K)
    log_unnorm = alpha * top_k_expanded + log_zeta_loo               # (batch, M, K)
    log_Z = torch.logsumexp(log_unnorm, dim=-1, keepdim=True)        # (batch, M, 1)
    p_loo_all = torch.exp(log_unnorm - log_Z)                        # (batch, M, K)

    p_loo_sum = p_loo_all.sum(dim=1)                                 # (batch, K)

    # Jackknife combination
    p_jk = M * p_full - ((M - 1.0) / M) * p_loo_sum

    # Clamp and renormalize (jackknife can produce negative values)
    p_jk = p_jk.clamp(min=0.0)
    p_jk_sum = p_jk.sum(dim=-1, keepdim=True)
    # Fall back to uncorrected distribution if jackknife zeroed everything out
    degenerate = (p_jk_sum == 0.0).squeeze(-1)
    if degenerate.any():
        p_jk[degenerate] = p_full[degenerate]
        p_jk_sum = p_jk.sum(dim=-1, keepdim=True)
    p_jk = p_jk / p_jk_sum

    return p_jk.squeeze(0)
