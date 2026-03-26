"""Tests for scaling factor computation and jackknife correction."""

import torch
import pytest

from scalable_power_sampling.utils import log_mean_exp, log_sum_exp, log_softmax_top_k
from scalable_power_sampling.scaling import (
    compute_log_scaling_factors,
    compute_power_distribution,
    jackknife_power_distribution,
)


class TestLogSpaceMath:
    def test_log_sum_exp_matches_naive(self):
        x = torch.randn(5)
        expected = torch.log(torch.exp(x).sum())
        result = log_sum_exp(x, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_log_mean_exp_matches_naive(self):
        x = torch.randn(5)
        expected = torch.log(torch.exp(x).mean())
        result = log_mean_exp(x, dim=0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_log_mean_exp_large_values(self):
        """Verify numerical stability with large magnitudes."""
        x = torch.tensor([-500.0, -501.0, -499.0])
        result = log_mean_exp(x, dim=0)
        assert torch.isfinite(result)

    def test_log_softmax_top_k(self):
        logits = torch.randn(1, 100)
        log_probs, indices = log_softmax_top_k(logits, k=5)
        assert log_probs.shape == (1, 5)
        assert indices.shape == (1, 5)
        # All log-probs should be <= 0
        assert (log_probs <= 0).all()
        # Indices should be sorted by probability (descending)
        assert (log_probs[0, :-1] >= log_probs[0, 1:]).all()


class TestScalingFactors:
    def test_shape(self):
        rollout_ll = torch.randn(8, 8)  # K=8, M=8
        log_zeta = compute_log_scaling_factors(rollout_ll, alpha=4.0)
        assert log_zeta.shape == (8,)

    def test_alpha_1_gives_zero(self):
        """With alpha=1, scaling factors should be log(1)=0 (no sharpening)."""
        rollout_ll = torch.randn(4, 8)
        log_zeta = compute_log_scaling_factors(rollout_ll, alpha=1.0)
        # (alpha-1) * anything = 0, so log_mean_exp(zeros) = 0
        assert torch.allclose(log_zeta, torch.zeros_like(log_zeta), atol=1e-6)

    def test_higher_alpha_favors_higher_likelihood(self):
        """Higher alpha should amplify differences in scaling factors."""
        rollout_ll = torch.tensor([
            [-1.0, -1.5, -1.2, -0.8],  # candidate 0: higher likelihood rollouts
            [-5.0, -6.0, -5.5, -4.5],  # candidate 1: lower likelihood rollouts
        ])
        log_zeta_low = compute_log_scaling_factors(rollout_ll, alpha=2.0)
        log_zeta_high = compute_log_scaling_factors(rollout_ll, alpha=8.0)
        # Both should prefer candidate 0, but high alpha more strongly
        diff_low = log_zeta_low[0] - log_zeta_low[1]
        diff_high = log_zeta_high[0] - log_zeta_high[1]
        assert diff_high > diff_low


class TestPowerDistribution:
    def test_sums_to_one(self):
        top_k_log_probs = torch.log_softmax(torch.randn(1, 8), dim=-1)
        log_zeta = torch.randn(1, 8)
        probs = compute_power_distribution(top_k_log_probs, log_zeta, alpha=4.0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_alpha_1_no_zeta_recovers_base(self):
        """With alpha=1 and uniform zeta, should recover base distribution."""
        top_k_log_probs = torch.log_softmax(torch.randn(1, 8), dim=-1)
        log_zeta = torch.zeros(1, 8)
        probs = compute_power_distribution(top_k_log_probs, log_zeta, alpha=1.0)
        base_probs = torch.exp(top_k_log_probs)
        base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)
        assert torch.allclose(probs, base_probs, atol=1e-5)


class TestJackknife:
    def test_output_is_valid_distribution(self):
        top_k_log_probs = torch.log_softmax(torch.randn(8), dim=-1)
        rollout_ll = torch.randn(8, 8) * 2  # K=8, M=8
        probs = jackknife_power_distribution(top_k_log_probs, rollout_ll, alpha=4.0)
        assert probs.shape[-1] == 8
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
        assert (probs >= 0).all()

    def test_jackknife_close_to_full_with_many_rollouts(self):
        """With many rollouts, jackknife and full estimate should converge."""
        torch.manual_seed(42)
        top_k_log_probs = torch.log_softmax(torch.randn(8), dim=-1)
        rollout_ll = torch.randn(8, 64)  # M=64 rollouts
        probs_jk = jackknife_power_distribution(
            top_k_log_probs, rollout_ll, alpha=4.0
        )
        log_zeta = compute_log_scaling_factors(rollout_ll, alpha=4.0)
        probs_full = compute_power_distribution(
            top_k_log_probs.unsqueeze(0), log_zeta.unsqueeze(0), alpha=4.0
        )
        assert torch.allclose(probs_jk.view(-1), probs_full.view(-1), atol=0.05)
