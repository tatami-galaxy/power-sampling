"""Power-SMC sampling for HuggingFace causal LMs.

Implements Algorithm 1 from "Power-SMC: Low-Latency Sequence-Level Power
Sampling for Training-Free LLM Reasoning" (Zhao, Heng & Doucet, 2025).

Instead of rollout-based scaling factors (Scalable Power Sampling), Power-SMC
maintains N particles in parallel with importance weights that correct toward
the sequence-level power distribution pi_alpha(y|x) ~ p(y|x)^alpha.

Key features:
  - Token-level proposal at temperature tau=1/alpha (proven locally optimal)
  - ESS-triggered systematic resampling with KV cache reordering
  - Linear alpha-ramping to reduce early weight divergence
  - Absorbing EOS state: done particles contribute no further weight updates
"""

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from scalable_power_sampling.rollouts import (
    _batch_select_kv_cache,
    _expand_kv_cache,
)


def systematic_resample(log_weights: Tensor) -> Tensor:
    """Systematic resampling from log-weights.

    Draws ancestor indices using a single uniform offset with evenly spaced
    positions on the weight CDF. Unbiased and lower-variance than multinomial
    resampling.

    Args:
        log_weights: Unnormalized log-weights, shape (N,).

    Returns:
        Ancestor indices, shape (N,), dtype long.
    """
    N = log_weights.shape[0]
    w = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
    cdf = torch.cumsum(w, dim=0)
    u0 = torch.rand(1, device=log_weights.device, dtype=w.dtype) / N
    positions = u0 + torch.arange(
        N, device=log_weights.device, dtype=w.dtype
    ) / N
    indices = torch.searchsorted(cdf, positions).clamp(max=N - 1)
    return indices.long()


def _alpha_ramp(t: int, alpha_final: float, ramp_tokens: int) -> float:
    """Linear alpha schedule from 1.0 to alpha_final over ramp_tokens steps.

    Args:
        t: Current generation step (0-indexed).
        alpha_final: Target power exponent.
        ramp_tokens: Number of tokens over which to ramp. <=1 disables.

    Returns:
        Current alpha value.
    """
    if ramp_tokens <= 1:
        return alpha_final
    if t < ramp_tokens:
        return 1.0 + (alpha_final - 1.0) * (t + 1) / ramp_tokens
    return alpha_final


class SMCPowerSampler:
    """Power-SMC sampling for autoregressive language models.

    Maintains N weighted particles that generate tokens in parallel. At each
    step, tokens are sampled from a temperature-scaled proposal (tau=1/alpha_t),
    and importance weights correct toward the power distribution. When the
    effective sample size drops below a threshold, systematic resampling
    duplicates high-weight particles and discards low-weight ones.

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        alpha: Power exponent (>=1). Default 4.0.
        n_particles: Number of SMC particles (N). Default 64.
        ess_threshold: Resample when ESS < threshold * N. Default 0.5.
        block_size: Check ESS every block_size tokens. Default 64.
        alpha_ramp_tokens: Linear ramp from alpha=1 to target alpha over this
            many tokens. Set to 0 or 1 to disable. Default 100.
        max_new_tokens: Maximum tokens to generate. Default 2048.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float = 4.0,
        n_particles: int = 64,
        ess_threshold: float = 0.5,
        block_size: int = 64,
        alpha_ramp_tokens: int = 100,
        max_new_tokens: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.block_size = block_size
        self.alpha_ramp_tokens = alpha_ramp_tokens
        self.max_new_tokens = max_new_tokens

        self.model.eval()
        self.device = next(model.parameters()).device

        eos = tokenizer.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        self.eos_token_id = eos

    @torch.no_grad()
    def generate(
        self,
        prompt: str | None = None,
        input_ids: Tensor | None = None,
        verbose: bool = False,
    ) -> dict:
        """Generate a completion using Power-SMC.

        Args:
            prompt: Text prompt to complete.
            input_ids: Pre-tokenized input, shape (1, seq_len).
            verbose: If True, print ESS and resampling events.

        Returns:
            dict with keys:
                - "text": Generated text (excluding prompt).
                - "input_ids": Full token sequence including prompt.
                - "num_tokens_generated": Number of new tokens.
        """
        if input_ids is None:
            assert prompt is not None, "Provide either prompt or input_ids"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        prompt_len = input_ids.shape[1]
        N = self.n_particles

        # --- Shared prompt cache (forward at batch=1, then replicate) ---
        out = self.model(input_ids, use_cache=True)
        cache = _expand_kv_cache(out.past_key_values, N)
        logits = out.logits[:, -1, :].expand(N, -1).clone()  # (N, V)

        # --- Particle state ---
        generated = torch.zeros(
            N, self.max_new_tokens, dtype=torch.long, device=self.device
        )
        gen_len = torch.zeros(N, dtype=torch.long, device=self.device)
        log_weights = torch.zeros(N, device=self.device)
        cum_logp = torch.zeros(N, device=self.device)
        done = torch.zeros(N, dtype=torch.bool, device=self.device)
        prev_alpha = 1.0 if self.alpha_ramp_tokens > 1 else self.alpha

        # --- Main generation loop ---
        for t in range(self.max_new_tokens):
            alpha_t = _alpha_ramp(t, self.alpha, self.alpha_ramp_tokens)

            # Alpha-ramping weight correction (Appendix B of Power-SMC):
            # when alpha changes, reweight by p(y_{1:t-1})^{alpha_t - alpha_{t-1}}
            delta = alpha_t - prev_alpha
            if delta != 0.0:
                log_weights[~done] += delta * cum_logp[~done]

            # Proposal: temperature tau = 1/alpha_t (Theorem 1: locally optimal)
            tau = 1.0 / alpha_t if alpha_t > 0 else 1.0
            log_p = torch.log_softmax(logits, dim=-1)  # base log p
            log_q = torch.log_softmax(logits / tau, dim=-1)  # proposal log q

            # Sample tokens from proposal
            next_tokens = torch.multinomial(torch.exp(log_q), 1)  # (N, 1)

            # Done particles: force EOS, absorbing state (Appendix D)
            if done.any():
                next_tokens[done] = self.eos_token_id

            # Per-token log-probs under base and proposal
            token_logp = log_p.gather(1, next_tokens).squeeze(-1)  # (N,)
            token_logq = log_q.gather(1, next_tokens).squeeze(-1)  # (N,)

            # Zero out for done particles (weight unchanged in absorbing state)
            token_logp = token_logp.masked_fill(done, 0.0)
            token_logq = token_logq.masked_fill(done, 0.0)

            # Incremental importance weight (Eq. 8 of Power-SMC):
            #   log w_t = alpha_t * log p(y_t) - log q(y_t)
            #           = (alpha_t - 1) * log p(y_t) + (log p(y_t) - log q(y_t))
            log_weights += (alpha_t - 1.0) * token_logp + (
                token_logp - token_logq
            )
            cum_logp += token_logp

            # Store generated tokens and track lengths
            generated[:, t] = next_tokens.squeeze(-1)
            gen_len[~done] = t + 1

            # Update done status
            if self.eos_token_id is not None:
                done = done | (next_tokens.squeeze(-1) == self.eos_token_id)

            if done.all():
                if verbose:
                    print(f"  All particles done at step {t + 1}")
                break

            # Forward pass for next step (all particles including done —
            # done particles emit EOS but we still advance the cache so
            # batch indexing stays aligned)
            out = self.model(
                next_tokens, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]  # (N, V)

            # ESS check at block boundaries
            if (t + 1) % self.block_size == 0:
                w = torch.exp(
                    log_weights - torch.logsumexp(log_weights, dim=0)
                )
                ess = 1.0 / (w**2).sum().item()

                if verbose:
                    print(f"  Step {t + 1}: ESS = {ess:.1f}/{N}")

                if ess < self.ess_threshold * N:
                    ancestors = systematic_resample(log_weights)

                    # Reorder all particle state by ancestor indices
                    generated = generated[ancestors]
                    gen_len = gen_len[ancestors]
                    done = done[ancestors]
                    cum_logp = cum_logp[ancestors]
                    cache = _batch_select_kv_cache(cache, ancestors)
                    logits = logits[ancestors]

                    log_weights.zero_()

                    if verbose:
                        print(
                            f"    Resampled "
                            f"(ESS={ess:.1f} < "
                            f"{self.ess_threshold * N:.1f})"
                        )

            prev_alpha = alpha_t

        # --- Final alpha catch-up (if ramp didn't reach target) ---
        if prev_alpha != self.alpha:
            delta_final = self.alpha - prev_alpha
            log_weights += delta_final * cum_logp

        # --- Select one particle proportional to final weights ---
        w = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
        selected = torch.multinomial(w, 1).item()

        n_gen = gen_len[selected].item()
        generated_ids = generated[selected, :n_gen].tolist()

        # Truncate at first EOS (inclusive)
        if self.eos_token_id is not None and self.eos_token_id in generated_ids:
            eos_pos = generated_ids.index(self.eos_token_id)
            generated_ids = generated_ids[: eos_pos + 1]

        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        full_ids = input_ids[0].tolist() + generated_ids

        return {
            "text": generated_text,
            "input_ids": full_ids,
            "num_tokens_generated": len(generated_ids),
        }

    def __repr__(self) -> str:
        return (
            f"SMCPowerSampler(alpha={self.alpha}, "
            f"n_particles={self.n_particles}, "
            f"ess_threshold={self.ess_threshold}, "
            f"block_size={self.block_size}, "
            f"alpha_ramp_tokens={self.alpha_ramp_tokens})"
        )
