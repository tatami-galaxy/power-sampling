"""Main PowerSampler: scalable power sampling for HuggingFace causal LMs."""

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from scalable_power_sampling.rollouts import generate_rollouts
from scalable_power_sampling.scaling import jackknife_power_distribution
from scalable_power_sampling.utils import log_softmax_top_k


class PowerSampler:
    """Scalable power sampling for autoregressive language models.

    Implements the single-token algorithm from "Scalable Power Sampling:
    Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution
    Sharpening" (Ji et al., 2025).

    At each token position:
      1. Select top-K candidate tokens by base model probability
      2. For each candidate, generate M rollouts of H tokens
      3. Compute scaling factors from rollout likelihoods
      4. Apply jackknife bias correction
      5. Sample from the power distribution p^alpha * zeta

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        alpha: Power exponent. Higher values sharpen more. Default 4.0.
        top_k: Number of candidate tokens per position. Default 8.
        num_rollouts: Number of rollouts per candidate. Default 8.
        lookahead: Rollout horizon in tokens. Default 192.
        max_new_tokens: Maximum tokens to generate. Default 3072.
        use_jackknife: Whether to apply jackknife bias correction. Default True.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float = 4.0,
        top_k: int = 8,
        num_rollouts: int = 8,
        lookahead: int = 192,
        max_new_tokens: int = 3072,
        use_jackknife: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.top_k = top_k
        self.num_rollouts = num_rollouts
        self.lookahead = lookahead
        self.max_new_tokens = max_new_tokens
        self.use_jackknife = use_jackknife

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
        """Generate a completion using scalable power sampling.

        Either prompt or input_ids must be provided.

        Args:
            prompt: Text prompt to complete.
            input_ids: Pre-tokenized input, shape (1, seq_len).
            verbose: If True, print progress every 10 tokens.

        Returns:
            dict with keys:
                - "text": Generated text (excluding prompt).
                - "input_ids": Full sequence including prompt.
                - "num_tokens_generated": Number of new tokens.
                - "token_log_probs": Log-prob of each sampled token under
                  the power distribution.
        """
        if input_ids is None:
            assert prompt is not None, "Provide either prompt or input_ids"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        prompt_len = input_ids.shape[1]
        generated_log_probs = []
        prefix_cache = None

        for step in range(self.max_new_tokens):
            # 1. Forward pass on current sequence to get logits + cache
            if prefix_cache is None:
                out = self.model(input_ids, use_cache=True)
                prefix_cache = out.past_key_values
                logits = out.logits[:, -1, :]  # (1, vocab_size)
            else:
                # Only feed the last token, reuse cache
                last_token = input_ids[:, -1:]
                out = self.model(last_token, past_key_values=prefix_cache, use_cache=True)
                prefix_cache = out.past_key_values
                logits = out.logits[:, -1, :]  # (1, vocab_size)

            # 2. Select top-K candidates
            top_k_log_probs, top_k_indices = log_softmax_top_k(logits, self.top_k)
            # top_k_log_probs: (1, K), top_k_indices: (1, K)

            if self.alpha == 1.0:
                # No sharpening: sample from base distribution restricted to top-K
                probs = torch.exp(top_k_log_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(probs.squeeze(0), 1)
                sampled_log_prob = top_k_log_probs[0, idx.item()].item()
            else:
                # 3. Generate rollouts for each candidate
                rollout_ll, _ = generate_rollouts(
                    model=self.model,
                    prefix_ids=input_ids,
                    candidate_token_ids=top_k_indices.squeeze(0),
                    num_rollouts=self.num_rollouts,
                    lookahead=self.lookahead,
                    past_key_values=prefix_cache,
                )
                # rollout_ll: (K, M)

                # 4. Compute power distribution (with or without jackknife)
                if self.use_jackknife:
                    power_probs = jackknife_power_distribution(
                        top_k_log_probs.squeeze(0),
                        rollout_ll,
                        self.alpha,
                    )
                else:
                    from scalable_power_sampling.scaling import (
                        compute_log_scaling_factors,
                        compute_power_distribution,
                    )
                    log_zeta = compute_log_scaling_factors(rollout_ll, self.alpha)
                    power_probs = compute_power_distribution(
                        top_k_log_probs, log_zeta.unsqueeze(0), self.alpha
                    )

                # power_probs: (1, K) or (K,)
                power_probs = power_probs.view(-1)

                # 5. Sample from power distribution
                idx = torch.multinomial(power_probs, 1)
                sampled_log_prob = torch.log(power_probs[idx.item()]).item()

            # Map back to vocabulary token ID
            next_token_id = top_k_indices[0, idx.item()].unsqueeze(0).unsqueeze(0)  # (1, 1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            generated_log_probs.append(sampled_log_prob)

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{self.max_new_tokens}")

            # Check EOS
            if self.eos_token_id is not None and next_token_id.item() == self.eos_token_id:
                if verbose:
                    print(f"  EOS at step {step + 1}")
                break

        generated_text = self.tokenizer.decode(
            input_ids[0, prompt_len:], skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "input_ids": input_ids,
            "num_tokens_generated": input_ids.shape[1] - prompt_len,
            "token_log_probs": generated_log_probs,
        }

    def __repr__(self) -> str:
        return (
            f"PowerSampler(alpha={self.alpha}, top_k={self.top_k}, "
            f"num_rollouts={self.num_rollouts}, lookahead={self.lookahead}, "
            f"jackknife={self.use_jackknife})"
        )
