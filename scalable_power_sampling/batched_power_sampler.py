"""Batched power sampling for HuggingFace causal LMs (Algorithm 2).

Instead of sampling one token at a time, samples chunks of B tokens per step,
reducing the number of power-sampling decisions from T to ceil(T/B).
"""

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from scalable_power_sampling.rollouts import (
    _batch_select_kv_cache,
    _expand_kv_cache,
    _select_kv_cache,
)
from scalable_power_sampling.scaling import (
    compute_log_scaling_factors,
    compute_power_distribution,
    jackknife_power_distribution,
)


class BatchedPowerSampler:
    """Batched scalable power sampling for autoregressive language models.

    Implements Algorithm 2 from "Scalable Power Sampling: Unlocking Efficient,
    Training-Free Reasoning for LLMs via Distribution Sharpening"
    (Ji et al., 2025).

    At each batch step:
      1. Generate L candidate chunks of B tokens from the base model
      2. Select top-K chunks by likelihood
      3. For each candidate chunk, generate M rollouts of H tokens
      4. Compute scaling factors and jackknife bias correction
      5. Sample one chunk from the power distribution
    The final remainder (T mod B tokens) uses low-temperature sampling.

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        alpha: Power exponent (>=1). Higher values sharpen more. Default 4.0.
        batch_size: Number of tokens per chunk (B). Default 8.
        num_candidates: Candidate chunks to generate per step (L). Default 32.
        top_k: Candidates to keep after filtering (K). Default 8.
        num_rollouts: Rollouts per candidate (M). Default 8.
        lookahead: Rollout horizon in tokens (H). Default 192.
        max_new_tokens: Maximum tokens to generate (T). Default 3072.
        use_jackknife: Apply jackknife bias correction. Default True.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        alpha: float = 4.0,
        batch_size: int = 8,
        num_candidates: int = 32,
        top_k: int = 8,
        num_rollouts: int = 8,
        lookahead: int = 192,
        max_new_tokens: int = 3072,
        use_jackknife: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_candidates = num_candidates
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
        """Generate a completion using batched power sampling.

        Either prompt or input_ids must be provided.

        Args:
            prompt: Text prompt to complete.
            input_ids: Pre-tokenized input, shape (1, seq_len).
            verbose: If True, print progress.

        Returns:
            dict with keys:
                - "text": Generated text (excluding prompt).
                - "input_ids": Full sequence including prompt.
                - "num_tokens_generated": Number of new tokens.
        """
        if input_ids is None:
            assert prompt is not None, "Provide either prompt or input_ids"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        prompt_len = input_ids.shape[1]
        B = self.batch_size
        num_full_batches = self.max_new_tokens // B
        remainder = self.max_new_tokens % B

        # Initial forward pass: prefix cache + logits for first chunk position
        out = self.model(input_ids, use_cache=True)
        prefix_cache = out.past_key_values
        prefix_logits = out.logits[:, -1, :]  # (1, vocab_size)

        eos_hit = False

        for step in range(num_full_batches):
            # --- Line 4: Generate L candidate chunks of B tokens ---
            chunk_ids, chunk_log_probs, chunk_cache, chunk_logits = (
                self._generate_candidate_chunks(prefix_cache, prefix_logits, B)
            )

            # --- Line 5: Select top-K candidates by likelihood ---
            K = min(self.top_k, chunk_ids.shape[0])
            top_k_vals, top_k_idx = chunk_log_probs.topk(K)
            top_k_ids = chunk_ids[top_k_idx]          # (K, B)
            top_k_log_probs = top_k_vals               # (K,)
            top_k_cache = _batch_select_kv_cache(chunk_cache, top_k_idx)
            top_k_logits = chunk_logits[top_k_idx]     # (K, vocab_size)
            del chunk_cache

            if self.alpha == 1.0:
                # No sharpening: sample proportional to base likelihood
                probs = torch.exp(top_k_log_probs - torch.logsumexp(top_k_log_probs, dim=-1))
                idx = torch.multinomial(probs, 1).item()
            else:
                # --- Lines 6-9: Generate rollouts for each candidate ---
                rollout_ll = self._generate_chunk_rollouts(
                    top_k_cache, top_k_logits
                )  # (K, M)

                # --- Lines 10-16: Power distribution with jackknife ---
                if self.use_jackknife:
                    power_probs = jackknife_power_distribution(
                        top_k_log_probs, rollout_ll, self.alpha
                    )
                else:
                    log_zeta = compute_log_scaling_factors(rollout_ll, self.alpha)
                    power_probs = compute_power_distribution(
                        top_k_log_probs.unsqueeze(0),
                        log_zeta.unsqueeze(0),
                        self.alpha,
                    )

                power_probs = power_probs.view(-1)
                idx = torch.multinomial(power_probs, 1).item()

            # --- Line 17: Concatenate selected chunk ---
            selected_chunk = top_k_ids[idx]  # (B,)
            input_ids = torch.cat([input_ids, selected_chunk.unsqueeze(0)], dim=-1)
            prefix_cache = _select_kv_cache(top_k_cache, idx)
            prefix_logits = top_k_logits[idx].unsqueeze(0)  # (1, vocab_size)

            if verbose and (step + 1) % 5 == 0:
                print(
                    f"  Batch step {step + 1}/{num_full_batches} "
                    f"({(step + 1) * B} tokens)"
                )

            # Check EOS within selected chunk
            if self.eos_token_id is not None:
                eos_mask = selected_chunk == self.eos_token_id
                if eos_mask.any():
                    eos_pos = eos_mask.nonzero(as_tuple=False)[0].item()
                    input_ids = input_ids[:, : prompt_len + step * B + eos_pos + 1]
                    eos_hit = True
                    if verbose:
                        print(f"  EOS at batch step {step + 1}, position {eos_pos}")
                    break

        # --- Lines 19-22: Remainder via low-temperature sampling (Eq. 16) ---
        if not eos_hit and remainder > 0:
            chunk_ids, chunk_log_probs, _, _ = self._generate_candidate_chunks(
                prefix_cache, prefix_logits, remainder
            )
            K = min(self.top_k, chunk_ids.shape[0])
            top_k_vals, top_k_idx = chunk_log_probs.topk(K)
            top_k_ids = chunk_ids[top_k_idx]

            if self.alpha > 1.0:
                # Eq. 16: p_low_temp(chunk) = p^alpha(chunk) / Z
                log_unnorm = self.alpha * top_k_vals
                probs = torch.exp(log_unnorm - torch.logsumexp(log_unnorm, dim=-1))
            else:
                probs = torch.exp(top_k_vals - torch.logsumexp(top_k_vals, dim=-1))

            idx = torch.multinomial(probs, 1).item()
            selected_chunk = top_k_ids[idx]
            input_ids = torch.cat([input_ids, selected_chunk.unsqueeze(0)], dim=-1)

            if self.eos_token_id is not None:
                eos_mask = selected_chunk == self.eos_token_id
                if eos_mask.any():
                    eos_pos = eos_mask.nonzero(as_tuple=False)[0].item()
                    input_ids = input_ids[:, : input_ids.shape[1] - remainder + eos_pos + 1]

        generated_text = self.tokenizer.decode(
            input_ids[0, prompt_len:], skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "input_ids": input_ids,
            "num_tokens_generated": input_ids.shape[1] - prompt_len,
        }

    @torch.no_grad()
    def _generate_candidate_chunks(
        self,
        prefix_cache,
        prefix_logits: Tensor,
        chunk_len: int,
    ) -> tuple[Tensor, Tensor, object, Tensor]:
        """Generate L candidate chunks by sampling from the base model.

        Args:
            prefix_cache: KV-cache for the prefix (batch dim 1).
            prefix_logits: Logits from the last prefix position, (1, vocab).
            chunk_len: Number of tokens per chunk (B).

        Returns:
            chunk_ids: Token IDs, shape (L, chunk_len).
            chunk_log_probs: Sum of log-probs per chunk, shape (L,).
            cache: KV-cache after generation, batch dim L.
            last_logits: Logits after the last chunk token, (L, vocab).
        """
        L = self.num_candidates

        cache = _expand_kv_cache(prefix_cache, L)
        logits = prefix_logits.expand(L, -1)  # (L, vocab)

        chunk_ids = torch.zeros(L, chunk_len, dtype=torch.long, device=self.device)
        cum_log_probs = torch.zeros(L, device=self.device)
        active = torch.ones(L, dtype=torch.bool, device=self.device)

        for b in range(chunk_len):
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1)  # (L, 1)

            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)  # (L,)

            cum_log_probs += token_log_probs * active.float()
            chunk_ids[:, b] = next_tokens.squeeze(-1)

            if self.eos_token_id is not None:
                active = active & (next_tokens.squeeze(-1) != self.eos_token_id)

            out = self.model(next_tokens, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]  # (L, vocab)

        return chunk_ids, cum_log_probs, cache, logits

    @torch.no_grad()
    def _generate_chunk_rollouts(
        self,
        candidate_caches,
        candidate_logits: Tensor,
    ) -> Tensor:
        """Generate M rollouts of H tokens for each of K candidates.

        Processes one candidate at a time to manage VRAM.

        Args:
            candidate_caches: KV-cache with batch dim K.
            candidate_logits: Logits after each candidate chunk, (K, vocab).

        Returns:
            rollout_log_likelihoods: Shape (K, M).
        """
        K = candidate_logits.shape[0]
        M = self.num_rollouts
        H = self.lookahead

        rollout_ll = torch.zeros(K, M, device=self.device)

        for k in range(K):
            single_cache = _select_kv_cache(candidate_caches, k)
            rollout_cache = _expand_kv_cache(single_cache, M)
            logits = candidate_logits[k].unsqueeze(0).expand(M, -1)  # (M, vocab)

            cum_log_prob = torch.zeros(M, device=self.device)
            active = torch.ones(M, dtype=torch.bool, device=self.device)

            for h in range(H):
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)  # (M, 1)

                log_probs = torch.log_softmax(logits, dim=-1)
                sampled_log_probs = log_probs.gather(1, next_tokens).squeeze(-1)

                cum_log_prob += sampled_log_probs * active.float()

                if self.eos_token_id is not None:
                    active = active & (next_tokens.squeeze(-1) != self.eos_token_id)
                    if not active.any():
                        break

                out = self.model(
                    next_tokens, past_key_values=rollout_cache, use_cache=True
                )
                rollout_cache = out.past_key_values
                logits = out.logits[:, -1, :]  # (M, vocab)

            rollout_ll[k] = cum_log_prob

        return rollout_ll

    def __repr__(self) -> str:
        return (
            f"BatchedPowerSampler(alpha={self.alpha}, batch_size={self.batch_size}, "
            f"num_candidates={self.num_candidates}, top_k={self.top_k}, "
            f"num_rollouts={self.num_rollouts}, lookahead={self.lookahead}, "
            f"jackknife={self.use_jackknife})"
        )
