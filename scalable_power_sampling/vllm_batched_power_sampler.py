"""vLLM-accelerated batched power sampling (Algorithm 2).

Uses vLLM for all generation and log-prob computation, eliminating manual
KV-cache management. vLLM's PagedAttention, CUDA graphs, and automatic
prefix caching handle the heavy lifting.
"""

import time
import torch
from torch import Tensor

from vllm import LLM, SamplingParams

from scalable_power_sampling.scaling import (
    compute_log_scaling_factors,
    compute_power_distribution,
    jackknife_power_distribution,
)


class VLLMBatchedPowerSampler:
    """vLLM-accelerated batched power sampling.

    Implements Algorithm 2 from "Scalable Power Sampling" (Ji et al., 2025)
    using vLLM for generation. Instead of manual KV-cache manipulation,
    each step issues two vLLM generate() calls:
      1. n=L completions of B tokens for candidate chunks
      2. K prompts × n=M completions of H tokens for rollouts

    vLLM automatically shares prefix KV-cache across candidates and rollouts.

    Args:
        model_name: HuggingFace model name or local path.
        alpha: Power exponent (>=1). Default 4.0.
        batch_size: Tokens per chunk (B). Default 192.
        num_candidates: Candidate chunks per step (L). Default 32.
        top_k: Candidates to keep (K). Default 8.
        num_rollouts: Rollouts per candidate (M). Default 8.
        lookahead: Rollout horizon in tokens (H). Default 192.
        max_new_tokens: Maximum tokens to generate (T). Default 3072.
        use_jackknife: Apply jackknife bias correction. Default True.
        tensor_parallel_size: GPUs for tensor parallelism. Default 1.
        max_model_len: Max context length for vLLM KV cache. Default 4096.
        dtype: Model dtype. Default "bfloat16".
        length_normalize: If True, normalize cumulative log-probs by sequence
            length before scoring. Prevents length bias when chunks or rollouts
            terminate early at EOS. Default False.
    """

    def __init__(
        self,
        model_name: str,
        alpha: float = 4.0,
        batch_size: int = 192,
        num_candidates: int = 32,
        top_k: int = 8,
        num_rollouts: int = 8,
        lookahead: int = 192,
        max_new_tokens: int = 3072,
        use_jackknife: bool = True,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        length_normalize: bool = False,
    ):
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_rollouts = num_rollouts
        self.lookahead = lookahead
        self.max_new_tokens = max_new_tokens
        self.use_jackknife = use_jackknife
        self.length_normalize = length_normalize

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, list):
            self.eos_token_ids = set(eos)
        elif eos is not None:
            self.eos_token_ids = {eos}
        else:
            self.eos_token_ids = set()

    def generate(
        self,
        prompt: str | None = None,
        input_ids: list[int] | None = None,
        verbose: bool = False,
    ) -> dict:
        """Generate a completion using vLLM-accelerated batched power sampling.

        Args:
            prompt: Text prompt to complete.
            input_ids: Pre-tokenized input as a list of ints.
            verbose: If True, print progress.

        Returns:
            dict with keys: "text", "input_ids", "num_tokens_generated".
        """
        if input_ids is None:
            assert prompt is not None, "Provide either prompt or input_ids"
            input_ids = self.tokenizer.encode(prompt)
        elif isinstance(input_ids, Tensor):
            input_ids = input_ids.view(-1).tolist()

        prefix_ids = list(input_ids)
        prompt_len = len(prefix_ids)
        B = self.batch_size
        num_full_batches = self.max_new_tokens // B
        remainder = self.max_new_tokens % B

        eos_hit = False

        for step in range(num_full_batches):
            # --- Line 4: Generate L candidate chunks of B tokens ---
            chunk_ids, chunk_log_probs = self._generate_candidate_chunks(
                prefix_ids, B
            )
            # chunk_ids: list of L lists, chunk_log_probs: (L,)

            # --- Line 5: Select top-K by likelihood ---
            K = min(self.top_k, len(chunk_ids))
            top_k_vals, top_k_idx = chunk_log_probs.topk(K)
            top_k_chunks = [chunk_ids[i] for i in top_k_idx.tolist()]
            top_k_log_probs = top_k_vals  # (K,)
            remaining_after_chunk = self.max_new_tokens - ((step + 1) * B)
            rollout_len = min(self.lookahead, remaining_after_chunk)

            # Identify candidates without EOS (rollouts after EOS are
            # meaningless — the future is empty so zeta=1, i.e. log_zeta=0,
            # which reduces to pure low-temperature for that candidate)
            if self.eos_token_ids:
                eos_free = [
                    i
                    for i, c in enumerate(top_k_chunks)
                    if self.eos_token_ids.isdisjoint(c)
                ]
            else:
                eos_free = list(range(K))

            if not eos_free or rollout_len <= 0:
                # If all candidates terminate, or there is no future left to
                # roll out, power sampling reduces to low-temperature sampling.
                log_unnorm = self.alpha * top_k_log_probs
                probs = torch.exp(
                    log_unnorm - torch.logsumexp(log_unnorm, dim=-1)
                )
                idx = torch.multinomial(probs, 1).item()
            else:
                # --- Lines 6-9: Generate rollouts for EOS-free candidates ---
                eos_free_chunks = [top_k_chunks[i] for i in eos_free]
                eos_free_ll = self._generate_rollouts(
                    prefix_ids, eos_free_chunks, rollout_len
                )  # (len(eos_free), M)

                # Full rollout_ll: zeros for EOS candidates (log_zeta=0)
                rollout_ll = torch.zeros(K, self.num_rollouts)
                for j, i in enumerate(eos_free):
                    rollout_ll[i] = eos_free_ll[j]

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
            selected_chunk = top_k_chunks[idx]
            prefix_ids = prefix_ids + selected_chunk

            if verbose and (step + 1) % 5 == 0:
                print(
                    f"  Batch step {step + 1}/{num_full_batches} "
                    f"({(step + 1) * B} tokens)"
                )

            # Check EOS within selected chunk
            if self.eos_token_ids:
                chunk_start = len(prefix_ids) - len(selected_chunk)
                for pos, tid in enumerate(selected_chunk):
                    if tid in self.eos_token_ids:
                        prefix_ids = prefix_ids[: chunk_start + pos + 1]
                        eos_hit = True
                        if verbose:
                            print(
                                f"  EOS at batch step {step + 1}, position {pos}"
                            )
                        break
            if eos_hit:
                break

        # --- Lines 19-22: Remainder via low-temperature sampling (Eq. 16) ---
        if not eos_hit and remainder > 0:
            chunk_ids, chunk_log_probs = self._generate_candidate_chunks(
                prefix_ids, remainder
            )
            K = min(self.top_k, len(chunk_ids))
            top_k_vals, top_k_idx = chunk_log_probs.topk(K)
            top_k_chunks = [chunk_ids[i] for i in top_k_idx.tolist()]

            log_unnorm = self.alpha * top_k_vals
            probs = torch.exp(
                log_unnorm - torch.logsumexp(log_unnorm, dim=-1)
            )

            idx = torch.multinomial(probs, 1).item()
            selected_chunk = top_k_chunks[idx]
            prefix_ids = prefix_ids + selected_chunk

            if self.eos_token_ids:
                chunk_start = len(prefix_ids) - len(selected_chunk)
                for pos, tid in enumerate(selected_chunk):
                    if tid in self.eos_token_ids:
                        prefix_ids = prefix_ids[: chunk_start + pos + 1]
                        break

        generated_ids = prefix_ids[prompt_len:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            "text": generated_text,
            "input_ids": prefix_ids,
            "num_tokens_generated": len(generated_ids),
        }

    def _generate_candidate_chunks(
        self,
        prefix_ids: list[int],
        chunk_len: int,
    ) -> tuple[list[list[int]], Tensor]:
        """Generate L candidate chunks via vLLM.

        Args:
            prefix_ids: Token IDs for the current prefix.
            chunk_len: Tokens per chunk (B).

        Returns:
            chunk_ids: List of L token-ID lists, each of length <= chunk_len.
            chunk_log_probs: Cumulative log-prob per chunk, shape (L,).
        """
        params = SamplingParams(
            n=self.num_candidates,
            max_tokens=chunk_len,
            temperature=1.0,
            logprobs=0,
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": prefix_ids}],
            sampling_params=params,
            use_tqdm=False,
        )

        all_ids: list[list[int]] = []
        all_lps: list[float] = []

        for completion in outputs[0].outputs:
            ids = list(completion.token_ids)
            assert completion.cumulative_logprob is not None, (
                "cumulative_logprob is None — set logprobs>=1 in SamplingParams"
            )
            lp = completion.cumulative_logprob
            if self.length_normalize and len(ids) > 0:
                lp = lp / len(ids)
            all_ids.append(ids)
            all_lps.append(lp)

        return all_ids, torch.tensor(all_lps)


    def _generate_rollouts(
        self,
        prefix_ids: list[int],
        candidate_chunks: list[list[int]],
        rollout_len: int,
    ) -> Tensor:
        """Generate M rollouts of H tokens for each of K candidate chunks.

        Submits K prompts (prefix + chunk_k) with n=M each. vLLM batches
        them efficiently and reuses the shared prefix cache.

        Args:
            prefix_ids: Token IDs for the shared prefix.
            candidate_chunks: K lists of token IDs, one per candidate.
            rollout_len: Number of future tokens to roll out.

        Returns:
            rollout_ll: Cumulative log-probs, shape (K, M).
        """
        K = len(candidate_chunks)
        M = self.num_rollouts

        # Build K prompts: prefix + each candidate chunk
        rollout_prompts = [
            {"prompt_token_ids": prefix_ids + chunk}
            for chunk in candidate_chunks
        ]

        params = SamplingParams(
            n=M,
            max_tokens=rollout_len,
            temperature=1.0,
            logprobs=0,
            ignore_eos=False,
        )
        outputs = self.llm.generate(
            rollout_prompts,
            sampling_params=params,
            use_tqdm=False,
        )

        rollout_ll = torch.zeros(K, M)
        for k, request_output in enumerate(outputs):
            for m, completion in enumerate(request_output.outputs):
                assert completion.cumulative_logprob is not None, (
                    "cumulative_logprob is None — set logprobs>=1 in SamplingParams"
                )
                ll = completion.cumulative_logprob
                if self.length_normalize:
                    n_tokens = len(completion.token_ids)
                    if n_tokens > 0:
                        ll = ll / n_tokens
                rollout_ll[k, m] = ll

        return rollout_ll


    def __repr__(self) -> str:
        return (
            f"VLLMBatchedPowerSampler(alpha={self.alpha}, "
            f"batch_size={self.batch_size}, "
            f"num_candidates={self.num_candidates}, top_k={self.top_k}, "
            f"num_rollouts={self.num_rollouts}, lookahead={self.lookahead}, "
            f"jackknife={self.use_jackknife}, "
            f"length_normalize={self.length_normalize})"
        )
