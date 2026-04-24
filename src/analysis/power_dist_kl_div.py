"""
Measure how close training checkpoints are to the power distribution of the
initial base model, using cross-entropy under power samples as a proxy for KL.

Metric:  E_{x ~ p_power}[log p_theta(x)]
   - Higher means checkpoint theta assigns more mass to power-preferred sequences.
   - The difference  E[log p_theta(x)] - E[log p_base(x)]  tracks how much the
     checkpoint has shifted toward the power distribution.
   - E[log p_theta(x)] - alpha * E[log p_base(x)]  is monotonically related to
     -KL(p_theta || p_power) (differs only by the constant log Z_alpha).

Pipeline:
  1. Generate reference sequences from p_power (base model + power sampling)
  2. Score each sequence under the base model and each checkpoint
  3. Report metrics

Usage:
    uv run python -m scripts.power_dist_kl_div.py \
        --base_model Qwen/Qwen3-1.7B \
        --checkpoint_dir checkpoints/ \
        --chat_template_model Qwen/Qwen3-1.7B-Instruct \
        --max_samples 50 \
        --alpha 4.0 --top_k 8 --num_rollouts 8 --lookahead 32 \
        --batch_size 8 --num_candidates 32 \
        --output_dir results/kl_analysis
"""

import argparse
import json
import os
import random
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.utils import DATASET_REGISTRY_EVAL


# ---------------------------------------------------------------------------
# Prompt formatting (same as run_eval.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


# ---------------------------------------------------------------------------
# Step 1: Generate reference sequences from the power distribution
# ---------------------------------------------------------------------------

def generate_power_samples(
    base_model: str,
    problems: list[dict],
    chat_template_tokenizer,
    alpha: float,
    batch_size: int,
    num_candidates: int,
    top_k: int,
    num_rollouts: int,
    lookahead: int,
    max_tokens: int,
    confidence_threshold: float | None,
    tensor_parallel_size: int,
    max_model_len: int,
) -> list[dict]:
    """Generate sequences from the power distribution of the base model."""
    from scalable_power_sampling import VLLMBatchedPowerSampler

    sampler = VLLMBatchedPowerSampler(
        model_name=base_model,
        alpha=alpha,
        batch_size=batch_size,
        num_candidates=num_candidates,
        top_k=top_k,
        num_rollouts=num_rollouts,
        lookahead=lookahead,
        max_new_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        confidence_threshold=confidence_threshold,
    )
    tokenizer = sampler.tokenizer
    template_tok = chat_template_tokenizer or tokenizer

    samples = []
    pbar = tqdm(problems, desc="Generating power samples", unit="problem")
    for prob in pbar:
        messages = format_prompt(prob["problem"])
        prompt_text = template_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt_text)

        t0 = time.time()
        out = sampler.generate(input_ids=prompt_ids, verbose=False)
        elapsed = time.time() - t0

        # Store the full token sequence (prompt + response)
        full_ids = out["input_ids"]
        if isinstance(full_ids, torch.Tensor):
            full_ids = full_ids.view(-1).tolist()

        samples.append({
            "problem": prob["problem"],
            "answer": prob.get("answer", ""),
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "response_text": out["text"],
            "num_tokens_generated": out["num_tokens_generated"],
            "generation_time_s": elapsed,
        })

        pbar.set_postfix(tokens=out["num_tokens_generated"], time=f"{elapsed:.1f}s")

    # Clean up the sampler's LLM to free GPU memory before scoring
    del sampler
    torch.cuda.empty_cache()

    return samples


# ---------------------------------------------------------------------------
# Step 2: Score sequences under a model using vLLM prompt_logprobs
# ---------------------------------------------------------------------------

def score_sequences(
    model_name: str,
    samples: list[dict],
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> list[float]:
    """Compute log p_model(response | prompt) for each sample.

    Uses vLLM's prompt_logprobs to get per-token log-probs of the full
    sequence, then sums over just the response tokens.
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    # Feed each full sequence as a "prompt" with prompt_logprobs enabled.
    # max_tokens=1 so vLLM returns prompt logprobs without generating much.
    scoring_params = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        prompt_logprobs=0,
    )

    prompts = [
        {"prompt_token_ids": s["full_ids"]} for s in samples
    ]

    outputs = llm.generate(prompts, scoring_params, use_tqdm=False)

    sequence_log_probs = []
    for sample, output in zip(samples, outputs):
        prompt_len = len(sample["prompt_ids"])
        # prompt_logprobs[i] is the log-prob of token i given tokens 0..i-1
        # Index 0 has no conditioning, so logprobs start meaningfully at index 1.
        # We want the sum over response tokens: positions prompt_len .. end
        prompt_logprobs = output.prompt_logprobs  # list of dicts, one per token

        log_prob_sum = 0.0
        full_ids = sample["full_ids"]
        for i in range(prompt_len, len(full_ids)):
            if prompt_logprobs[i] is not None:
                token_id = full_ids[i]
                if token_id in prompt_logprobs[i]:
                    log_prob_sum += prompt_logprobs[i][token_id].logprob
                else:
                    # Token not in top-k logprobs; this shouldn't happen with
                    # prompt_logprobs=0 which should include the actual token
                    log_prob_sum += -100.0  # large penalty

        sequence_log_probs.append(log_prob_sum)

    del llm
    torch.cuda.empty_cache()

    return sequence_log_probs


# ---------------------------------------------------------------------------
# Step 3: Analyze and report
# ---------------------------------------------------------------------------

def analyze_results(
    base_scores: list[float],
    checkpoint_scores: dict[str, list[float]],
    alpha: float,
    samples: list[dict],
) -> dict:
    """Compute metrics and print report."""
    n = len(base_scores)
    mean_base = sum(base_scores) / n

    # Per-token normalization for readability
    total_tokens = sum(s["num_tokens_generated"] for s in samples)
    mean_base_per_tok = sum(base_scores) / total_tokens

    print(f"\n{'='*70}")
    print(f"Power Distribution KL Analysis  (alpha={alpha}, N={n} samples)")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'E[log p]':>10} {'per-tok':>10} {'delta':>10}")
    print(f"{'-'*70}")
    print(f"{'base':45} {mean_base:>10.2f} {mean_base_per_tok:>10.4f} {'--':>10}")

    results = {
        "alpha": alpha,
        "num_samples": n,
        "total_response_tokens": total_tokens,
        "base": {
            "mean_log_prob": mean_base,
            "mean_log_prob_per_token": mean_base_per_tok,
            "per_sample_log_probs": base_scores,
        },
        "checkpoints": {},
    }

    for ckpt_name in sorted(checkpoint_scores.keys()):
        scores = checkpoint_scores[ckpt_name]
        mean_ckpt = sum(scores) / n
        mean_ckpt_per_tok = sum(scores) / total_tokens
        delta = mean_ckpt - mean_base
        delta_per_tok = mean_ckpt_per_tok - mean_base_per_tok

        # Proxy for -KL(p_theta || p_power) + const
        # = E_{p_power}[log p_theta(x)] - alpha * E_{p_power}[log p_base(x)]
        kl_proxy = mean_ckpt - alpha * mean_base

        print(f"{ckpt_name:45} {mean_ckpt:>10.2f} {mean_ckpt_per_tok:>10.4f} {delta_per_tok:>+10.4f}")

        results["checkpoints"][ckpt_name] = {
            "mean_log_prob": mean_ckpt,
            "mean_log_prob_per_token": mean_ckpt_per_tok,
            "delta_vs_base": delta,
            "delta_vs_base_per_token": delta_per_tok,
            "neg_kl_proxy": kl_proxy,
            "per_sample_log_probs": scores,
        }

    print(f"{'='*70}")
    print(f"\ndelta = E[log p_theta] - E[log p_base]  (positive = closer to p_power)")
    print(f"All log-probs are natural log, summed over response tokens.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure checkpoint proximity to the base model's power distribution"
    )
    parser.add_argument("--base_model", required=True,
                        help="Base model name or path")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing checkpoint-N subdirectories")
    parser.add_argument("--chat_template_model", default=None,
                        help="Model to load chat template from (for base models without one)")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Number of problems to sample from the dataset")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum tokens to generate per power sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/kl_analysis")

    # Power sampling hyperparameters
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=8, help="Tokens per chunk (B)")
    parser.add_argument("--num_candidates", type=int, default=32, help="Candidates per step (L)")
    parser.add_argument("--top_k", type=int, default=8, help="Candidates to keep (K)")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Rollouts per candidate (M)")
    parser.add_argument("--lookahead", type=int, default=32, help="Rollout horizon (H)")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Skip rollouts when top candidate is confident enough")

    # vLLM options
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)

    # Option to load pre-generated samples instead of regenerating
    parser.add_argument("--load_samples", default=None,
                        help="Path to a previously saved samples JSON (skip generation)")

    args = parser.parse_args()

    # --- Load chat template tokenizer ---
    chat_template_tokenizer = None
    if args.chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model, trust_remote_code=True
        )
        print(f"Using chat template from: {args.chat_template_model}")

    # --- Load or generate power samples ---
    if args.load_samples:
        print(f"Loading pre-generated samples from {args.load_samples}")
        with open(args.load_samples) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples")
    else:
        # Load dataset
        problems = DATASET_REGISTRY_EVAL["math500"]()
        print(f"Loaded {len(problems)} problems from math500")

        if args.max_samples and args.max_samples < len(problems):
            random.seed(args.seed)
            problems = random.sample(problems, args.max_samples)
            print(f"Subsampled to {len(problems)} problems (seed={args.seed})")

        print(f"\n--- Step 1: Generate power samples from base model ---")
        samples = generate_power_samples(
            base_model=args.base_model,
            problems=problems,
            chat_template_tokenizer=chat_template_tokenizer,
            alpha=args.alpha,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            top_k=args.top_k,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            max_tokens=args.max_tokens,
            confidence_threshold=args.confidence_threshold,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )

        # Save samples for reuse
        os.makedirs(args.output_dir, exist_ok=True)
        samples_path = os.path.join(args.output_dir, "power_samples.json")
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved {len(samples)} power samples to {samples_path}")

    # --- Step 2: Score under base model ---
    print(f"\n--- Step 2: Score sequences under base model ---")
    base_scores = score_sequences(
        args.base_model, samples,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    # --- Step 3: Discover and score checkpoints ---
    checkpoint_dir = args.checkpoint_dir
    ckpt_names = sorted([
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint")
    ])

    if not ckpt_names:
        print(f"No checkpoint-* directories found in {checkpoint_dir}")
        return

    print(f"\nFound {len(ckpt_names)} checkpoints: {', '.join(ckpt_names)}")

    checkpoint_scores = {}
    for ckpt_name in ckpt_names:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        print(f"\n--- Scoring: {ckpt_name} ---")
        scores = score_sequences(
            ckpt_path, samples,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        checkpoint_scores[ckpt_name] = scores

    # --- Step 4: Analyze and report ---
    results = analyze_results(base_scores, checkpoint_scores, args.alpha, samples)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "kl_analysis.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
