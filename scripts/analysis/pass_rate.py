"""
Compare pass@K between base sampling and power sampling.

For each problem, generates K solutions under both the base distribution
(vLLM, temperature=1.0) and the power distribution (VLLMBatchedPowerSampler),
then computes the unbiased pass@K estimator.

Usage:
    # Base vs power sampling, K=8, on 50 MATH500 problems
    uv run python -m scripts.pass_rate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset math500 \
        --num_samples 50 \
        --K 8 \
        --alpha 4.0 --top_k 8 --num_rollouts 8 --lookahead 32

    # Sweep multiple K values
    uv run python -m scripts.pass_rate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset math500 \
        --K 1 2 4 8 \
        --alpha 4.0

    # Answer-conditioned: model sees the correct answer in the prompt
    uv run python -m scripts.pass_rate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset math500 \
        --K 1 2 4 8 \
        --answer_conditioned
"""

import argparse
import json
import math
import os
import random
import time

from tqdm import tqdm
from vllm import LLM, SamplingParams

from scripts.utils import (
    DATASET_REGISTRY_EVAL,
    extract_boxed_answer,
    is_equiv,
)


SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)

ANSWER_CONDITIONED_SYSTEM_PROMPT = (
    "You are a helpful math assistant. You are given a math problem along with its "
    "correct answer. Write a full step-by-step solution to the problem which "
    "concludes with the correct answer. Put the final answer in \\boxed{}."
)


def format_prompt(problem: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def format_prompt_answer_conditioned(problem: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_CONDITIONED_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nCorrect answer: {answer}"},
    ]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Uses log-space computation for numerical stability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(
        math.lgamma(n - c + 1) + math.lgamma(n - k + 1)
        - math.lgamma(n - c - k + 1) - math.lgamma(n + 1)
    )


def generate_base_solutions(
    llm: LLM,
    problems: list[dict],
    K: int,
    max_tokens: int,
    chat_template_tokenizer=None,
    answer_conditioned: bool = False,
) -> list[list[dict]]:
    """Generate K solutions per problem using base sampling (temperature=1)."""
    tokenizer = chat_template_tokenizer or llm.get_tokenizer()
    prompts = []
    for p in problems:
        if answer_conditioned:
            messages = format_prompt_answer_conditioned(p["problem"], p["answer"])
        else:
            messages = format_prompt(p["problem"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    params = SamplingParams(
        n=K,
        temperature=1.0,
        max_tokens=max_tokens,
    )

    print(f"Generating {K} base solutions for {len(problems)} problems...")
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"Base generation took {elapsed:.1f}s")

    all_solutions = []
    for prob, output in zip(problems, outputs):
        solutions = []
        for completion in output.outputs:
            pred = extract_boxed_answer(completion.text)
            correct = is_equiv(pred, prob["answer"]) if pred else False
            solutions.append({
                "text": completion.text,
                "pred_answer": pred,
                "correct": correct,
            })
        all_solutions.append(solutions)

    return all_solutions


def generate_power_solutions(
    model_name: str,
    problems: list[dict],
    K: int,
    max_tokens: int,
    alpha: float,
    top_k: int,
    num_rollouts: int,
    lookahead: int,
    batch_size: int,
    num_candidates: int,
    tensor_parallel_size: int,
    max_model_len: int,
    confidence_threshold: float | None,
    chat_template_tokenizer=None,
    answer_conditioned: bool = False,
) -> list[list[dict]]:
    """Generate K solutions per problem using power sampling."""
    from scalable_power_sampling import VLLMBatchedPowerSampler

    sampler = VLLMBatchedPowerSampler(
        model_name=model_name,
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
    tokenizer = chat_template_tokenizer or sampler.tokenizer

    print(
        f"Generating {K} power solutions for {len(problems)} problems "
        f"(alpha={alpha}, top_k={top_k}, M={num_rollouts}, H={lookahead})..."
    )
    t0 = time.time()

    all_solutions = []
    pbar = tqdm(problems, desc="power_sampling", unit="problem")
    for prob in pbar:
        if answer_conditioned:
            messages = format_prompt_answer_conditioned(prob["problem"], prob["answer"])
        else:
            messages = format_prompt(prob["problem"])
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = sampler.tokenizer.encode(prompt_text)

        solutions = []
        num_correct = 0
        for k in range(K):
            out = sampler.generate(input_ids=input_ids, verbose=False)
            pred = extract_boxed_answer(out["text"])
            correct = is_equiv(pred, prob["answer"]) if pred else False
            num_correct += int(correct)
            solutions.append({
                "text": out["text"],
                "pred_answer": pred,
                "correct": correct,
                "num_tokens_generated": out["num_tokens_generated"],
            })

        all_solutions.append(solutions)
        pbar.set_postfix(correct=f"{num_correct}/{K}")

    elapsed = time.time() - t0
    print(f"Power sampling took {elapsed:.1f}s")

    return all_solutions


def compute_pass_at_k(
    all_solutions: list[list[dict]],
    k_values: list[int],
) -> dict[int, float]:
    """Compute pass@k for each k, averaged across problems."""
    results = {}
    for k in k_values:
        scores = []
        for solutions in all_solutions:
            n = len(solutions)
            if k > n:
                continue
            c = sum(s["correct"] for s in solutions)
            scores.append(pass_at_k(n, c, k))
        results[k] = sum(scores) / len(scores) if scores else 0.0
    return results


def print_comparison(
    base_pass_at_k: dict[int, float],
    power_pass_at_k: dict[int, float],
    base_solutions: list[list[dict]],
    power_solutions: list[list[dict]],
):
    """Print side-by-side pass@k comparison."""
    # Per-problem correctness counts
    base_total_correct = sum(
        sum(s["correct"] for s in sols) for sols in base_solutions
    )
    power_total_correct = sum(
        sum(s["correct"] for s in sols) for sols in power_solutions
    )
    base_total = sum(len(sols) for sols in base_solutions)
    power_total = sum(len(sols) for sols in power_solutions)

    print(f"\n{'='*60}")
    print("Raw correctness rate")
    print(f"  Base:  {base_total_correct}/{base_total} = "
          f"{base_total_correct/base_total*100:.1f}%")
    print(f"  Power: {power_total_correct}/{power_total} = "
          f"{power_total_correct/power_total*100:.1f}%")

    print(f"\n{'k':<6} {'Base pass@k':>14} {'Power pass@k':>14} {'Delta':>10}")
    print("-" * 48)
    all_k = sorted(set(base_pass_at_k) | set(power_pass_at_k))
    for k in all_k:
        b = base_pass_at_k.get(k, float("nan"))
        p = power_pass_at_k.get(k, float("nan"))
        delta = p - b
        print(f"{k:<6} {b:>13.1%} {p:>13.1%} {delta:>+9.1%}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare pass@K between base and power sampling"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset", default="math500", choices=list(DATASET_REGISTRY_EVAL.keys()),
    )
    parser.add_argument("--levels", nargs="*", type=int, default=None)
    parser.add_argument(
        "--K", nargs="+", type=int, default=[1, 2, 4, 8],
        help="K values for pass@K. Generates max(K) solutions per problem.",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset of N problems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/pass_rate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)

    # Power sampling config
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    parser.add_argument("--chat_template_model", type=str, default=None)
    parser.add_argument("--answer_conditioned", action="store_true",
                        help="Include the correct answer in the prompt")
    parser.add_argument("--base_only", action="store_true",
                        help="Only run base sampling (skip power sampling)")
    parser.add_argument("--power_only", action="store_true",
                        help="Only run power sampling (skip base sampling)")

    args = parser.parse_args()

    # Load dataset
    problems = DATASET_REGISTRY_EVAL[args.dataset](levels=args.levels)
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    if args.num_samples is not None and args.num_samples < len(problems):
        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
        print(f"Subsampled to {len(problems)} problems (seed={args.seed})")

    max_k = max(args.K)

    # Load chat template tokenizer if needed
    chat_template_tokenizer = None
    if args.chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model, trust_remote_code=True
        )

    # --- Base sampling ---
    base_solutions = None
    base_pass = {}
    if not args.power_only:
        llm_kwargs = dict(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        if args.max_model_len:
            llm_kwargs["max_model_len"] = args.max_model_len
        llm = LLM(**llm_kwargs)

        base_solutions = generate_base_solutions(
            llm, problems, max_k, args.max_tokens,
            chat_template_tokenizer=chat_template_tokenizer,
            answer_conditioned=args.answer_conditioned,
        )
        base_pass = compute_pass_at_k(base_solutions, args.K)

        # Free vLLM instance before loading power sampler
        del llm

    # --- Power sampling ---
    power_solutions = None
    power_pass = {}
    if not args.base_only:
        power_solutions = generate_power_solutions(
            model_name=args.model,
            problems=problems,
            K=max_k,
            max_tokens=args.max_tokens,
            alpha=args.alpha,
            top_k=args.top_k,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            confidence_threshold=args.confidence_threshold,
            chat_template_tokenizer=chat_template_tokenizer,
            answer_conditioned=args.answer_conditioned,
        )
        power_pass = compute_pass_at_k(power_solutions, args.K)

    # --- Report ---
    if base_solutions and power_solutions:
        print_comparison(base_pass, power_pass, base_solutions, power_solutions)
    elif base_solutions:
        print(f"\n{'k':<6} {'Base pass@k':>14}")
        print("-" * 22)
        for k in sorted(base_pass):
            print(f"{k:<6} {base_pass[k]:>13.1%}")
    elif power_solutions:
        print(f"\n{'k':<6} {'Power pass@k':>14}")
        print("-" * 22)
        for k in sorted(power_pass):
            print(f"{k:<6} {power_pass[k]:>13.1%}")

    # --- Save ---
    model_slug = args.model.replace("/", "_")
    subdir = "answer_conditioned" if args.answer_conditioned else "unconditioned"
    output_dir = os.path.join(args.output_dir, args.dataset, model_slug, subdir)
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "model": args.model,
        "dataset": args.dataset,
        "answer_conditioned": args.answer_conditioned,
        "num_problems": len(problems),
        "max_k": max_k,
        "k_values": args.K,
        "seed": args.seed,
    }
    if base_solutions:
        output["base"] = {
            "pass_at_k": {str(k): v for k, v in base_pass.items()},
            "raw_correct_rate": sum(
                sum(s["correct"] for s in sols) for sols in base_solutions
            ) / sum(len(sols) for sols in base_solutions),
            "solutions": [
                [
                    {"pred_answer": s["pred_answer"], "correct": s["correct"]}
                    for s in sols
                ]
                for sols in base_solutions
            ],
        }
    if power_solutions:
        output["power"] = {
            "config": {
                "alpha": args.alpha,
                "top_k": args.top_k,
                "num_rollouts": args.num_rollouts,
                "lookahead": args.lookahead,
                "batch_size": args.batch_size,
                "num_candidates": args.num_candidates,
                "confidence_threshold": args.confidence_threshold,
            },
            "pass_at_k": {str(k): v for k, v in power_pass.items()},
            "raw_correct_rate": sum(
                sum(s["correct"] for s in sols) for sols in power_solutions
            ) / sum(len(sols) for sols in power_solutions),
            "solutions": [
                [
                    {"pred_answer": s["pred_answer"], "correct": s["correct"]}
                    for s in sols
                ]
                for sols in power_solutions
            ],
        }

    results_path = os.path.join(output_dir, "pass_rate.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
