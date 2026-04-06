"""
Pairwise preference evaluation: base vs power sampling solution quality.

Two-phase design:
  Phase 1 (generate): Produce one solution per problem from each distribution.
  Phase 2 (judge):    Load an LLM judge, run pairwise comparisons in both
                      orderings to control for position bias.

Phases are independent — you can regenerate without re-judging, or re-judge
with a different model/prompt without regenerating.

Usage:
    # Full pipeline (generate + judge)
    uv run python -m scripts.judge_quality \
        --model Qwen/Qwen2.5-7B-Instruct \
        --judge_model Qwen/Qwen2.5-32B-Instruct \
        --dataset math500 --num_samples 50

    # Generate only (saves solutions to disk)
    uv run python -m scripts.judge_quality \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset math500 --num_samples 50 \
        --generate_only

    # Judge from existing solutions file
    uv run python -m scripts.judge_quality \
        --judge_model Qwen/Qwen2.5-32B-Instruct \
        --judge_from results/judge/math500/.../solutions.json

    # Answer-conditioned: both base and power see the correct answer
    uv run python -m scripts.judge_quality \
        --model Qwen/Qwen2.5-7B-Instruct \
        --judge_model Qwen/Qwen2.5-32B-Instruct \
        --dataset math500 --num_samples 50 \
        --answer_conditioned

    # Base sees the answer, power does not (SDFT teacher proxy vs power dist)
    uv run python -m scripts.judge_quality \
        --model Qwen/Qwen2.5-7B-Instruct \
        --judge_model Qwen/Qwen2.5-32B-Instruct \
        --dataset math500 --num_samples 50 \
        --base_answer_conditioned
"""

import argparse
import json
import os
import random
import re
import statistics
import time

from tqdm import tqdm
from vllm import LLM, SamplingParams

from scripts.utils import (
    DATASET_REGISTRY_EVAL,
    extract_boxed_answer,
    is_equiv,
)


# ---------------------------------------------------------------------------
# Prompt formatting
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


ANSWER_CONDITIONED_SYSTEM_PROMPT = (
    "You are a helpful math assistant. You are given a math problem along with its "
    "correct answer. Write a full step-by-step solution to the problem which "
    "concludes with the correct answer. Put the final answer in \\boxed{}."
)


def format_prompt_answer_conditioned(problem: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_CONDITIONED_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nCorrect answer: {answer}"},
    ]


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are an expert mathematics judge."
)

JUDGE_USER_TEMPLATE = """\
You are given a math problem and two candidate solutions. \
Evaluate the mathematical reasoning in each solution.

Focus on:
- Logical coherence: Does each step follow from the previous?
- Mathematical accuracy: Are intermediate calculations and transformations correct?
- Completeness: Are important steps justified rather than skipped?

Do NOT consider formatting, verbosity, or writing style. \
A concise correct argument is better than a verbose flawed one.

## Problem
{problem}

## Solution A
{solution_a}

## Solution B
{solution_b}

Assess the reasoning quality of both solutions, then choose exactly one verdict:
- **Verdict: A** — Solution A has clearly better reasoning than B.
- **Verdict: B** — Solution B has clearly better reasoning than A.
- **Verdict: Both Good** — Both solutions have correct, coherent reasoning \
with no meaningful quality difference. Only choose this if both are \
mathematically valid with no errors in intermediate steps.
- **Verdict: Both Bad** — Both solutions have significant flaws in their \
reasoning (wrong intermediate steps, logical gaps, or incorrect arguments).

First explain your assessment briefly, then state your verdict on the \
final line using one of the exact formats above."""


VERDICT_PATTERN = re.compile(
    r"\*\*Verdict:\s*(A|B|Both Good|Both Bad)\s*\*\*", re.IGNORECASE
)
VERDICT_FALLBACK = re.compile(
    r"Verdict:\s*(A|B|Both Good|Both Bad)\s*$", re.IGNORECASE | re.MULTILINE
)

# Canonical forms for reconciliation
_VERDICT_NORM = {
    "a": "A",
    "b": "B",
    "both good": "Both Good",
    "both bad": "Both Bad",
}


def parse_verdict(text: str) -> str | None:
    """Extract verdict from judge response.

    Returns 'A', 'B', 'Both Good', 'Both Bad', or None.
    """
    match = VERDICT_PATTERN.search(text)
    if match:
        return _VERDICT_NORM[match.group(1).lower()]
    match = VERDICT_FALLBACK.search(text.strip())
    if match:
        return _VERDICT_NORM[match.group(1).lower()]
    return None


# ---------------------------------------------------------------------------
# Phase 1: Generate solutions
# ---------------------------------------------------------------------------

def generate_solutions(
    model_name: str,
    problems: list[dict],
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
    base_answer_conditioned: bool = False,
    power_answer_conditioned: bool = False,
) -> list[dict]:
    """Generate one base and one power solution per problem.

    Base and power can be independently conditioned on the correct answer,
    enabling comparisons like answer-conditioned base vs unconditioned power.

    Returns list of dicts with keys: problem, answer, base_solution,
    power_solution, base_pred, power_pred, base_correct, power_correct.
    """

    # --- Base solutions via vLLM ---
    cond_label = " (answer-conditioned)" if base_answer_conditioned else ""
    print(f"\n=== Phase 1a: Base sampling{cond_label} ===")
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = chat_template_tokenizer or llm.get_tokenizer()

    def _build_prompt(prob):
        if base_answer_conditioned:
            messages = format_prompt_answer_conditioned(prob["problem"], prob["answer"])
        else:
            messages = format_prompt(prob["problem"])
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    prompts = [_build_prompt(p) for p in problems]

    params = SamplingParams(temperature=1.0, max_tokens=max_tokens)
    t0 = time.time()
    base_outputs = llm.generate(prompts, params, use_tqdm=True)
    print(f"Base generation: {time.time() - t0:.1f}s")

    del llm  # free GPU for power sampler

    # --- Power solutions via VLLMBatchedPowerSampler ---
    cond_label = " (answer-conditioned)" if power_answer_conditioned else ""
    print(f"\n=== Phase 1b: Power sampling{cond_label} ===")
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
    ps_tokenizer = chat_template_tokenizer or sampler.tokenizer

    def _build_ps_prompt(prob):
        if power_answer_conditioned:
            messages = format_prompt_answer_conditioned(prob["problem"], prob["answer"])
        else:
            messages = format_prompt(prob["problem"])
        return ps_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    power_texts = []
    t0 = time.time()
    pbar = tqdm(problems, desc="power_sampling", unit="problem")
    for prob in pbar:
        prompt_text = _build_ps_prompt(prob)
        input_ids = sampler.tokenizer.encode(prompt_text)
        out = sampler.generate(input_ids=input_ids, verbose=False)
        power_texts.append(out["text"])
    print(f"Power generation: {time.time() - t0:.1f}s")

    del sampler

    # --- Combine ---
    pairs = []
    for i, prob in enumerate(problems):
        base_text = base_outputs[i].outputs[0].text
        power_text = power_texts[i]

        base_pred = extract_boxed_answer(base_text)
        power_pred = extract_boxed_answer(power_text)

        pairs.append({
            "problem": prob["problem"],
            "answer": prob["answer"],
            "level": prob.get("level", 0),
            "subject": prob.get("subject", ""),
            "base_answer_conditioned": base_answer_conditioned,
            "power_answer_conditioned": power_answer_conditioned,
            "base_solution": base_text,
            "power_solution": power_text,
            "base_pred": base_pred,
            "power_pred": power_pred,
            "base_correct": is_equiv(base_pred, prob["answer"]) if base_pred else False,
            "power_correct": is_equiv(power_pred, prob["answer"]) if power_pred else False,
            "base_len": len(base_text),
            "power_len": len(power_text),
        })

    return pairs


# ---------------------------------------------------------------------------
# Phase 2: Judge
# ---------------------------------------------------------------------------

def judge_solutions(
    pairs: list[dict],
    judge_model: str,
    judge_max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
) -> list[dict]:
    """Run pairwise LLM judge in both orderings for each pair.

    For each problem, runs two comparisons:
      - Forward: A=base, B=power
      - Reverse: A=power, B=base

    If both verdicts agree on the same underlying solution, that's the winner.
    If they disagree (position bias), marked as inconsistent.
    """
    print(f"\n=== Phase 2: Judging with {judge_model} ===")

    llm = LLM(
        model=judge_model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts for both orderings
    forward_prompts = []  # A=base, B=power
    reverse_prompts = []  # A=power, B=base

    for pair in pairs:
        # Forward
        fwd_msg = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                problem=pair["problem"],
                solution_a=pair["base_solution"],
                solution_b=pair["power_solution"],
            )},
        ]
        forward_prompts.append(
            tokenizer.apply_chat_template(
                fwd_msg, tokenize=False, add_generation_prompt=True
            )
        )

        # Reverse
        rev_msg = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                problem=pair["problem"],
                solution_a=pair["power_solution"],
                solution_b=pair["base_solution"],
            )},
        ]
        reverse_prompts.append(
            tokenizer.apply_chat_template(
                rev_msg, tokenize=False, add_generation_prompt=True
            )
        )

    # Run all judge calls in one batch
    all_prompts = forward_prompts + reverse_prompts
    params = SamplingParams(temperature=0.0, max_tokens=judge_max_tokens)

    t0 = time.time()
    outputs = llm.generate(all_prompts, params, use_tqdm=True)
    print(f"Judging took {time.time() - t0:.1f}s")

    n = len(pairs)
    fwd_outputs = outputs[:n]
    rev_outputs = outputs[n:]

    # Parse and reconcile verdicts
    results = []
    for i, pair in enumerate(pairs):
        fwd_text = fwd_outputs[i].outputs[0].text
        rev_text = rev_outputs[i].outputs[0].text

        fwd_verdict = parse_verdict(fwd_text)
        rev_verdict = parse_verdict(rev_text)

        # Map verdicts to underlying solutions / categories
        # Forward: A=base, B=power
        # Reverse: A=power, B=base
        # "Both Good" / "Both Bad" are position-independent
        def _map_verdict(verdict, is_forward):
            if verdict == "Both Good":
                return "both_good"
            if verdict == "Both Bad":
                return "both_bad"
            if is_forward:
                return "base" if verdict == "A" else "power" if verdict == "B" else None
            else:
                return "power" if verdict == "A" else "base" if verdict == "B" else None

        fwd_mapped = _map_verdict(fwd_verdict, is_forward=True)
        rev_mapped = _map_verdict(rev_verdict, is_forward=False)

        if fwd_mapped and rev_mapped and fwd_mapped == rev_mapped:
            winner = fwd_mapped
        else:
            winner = "inconsistent"

        results.append({
            **pair,
            "fwd_judge_response": fwd_text,
            "rev_judge_response": rev_text,
            "fwd_verdict": fwd_verdict,
            "rev_verdict": rev_verdict,
            "fwd_mapped": fwd_mapped,
            "rev_mapped": rev_mapped,
            "winner": winner,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _count_winners(results: list[dict]) -> dict[str, int]:
    """Count occurrences of each winner category."""
    counts = {"base": 0, "power": 0, "both_good": 0, "both_bad": 0, "inconsistent": 0}
    for r in results:
        counts[r["winner"]] = counts.get(r["winner"], 0) + 1
    return counts


def print_report(results: list[dict]):
    """Print judge preference summary."""
    n = len(results)
    counts = _count_winners(results)

    base_correct = sum(1 for r in results if r["base_correct"])
    power_correct = sum(1 for r in results if r["power_correct"])

    # Derive labels from conditioning metadata
    r0 = results[0] if results else {}
    base_ac = r0.get("base_answer_conditioned", r0.get("answer_conditioned", False))
    power_ac = r0.get("power_answer_conditioned", r0.get("answer_conditioned", False))
    base_label = "Base+answer" if base_ac else "Base"
    power_label = "Power+answer" if power_ac else "Power"

    print(f"\n{'='*60}")
    print(f"Results ({n} problems)")
    if base_ac or power_ac:
        print(f"  {base_label} vs {power_label}")
    print(f"{'='*60}")

    # Response length stats
    base_lens = [r["base_len"] for r in results if "base_len" in r]
    power_lens = [r["power_len"] for r in results if "power_len" in r]

    if base_lens and power_lens:
        print(f"\nResponse length (chars):")
        print(f"  {'':>16} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
        print(f"  {base_label:>16} {statistics.mean(base_lens):>8.0f} "
              f"{statistics.median(base_lens):>8.0f} "
              f"{min(base_lens):>8} {max(base_lens):>8}")
        print(f"  {power_label:>16} {statistics.mean(power_lens):>8.0f} "
              f"{statistics.median(power_lens):>8.0f} "
              f"{min(power_lens):>8} {max(power_lens):>8}")

    print(f"\nCorrectness:")
    print(f"  {base_label}:  {base_correct}/{n} ({base_correct/n*100:.1f}%)")
    print(f"  {power_label}: {power_correct}/{n} ({power_correct/n*100:.1f}%)")

    print(f"\nJudge verdict:")
    print(f"  {base_label+' preferred:':<22} {counts['base']:>4}/{n} ({counts['base']/n*100:.1f}%)")
    print(f"  {power_label+' preferred:':<22} {counts['power']:>4}/{n} ({counts['power']/n*100:.1f}%)")
    print(f"  {'Both good:':<22} {counts['both_good']:>4}/{n} ({counts['both_good']/n*100:.1f}%)")
    print(f"  {'Both bad:':<22} {counts['both_bad']:>4}/{n} ({counts['both_bad']/n*100:.1f}%)")
    print(f"  {'Inconsistent:':<22} {counts['inconsistent']:>4}/{n} ({counts['inconsistent']/n*100:.1f}%)")

    # Win rate among decisive verdicts (base or power preferred)
    decisive = counts["base"] + counts["power"]
    if decisive:
        print(
            f"  {power_label} win rate (decisive only): "
            f"{counts['power']}/{decisive} ({counts['power']/decisive*100:.1f}%)"
        )

    # Breakdown by correctness
    categories = {
        "both_correct": [
            r for r in results if r["base_correct"] and r["power_correct"]
        ],
        "only_power_correct": [
            r for r in results if not r["base_correct"] and r["power_correct"]
        ],
        "only_base_correct": [
            r for r in results if r["base_correct"] and not r["power_correct"]
        ],
        "both_wrong": [
            r for r in results if not r["base_correct"] and not r["power_correct"]
        ],
    }

    print(f"\nBreakdown by correctness:")
    print(f"  {'Category':<24} {'N':>4} {'Base':>6} {'Power':>6} "
          f"{'Good':>6} {'Bad':>6} {'Incon':>6}")
    print(f"  {'-'*60}")
    for cat_name, cat_results in categories.items():
        if not cat_results:
            print(f"  {cat_name:<24} {0:>4}")
            continue
        c = _count_winners(cat_results)
        print(
            f"  {cat_name:<24} {len(cat_results):>4} "
            f"{c['base']:>6} {c['power']:>6} "
            f"{c['both_good']:>6} {c['both_bad']:>6} {c['inconsistent']:>6}"
        )

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise preference evaluation: base vs power sampling"
    )

    # Generation args
    parser.add_argument("--model", type=str, default=None,
                        help="Model to generate solutions with")
    parser.add_argument("--dataset", default="math500",
                        choices=list(DATASET_REGISTRY_EVAL.keys()))
    parser.add_argument("--levels", nargs="*", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--chat_template_model", type=str, default=None)

    # Power sampling config
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    # Judge args
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        help="HuggingFace model for judging (e.g. Qwen/Qwen2.5-32B-Instruct)")
    parser.add_argument("--judge_max_tokens", type=int, default=4096)
    parser.add_argument("--judge_tensor_parallel_size", type=int, default=None,
                        help="TP size for judge (defaults to --tensor_parallel_size)")
    parser.add_argument("--judge_max_model_len", type=int, default=8192)

    # Prompt conditioning
    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument(
        "--answer_conditioned", action="store_true",
        help="Include the correct answer in the prompt for both "
             "base and power sampling",
    )
    cond_group.add_argument(
        "--base_answer_conditioned", action="store_true",
        help="Only base sees the correct answer; power sampling does not. "
             "Compares answer-conditioned base (SDFT teacher proxy) vs "
             "unconditioned power distribution.",
    )

    # Workflow control
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate solutions, skip judging")
    parser.add_argument("--judge_from", type=str, default=None,
                        help="Path to existing solutions.json to judge (skip generation)")

    parser.add_argument("--output_dir", default="results/judge")

    args = parser.parse_args()

    judge_tp = args.judge_tensor_parallel_size or args.tensor_parallel_size

    # Load chat template tokenizer if needed
    chat_template_tokenizer = None
    if args.chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model, trust_remote_code=True
        )

    # --- Phase 1: Generate or load solutions ---
    if args.judge_from:
        print(f"Loading solutions from {args.judge_from}")
        with open(args.judge_from) as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} solution pairs")
    else:
        assert args.model, "--model is required for generation"

        problems = DATASET_REGISTRY_EVAL[args.dataset](levels=args.levels)
        print(f"Loaded {len(problems)} problems from {args.dataset}")

        if args.num_samples and args.num_samples < len(problems):
            random.seed(args.seed)
            problems = random.sample(problems, args.num_samples)
            print(f"Subsampled to {len(problems)} problems (seed={args.seed})")

        # Resolve per-method conditioning flags
        base_ac = args.answer_conditioned or args.base_answer_conditioned
        power_ac = args.answer_conditioned  # only True when both are conditioned

        pairs = generate_solutions(
            model_name=args.model,
            problems=problems,
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
            base_answer_conditioned=base_ac,
            power_answer_conditioned=power_ac,
        )

        # Save solutions
        model_slug = args.model.replace("/", "_")
        if args.answer_conditioned:
            subdir = "answer_conditioned"
        elif args.base_answer_conditioned:
            subdir = "base_answer_conditioned"
        else:
            subdir = "unconditioned"
        sol_dir = os.path.join(args.output_dir, args.dataset, model_slug, subdir)
        os.makedirs(sol_dir, exist_ok=True)
        sol_path = os.path.join(sol_dir, "solutions.json")
        with open(sol_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"\nSaved solutions: {sol_path}")

    if args.generate_only:
        # Print correctness stats and exit
        n = len(pairs)
        bc = sum(1 for p in pairs if p["base_correct"])
        pc = sum(1 for p in pairs if p["power_correct"])
        print(f"\nCorrectness: base={bc}/{n} ({bc/n*100:.1f}%), "
              f"power={pc}/{n} ({pc/n*100:.1f}%)")
        return

    # --- Phase 2: Judge ---
    assert args.judge_model, "--judge_model is required for judging"

    results = judge_solutions(
        pairs=pairs,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        tensor_parallel_size=judge_tp,
        max_model_len=args.judge_max_model_len,
    )

    print_report(results)

    # Save full results
    if args.judge_from:
        out_dir = os.path.dirname(args.judge_from)
    else:
        model_slug = args.model.replace("/", "_")
        if args.answer_conditioned:
            subdir = "answer_conditioned"
        elif args.base_answer_conditioned:
            subdir = "base_answer_conditioned"
        else:
            subdir = "unconditioned"
        out_dir = os.path.join(args.output_dir, args.dataset, model_slug, subdir)
    os.makedirs(out_dir, exist_ok=True)

    judge_slug = args.judge_model.replace("/", "_")
    results_path = os.path.join(out_dir, f"judge_{judge_slug}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved judge results: {results_path}")

    # Save summary
    n = len(results)
    counts = _count_winners(results)
    decisive = counts["base"] + counts["power"]
    r0 = results[0] if results else {}
    summary = {
        "judge_model": args.judge_model,
        "num_problems": n,
        "base_answer_conditioned": r0.get("base_answer_conditioned", False),
        "power_answer_conditioned": r0.get("power_answer_conditioned", False),
        "base_correct": sum(1 for r in results if r["base_correct"]),
        "power_correct": sum(1 for r in results if r["power_correct"]),
        "verdicts": counts,
        "power_win_rate_decisive": (
            counts["power"] / decisive if decisive else None
        ),
        "response_length": {
            "base_mean": statistics.mean(bl) if (bl := [r["base_len"] for r in results if "base_len" in r]) else None,
            "base_median": statistics.median(bl) if bl else None,
            "power_mean": statistics.mean(pl) if (pl := [r["power_len"] for r in results if "power_len" in r]) else None,
            "power_median": statistics.median(pl) if pl else None,
        },
    }
    summary_path = os.path.join(out_dir, f"judge_{judge_slug}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
