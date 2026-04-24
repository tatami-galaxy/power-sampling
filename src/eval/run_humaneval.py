"""
Evaluate base vs. power-sampled generation on HumanEval.

Scoring uses the official `human_eval.execution.check_correctness`, which runs
each completion in a subprocess with a reliability guard and a timeout.

Usage:
    # Base model only (greedy)
    uv run python -m src.eval.run_humaneval \
        --model Qwen/Qwen2.5-7B-Instruct

    # Base + power sampling
    uv run python -m src.eval.run_humaneval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --power_sampling --use_vllm

    # Quick test on 10 problems
    uv run python -m src.eval.run_humaneval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --power_sampling --use_vllm --num_samples 10
"""

import argparse
import json
import os
import re
import time

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful Python programming assistant. "
    "Return only code, wrapped in a ```python ... ``` block."
)


def format_prompt(problem_text: str) -> list[dict]:
    user = (
        "Write a Python function to solve the following problem:\n\n"
        + problem_text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract code from a ```python ... ``` fence, or return text as-is."""
    matches = _FENCE_RE.findall(text)
    if matches:
        # Prefer the last fenced block (models sometimes show examples first)
        return matches[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_humaneval() -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            "entry_point": r["entry_point"],
            "test": r["test"],
            "canonical_solution": r["canonical_solution"],
        }
        for r in ds
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def destroy_vllm_model(llm: LLM):
    """Destroy a vLLM model and free all GPU memory."""
    import gc
    import torch
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print("Freed GPU memory from previous model")


def generate_base(
    model_name: str,
    problems: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    chat_template_model: str | None = None,
) -> list[str]:
    """Generate one completion per problem via vLLM."""
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    template_tok = tokenizer
    if chat_template_model:
        from transformers import AutoTokenizer
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )

    prompts = []
    for p in problems:
        messages = format_prompt(p["prompt"])
        text = template_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - t0

    responses = [o.outputs[0].text for o in outputs]
    print(f"Base generation: {len(prompts)} problems in {elapsed:.1f}s")

    destroy_vllm_model(llm)
    return responses


def generate_power_sampling(
    model_name: str,
    problems: list[dict],
    alpha: float = 4.0,
    batch_size: int = 8,
    num_candidates: int = 32,
    top_k: int = 8,
    num_rollouts: int = 8,
    lookahead: int = 64,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    confidence_threshold: float | None = None,
    chat_template_model: str | None = None,
) -> list[str]:
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
    tokenizer = sampler.tokenizer

    template_tok = tokenizer
    if chat_template_model:
        from transformers import AutoTokenizer
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )

    responses = []
    t0 = time.time()
    pbar = tqdm(problems, desc="power_sampling", unit="problem")
    for prob in pbar:
        messages = format_prompt(prob["prompt"])
        text = template_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text)

        sample_t0 = time.time()
        out = sampler.generate(input_ids=input_ids, verbose=False)
        sample_elapsed = time.time() - sample_t0

        responses.append(out["text"])
        pbar.set_postfix(
            tokens=out["num_tokens_generated"],
            time=f"{sample_elapsed:.1f}s",
        )

    elapsed = time.time() - t0
    print(f"Power sampling: {len(problems)} problems in {elapsed:.1f}s")
    return responses


# ---------------------------------------------------------------------------
# Scoring (official human_eval sandbox)
# ---------------------------------------------------------------------------

def score_completions(
    problems: list[dict],
    responses: list[str],
    timeout: float = 10.0,
) -> list[dict]:
    """Run each extracted completion against the unit tests.

    Returns one dict per problem with keys:
        task_id, passed, code, response, result.
    """
    from human_eval.execution import check_correctness

    out = []
    for i, (prob, response) in enumerate(
        tqdm(list(zip(problems, responses)), desc="scoring", unit="problem")
    ):
        code = extract_code(response)
        # Model returns the full function; blank the HumanEval prompt so
        # check_correctness just concatenates our code with the test block.
        human_eval_problem = {
            "task_id": prob["task_id"],
            "prompt": "",
            "entry_point": prob["entry_point"],
            "test": prob["test"],
        }
        result = check_correctness(
            human_eval_problem, code, timeout=timeout, completion_id=i
        )
        out.append({
            "task_id": prob["task_id"],
            "entry_point": prob["entry_point"],
            "passed": bool(result["passed"]),
            "result": result["result"],
            "code": code,
            "response": response,
        })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(tag: str, scored: list[dict]):
    total = len(scored)
    passed = sum(r["passed"] for r in scored)
    print(f"\n{'='*60}")
    print(f"{tag}: pass@1 = {passed}/{total} = {passed/total*100:.1f}%")
    print(f"{'='*60}")
    # Top failure modes
    fails = [r["result"] for r in scored if not r["passed"]]
    if fails:
        buckets = {"failed": 0, "timed out": 0, "error": 0, "other": 0}
        for r in fails:
            if r.startswith("passed"):
                continue
            if r.startswith("failed:"):
                buckets["failed"] += 1
            elif "timed out" in r:
                buckets["timed out"] += 1
            elif r == "":
                buckets["other"] += 1
            else:
                buckets["error"] += 1
        print(f"  Failures: " + ", ".join(f"{k}={v}" for k, v in buckets.items() if v))


def print_comparison(base_scored: list[dict], power_scored: list[dict]):
    base = {r["task_id"]: r["passed"] for r in base_scored}
    power = {r["task_id"]: r["passed"] for r in power_scored}
    shared = set(base) & set(power)

    both_right = sum(1 for t in shared if base[t] and power[t])
    both_wrong = sum(1 for t in shared if not base[t] and not power[t])
    base_wrong_power_right = sum(1 for t in shared if not base[t] and power[t])
    base_right_power_wrong = sum(1 for t in shared if base[t] and not power[t])

    print(f"\n{'='*60}")
    print(f"Baseline vs Power ({len(shared)} shared problems)")
    print(f"{'='*60}")
    print(f"  Both pass:                    {both_right}")
    print(f"  Both fail:                    {both_wrong}")
    print(f"  Baseline fail -> Power pass:  {base_wrong_power_right}")
    print(f"  Baseline pass -> Power fail:  {base_right_power_wrong}")
    print(f"  Net delta:                    {base_wrong_power_right - base_right_power_wrong:+d}")
    print(f"{'='*60}")


def save_results(output_dir: str, tag: str, scored: list[dict]):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}_results.jsonl")
    with open(path, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs. power sampling on HumanEval"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--chat_template_model", default=None,
                        help="Borrow chat template from this model (for base models)")
    parser.add_argument("--output_dir", default="results/humaneval")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset (for quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Per-problem execution timeout (seconds)")

    # Power sampling
    parser.add_argument("--power_sampling", action="store_true")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=64,
                        help="Rollout horizon. Default 64 for HumanEval "
                             "(completions are short; 192 would be mostly EOS).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    args = parser.parse_args()

    # --- Load data ---
    problems = load_humaneval()
    print(f"Loaded {len(problems)} HumanEval problems")

    if args.num_samples is not None and args.num_samples < len(problems):
        import random
        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
        print(f"Subsampled to {args.num_samples} problems (seed={args.seed})")

    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_slug)

    # --- Base generation ---
    print(f"\n{'='*60}\nBase generation\n{'='*60}")
    base_responses = generate_base(
        model_name=args.model,
        problems=problems,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        chat_template_model=args.chat_template_model,
    )
    base_scored = score_completions(problems, base_responses, timeout=args.timeout)
    print_report("Base", base_scored)
    save_results(output_dir, "base", base_scored)

    # --- Power sampling ---
    power_scored = None
    if args.power_sampling:
        print(f"\n{'='*60}\nPower sampling generation\n{'='*60}")
        power_responses = generate_power_sampling(
            model_name=args.model,
            problems=problems,
            alpha=args.alpha,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            top_k=args.top_k,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            confidence_threshold=args.confidence_threshold,
            chat_template_model=args.chat_template_model,
        )
        power_scored = score_completions(problems, power_responses, timeout=args.timeout)
        print_report("Power", power_scored)
        save_results(output_dir, "power", power_scored)
        print_comparison(base_scored, power_scored)

    # --- Summary ---
    summary = {
        "model": args.model,
        "num_problems": len(problems),
        "base": {
            "temperature": args.temperature,
            "pass@1": sum(r["passed"] for r in base_scored) / len(base_scored),
        },
    }
    if power_scored is not None:
        summary["power_sampling"] = {
            "alpha": args.alpha,
            "top_k": args.top_k,
            "num_rollouts": args.num_rollouts,
            "lookahead": args.lookahead,
            "batch_size": args.batch_size,
            "num_candidates": args.num_candidates,
            "pass@1": sum(r["passed"] for r in power_scored) / len(power_scored),
        }

    summary_path = os.path.join(output_dir, "humaneval_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
