"""
Evaluate base vs. power-sampled generation on IFBench.

Requires the IFBench repo to be cloned alongside this project:
    git clone https://github.com/allenai/IFBench.git ../IFBench

Usage:
    # Base model only (greedy)
    uv run python -m scripts.run_ifbench \
        --model Qwen/Qwen2.5-7B-Instruct

    # Base + power sampling comparison
    uv run python -m scripts.run_ifbench \
        --model Qwen/Qwen2.5-7B-Instruct \
        --power_sampling --use_vllm

    # Quick test on 10 samples
    uv run python -m scripts.run_ifbench \
        --model Qwen/Qwen2.5-7B-Instruct \
        --power_sampling --use_vllm --num_samples 10
"""

import argparse
import json
import os
import sys
import time

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# IFBench evaluation helpers
# ---------------------------------------------------------------------------

def get_ifbench_dir() -> str:
    """Locate the IFBench repo (expected at ../IFBench relative to project root)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "IFBench"),
        os.path.expanduser("~/IFBench"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "run_eval.py")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "IFBench repo not found. Clone it with:\n"
        "  git clone https://github.com/allenai/IFBench.git ../IFBench"
    )


def load_ifbench_test() -> list[dict]:
    """Load IFBench test data from HuggingFace."""
    ds = load_dataset("allenai/IFBench_test", split="train")
    return [row for row in ds]


def run_ifbench_eval(
    ifbench_dir: str,
    test_data: list[dict],
    responses: dict[str, str],
    tag: str,
    output_dir: str,
) -> dict:
    """Run IFBench strict/loose evaluation and return accuracy scores.

    Args:
        ifbench_dir: Path to cloned IFBench repo.
        test_data: List of IFBench test examples.
        responses: Mapping from prompt text to response text.
        tag: Label for this run (e.g. "base_greedy", "power_sampling").
        output_dir: Where to write result files.

    Returns:
        Dict with strict_accuracy, loose_accuracy, and per-constraint details.
    """
    # Add IFBench to path so we can import its evaluation code
    if ifbench_dir not in sys.path:
        sys.path.insert(0, ifbench_dir)

    import evaluation_lib  # type: ignore

    # Build InputExample list and prompt->response dict
    inputs = []
    prompt_to_response = {}
    for row in test_data:
        prompt = row["prompt"]
        if prompt not in responses:
            continue
        inp = evaluation_lib.InputExample(
            key=int(row["key"]),
            instruction_id_list=row["instruction_id_list"],
            prompt=prompt,
            kwargs=row["kwargs"],
        )
        inputs.append(inp)
        prompt_to_response[prompt] = responses[prompt]

    if not inputs:
        print(f"  WARNING: No matching prompts found for {tag}")
        return {"strict_accuracy": 0.0, "loose_accuracy": 0.0}

    results = {}
    for eval_fn, mode in [
        (evaluation_lib.test_instruction_following_strict, "strict"),
        (evaluation_lib.test_instruction_following_loose, "loose"),
    ]:
        outputs = [eval_fn(inp, prompt_to_response) for inp in inputs]
        follow_all = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all) / len(follow_all)
        results[f"{mode}_accuracy"] = accuracy

        # Per-instruction breakdown
        instruction_counts: dict[str, dict[str, int]] = {}
        for o in outputs:
            for iid, followed in zip(o.instruction_id_list, o.follow_instruction_list):
                if iid not in instruction_counts:
                    instruction_counts[iid] = {"followed": 0, "total": 0}
                instruction_counts[iid]["total"] += 1
                instruction_counts[iid]["followed"] += int(followed)
        results[f"{mode}_per_instruction"] = {
            iid: {**v, "accuracy": v["followed"] / v["total"]}
            for iid, v in sorted(instruction_counts.items())
        }

        # Write detailed eval output
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{tag}_{mode}.jsonl")
        evaluation_lib.write_outputs(out_path, outputs)

    return results


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
    prompts: list[str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> list[str]:
    """Generate responses using standard vLLM sampling."""
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Apply chat template
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate(formatted, params, use_tqdm=True)
    elapsed = time.time() - t0

    responses = [o.outputs[0].text for o in outputs]
    print(f"Base generation: {len(prompts)} prompts in {elapsed:.1f}s")

    destroy_vllm_model(llm)
    return responses


def generate_power_sampling(
    model_name: str,
    prompts: list[str],
    alpha: float = 4.0,
    batch_size: int = 8,
    num_candidates: int = 32,
    top_k: int = 8,
    num_rollouts: int = 8,
    lookahead: int = 192,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    confidence_threshold: float | None = None,
) -> list[str]:
    """Generate responses using VLLMBatchedPowerSampler."""
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

    responses = []
    t0 = time.time()
    pbar = tqdm(prompts, desc="power_sampling", unit="prompt")
    for p in pbar:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
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
    print(f"Power sampling: {len(prompts)} prompts in {elapsed:.1f}s")
    return responses


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(base_results: dict, ps_results: dict | None = None):
    """Print a comparison table of strict/loose accuracy."""
    print(f"\n{'='*64}")
    print("IFBench Results")
    print(f"{'='*64}")

    header = f"{'Metric':<25} {'Base':>10}"
    if ps_results:
        header += f" {'Power':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for mode in ["strict", "loose"]:
        base_acc = base_results[f"{mode}_accuracy"]
        row = f"{mode + ' accuracy':<25} {base_acc:>10.1%}"
        if ps_results:
            ps_acc = ps_results[f"{mode}_accuracy"]
            delta = ps_acc - base_acc
            sign = "+" if delta >= 0 else ""
            row += f" {ps_acc:>10.1%} {sign}{delta:>9.1%}"
        print(row)

    # Per-instruction breakdown for strict
    print(f"\n{'='*64}")
    print("Per-instruction strict accuracy")
    print(f"{'='*64}")

    base_per = base_results.get("strict_per_instruction", {})
    ps_per = ps_results.get("strict_per_instruction", {}) if ps_results else {}
    all_instructions = sorted(set(base_per) | set(ps_per))

    header = f"{'Instruction':<40} {'Base':>10}"
    if ps_results:
        header += f" {'Power':>10}"
    print(header)
    print("-" * len(header))

    for iid in all_instructions:
        b = base_per.get(iid, {})
        row = f"{iid:<40} {b.get('accuracy', 0):>10.1%}"
        if ps_results:
            p = ps_per.get(iid, {})
            row += f" {p.get('accuracy', 0):>10.1%}"
        print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base vs. power sampling on IFBench"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output_dir", default="results/ifbench")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for base generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset (for quick tests)")
    parser.add_argument("--seed", type=int, default=42)

    # Power sampling
    parser.add_argument("--power_sampling", action="store_true",
                        help="Also evaluate with power sampling")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM backend for power sampling")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=None)

    args = parser.parse_args()

    # --- Load data ---
    print("Loading IFBench test data...")
    test_data = load_ifbench_test()
    print(f"Loaded {len(test_data)} test examples")

    if args.num_samples is not None and args.num_samples < len(test_data):
        import random
        random.seed(args.seed)
        test_data = random.sample(test_data, args.num_samples)
        print(f"Subsampled to {args.num_samples} examples (seed={args.seed})")

    prompts = [row["prompt"] for row in test_data]

    # --- Locate IFBench repo for evaluation ---
    ifbench_dir = get_ifbench_dir()
    print(f"Using IFBench evaluation from: {ifbench_dir}")

    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, model_slug)

    # --- Base generation ---
    print(f"\n{'='*64}")
    print("Generating base responses...")
    print(f"{'='*64}")
    base_responses = generate_base(
        model_name=args.model,
        prompts=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    base_prompt_to_response = dict(zip(prompts, base_responses))

    # Save raw responses
    os.makedirs(output_dir, exist_ok=True)
    base_resp_path = os.path.join(output_dir, "base_responses.jsonl")
    with open(base_resp_path, "w") as f:
        for prompt, response in zip(prompts, base_responses):
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

    # Evaluate base
    print("\nEvaluating base responses...")
    base_results = run_ifbench_eval(
        ifbench_dir, test_data, base_prompt_to_response, "base", output_dir
    )

    # --- Power sampling generation ---
    ps_results = None
    if args.power_sampling:
        print(f"\n{'='*64}")
        print("Generating power sampling responses...")
        print(f"{'='*64}")
        ps_responses = generate_power_sampling(
            model_name=args.model,
            prompts=prompts,
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
        )
        ps_prompt_to_response = dict(zip(prompts, ps_responses))

        # Save raw responses
        ps_resp_path = os.path.join(output_dir, "power_sampling_responses.jsonl")
        with open(ps_resp_path, "w") as f:
            for prompt, response in zip(prompts, ps_responses):
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

        # Evaluate power sampling
        print("\nEvaluating power sampling responses...")
        ps_results = run_ifbench_eval(
            ifbench_dir, test_data, ps_prompt_to_response,
            "power_sampling", output_dir,
        )

    # --- Report ---
    print_comparison(base_results, ps_results)

    # Save summary
    summary = {
        "model": args.model,
        "num_examples": len(test_data),
        "base": {
            "temperature": args.temperature,
            "strict_accuracy": base_results["strict_accuracy"],
            "loose_accuracy": base_results["loose_accuracy"],
        },
    }
    if ps_results:
        summary["power_sampling"] = {
            "alpha": args.alpha,
            "top_k": args.top_k,
            "num_rollouts": args.num_rollouts,
            "lookahead": args.lookahead,
            "batch_size": args.batch_size,
            "num_candidates": args.num_candidates,
            "strict_accuracy": ps_results["strict_accuracy"],
            "loose_accuracy": ps_results["loose_accuracy"],
        }

    summary_path = os.path.join(output_dir, "ifbench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
