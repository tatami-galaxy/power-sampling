"""
Generate power-sampled demonstrations for SDFT distillation.

Uses VLLMBatchedPowerSampler to generate high-quality solutions from the
model's own sharpened distribution pi_alpha, then verifies correctness
against gold answers.  Saves results as JSONL compatible with train_sdft.py.

Output files:
  {output_dir}/power_demos_all.jsonl       — all generations
  {output_dir}/power_demos_correct.jsonl   — filtered to verified correct

Each line has: problem, answer, solution, correct, pred_answer,
num_tokens_generated, alpha, difficulty (if available).

The script is resumable: on restart it skips problems already in the output.
Supports data-parallel generation across multiple GPUs.

Usage:
    # Generate from DeepMath
    uv run python -m scripts.rl.generate_power_demos \
        --model Qwen/Qwen3-4B \
        --dataset deepmath \
        --alpha 4.0 \
        --max_problems 1000

    # Generate from Polaris, specific difficulties
    uv run python -m scripts.rl.generate_power_demos \
        --model Qwen/Qwen3-4B \
        --dataset polaris \
        --difficulty 1/8 2/8 3/8 \
        --alpha 3.0

    # Multi-GPU data parallelism (auto-detects GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m scripts.rl.generate_power_demos \
        --model Qwen/Qwen3-4B \
        --dataset deepmath \
        --max_problems 1000

    # Multi-GPU with tensor parallelism (TP=2, 4 GPUs → 2 workers)
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m scripts.rl.generate_power_demos \
        --model Qwen/Qwen3-8B \
        --dataset deepmath \
        --tensor_parallel_size 2

    # Explicit worker count
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m scripts.rl.generate_power_demos \
        --model Qwen/Qwen3-4B \
        --dataset deepmath \
        --num_workers 4
"""

import argparse
import json
import multiprocessing as mp
import os
import time

from datasets import load_dataset
from tqdm import tqdm

from scripts.utils import extract_boxed_answer, is_equiv


# ---------------------------------------------------------------------------
# Prompt formatting (same as eval / training scripts)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# GPU detection (no CUDA init — safe to call before spawn)
# ---------------------------------------------------------------------------

def _get_visible_gpus() -> list[int]:
    """Return list of GPU indices from CUDA_VISIBLE_DEVICES, or detect via nvidia-smi."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x.strip()) for x in env.split(",") if x.strip()]
    # Fallback: count GPUs without initializing CUDA
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
    except Exception:
        return [0]


# ---------------------------------------------------------------------------
# Dataset loading (returns list[dict] with problem, answer, difficulty)
# ---------------------------------------------------------------------------

def load_problems(dataset: str, max_problems: int | None, sources: list[str] | None,
                  difficulty: list[str] | None, seed: int) -> list[dict]:
    """Load problems from the specified dataset as a list of dicts."""

    if dataset == "deepmath":
        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        problems = []
        for row in ds:
            if row["final_answer"] and row["final_answer"].strip():
                problems.append({
                    "problem": row["question"],
                    "answer": row["final_answer"],
                    "difficulty": str(round(row["difficulty"], 1)) if row.get("difficulty") is not None else None,
                })

    elif dataset == "polaris":
        ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        problems = []
        for row in ds:
            if row["answer"] and row["answer"].strip():
                if difficulty and row.get("difficulty") not in difficulty:
                    continue
                problems.append({
                    "problem": row["problem"],
                    "answer": row["answer"],
                    "difficulty": row.get("difficulty"),
                })

    elif dataset == "numinamath":
        ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
        problems = []
        for row in ds:
            if row["problem_is_valid"] != "Yes":
                continue
            if not row["answer"] or not row["answer"].strip():
                continue
            if sources and row["source"] not in sources:
                continue
            problems.append({
                "problem": row["problem"],
                "answer": row["answer"],
                "difficulty": row.get("source"),
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Deterministic shuffle + limit
    import random
    random.seed(seed)
    random.shuffle(problems)
    if max_problems:
        problems = problems[:max_problems]

    return problems


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results(output_path: str) -> set[str]:
    """Load problem texts already generated (for resume)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["problem"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ---------------------------------------------------------------------------
# Worker: generates solutions for a shard of problems on assigned GPUs
# ---------------------------------------------------------------------------

def _worker_fn(rank: int, gpu_ids: list[int], problems: list[dict],
               args_dict: dict, output_path: str):
    """Run in a spawned subprocess. Sets CUDA_VISIBLE_DEVICES, creates its own
    sampler, generates for its shard, writes results to output_path."""

    # Must set before any CUDA/torch/vLLM import
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    from transformers import AutoTokenizer
    from scalable_power_sampling import VLLMBatchedPowerSampler

    # Reconstruct args
    args = argparse.Namespace(**args_dict)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model, trust_remote_code=True)
        tokenizer.chat_template = template_tok.chat_template

    sampler = VLLMBatchedPowerSampler(
        model_name=args.model,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        num_rollouts=args.num_rollouts,
        lookahead=args.lookahead,
        max_new_tokens=args.max_tokens,
        # Worker sees only its assigned GPUs (remapped to 0..TP-1)
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        confidence_threshold=args.confidence_threshold,
    )

    print(f"[Worker {rank}] GPUs {gpu_ids}, {len(problems)} problems, {sampler}")

    correct_count = 0
    total_count = 0
    t0 = time.time()

    with open(output_path, "w") as f_out:
        desc = f"worker {rank}"
        pbar = tqdm(problems, desc=desc, unit="problem", position=rank)
        for prob in pbar:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt_text)

            sample_t0 = time.time()
            out = sampler.generate(input_ids=input_ids, verbose=False)
            sample_elapsed = time.time() - sample_t0

            response = out["text"]
            pred_answer = extract_boxed_answer(response)
            correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False

            result = {
                "problem": prob["problem"],
                "answer": prob["answer"],
                "solution": response,
                "correct": correct,
                "pred_answer": pred_answer,
                "num_tokens_generated": out["num_tokens_generated"],
                "sample_time_s": round(sample_elapsed, 2),
                "alpha": args.alpha,
            }
            if prob.get("difficulty") is not None:
                result["difficulty"] = prob["difficulty"]

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            total_count += 1
            correct_count += int(correct)

            pbar.set_postfix(
                correct=f"{correct_count}/{total_count}",
                tokens=out["num_tokens_generated"],
                time=f"{sample_elapsed:.1f}s",
            )

    elapsed = time.time() - t0
    print(f"[Worker {rank}] Done: {total_count} problems, {correct_count} correct, {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Sequential generation (single worker, used when num_workers=1)
# ---------------------------------------------------------------------------

def _generate_sequential(problems: list[dict], args, all_path: str, num_already_done: int):
    """Generate solutions sequentially. Appends to all_path."""
    from transformers import AutoTokenizer
    from scalable_power_sampling import VLLMBatchedPowerSampler

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model, trust_remote_code=True)
        tokenizer.chat_template = template_tok.chat_template
        print(f"Using chat template from: {args.chat_template_model}")

    sampler = VLLMBatchedPowerSampler(
        model_name=args.model,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        num_rollouts=args.num_rollouts,
        lookahead=args.lookahead,
        max_new_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"\n{sampler}\n")

    t0 = time.time()
    correct_count = 0
    total_count = num_already_done
    total_tokens = 0

    with open(all_path, "a") as f_all:
        pbar = tqdm(problems, desc="power sampling", unit="problem")
        for prob in pbar:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt_text)

            sample_t0 = time.time()
            out = sampler.generate(input_ids=input_ids, verbose=False)
            sample_elapsed = time.time() - sample_t0

            response = out["text"]
            pred_answer = extract_boxed_answer(response)
            correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False

            result = {
                "problem": prob["problem"],
                "answer": prob["answer"],
                "solution": response,
                "correct": correct,
                "pred_answer": pred_answer,
                "num_tokens_generated": out["num_tokens_generated"],
                "sample_time_s": round(sample_elapsed, 2),
                "alpha": args.alpha,
            }
            if prob.get("difficulty") is not None:
                result["difficulty"] = prob["difficulty"]

            f_all.write(json.dumps(result) + "\n")
            f_all.flush()

            total_count += 1
            correct_count += int(correct)
            total_tokens += out["num_tokens_generated"]

            pbar.set_postfix(
                correct=f"{correct_count}/{total_count}",
                acc=f"{correct_count/total_count*100:.1f}%",
                tokens=out["num_tokens_generated"],
                time=f"{sample_elapsed:.1f}s",
            )

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s")
    if problems:
        print(f"  Generated: {len(problems)}, Correct: {correct_count}")
        print(f"  Avg tokens: {total_tokens / len(problems):.0f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(args):
    # --- Load problems ---
    problems = load_problems(
        dataset=args.dataset,
        max_problems=args.max_problems,
        sources=args.sources,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # --- Output paths ---
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, args.dataset, model_slug, f"alpha_{args.alpha}")
    os.makedirs(output_dir, exist_ok=True)

    all_path = os.path.join(output_dir, "power_demos_all.jsonl")
    correct_path = os.path.join(output_dir, "power_demos_correct.jsonl")

    # --- Resume: skip already-generated problems ---
    done = load_existing_results(all_path)
    remaining = [p for p in problems if p["problem"] not in done]
    print(f"Already generated: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to generate. Exiting.")
        _write_correct_subset(all_path, correct_path)
        _print_summary(all_path)
        return

    # --- Determine parallelism ---
    gpus = _get_visible_gpus()
    tp = args.tensor_parallel_size

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = len(gpus) // tp

    num_workers = max(1, min(num_workers, len(gpus) // tp))

    if num_workers <= 1:
        # --- Sequential path ---
        print(f"Running sequentially on {len(gpus)} GPU(s) (TP={tp})")
        _generate_sequential(remaining, args, all_path, num_already_done=len(done))
    else:
        # --- Parallel path ---
        print(f"Running {num_workers} workers on {len(gpus)} GPUs (TP={tp})")

        # Assign GPUs: worker i gets gpus[i*tp : (i+1)*tp]
        gpu_assignments = [gpus[i * tp : (i + 1) * tp] for i in range(num_workers)]

        # Partition problems round-robin for balanced shards
        shards = [[] for _ in range(num_workers)]
        for i, prob in enumerate(remaining):
            shards[i % num_workers].append(prob)

        for i in range(num_workers):
            print(f"  Worker {i}: GPUs {gpu_assignments[i]}, {len(shards[i])} problems")

        # Worker output files
        worker_paths = [
            os.path.join(output_dir, f"power_demos_worker_{i}.jsonl")
            for i in range(num_workers)
        ]

        args_dict = vars(args)

        # Spawn workers
        mp.set_start_method("spawn", force=True)
        processes = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(i, gpu_assignments[i], shards[i], args_dict, worker_paths[i]),
            )
            processes.append(p)
            p.start()

        # Wait for all workers
        for p in processes:
            p.join()

        # Check for failures
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"WARNING: Workers {failed} failed (non-zero exit code)")

        # Merge worker outputs into main file
        merged = 0
        with open(all_path, "a") as f_out:
            for wp in worker_paths:
                if os.path.exists(wp):
                    with open(wp) as f_in:
                        for line in f_in:
                            f_out.write(line)
                            merged += 1
                    os.remove(wp)
        print(f"Merged {merged} results from {num_workers} workers")

    # --- Write correct-only subset ---
    _write_correct_subset(all_path, correct_path)
    _print_summary(all_path)

    # --- Save generation config ---
    config_path = os.path.join(output_dir, "generate_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")


def _write_correct_subset(all_path: str, correct_path: str):
    """Read all results and write correct-only subset."""
    total = 0
    correct = 0
    with open(all_path) as f_in, open(correct_path, "w") as f_out:
        for line in f_in:
            total += 1
            row = json.loads(line)
            if row.get("correct"):
                f_out.write(line)
                correct += 1
    print(f"Wrote {correct}/{total} correct demos to {correct_path}")


def _print_summary(all_path: str):
    """Print summary statistics from the results file."""
    results = []
    with open(all_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return

    total = len(results)
    correct = sum(r["correct"] for r in results)
    tokens = [r["num_tokens_generated"] for r in results]
    times = [r.get("sample_time_s", 0) for r in results]

    print(f"\n{'='*60}")
    print(f"Summary: {total} demos, {correct} correct ({correct/total*100:.1f}%)")
    print(f"  Tokens: mean={sum(tokens)/total:.0f}, min={min(tokens)}, max={max(tokens)}")
    print(f"  Time:   mean={sum(times)/total:.1f}s")

    # By difficulty
    difficulties = {}
    for r in results:
        d = r.get("difficulty")
        if d is not None:
            if d not in difficulties:
                difficulties[d] = {"correct": 0, "total": 0}
            difficulties[d]["total"] += 1
            difficulties[d]["correct"] += int(r["correct"])

    if difficulties:
        print(f"\n  By difficulty:")
        for d in sorted(difficulties.keys()):
            s = difficulties[d]
            print(f"    {d}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate power-sampled demonstrations for SDFT distillation"
    )

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from")

    # Data
    parser.add_argument("--dataset", type=str, default="deepmath",
                        choices=["deepmath", "polaris", "numinamath"])
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Max problems to load from dataset before generation")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="NuminaMath source filter")
    parser.add_argument("--difficulty", nargs="*", default=None,
                        help="Polaris difficulty filter (e.g. 1/8 2/8)")
    parser.add_argument("--seed", type=int, default=42)

    # Power sampling
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Tokens per chunk (B)")
    parser.add_argument("--num_candidates", type=int, default=32,
                        help="Candidate chunks per step (L)")
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens per generation")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Skip rollouts when top-1 vs top-2 gap exceeds this")

    # vLLM / parallelism
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="GPUs per sampler instance (tensor parallelism)")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: auto = num_gpus // tensor_parallel_size)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/power_demos")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
