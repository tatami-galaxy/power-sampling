"""
Generate baseline demonstrations (plain vLLM sampling — no power distribution)
for comparison against power-sampled demos from generate_power_demos.py.

Identical data-loading, resume, multi-GPU, and output-file structure — just
swaps the sampler for standard autoregressive sampling at configurable
temperature / top_p.

Output files:
  {output_dir}/demos_all.jsonl       — all generations
  {output_dir}/demos_correct.jsonl   — filtered to verified correct (skipped if no gold)

Each line has: problem, answer, solution, correct, pred_answer,
num_tokens_generated, temperature, top_p, difficulty (if available).

Usage:
    uv run python -m scripts.rl.generate_demos \
        --model Qwen/Qwen3-4B \
        --dataset synthetic \
        --synthetic_file results/synthetic_questions/Qwen_Qwen3-4B/questions_accepted.jsonl \
        --temperature 1.0 --top_p 0.95
"""

import argparse
import json
import multiprocessing as mp
import os
import time

from datasets import load_dataset
from tqdm import tqdm

from src.utils import extract_boxed_answer, is_equiv


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
                  difficulty: list[str] | None, seed: int,
                  synthetic_file: str | None = None) -> list[dict]:
    """Load problems from the specified dataset as a list of dicts.

    For ``dataset == "synthetic"``, reads a JSONL file with ``problem`` (and
    optionally ``topic``) fields. No gold answer is expected; ``answer`` is
    set to ``None`` and downstream correctness checks are skipped.
    """

    if dataset == "synthetic":
        if not synthetic_file:
            raise ValueError("--synthetic_file is required when --dataset synthetic")
        problems = []
        with open(synthetic_file) as f:
            for line in f:
                row = json.loads(line)
                if not row.get("problem"):
                    continue
                problems.append({
                    "problem": row["problem"],
                    "answer": None,
                    "difficulty": row.get("topic"),
                })

    elif dataset == "deepmath":
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
               args_dict: dict, output_path: str, counter, max_responses):
    """Run in a spawned subprocess. Sets CUDA_VISIBLE_DEVICES, creates its own
    sampler, generates for its shard, writes results to output_path.

    `counter` is a shared mp.Value('i') tracking total saved rows across all
    workers (seeded with rows already on disk). If `max_responses` is set,
    the worker exits once counter.value reaches it.
    """

    # Must set before any CUDA/torch/vLLM import
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Reconstruct args
    args = argparse.Namespace(**args_dict)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model, trust_remote_code=True)
        tokenizer.chat_template = template_tok.chat_template

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"[Worker {rank}] GPUs {gpu_ids}, {len(problems)} problems, "
          f"temp={args.temperature} top_p={args.top_p}")

    correct_count = 0
    saved_count = 0
    t0 = time.time()

    with open(output_path, "w") as f_out:
        desc = f"worker {rank}"
        pbar = tqdm(total=len(problems), desc=desc, unit="response", position=rank)
        for prob in problems:
            if max_responses is not None:
                with counter.get_lock():
                    if counter.value >= max_responses:
                        break

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt_text)

            sample_t0 = time.time()
            req_outs = llm.generate(
                [{"prompt_token_ids": input_ids}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            completion = req_outs[0].outputs[0]
            response = completion.text
            num_tokens = len(completion.token_ids)
            sample_elapsed = time.time() - sample_t0

            pred_answer = extract_boxed_answer(response)
            if not pred_answer:
                continue
            if prob.get("answer") is None:
                correct = None  # no gold — skip verification
            else:
                correct = is_equiv(pred_answer, prob["answer"])

            # Atomically claim a save slot (or bail if cap reached concurrently)
            if max_responses is not None:
                with counter.get_lock():
                    if counter.value >= max_responses:
                        break
                    counter.value += 1
                    global_saved = counter.value
            else:
                with counter.get_lock():
                    counter.value += 1
                    global_saved = counter.value

            result = {
                "problem": prob["problem"],
                "answer": prob["answer"],
                "solution": response,
                "correct": correct,
                "pred_answer": pred_answer,
                "num_tokens_generated": num_tokens,
                "sample_time_s": round(sample_elapsed, 2),
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if prob.get("difficulty") is not None:
                result["difficulty"] = prob["difficulty"]

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            saved_count += 1
            correct_count += int(correct is True)

            pbar.update(1)
            postfix = {
                "correct": ("n/a" if correct is None else f"{correct_count}/{saved_count}"),
                "tokens": num_tokens,
                "time": f"{sample_elapsed:.1f}s",
            }
            if max_responses is not None:
                postfix["global"] = f"{global_saved}/{max_responses}"
            else:
                postfix["global"] = str(global_saved)
            pbar.set_postfix(**postfix)
        pbar.close()

    elapsed = time.time() - t0
    print(f"[Worker {rank}] Done: {saved_count} saved, {correct_count} correct, {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Sequential generation (single worker, used when num_workers=1)
# ---------------------------------------------------------------------------

def _generate_sequential(problems: list[dict], args, all_path: str, num_already_done: int):
    """Generate solutions sequentially. Appends to all_path."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(args.chat_template_model, trust_remote_code=True)
        tokenizer.chat_template = template_tok.chat_template
        print(f"Using chat template from: {args.chat_template_model}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    print(f"\nvLLM sampler (temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens})\n")

    t0 = time.time()
    correct_count = 0
    saved_count = 0
    total_tokens = 0
    max_responses = args.max_responses

    with open(all_path, "a") as f_all:
        pbar = tqdm(total=len(problems), desc="power sampling", unit="response")
        for prob in problems:
            if max_responses is not None and num_already_done + saved_count >= max_responses:
                break

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(prompt_text)

            sample_t0 = time.time()
            req_outs = llm.generate(
                [{"prompt_token_ids": input_ids}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            completion = req_outs[0].outputs[0]
            response = completion.text
            num_tokens = len(completion.token_ids)
            sample_elapsed = time.time() - sample_t0

            pred_answer = extract_boxed_answer(response)
            if not pred_answer:
                continue
            if prob.get("answer") is None:
                correct = None  # no gold — skip verification
            else:
                correct = is_equiv(pred_answer, prob["answer"])

            result = {
                "problem": prob["problem"],
                "answer": prob["answer"],
                "solution": response,
                "correct": correct,
                "pred_answer": pred_answer,
                "num_tokens_generated": num_tokens,
                "sample_time_s": round(sample_elapsed, 2),
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if prob.get("difficulty") is not None:
                result["difficulty"] = prob["difficulty"]

            f_all.write(json.dumps(result) + "\n")
            f_all.flush()

            saved_count += 1
            correct_count += int(correct is True)
            total_tokens += num_tokens
            global_saved = num_already_done + saved_count

            pbar.update(1)
            if correct is None:
                postfix = {
                    "correct": "n/a",
                    "tokens": num_tokens,
                    "time": f"{sample_elapsed:.1f}s",
                }
            else:
                postfix = {
                    "correct": f"{correct_count}/{saved_count}",
                    "acc": f"{correct_count/saved_count*100:.1f}%",
                    "tokens": num_tokens,
                    "time": f"{sample_elapsed:.1f}s",
                }
            if max_responses is not None:
                postfix["global"] = f"{global_saved}/{max_responses}"
            pbar.set_postfix(**postfix)
        pbar.close()

    elapsed = time.time() - t0
    print(f"\nGeneration complete in {elapsed:.1f}s")
    if saved_count:
        print(f"  Saved: {saved_count}, Correct: {correct_count}")
        print(f"  Avg tokens: {total_tokens / saved_count:.0f}")


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
        synthetic_file=args.synthetic_file,
    )
    src = args.synthetic_file if args.dataset == "synthetic" else args.dataset
    print(f"Loaded {len(problems)} problems from {src}")

    # --- Output paths ---
    model_slug = args.model.replace("/", "_")
    if args.dataset == "synthetic":
        stem = os.path.splitext(os.path.basename(args.synthetic_file))[0]
        dataset_slug = f"synthetic_{stem}"
    else:
        dataset_slug = args.dataset
    run_slug = f"temp_{args.temperature}_topp_{args.top_p}"
    output_dir = os.path.join(args.output_dir, dataset_slug, model_slug, run_slug)
    os.makedirs(output_dir, exist_ok=True)

    all_path = os.path.join(output_dir, "demos_all.jsonl")
    correct_path = os.path.join(output_dir, "demos_correct.jsonl")

    # --- Resume: skip already-generated problems ---
    done = load_existing_results(all_path)
    remaining = [p for p in problems if p["problem"] not in done]
    print(f"Already generated: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to generate. Exiting.")
        _write_correct_subset(all_path, correct_path)
        _print_summary(all_path)
        return

    if args.max_responses is not None and len(done) >= args.max_responses:
        print(f"Already have {len(done)} >= max_responses ({args.max_responses}). Exiting.")
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
            os.path.join(output_dir, f"demos_worker_{i}.jsonl")
            for i in range(num_workers)
        ]

        args_dict = vars(args)

        # Spawn workers
        mp.set_start_method("spawn", force=True)

        # Shared counter seeded with rows already on disk; workers stop when
        # counter.value >= args.max_responses.
        saved_counter = mp.Value("i", len(done))

        processes = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker_fn,
                args=(i, gpu_assignments[i], shards[i], args_dict, worker_paths[i],
                      saved_counter, args.max_responses),
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
    has_verifier = False
    with open(all_path) as f_in, open(correct_path, "w") as f_out:
        for line in f_in:
            total += 1
            row = json.loads(line)
            if row.get("correct") is not None:
                has_verifier = True
            if row.get("correct") is True:
                f_out.write(line)
                correct += 1
    if has_verifier:
        print(f"Wrote {correct}/{total} correct demos to {correct_path}")
    else:
        print(f"No gold answers — correct-only subset skipped (train on {all_path})")


def _print_summary(all_path: str):
    """Print summary statistics from the results file."""
    results = []
    with open(all_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return

    total = len(results)
    has_verifier = any(r.get("correct") is not None for r in results)
    correct = sum(1 for r in results if r.get("correct") is True)
    tokens = [r["num_tokens_generated"] for r in results]
    times = [r.get("sample_time_s", 0) for r in results]

    print(f"\n{'='*60}")
    if has_verifier:
        print(f"Summary: {total} demos, {correct} correct ({correct/total*100:.1f}%)")
    else:
        print(f"Summary: {total} demos (no gold answers — correctness not evaluated)")
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
            difficulties[d]["correct"] += int(r.get("correct") is True)

    if difficulties:
        print(f"\n  By difficulty:")
        for d in sorted(difficulties.keys()):
            s = difficulties[d]
            if has_verifier:
                print(f"    {d}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")
            else:
                print(f"    {d}: {s['total']}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline demonstrations (plain vLLM sampling)"
    )

    # Model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="HF model to borrow chat template from")

    # Data
    parser.add_argument("--dataset", type=str, default="deepmath",
                        choices=["deepmath", "polaris", "numinamath", "synthetic"])
    parser.add_argument("--synthetic_file", type=str, default=None,
                        help="JSONL of synthetic questions (problem, optionally topic). "
                             "Required when --dataset synthetic. No gold answers expected; "
                             "correctness is not evaluated.")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Max problems to load from dataset before generation")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="NuminaMath source filter")
    parser.add_argument("--difficulty", nargs="*", default=None,
                        help="Polaris difficulty filter (e.g. 1/8 2/8)")
    parser.add_argument("--seed", type=int, default=42)

    # Sampling
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens per generation")
    parser.add_argument("--max_responses", type=int, default=None,
                        help="Stop after this many saved rows globally (across all workers, "
                             "counting any rows already on disk from a prior run).")

    # vLLM / parallelism
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="GPUs per LLM instance (tensor parallelism)")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: auto = num_gpus // tensor_parallel_size)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/demos")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
