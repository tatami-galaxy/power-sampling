"""
Evaluate base vs. power-sampled generation on SciKnowEval.

Supports the Chemistry L-3 subset (default) and other domain/level combinations.
Handles both MCQ (multiple-choice) and filling (free-text) question types.

Usage:
    # Chemistry L3 (default, matches SDFT paper)
    uv run python -m src.eval.run_sciknoweval \
        --model Qwen/Qwen2.5-7B-Instruct

    # Base + power sampling
    uv run python -m src.eval.run_sciknoweval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --power_sampling

    # Different domain/level
    uv run python -m src.eval.run_sciknoweval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --domain Physics --level L3

    # Only MCQ questions
    uv run python -m src.eval.run_sciknoweval \
        --model Qwen/Qwen2.5-7B-Instruct \
        --question_type mcq
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.utils import extract_boxed_answer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

MCQ_SYSTEM_PROMPT = (
    "You are a helpful science assistant. For the following multiple choice question, "
    "reason step by step and put your final answer letter in \\boxed{}."
)

FILLING_SYSTEM_PROMPT = (
    "You are a helpful science assistant. Solve the following problem step by step. "
    "Put your final answer in \\boxed{}."
)


def format_mcq_prompt(prob: dict) -> list[dict]:
    labels = prob["choices_label"]
    texts = prob["choices_text"]
    choice_block = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    user = f"{prob['question']}\n\n{choice_block}"
    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_filling_prompt(prob: dict) -> list[dict]:
    return [
        {"role": "system", "content": FILLING_SYSTEM_PROMPT},
        {"role": "user", "content": prob["question"]},
    ]


def format_messages(prob: dict) -> list[dict]:
    if prob["type"] == "mcq":
        return format_mcq_prompt(prob)
    return format_filling_prompt(prob)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sciknoweval(
    domain: str = "Chemistry",
    level: str = "L3",
    question_type: str | None = None,
) -> list[dict]:
    ds = load_dataset("hicai-zju/SciKnowEval", "v2", split="test")

    problems = []
    for row in ds:
        if row["domain"] != domain:
            continue
        if row["details"]["level"] != level:
            continue

        is_mcq = row["type"].startswith("mcq") or row["type"] == "true_false"
        q_type = "mcq" if is_mcq else "filling"

        if question_type and q_type != question_type:
            continue

        prob = {
            "question": row["question"],
            "type": q_type,
            "subtask": row["details"]["subtask"],
            "source": row["details"]["source"],
        }

        if is_mcq:
            prob["choices_label"] = row["choices"]["label"]
            prob["choices_text"] = row["choices"]["text"]
            prob["correct_label"] = row["answerKey"]
            prob["answer"] = None
        else:
            prob["choices_label"] = []
            prob["choices_text"] = []
            prob["correct_label"] = None
            prob["answer"] = row["answer"]

        problems.append(prob)

    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def destroy_vllm_model(llm: LLM):
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
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    chat_template_model: str | None = None,
    enable_thinking: bool | None = None,
) -> list[str]:
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

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompts = []
    for p in problems:
        messages = format_messages(p)
        text = template_tok.apply_chat_template(messages, **template_kwargs)
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
    lookahead: int = 192,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    chat_template_model: str | None = None,
    enable_thinking: bool | None = None,
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
    )
    tokenizer = sampler.tokenizer

    template_tok = tokenizer
    if chat_template_model:
        from transformers import AutoTokenizer
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    responses = []
    t0 = time.time()
    pbar = tqdm(problems, desc="power_sampling", unit="problem")
    for prob in pbar:
        messages = format_messages(prob)
        text = template_tok.apply_chat_template(messages, **template_kwargs)
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
# Scoring
# ---------------------------------------------------------------------------

def _normalize_chem(s: str) -> str:
    """Light normalization for chemical equation comparison."""
    s = s.strip()
    # Remove whitespace around operators
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\s*->\s*", " = ", s)
    s = re.sub(r"\s*→\s*", " = ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer_label(response: str, labels: list[str]) -> str | None:
    """Extract the answer letter from a response.

    Tries \\boxed{} first, then falls back to the last standalone label."""
    boxed = extract_boxed_answer(response)
    if boxed and boxed.strip().upper() in labels:
        return boxed.strip().upper()
    label_pattern = "|".join(re.escape(l) for l in labels)
    matches = re.findall(rf"\b({label_pattern})\b", response)
    if matches:
        return matches[-1].upper()
    return None


def score_responses(
    problems: list[dict],
    responses: list[str],
) -> list[dict]:
    out = []
    for prob, response in zip(problems, responses):
        if prob["type"] == "mcq":
            pred = extract_answer_label(response, prob["choices_label"])
            correct = pred == prob["correct_label"] if pred else False
            out.append({
                "question": prob["question"][:200],
                "type": prob["type"],
                "subtask": prob["subtask"],
                "correct_label": prob["correct_label"],
                "pred_label": pred,
                "correct": correct,
                "response": response,
            })
        else:
            # Filling: extract from \boxed{} and compare
            pred = extract_boxed_answer(response)
            if pred:
                correct = _normalize_chem(pred) == _normalize_chem(prob["answer"])
            else:
                # Fallback: check if the gold answer appears in the response
                correct = _normalize_chem(prob["answer"]) in _normalize_chem(response)
            out.append({
                "question": prob["question"][:200],
                "type": prob["type"],
                "subtask": prob["subtask"],
                "correct_label": prob["answer"][:100] if prob["answer"] else None,
                "pred_label": pred[:100] if pred else None,
                "correct": correct,
                "response": response,
            })
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(tag: str, scored: list[dict]):
    total = len(scored)
    correct = sum(r["correct"] for r in scored)
    no_answer = sum(1 for r in scored if r["pred_label"] is None)

    print(f"\n{'='*60}")
    print(f"{tag}: {correct}/{total} = {correct/total*100:.1f}%")
    if no_answer:
        print(f"  Extraction failures: {no_answer}/{total}")
    print(f"{'='*60}")

    # By type
    for qtype in ["mcq", "filling"]:
        typed = [r for r in scored if r["type"] == qtype]
        if typed:
            tc = sum(r["correct"] for r in typed)
            print(f"  {qtype}: {tc}/{len(typed)} = {tc/len(typed)*100:.1f}%")

    # By subtask
    by_sub = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in scored:
        by_sub[r["subtask"]]["total"] += 1
        by_sub[r["subtask"]]["correct"] += int(r["correct"])

    if len(by_sub) > 1:
        print(f"\n{'Subtask':<40} {'Correct':>8} {'Total':>6} {'Acc':>8}")
        print("-" * 65)
        for sub in sorted(by_sub):
            d = by_sub[sub]
            acc = d["correct"] / d["total"] * 100 if d["total"] else 0
            print(f"{sub:<40} {d['correct']:>8} {d['total']:>6} {acc:>7.1f}%")


def print_comparison(base_scored: list[dict], power_scored: list[dict]):
    base = {r["question"]: r["correct"] for r in base_scored}
    power = {r["question"]: r["correct"] for r in power_scored}
    shared = set(base) & set(power)

    both_right = sum(1 for q in shared if base[q] and power[q])
    both_wrong = sum(1 for q in shared if not base[q] and not power[q])
    base_wrong_power_right = sum(1 for q in shared if not base[q] and power[q])
    base_right_power_wrong = sum(1 for q in shared if base[q] and not power[q])

    print(f"\n{'='*60}")
    print(f"Baseline vs Power ({len(shared)} shared problems)")
    print(f"{'='*60}")
    print(f"  Both correct:                 {both_right}")
    print(f"  Both wrong:                   {both_wrong}")
    print(f"  Baseline wrong -> Power right: {base_wrong_power_right}")
    print(f"  Baseline right -> Power wrong: {base_right_power_wrong}")
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
        description="Evaluate base vs. power sampling on SciKnowEval"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--chat_template_model", default=None)
    parser.add_argument("--output_dir", default="results/sciknoweval")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate on a random subset (for quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None)

    # Dataset filters
    parser.add_argument("--domain", default="Chemistry",
                        help="Domain filter (default: Chemistry)")
    parser.add_argument("--level", default="L3",
                        help="Level filter (default: L3)")
    parser.add_argument("--question_type", default=None, choices=["mcq", "filling"],
                        help="Filter to a specific question type (default: all)")

    # Power sampling
    parser.add_argument("--power_sampling", action="store_true")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)

    args = parser.parse_args()

    # --- Load data ---
    problems = load_sciknoweval(
        domain=args.domain, level=args.level, question_type=args.question_type,
    )
    subset_tag = f"{args.domain}_{args.level}"
    if args.question_type:
        subset_tag += f"_{args.question_type}"
    print(f"Loaded {len(problems)} SciKnowEval problems ({subset_tag})")

    mcq_count = sum(1 for p in problems if p["type"] == "mcq")
    fill_count = len(problems) - mcq_count
    print(f"  MCQ: {mcq_count}, Filling: {fill_count}")

    if not problems:
        print("No problems found. Check --domain and --level filters.")
        return

    if args.num_samples is not None and args.num_samples < len(problems):
        random.seed(args.seed)
        problems = random.sample(problems, args.num_samples)
        print(f"Subsampled to {args.num_samples} problems (seed={args.seed})")

    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, subset_tag, model_slug)

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
        enable_thinking=args.enable_thinking,
    )
    base_scored = score_responses(problems, base_responses)
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
            chat_template_model=args.chat_template_model,
            enable_thinking=args.enable_thinking,
        )
        power_scored = score_responses(problems, power_responses)
        print_report("Power", power_scored)
        save_results(output_dir, "power", power_scored)
        print_comparison(base_scored, power_scored)

    # --- Summary ---
    summary = {
        "model": args.model,
        "subset": subset_tag,
        "num_problems": len(problems),
        "num_mcq": mcq_count,
        "num_filling": fill_count,
        "base": {
            "temperature": args.temperature,
            "accuracy": sum(r["correct"] for r in base_scored) / len(base_scored),
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
            "accuracy": sum(r["correct"] for r in power_scored) / len(power_scored),
        }

    summary_path = os.path.join(output_dir, "sciknoweval_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
