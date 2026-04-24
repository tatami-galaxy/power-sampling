"""
Epistemic marker analysis: measure how power sampling affects uncertainty
expressions in reasoning traces.

Based on Kim et al. (2025) "Why Does Self-Distillation (Sometimes) Degrade
the Reasoning Capability of LLMs?" which identifies 10 epistemic markers
that signal self-correction in reasoning:
    T = {wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}

Modes:
  1. Analyse existing results (no GPU needed)
  2. Generate fresh results across multiple alpha values, then analyse
  3. Generate teacher-conditioned results (SDFT self-teacher) alongside
     base / power results to compare epistemic suppression

Usage:
    # Analyse existing base vs power results
    uv run python -m scripts.epistemic_analysis \
        --base_results results/math500/Qwen_Qwen3-4B/Qwen_Qwen3-4B_results.json \
        --power_results results/math500/Qwen_Qwen3-4B/vllm_batched_power_sampling/Qwen_Qwen3-4B_results.json

    # Analyse multiple alpha sweeps from a directory
    uv run python -m scripts.epistemic_analysis \
        --results_dir results/math500/Qwen_Qwen3-4B

    # Generate + analyse across alpha values
    uv run python -m scripts.epistemic_analysis \
        --model Qwen/Qwen3-4B \
        --dataset math500 --num_samples 50 \
        --alphas 1.0 2.0 4.0 6.0 8.0 \
        --use_vllm

    # Include SDFT teacher-conditioned generations
    uv run python -m scripts.epistemic_analysis \
        --model Qwen/Qwen3-4B \
        --dataset math500 --num_samples 50 \
        --alphas 1.0 2.0 4.0 6.0 8.0 \
        --include_teacher --use_vllm

    # Teacher-conditioned only (no power sweep)
    uv run python -m scripts.epistemic_analysis \
        --model Qwen/Qwen3-4B \
        --dataset math500 --num_samples 50 \
        --alphas 1.0 --include_teacher --use_vllm
"""

import argparse
import json
import os
import re
import statistics
from glob import glob

# ---------------------------------------------------------------------------
# Epistemic markers (Kim et al., 2025)
# ---------------------------------------------------------------------------

EPISTEMIC_MARKERS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]

# Compile case-insensitive whole-word patterns.
# \b handles word boundaries so "actually" doesn't match inside another word.
MARKER_PATTERNS = {
    marker: re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE)
    for marker in EPISTEMIC_MARKERS
}


def count_markers(text: str) -> dict[str, int]:
    """Count occurrences of each epistemic marker in *text*."""
    return {
        marker: len(pattern.findall(text))
        for marker, pattern in MARKER_PATTERNS.items()
    }


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Per-result-set analysis
# ---------------------------------------------------------------------------

def analyse_results(results: list[dict], label: str = "") -> dict:
    """Compute epistemic marker statistics for a list of result dicts.

    Each result must have at least a ``response`` key (the reasoning trace).
    Optional keys: ``correct``, ``level``, ``subject``.
    """
    per_response = []
    for r in results:
        text = r.get("response", "")
        counts = count_markers(text)
        wc = word_count(text)
        total = sum(counts.values())
        per_response.append({
            "counts": counts,
            "total_markers": total,
            "word_count": wc,
            "density_per_1k": (total / wc * 1000) if wc > 0 else 0.0,
            "correct": r.get("correct"),
            "level": r.get("level"),
            "subject": r.get("subject"),
        })

    n = len(per_response)
    if n == 0:
        return {"label": label, "n": 0}

    # Aggregate marker counts
    marker_totals = {m: 0 for m in EPISTEMIC_MARKERS}
    for pr in per_response:
        for m in EPISTEMIC_MARKERS:
            marker_totals[m] += pr["counts"][m]

    total_words = sum(pr["word_count"] for pr in per_response)
    total_markers = sum(pr["total_markers"] for pr in per_response)
    densities = [pr["density_per_1k"] for pr in per_response]

    # Per-marker density (per 1k words across corpus)
    marker_density = {
        m: (marker_totals[m] / total_words * 1000) if total_words > 0 else 0.0
        for m in EPISTEMIC_MARKERS
    }

    # Density by correctness
    density_by_correct = {}
    for correct_val in [True, False]:
        subset = [pr for pr in per_response if pr["correct"] == correct_val]
        if subset:
            density_by_correct[str(correct_val)] = {
                "n": len(subset),
                "mean_density": statistics.mean(pr["density_per_1k"] for pr in subset),
            }

    # Density by difficulty level
    density_by_level = {}
    levels = sorted(set(pr["level"] for pr in per_response if pr["level"] is not None))
    for level in levels:
        subset = [pr for pr in per_response if pr["level"] == level]
        if subset:
            density_by_level[str(level)] = {
                "n": len(subset),
                "mean_density": statistics.mean(pr["density_per_1k"] for pr in subset),
            }

    return {
        "label": label,
        "n": n,
        "total_words": total_words,
        "mean_word_count": total_words / n,
        "total_markers": total_markers,
        "mean_density_per_1k": statistics.mean(densities),
        "median_density_per_1k": statistics.median(densities),
        "std_density_per_1k": statistics.stdev(densities) if n > 1 else 0.0,
        "marker_totals": marker_totals,
        "marker_density_per_1k": marker_density,
        "density_by_correct": density_by_correct,
        "density_by_level": density_by_level,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_analysis(analysis: dict):
    """Pretty-print one analysis block."""
    label = analysis["label"]
    n = analysis["n"]
    if n == 0:
        print(f"\n[{label}] — no results")
        return

    print(f"\n{'='*65}")
    print(f"  {label}  (n={n})")
    print(f"{'='*65}")
    print(f"  Mean response length : {analysis['mean_word_count']:.0f} words")
    print(f"  Total epistemic markers : {analysis['total_markers']}")
    print(f"  Density (per 1k words) : {analysis['mean_density_per_1k']:.2f} "
          f"(median {analysis['median_density_per_1k']:.2f}, "
          f"std {analysis['std_density_per_1k']:.2f})")

    # Per-marker breakdown
    print(f"\n  {'Marker':<16} {'Count':>6} {'Density':>10}")
    print(f"  {'-'*34}")
    for m in EPISTEMIC_MARKERS:
        cnt = analysis["marker_totals"][m]
        d = analysis["marker_density_per_1k"][m]
        print(f"  {m:<16} {cnt:>6} {d:>9.2f}")

    # By correctness
    if analysis["density_by_correct"]:
        print(f"\n  Density by correctness:")
        for k, v in analysis["density_by_correct"].items():
            print(f"    correct={k:<6} n={v['n']:<4}  density={v['mean_density']:.2f}")

    # By level
    if analysis["density_by_level"]:
        print(f"\n  Density by difficulty level:")
        for k, v in sorted(analysis["density_by_level"].items(), key=lambda x: x[0]):
            print(f"    Level {k:<3}  n={v['n']:<4}  density={v['mean_density']:.2f}")


def print_comparison_table(analyses: list[dict]):
    """Print a compact comparison across multiple conditions (e.g. alpha sweep)."""
    if len(analyses) < 2:
        return

    print(f"\n{'='*75}")
    print("  COMPARISON TABLE")
    print(f"{'='*75}")

    # Header
    labels = [a["label"] for a in analyses]
    col_w = max(12, max(len(l) for l in labels) + 2)
    header = f"  {'Marker':<16}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"  {'-'*(16 + col_w * len(labels))}")

    # Per-marker rows
    for m in EPISTEMIC_MARKERS:
        row = f"  {m:<16}"
        for a in analyses:
            d = a["marker_density_per_1k"].get(m, 0.0)
            row += f"{d:>{col_w}.2f}"
        print(row)

    # Total density row
    row = f"  {'TOTAL':<16}"
    for a in analyses:
        row += f"{a['mean_density_per_1k']:>{col_w}.2f}"
    print(f"  {'-'*(16 + col_w * len(labels))}")
    print(row)

    # Response length row
    row = f"  {'Avg words':<16}"
    for a in analyses:
        row += f"{a['mean_word_count']:>{col_w}.0f}"
    print(row)

    # Accuracy row (if available)
    has_acc = any(a["density_by_correct"] for a in analyses)
    if has_acc:
        row = f"  {'Accuracy':<16}"
        for a in analyses:
            correct_n = a["density_by_correct"].get("True", {}).get("n", 0)
            acc = correct_n / a["n"] * 100 if a["n"] > 0 else 0
            row += f"{acc:>{col_w - 1}.1f}%"
        print(row)


# ---------------------------------------------------------------------------
# Loading existing results
# ---------------------------------------------------------------------------

def load_results_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def infer_label_from_path(path: str) -> str:
    """Guess a short label from the result file path."""
    parts = path.replace("\\", "/").split("/")
    # Look for an alpha mention in parent dirs or filename
    for p in reversed(parts):
        if "alpha" in p.lower():
            return p
    # Check for method dir
    if len(parts) >= 2:
        parent = parts[-2]
        if parent in ("vllm_batched_power_sampling", "batched_power_sampling", "power_sampling"):
            return parent.replace("_", " ")
    # Fallback: filename stem
    return os.path.splitext(parts[-1])[0]


def discover_results_in_dir(results_dir: str) -> list[tuple[str, str]]:
    """Find all *_results.json in a directory tree. Returns (path, label) pairs."""
    pattern = os.path.join(results_dir, "**", "*_results.json")
    paths = sorted(glob(pattern, recursive=True))
    out = []
    for p in paths:
        rel = os.path.relpath(p, results_dir)
        parts = rel.replace("\\", "/").split("/")
        if len(parts) == 1:
            label = "base"
        else:
            label = "/".join(parts[:-1])
        out.append((p, label))
    return out


# ---------------------------------------------------------------------------
# Generation mode
# ---------------------------------------------------------------------------

def generate_for_alpha_sweep(
    model_name: str,
    dataset: str,
    alphas: list[float],
    num_samples: int | None,
    seed: int,
    use_vllm: bool,
    top_k: int,
    num_rollouts: int,
    lookahead: int,
    batch_size: int,
    num_candidates: int,
    max_tokens: int,
    tensor_parallel_size: int,
    max_model_len: int,
    chat_template_model: str | None,
    output_dir: str,
    levels: list[int] | None,
    enable_thinking: bool | None = None,
) -> list[tuple[str, str]]:
    """Generate results for base + each alpha value. Returns (path, label) pairs."""
    from src.eval.run_eval import (
        evaluate_model,
        evaluate_model_power_sampling,
        save_results,
    )
    from src.utils import DATASET_REGISTRY_EVAL
    import random

    loader = DATASET_REGISTRY_EVAL[dataset]
    problems = loader(levels=levels)
    if num_samples is not None and num_samples < len(problems):
        random.seed(seed)
        problems = random.sample(problems, num_samples)
    print(f"Loaded {len(problems)} problems from {dataset}")

    chat_template_tokenizer = None
    if chat_template_model:
        from transformers import AutoTokenizer
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True,
        )

    model_slug = model_name.replace("/", "_")
    base_dir = os.path.join(output_dir, dataset, model_slug)

    result_files = []

    # --- Base (alpha=1.0, standard sampling) ---
    base_path = os.path.join(base_dir, f"{model_slug}_results.json")
    if os.path.exists(base_path):
        print(f"Base results already exist: {base_path}")
    else:
        eval_out = evaluate_model(
            model_name=model_name,
            problems=problems,
            max_tokens=max_tokens,
            temperature=1.0,  # sample, not greedy, for fair comparison
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len or None,
            chat_template_tokenizer=chat_template_tokenizer,
            enable_thinking=enable_thinking,
        )
        save_results(eval_out, base_dir)
    result_files.append((base_path, "base (α=1)"))

    # --- Power sampling at each alpha ---
    for alpha in alphas:
        if alpha == 1.0:
            continue  # base already covers this
        alpha_label = f"α={alpha}"
        alpha_dir = os.path.join(base_dir, f"power_alpha_{alpha}")
        results_path = os.path.join(alpha_dir, f"{model_slug}_results.json")

        if os.path.exists(results_path):
            print(f"Results exist for {alpha_label}: {results_path}")
        else:
            ps_out = evaluate_model_power_sampling(
                model_name=model_name,
                problems=problems,
                max_tokens=max_tokens,
                alpha=alpha,
                top_k=top_k,
                num_rollouts=num_rollouts,
                lookahead=lookahead,
                batched=not use_vllm,
                batch_size=batch_size,
                num_candidates=num_candidates,
                use_vllm=use_vllm,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len or None,
                chat_template_tokenizer=chat_template_tokenizer,
                enable_thinking=enable_thinking,
            )
            os.makedirs(alpha_dir, exist_ok=True)
            save_results(ps_out, alpha_dir)

        result_files.append((results_path, alpha_label))

    return result_files


# ---------------------------------------------------------------------------
# Teacher-conditioned generation (SDFT self-teacher)
# ---------------------------------------------------------------------------

def generate_teacher_conditioned(
    model_name: str,
    dataset: str,
    num_samples: int | None,
    seed: int,
    max_tokens: int,
    tensor_parallel_size: int,
    max_model_len: int,
    chat_template_model: str | None,
    output_dir: str,
    levels: list[int] | None,
    teacher_template: int,
    problems: list[dict] | None = None,
    enable_thinking: bool | None = None,
) -> tuple[str, str]:
    """Generate responses under the SDFT teacher context (question + solution).

    Uses the same model but with richer input context — the teacher template
    from train_sdft.py that includes the expert demonstration.  This lets us
    measure how much epistemic verbalization the teacher suppresses compared
    to the base (student) and power-sampled conditions.

    Returns (results_path, label).
    """
    from src.eval.run_eval import save_results
    from src.train_sdft import (
        SYSTEM_PROMPT as SDFT_SYSTEM_PROMPT,
        TEACHER_TEMPLATE_1,
        TEACHER_TEMPLATE_2,
    )
    from src.utils import DATASET_REGISTRY_EVAL, extract_boxed_answer, is_equiv
    import random
    import time

    template = TEACHER_TEMPLATE_1 if teacher_template == 1 else TEACHER_TEMPLATE_2
    template_label = f"teacher_t{teacher_template}"

    # ── Load problems ───────────────────────────────────────────────────────
    if problems is None:
        loader = DATASET_REGISTRY_EVAL[dataset]
        problems = loader(levels=levels)
        if num_samples is not None and num_samples < len(problems):
            random.seed(seed)
            problems = random.sample(problems, num_samples)

    # Validate that problems have solutions (needed for teacher context)
    missing = [i for i, p in enumerate(problems) if not p.get("solution")]
    if missing:
        raise ValueError(
            f"Teacher mode requires problems with 'solution' field. "
            f"{len(missing)}/{len(problems)} problems are missing solutions."
        )

    model_slug = model_name.replace("/", "_")
    base_dir = os.path.join(output_dir, dataset, model_slug)
    teacher_dir = os.path.join(base_dir, template_label)
    results_path = os.path.join(teacher_dir, f"{model_slug}_results.json")

    if os.path.exists(results_path):
        print(f"Teacher results already exist: {results_path}")
        return results_path, template_label

    # ── Build teacher-conditioned prompts ────────────────────────────────────
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if chat_template_model:
        template_tok = AutoTokenizer.from_pretrained(
            chat_template_model, trust_remote_code=True
        )
    else:
        template_tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    prompts = []
    for p in problems:
        solution = p["solution"].strip()
        if "\\boxed" not in solution:
            solution += f"\n\nThe answer is $\\boxed{{{p['answer']}}}$."

        teacher_messages = [
            {"role": "system", "content": SDFT_SYSTEM_PROMPT},
            {"role": "user", "content": template.format(
                question=p["problem"], demonstration=solution,
            )},
        ]
        text = template_tok.apply_chat_template(teacher_messages, **template_kwargs)
        prompts.append(text)

    # ── Generate ─────────────────────────────────────────────────────────────
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=max_tokens)

    print(f"\nGenerating teacher-conditioned responses ({template_label})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Teacher generation took {elapsed:.1f}s ({len(problems)/elapsed:.1f} problems/s)")

    # ── Score ─────────────────────────────────────────────────────────────────
    results = []
    for prob, output in zip(problems, outputs):
        response = output.outputs[0].text
        pred_answer = extract_boxed_answer(response)
        correct = is_equiv(pred_answer, prob["answer"]) if pred_answer else False
        results.append({
            **prob,
            "response": response,
            "pred_answer": pred_answer,
            "correct": correct,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(teacher_dir, exist_ok=True)
    save_results(
        {
            "model": model_name,
            "results": results,
            "elapsed_s": elapsed,
            "max_tokens": max_tokens,
        },
        teacher_dir,
    )
    print(f"Saved teacher results: {results_path}")

    # Clean up GPU memory before returning
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results_path, template_label


# ---------------------------------------------------------------------------
# Save analysis
# ---------------------------------------------------------------------------

def save_analysis(analyses: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Make JSON-serialisable (convert any non-standard types)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(output_path, "w") as f:
        json.dump(_clean(analyses), f, indent=2)
    print(f"\nSaved analysis: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse epistemic markers in base vs power sampling traces",
    )

    # --- Analysis from existing files ---
    parser.add_argument(
        "--base_results", type=str, default=None,
        help="Path to base model *_results.json",
    )
    parser.add_argument(
        "--power_results", nargs="*", default=None,
        help="Path(s) to power sampling *_results.json",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory to search recursively for *_results.json files",
    )

    # --- Generation mode ---
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--dataset", default="math500",
        help="Dataset name (math500, minerva_math, aime24, aime25, aime26, aime_2025)",
    )
    parser.add_argument("--levels", nargs="*", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[2.0, 4.0, 6.0, 8.0],
        help="Alpha values to sweep",
    )

    # SDFT teacher-conditioned generation
    parser.add_argument(
        "--include_teacher", action="store_true",
        help="Also generate under SDFT teacher context (question + expert solution)",
    )
    parser.add_argument(
        "--teacher_template", type=int, default=2, choices=[1, 2],
        help="Which teacher template to use (1 or 2, default: 2 matches active template in train_sdft.py)",
    )

    # Power sampling hyperparams
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_candidates", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--chat_template_model", type=str, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="For models with a toggleable thinking mode (e.g. Qwen3): "
                             "pass --enable-thinking or --no-enable-thinking to override "
                             "the template default. Leave unset to use the model default.")

    # Output
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--save_analysis_to", type=str, default=None,
        help="Path to save analysis JSON (auto-generated if not set)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Collect result files to analyse
    # -----------------------------------------------------------------
    result_files: list[tuple[str, str]] = []  # (path, label)

    if args.model:
        # Generation mode: run alpha sweep, then analyse
        result_files = generate_for_alpha_sweep(
            model_name=args.model,
            dataset=args.dataset,
            alphas=args.alphas,
            num_samples=args.num_samples,
            seed=args.seed,
            use_vllm=args.use_vllm,
            top_k=args.top_k,
            num_rollouts=args.num_rollouts,
            lookahead=args.lookahead,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            chat_template_model=args.chat_template_model,
            output_dir=args.output_dir,
            levels=args.levels,
            enable_thinking=args.enable_thinking,
        )

        # Teacher-conditioned generation (SDFT self-teacher)
        if args.include_teacher:
            # Load the same problem subset used by the alpha sweep
            from src.utils import DATASET_REGISTRY_EVAL
            import random as _rng

            loader = DATASET_REGISTRY_EVAL[args.dataset]
            problems = loader(levels=args.levels)
            if args.num_samples is not None and args.num_samples < len(problems):
                _rng.seed(args.seed)
                problems = _rng.sample(problems, args.num_samples)

            teacher_path, teacher_label = generate_teacher_conditioned(
                model_name=args.model,
                dataset=args.dataset,
                num_samples=args.num_samples,
                seed=args.seed,
                max_tokens=args.max_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                chat_template_model=args.chat_template_model,
                output_dir=args.output_dir,
                levels=args.levels,
                teacher_template=args.teacher_template,
                problems=problems,
                enable_thinking=args.enable_thinking,
            )
            result_files.append((teacher_path, teacher_label))
    elif args.results_dir:
        result_files = discover_results_in_dir(args.results_dir)
    else:
        # Explicit file paths
        if args.base_results:
            result_files.append((args.base_results, "base"))
        if args.power_results:
            for p in args.power_results:
                result_files.append((p, infer_label_from_path(p)))

    if not result_files:
        parser.error("Provide --model (generate mode), --results_dir, or --base_results/--power_results")

    # -----------------------------------------------------------------
    # Analyse
    # -----------------------------------------------------------------
    analyses = []
    for path, label in result_files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        results = load_results_json(path)
        analysis = analyse_results(results, label=label)
        analyses.append(analysis)

    # Print individual reports
    for a in analyses:
        print_analysis(a)

    # Print comparison table
    print_comparison_table(analyses)

    # Save
    save_path = args.save_analysis_to
    if save_path is None and args.model:
        model_slug = args.model.replace("/", "_")
        save_path = os.path.join(
            args.output_dir, args.dataset, model_slug,
            "epistemic_analysis.json",
        )
    if save_path is None and args.results_dir:
        save_path = os.path.join(args.results_dir, "epistemic_analysis.json")
    if save_path:
        save_analysis(analyses, save_path)


if __name__ == "__main__":
    main()
