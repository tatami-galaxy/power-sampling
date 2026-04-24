"""
Generate synthetic math questions from a curated topic list, filtered by
self-critique from the same model.

Pipeline:
  1. For each topic, sample `per_topic` candidate problems (temp=1.0, top_p=0.95)
  2. Ask the same model whether each problem is well-posed (greedy)
  3. Keep those whose critique starts with "Yes"

Output files:
  {output_dir}/{model_slug}/questions_all.jsonl       — all generations with verdict
  {output_dir}/{model_slug}/questions_accepted.jsonl  — accepted subset

Each line has: topic, problem, critique, accepted.

Usage:
    uv run python -m scripts.rl.generate_synthetic_questions \\
        --model Qwen/Qwen3-4B --per_topic 10
"""

import argparse
import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


TOPICS = [
    "advanced algebra",
    "number theory",
    "combinatorics",
    "probability theory",
    "geometry",
    "advanced calculus",
    "differential equations",
    "advanced linear algebra",
]


GENERATION_PROMPT = """You are an expert competition math problem writer. Write one original math problem on the topic of: {topic}.

Requirements:
- Has a single, unambiguous numerical answer (a number, fraction, or closed-form expression)
- State the problem clearly and completely in a single self-contained paragraph
- Difficulty: roughly AMC/AIME/early-olympiad level
- Do NOT include the solution, hints, or the final answer

Output only the problem statement, nothing else."""


CRITIQUE_PROMPT = """Problem:
{problem}

Is this problem well-posed, unambiguous, and does it have a unique answer? Answer with "Yes" or "No" on the first line, followed by a one-sentence reason."""


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_verdict(text: str) -> bool:
    """True iff the first non-empty line begins with 'yes' (case-insensitive)."""
    cleaned = _strip_thinking(text)
    if not cleaned:
        return False
    first_line = cleaned.splitlines()[0].strip()
    first_line = re.sub(r"^[^\w]+", "", first_line)  # drop leading markdown / punctuation
    return first_line.lower().startswith("yes")


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace for duplicate detection."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--per_topic", type=int, default=10,
                        help="Candidate problems per topic (total = per_topic * len(TOPICS))")
    parser.add_argument("--max_tokens_generate", type=int, default=512)
    parser.add_argument("--max_tokens_critique", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Allow Qwen3 thinking mode (default: off — faster)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="results/synthetic_questions")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_")
    out_dir = os.path.join(args.output_dir, model_slug)
    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "questions_all.jsonl")
    accepted_path = os.path.join(out_dir, "questions_accepted.jsonl")
    config_path = os.path.join(out_dir, "generate_config.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def _chat(user_msg: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=args.seed,
    )

    # ── 1. Generate candidate problems (per-sample seeds, one request each) ──
    total = len(TOPICS) * args.per_topic
    gen_prompts: list[str] = []
    gen_topics_flat: list[str] = []
    gen_params_list: list[SamplingParams] = []
    for topic_idx, topic in enumerate(TOPICS):
        for k in range(args.per_topic):
            gen_prompts.append(_chat(GENERATION_PROMPT.format(topic=topic)))
            gen_topics_flat.append(topic)
            gen_params_list.append(SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_generate,
                seed=args.seed + topic_idx * args.per_topic + k,
            ))
    print(f"Generating {total} problems over {len(TOPICS)} topics ({args.per_topic} each, per-sample seeds)...")

    gen_outputs = llm.generate(gen_prompts, gen_params_list)

    problems: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    empty = 0
    for topic, req_out in zip(gen_topics_flat, gen_outputs):
        text = _strip_thinking(req_out.outputs[0].text)
        if not text:
            empty += 1
            continue
        key = _normalize(text)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        problems.append({"topic": topic, "problem": text})

    print(f"Got {len(problems)} unique candidates "
          f"({duplicates} duplicates, {empty} empty dropped)")

    # ── 2. Self-critique ──────────────────────────────────────────────────────
    crit_prompts = [_chat(CRITIQUE_PROMPT.format(problem=p["problem"])) for p in problems]
    print(f"Critiquing {len(crit_prompts)} problems (greedy)...")

    crit_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens_critique,
    )
    crit_outputs = llm.generate(crit_prompts, crit_params)

    # ── 3. Merge, parse, write ───────────────────────────────────────────────
    accepted = 0
    with open(all_path, "w") as f_all, open(accepted_path, "w") as f_ok:
        for p, co in zip(problems, crit_outputs):
            critique_raw = co.outputs[0].text
            critique = _strip_thinking(critique_raw)
            verdict = _parse_verdict(critique_raw)
            row = {
                "topic": p["topic"],
                "problem": p["problem"],
                "critique": critique,
                "accepted": verdict,
            }
            f_all.write(json.dumps(row) + "\n")
            if verdict:
                f_ok.write(json.dumps(row) + "\n")
                accepted += 1

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generated: {len(problems)}")
    print(f"Accepted:  {accepted} ({100 * accepted / max(len(problems), 1):.1f}%)")
    print(f"  {all_path}")
    print(f"  {accepted_path}")

    by_topic: dict[str, dict[str, int]] = {}
    for p, co in zip(problems, crit_outputs):
        verdict = _parse_verdict(co.outputs[0].text)
        s = by_topic.setdefault(p["topic"], {"total": 0, "accepted": 0})
        s["total"] += 1
        s["accepted"] += int(verdict)
    print("\nPer-topic acceptance:")
    for t in TOPICS:
        s = by_topic.get(t, {"total": 0, "accepted": 0})
        print(f"  {s['accepted']:2d}/{s['total']:2d}  {t}")
    print(f"{'='*60}")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()
