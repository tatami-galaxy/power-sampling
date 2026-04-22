from collections import defaultdict
import re
from datasets import load_dataset
from math_verify import parse, verify
from math_verify.parser import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)

# ---------------------------------------------------------------------------
# Answer extraction and equivalence checking
# ---------------------------------------------------------------------------

PRED_EXTRACTION_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
]
GOLD_EXTRACTION_CONFIG = [
    LatexExtractionConfig(),
    ExprExtractionConfig(),
]


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _normalize(s: str) -> str:
    """Normalize a math answer string for string comparison."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove \text{} wrapper
    m = re.fullmatch(r"\\text\{(.+)\}", s)
    if m:
        s = m.group(1).strip()
    # Remove display commands
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = s.rstrip(".")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number (int, float, or simple fraction)."""
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        pass
    # a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    # \frac{a}{b}
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    # -\frac{a}{b}
    m = re.fullmatch(r"-\\frac\{(\d+)\}\{(\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return -int(m.group(1)) / int(m.group(2))
    return None


def is_equiv(pred: str, gold: str) -> bool:
    """Check equivalence using layered strategies:
    1. Normalized string match (fast, handles most cases)
    2. Numeric comparison (fractions, decimals)
    3. math_verify symbolic comparison (fallback for complex expressions)
    """
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    # 1. Exact string match after normalization
    if pred_n == gold_n:
        return True
    # 2. Numeric comparison
    pred_v = _try_parse_number(pred_n)
    gold_v = _try_parse_number(gold_n)
    if pred_v is not None and gold_v is not None:
        return abs(pred_v - gold_v) < 1e-6
    # 3. Symbolic comparison via math_verify
    try:
        gold_parsed = parse(gold, extraction_config=GOLD_EXTRACTION_CONFIG)
        pred_parsed = parse(pred, extraction_config=PRED_EXTRACTION_CONFIG)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataset loaders – each returns list[dict] with keys:
#   problem, answer, level (int), subject, unique_id (optional)
# ---------------------------------------------------------------------------

DATASET_REGISTRY_EVAL: dict[str, callable] = {}
DATASET_REGISTRY_TRAIN: dict[str, callable] = {}


def register_dataset_eval(name):
    def wrapper(fn):
        DATASET_REGISTRY_EVAL[name] = fn
        return fn
    return wrapper

def register_dataset_train(name):
    def wrapper(fn):
        DATASET_REGISTRY_TRAIN[name] = fn
        return fn
    return wrapper


@register_dataset_eval("minerva_math")
def load_minerva_math(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/minervamath", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["question"],
            "answer": row["answer"],
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": "",
        })
    return out


@register_dataset_eval("aime_2025")
def load_aime_2025(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/aime_2025", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": ", ".join(row["problem_type"]),
            "unique_id": f"aime2025_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("hmmt_feb_2025")
def load_hmmt_feb_2025(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/hmmt_feb_2025", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": ", ".join(row["problem_type"]),
            "unique_id": f"hmmt_feb2025_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("aime24")
def load_aime24(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/aime24", split="test")
    out = []
    for row in ds:
        answer = extract_boxed_answer(row["solution"]) or ""
        out.append({
            "problem": row["problem"],
            "answer": answer,
            "solution": row["solution"],
            "level": 0,
            "subject": "",
            "unique_id": f"aime24_{row['id']}",
        })
    return out


@register_dataset_eval("aime25")
def load_aime25(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("math-ai/aime25", split="test")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": f"aime25_{row['id']}",
        })
    return out


@register_dataset_eval("aime26")
def load_aime26(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("MathArena/aime_2026", split="train")
    out = []
    for row in ds:
        out.append({
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "solution": "",
            "level": 0,
            "subject": "",
            "unique_id": f"aime26_{row['problem_idx']}",
        })
    return out


@register_dataset_eval("math500")
def load_math500(levels: list[int] | None = None) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for row in ds:
        level = int(str(row["level"]).removeprefix("Level "))
        if levels and level not in levels:
            continue
        out.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "solution": row["solution"],
            "level": level,
            "subject": row["subject"],
            "unique_id": row.get("unique_id", ""),
        })
    return out


@register_dataset_train("deepmath")
def load_deepmath(
    max_samples: int | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load zwhe99/DeepMath-103K, exploding 3 solution columns into separate rows.

    Each example is tripled: one row per r1_solution_{1,2,3}. The columns are
    mapped to 'problem', 'solution', and 'answer' to match the existing format.
    """
    from datasets import concatenate_datasets

    ds = load_dataset("zwhe99/DeepMath-103K", split="train")

    # Explode: create 3 copies of each row, one per solution column
    def _make_split(sol_col):
        return ds.map(
            lambda x: {"problem": x["question"], "solution": x[sol_col], "answer": x["final_answer"]},
            remove_columns=ds.column_names,
            num_proc=4,
        )

    ds_exploded = concatenate_datasets([
        _make_split("r1_solution_1"),
        _make_split("r1_solution_2"),
        _make_split("r1_solution_3"),
    ])

    # Drop rows with empty solutions
    ds_exploded = ds_exploded.filter(
        lambda x: x["solution"] is not None and len(x["solution"].strip()) > 0,
        num_proc=4,
    )

    ds_exploded = ds_exploded.shuffle(seed=seed)

    if max_samples:
        ds_exploded = ds_exploded.select(range(min(max_samples, len(ds_exploded))))

    return ds_exploded


@register_dataset_train("openthoughts")
def load_openthoughts(
    max_samples: int | None = None,
    seed: int = 42,
) -> "Dataset":
    """Load OpenThoughts-114k (metadata subset), filtered to math domain.

    Maps deepseek_reasoning -> solution, extracts boxed answer from
    deepseek_solution (falls back to ground_truth_solution).
    """
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")

    # Filter to math domain
    ds = ds.filter(lambda x: x["domain"] == "math", num_proc=4)

    # Map to standard columns
    def _map_columns(example):
        # Try extracting boxed answer from deepseek_solution first
        answer = extract_boxed_answer(example["deepseek_solution"] or "")
        if not answer and example.get("ground_truth_solution"):
            answer = extract_boxed_answer(example["ground_truth_solution"])
        return {
            "problem": example["problem"],
            "solution": example["deepseek_reasoning"],
            "answer": answer or "",
        }

    ds = ds.map(_map_columns, remove_columns=ds.column_names, num_proc=4)

    # Drop rows with empty solution or answer
    ds = ds.filter(
        lambda x: (
            x["solution"] is not None
            and len(x["solution"].strip()) > 0
            and x["answer"] is not None
            and len(x["answer"].strip()) > 0
        ),
        num_proc=4,
    )

    ds = ds.shuffle(seed=seed)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds
