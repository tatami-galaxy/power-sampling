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


def _extract_braced(text: str, start: int) -> str | None:
    """Extract content inside balanced braces starting at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} or \\fbox{...} from text, handling nested braces."""
    for marker in ("\\boxed{", "\\fbox{"):
        idx = text.rfind(marker)
        if idx != -1:
            return _extract_braced(text, idx + len(marker) - 1)
    return None


# ---------------------------------------------------------------------------
# Normalization helpers (ported from Hendrycks MATH / reasoning-with-sampling)
# ---------------------------------------------------------------------------

_UNITS_RE = re.compile(
    r"(?:degree|cm|centimeter|meter|mile|second|minute|hour|day|week|month|year"
    r"|foot|feet|inch|yard)(es|s)? *(\^[0-9]+)?",
)


def _fix_fracs(s: str) -> str:
    r"""\\frac12 -> \\frac{1}{2}, etc."""
    parts = s.split("\\frac")
    if len(parts) < 2:
        return s
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if not part or part[0] == "{":
            out += part
            continue
        if len(part) < 2:
            out += part
            continue
        a, b = part[0], part[1]
        if b != "{":
            out += "{" + a + "}{" + b + "}" + part[2:]
        else:
            out += "{" + a + "}" + b + part[2:]
    return out


def _fix_sqrt(s: str) -> str:
    r"""\\sqrt3 -> \\sqrt{3}."""
    parts = s.split("\\sqrt")
    if len(parts) < 2:
        return s
    out = parts[0]
    for part in parts[1:]:
        if not part or part[0] == "{":
            out += "\\sqrt" + part
        else:
            out += "\\sqrt{" + part[0] + "}" + part[1:]
    return out


def _fix_a_slash_b(s: str) -> str:
    r"""Simple a/b -> \\frac{a}{b} when both are integers."""
    pieces = s.split("/")
    if len(pieces) != 2:
        return s
    try:
        a, b = int(pieces[0]), int(pieces[1])
        return f"\\frac{{{a}}}{{{b}}}"
    except ValueError:
        return s


def _strip_commas(s: str) -> str:
    """Remove properly-formatted thousands commas: 1,000,000 -> 1000000."""
    p = re.compile(r"(\d),(\d\d\d)(?=$|\D)")
    while True:
        new = p.sub(r"\1\2", s)
        if new == s:
            break
        s = new
    return s


def _normalize(s: str) -> str:
    """Normalize a math answer string for string comparison."""
    s = s.strip()

    # Remove dollar signs
    s = s.replace("\\$", "")
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    s = s.replace("$", "")

    # Remove percentage
    s = s.replace("\\%", "").replace("%", "")

    # Remove \text{} wrapper
    m = re.fullmatch(r"\\text\{(.+)\}", s)
    if m:
        s = m.group(1).strip()

    # Linebreaks and double backslash
    s = s.replace("\n", "")
    s = s.replace("\\\\", "\\")

    # Frac variants -> \frac
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")

    # Remove display commands
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")

    # Remove degree markers
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")

    # Remove units (including inside \text{})
    s = re.sub(r"\\text\{\s*[^}]*\}", "", s)
    s = _UNITS_RE.sub("", s)

    # Strip outer braces {expr}
    if len(s) > 1 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1]

    # Leading zero: .5 -> 0.5
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s

    # Variable prefix: "k = 5" -> "5"
    if len(s.split("=")) == 2 and len(s.split("=")[0].strip()) <= 2:
        s = s.split("=")[1]

    # Fix shorthand: \frac12, \sqrt3
    s = _fix_sqrt(s)
    s = _fix_fracs(s)

    # Thousands commas
    s = _strip_commas(s)

    # Remove all spaces
    s = s.replace(" ", "")

    # a/b -> \frac{a}{b} for simple integer fractions
    s = _fix_a_slash_b(s)

    s = s.rstrip(".")
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


_TUPLE_CHARS = "()[]"


def _split_tuple(s: str) -> list[str]:
    """Split a tuple/interval like '(a, b)' into elements ['a', 'b'].

    Returns a single-element list [s] if s is not a tuple.
    """
    s = _strip_commas(s)
    if (
        len(s) > 2
        and s[0] in _TUPLE_CHARS
        and s[-1] in _TUPLE_CHARS
        and all(ch not in s[1:-1] for ch in _TUPLE_CHARS)
    ):
        return [e.strip() for e in s[1:-1].split(",")]
    return [s]


def _is_equiv_single(pred_n: str, gold_n: str, pred_raw: str, gold_raw: str) -> bool:
    """Check equivalence of two scalar (non-tuple) normalized answers."""
    if pred_n == gold_n:
        return True
    pred_v = _try_parse_number(pred_n)
    gold_v = _try_parse_number(gold_n)
    if pred_v is not None and gold_v is not None:
        return abs(pred_v - gold_v) < 1e-6
    try:
        gold_parsed = parse(gold_raw, extraction_config=GOLD_EXTRACTION_CONFIG)
        pred_parsed = parse(pred_raw, extraction_config=PRED_EXTRACTION_CONFIG)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


def is_equiv(pred: str, gold: str) -> bool:
    """Check equivalence using layered strategies:
    1. Normalized string match (fast, handles most cases)
    2. Tuple/interval element-wise comparison
    3. Numeric comparison (fractions, decimals)
    4. math_verify symbolic comparison (fallback for complex expressions)
    """
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)

    # Fast path: full normalized match
    if pred_n == gold_n:
        return True

    # Split into tuple elements
    pred_elems = _split_tuple(pred_n)
    gold_elems = _split_tuple(gold_n)

    if len(pred_elems) != len(gold_elems):
        # Length mismatch — fall through to single-value comparison
        return _is_equiv_single(pred_n, gold_n, pred, gold)

    if len(pred_elems) > 1:
        # Tuple: require matching delimiters
        if pred_n[0] != gold_n[0] or pred_n[-1] != gold_n[-1]:
            return False
        return all(
            _is_equiv_single(pe, ge, pe, ge)
            for pe, ge in zip(pred_elems, gold_elems)
        )

    # Scalar: full comparison with raw strings for math_verify fallback
    return _is_equiv_single(pred_n, gold_n, pred, gold)


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
