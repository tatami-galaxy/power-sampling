"""
Answer-conditioned Self-Distillation Fine-Tuning (SDFT) for math reasoning.

Based on Shenfeld et al. (2025) "Self-Distillation Enables Continual Learning",
adapted for math tasks: instead of a full demonstration, the self-teacher is
conditioned only on the correct final answer.

Architecture:
  - vLLM: fast batch generation of student completions (one pass per round)
  - HF Transformers: forward passes for KL computation + gradient updates

Training loop (per round):
  1. Generate completions with vLLM from the current student
  2. For each example (gradient accumulated):
     a. Student forward: logits from pi_theta(. | x, y_{<t})
     b. Teacher forward: logits from pi_phi(. | x+answer, y_{<t})  [no grad]
     c. Loss = KL(student || stopgrad(teacher)) on completion tokens
     d. Backward, optimizer step, EMA update
  3. Save checkpoint

Usage:
    # Train with default settings
    uv run python -m scripts.train_sdft \
        --model Qwen/Qwen3-4B \
        --dataset math500 --num_samples 100

    # Full config
    uv run python -m scripts.train_sdft \
        --model Qwen/Qwen3-4B \
        --dataset math500 --num_samples 200 \
        --num_rounds 3 --lr 1e-5 --ema_alpha 0.01 \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing
"""

import argparse
import copy
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from vllm import LLM, SamplingParams

from scripts.run_eval import (
    DATASET_REGISTRY,
    SYSTEM_PROMPT,
    extract_boxed_answer,
    format_prompt,
    is_equiv,
)
from scripts.epistemic_analysis import count_markers, word_count


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_teacher_prompt(problem: str, answer: str) -> list[dict]:
    """Teacher sees the problem + correct final answer (not the solution path)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{problem}\n\nThe correct answer is {answer}."},
    ]


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def generate_completions(
    model_path: str,
    tokenizer,
    problems: list[dict],
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> list[dict]:
    """Generate one completion per problem using vLLM (temperature=1 for on-policy)."""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
    )

    prompts = []
    for p in problems:
        messages = format_prompt(p["problem"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    params = SamplingParams(temperature=1.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)

    completions = []
    for out in outputs:
        gen = out.outputs[0]
        completions.append({
            "text": gen.text,
            "token_ids": list(gen.token_ids),
        })

    del llm
    torch.cuda.empty_cache()
    return completions


# ---------------------------------------------------------------------------
# KL loss
# ---------------------------------------------------------------------------

def compute_kl_loss(
    student_model,
    teacher_model,
    student_ids: torch.Tensor,   # (1, s_prompt_len + comp_len)
    teacher_ids: torch.Tensor,   # (1, t_prompt_len + comp_len)
    completion_len: int,
    num_skip: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Reverse KL(student || stopgrad(teacher)) on completion tokens.

    Both models see the same completion tokens, prefixed by different prompts.
    Logits at position [prompt_len - 1] predict the first completion token.
    """
    device = student_ids.device

    # --- Student forward (with grad) ---
    s_out = student_model(input_ids=student_ids, use_cache=False)
    s_prompt_len = student_ids.size(1) - completion_len
    s_logits = s_out.logits[:, s_prompt_len - 1 : s_prompt_len + completion_len - 1, :]
    del s_out

    # --- Teacher forward (no grad) ---
    with torch.no_grad():
        t_out = teacher_model(input_ids=teacher_ids, use_cache=False)
        t_prompt_len = teacher_ids.size(1) - completion_len
        t_logits = t_out.logits[:, t_prompt_len - 1 : t_prompt_len + completion_len - 1, :]
        del t_out

    # Log-softmax in float32 for stability
    s_lp = F.log_softmax(s_logits.float(), dim=-1)
    del s_logits
    with torch.no_grad():
        t_lp = F.log_softmax(t_logits.float(), dim=-1)
    del t_logits

    # KL(student || teacher) per token = sum_v p_student(v) * log(p_student(v) / p_teacher(v))
    # F.kl_div(input=teacher_lp, target=student_lp, log_target=True)
    #   = exp(student_lp) * (student_lp - teacher_lp) per element
    kl_per_token = F.kl_div(
        t_lp, s_lp, log_target=True, reduction="none",
    ).sum(dim=-1).squeeze(0)  # (completion_len,)
    del t_lp

    # Loss mask: skip first N tokens (suppress learned artifacts per Shenfeld et al.)
    mask = torch.ones(completion_len, device=device)
    if num_skip > 0:
        mask[:num_skip] = 0.0

    loss = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)

    metrics = {
        "kl_mean": kl_per_token.detach().mean().item(),
        "kl_max": kl_per_token.detach().max().item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(student_model, teacher_model, alpha: float):
    """phi <- alpha * theta + (1 - alpha) * phi"""
    for s_param, t_param in zip(student_model.parameters(), teacher_model.parameters()):
        t_param.data.mul_(1.0 - alpha).add_(s_param.data, alpha=alpha)


# ---------------------------------------------------------------------------
# GPU / CPU offloading (to share GPU between vLLM and HF models)
# ---------------------------------------------------------------------------

def offload_to_cpu(student, teacher, optimizer):
    """Move models + optimizer state to CPU to free GPU for vLLM."""
    student.cpu()
    teacher.cpu()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    torch.cuda.empty_cache()


def reload_to_gpu(student, teacher, optimizer, device):
    """Move everything back to GPU after vLLM is done."""
    student.to(device)
    teacher.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# ---------------------------------------------------------------------------
# Per-round completion stats
# ---------------------------------------------------------------------------

def completion_stats(problems, completions):
    """Compute accuracy and epistemic density on generated completions."""
    correct = 0
    total_markers = 0
    total_words = 0
    lengths = []

    for prob, comp in zip(problems, completions):
        text = comp["text"]
        lengths.append(len(comp["token_ids"]))

        pred = extract_boxed_answer(text)
        if pred and is_equiv(pred, prob["answer"]):
            correct += 1

        counts = count_markers(text)
        total_markers += sum(counts.values())
        total_words += word_count(text)

    n = len(problems)
    density = (total_markers / total_words * 1000) if total_words > 0 else 0.0
    non_empty = sum(1 for l in lengths if l > 0)
    mean_len = sum(lengths) / n if n else 0

    return {
        "accuracy": correct / n if n else 0,
        "correct": correct,
        "total": n,
        "non_empty": non_empty,
        "mean_tokens": mean_len,
        "epistemic_density": density,
    }


# ---------------------------------------------------------------------------
# Training round
# ---------------------------------------------------------------------------

def train_round(
    student_model,
    teacher_model,
    optimizer,
    scheduler,
    tokenizer,
    problems,
    completions,
    *,
    grad_accum: int,
    num_skip: int,
    max_grad_norm: float,
    use_ema: bool,
    ema_alpha: float,
    device,
    round_idx: int,
) -> tuple[int, list[float]]:
    """Train one epoch on generated completions. Returns (num_steps, losses)."""
    student_model.train()
    teacher_model.eval()

    # Tokenize all examples up front (cheap — just prompt encoding)
    examples = []
    for prob, comp in zip(problems, completions):
        comp_ids = comp["token_ids"]
        if not comp_ids:
            continue

        s_msgs = format_prompt(prob["problem"])
        t_msgs = format_teacher_prompt(prob["problem"], prob["answer"])

        s_text = tokenizer.apply_chat_template(
            s_msgs, tokenize=False, add_generation_prompt=True,
        )
        t_text = tokenizer.apply_chat_template(
            t_msgs, tokenize=False, add_generation_prompt=True,
        )

        s_prompt_ids = tokenizer.encode(s_text, add_special_tokens=False)
        t_prompt_ids = tokenizer.encode(t_text, add_special_tokens=False)

        examples.append({
            "student_ids": s_prompt_ids + comp_ids,
            "teacher_ids": t_prompt_ids + comp_ids,
            "completion_len": len(comp_ids),
        })

    random.shuffle(examples)

    all_losses = []
    accum_losses = []
    accum_count = 0
    global_step = 0
    optimizer.zero_grad()

    pbar = tqdm(examples, desc=f"Round {round_idx}", unit="ex")
    for i, ex in enumerate(pbar):
        s_ids = torch.tensor([ex["student_ids"]], dtype=torch.long, device=device)
        t_ids = torch.tensor([ex["teacher_ids"]], dtype=torch.long, device=device)

        loss, metrics = compute_kl_loss(
            student_model, teacher_model,
            s_ids, t_ids,
            ex["completion_len"],
            num_skip,
        )

        (loss / grad_accum).backward()
        accum_losses.append(loss.item())
        accum_count += 1

        if accum_count == grad_accum or (i + 1) == len(examples):
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if use_ema:
                ema_update(student_model, teacher_model, ema_alpha)

            avg_loss = sum(accum_losses) / len(accum_losses)
            all_losses.append(avg_loss)
            lr = scheduler.get_last_lr()[0]
            global_step += 1

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                kl=f"{metrics['kl_mean']:.4f}",
                lr=f"{lr:.2e}",
            )

            accum_losses = []
            accum_count = 0

    return global_step, all_losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Answer-conditioned SDFT for math reasoning",
    )

    # Model
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--chat_template_model", type=str, default=None,
                        help="Load chat template from this model (for base models)")

    # Dataset
    parser.add_argument("--dataset", default="math500",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--levels", nargs="*", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--num_rounds", type=int, default=3,
                        help="Number of generate-then-train rounds")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_alpha", type=float, default=0.01,
                        help="EMA blending: phi <- alpha*theta + (1-alpha)*phi")
    parser.add_argument("--no_ema", action="store_true",
                        help="Disable EMA; teacher = student with different prompt")
    parser.add_argument("--num_loss_tokens_to_skip", type=int, default=3,
                        help="Mask loss on first N completion tokens (Shenfeld et al.)")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Generation
    parser.add_argument("--max_completion_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)

    # Output
    parser.add_argument("--output_dir", default="results/sdft")

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # ---- Dataset ----
    problems = DATASET_REGISTRY[args.dataset](levels=args.levels)
    if args.num_samples and args.num_samples < len(problems):
        problems = random.sample(problems, args.num_samples)
    print(f"Dataset: {args.dataset} ({len(problems)} problems)")

    # ---- Tokenizer ----
    tok_name = args.chat_template_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    # ---- Output dir ----
    model_slug = args.model.replace("/", "_")
    output_dir = os.path.join(args.output_dir, args.dataset, model_slug)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Load HF models ----
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    if args.no_ema:
        # Teacher shares architecture but is just the student (same weights, different prompt)
        # We still need a separate module for the no-grad forward pass
        print("EMA disabled — teacher = frozen copy of student (synced each round)")
        teacher = copy.deepcopy(student)
    else:
        print(f"Loading teacher model (EMA copy, alpha={args.ema_alpha})...")
        teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Optimizer & scheduler ----
    steps_per_round = max(1, len(problems) // args.gradient_accumulation_steps)
    total_steps = steps_per_round * args.num_rounds
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(student.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training plan: {args.num_rounds} rounds x ~{steps_per_round} steps = "
          f"~{total_steps} total steps (warmup={warmup_steps})")

    # ---- Training log ----
    log = {"config": vars(args), "rounds": []}
    current_model_path = args.model

    for round_idx in range(args.num_rounds):
        round_dir = os.path.join(output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        ckpt_dir = os.path.join(round_dir, "checkpoint")

        # Skip completed rounds (resume support)
        if os.path.exists(os.path.join(ckpt_dir, "config.json")):
            print(f"\nRound {round_idx}: checkpoint exists, skipping")
            current_model_path = ckpt_dir
            continue

        # ==== Phase 1: Generate completions with vLLM ====
        comp_path = os.path.join(round_dir, "completions.json")
        if os.path.exists(comp_path):
            print(f"\nRound {round_idx}: Loading existing completions")
            with open(comp_path) as f:
                completions = json.load(f)
        else:
            print(f"\nRound {round_idx}: Generating completions with vLLM...")

            # For round > 0, use the previous checkpoint
            if round_idx > 0:
                current_model_path = os.path.join(
                    output_dir, f"round_{round_idx - 1}", "checkpoint",
                )

            # Offload HF models to CPU → free GPU for vLLM
            offload_to_cpu(student, teacher, optimizer)

            completions = generate_completions(
                current_model_path,
                tokenizer,
                problems,
                max_tokens=args.max_completion_tokens,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
            )

            # Reload HF models to GPU
            reload_to_gpu(student, teacher, optimizer, device)

            with open(comp_path, "w") as f:
                json.dump(completions, f)

        # Completion stats
        stats = completion_stats(problems, completions)
        print(f"  Completions: {stats['non_empty']}/{stats['total']} non-empty, "
              f"mean {stats['mean_tokens']:.0f} tokens")
        print(f"  Accuracy:    {stats['correct']}/{stats['total']} "
              f"= {stats['accuracy']*100:.1f}%")
        print(f"  Epistemic:   {stats['epistemic_density']:.1f} markers/1k words")

        # If --no_ema, sync teacher to current student at start of each round
        if args.no_ema:
            with torch.no_grad():
                for s_p, t_p in zip(student.parameters(), teacher.parameters()):
                    t_p.data.copy_(s_p.data)

        # ==== Phase 2: Train ====
        t0 = time.time()
        steps, losses = train_round(
            student, teacher, optimizer, scheduler,
            tokenizer, problems, completions,
            grad_accum=args.gradient_accumulation_steps,
            num_skip=args.num_loss_tokens_to_skip,
            max_grad_norm=args.max_grad_norm,
            use_ema=not args.no_ema,
            ema_alpha=args.ema_alpha,
            device=device,
            round_idx=round_idx,
        )
        elapsed = time.time() - t0

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(f"  Training:  {steps} steps, avg loss {avg_loss:.4f}, {elapsed:.1f}s")

        # ==== Save checkpoint ====
        student.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        current_model_path = ckpt_dir
        print(f"  Checkpoint: {ckpt_dir}")

        round_log = {
            "round": round_idx,
            "steps": steps,
            "avg_loss": avg_loss,
            "losses": losses,
            "elapsed_s": elapsed,
            "completion_stats": stats,
            "checkpoint": ckpt_dir,
        }
        log["rounds"].append(round_log)

    # ---- Save log ----
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete.")
    print(f"  Log:              {log_path}")
    print(f"  Final checkpoint: {current_model_path}")


if __name__ == "__main__":
    main()
