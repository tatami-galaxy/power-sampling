"""Demo script: compare base sampling vs power sampling on a math prompt."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scalable_power_sampling import PowerSampler


def main():
    parser = argparse.ArgumentParser(description="Power sampling demo")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name (default: small model for testing)",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=32, help="Rollout horizon")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    if args.prompt is None:
        prompt = "Solve the following math problem step by step.\n\nProblem: What is the sum of all prime numbers less than 20?\n\nSolution:"
    else:
        prompt = args.prompt

    # --- Base model greedy ---
    print("\n=== Base Model (greedy) ===")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        base_out = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    base_text = tokenizer.decode(base_out[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(base_text)

    # --- Power sampling ---
    print(f"\n=== Power Sampling (alpha={args.alpha}, K={args.top_k}, M={args.num_rollouts}, H={args.lookahead}) ===")
    sampler = PowerSampler(
        model=model,
        tokenizer=tokenizer,
        alpha=args.alpha,
        top_k=args.top_k,
        num_rollouts=args.num_rollouts,
        lookahead=args.lookahead,
        max_new_tokens=args.max_new_tokens,
    )
    print(sampler)
    result = sampler.generate(input_ids=input_ids, verbose=True)
    print(result["text"])
    print(f"\n[Generated {result['num_tokens_generated']} tokens]")


if __name__ == "__main__":
    main()
