# Power Sampling

Implementation of "Scalable Power Sampling" (Ji, Tutunov, Zimmer & Bou Ammar, 2025),
with planned extension to self-distillation training.

## Project Structure

```
power-sampling/
├── scalable_power_sampling/   # Core library
│   ├── __init__.py
│   ├── power_sampler.py       # Main generation loop, top-level API
│   ├── rollouts.py            # Batched rollout generation + log-likelihood tracking
│   ├── scaling.py             # Scaling factor (zeta) computation + jackknife correction
│   └── utils.py               # Log-space math, top-k selection helpers
├── scripts/                   # Evaluation and demo scripts
├── tests/                     # Unit tests
├── pyproject.toml
└── CLAUDE.md
```

## Key Design Principles

- All probability computations in log-space for numerical stability
- KV-cache reuse: compute prefix cache once, fork for candidates and rollouts
- Per-candidate batching (M rollouts at a time) to manage VRAM
- Compatible with any HuggingFace AutoModelForCausalLM

## Algorithm Reference (Single-Token)

For each position t:
1. Forward pass on prefix -> logits -> top-K candidates
2. For each candidate: generate M rollouts of H tokens, tracking log-probs
3. Compute scaling factor: zeta(x_t) = (1/M) * sum_r exp((alpha-1) * sum log p(rollout_r))
4. Jackknife bias correction: leave-one-out estimates to reduce bias from O(1/M) to O(1/M^2)
5. Power distribution: p_pow(x_t) proportional to p^alpha(x_t) * zeta(x_t)
6. Sample next token from p_pow

## Default Hyperparameters (from paper)

- alpha = 4.0 (power exponent)
- K = 8 (top-K candidates)
- M = 8 (rollouts per candidate)
- H = 192 (lookahead horizon in tokens)
- T_max = 3072 (max generation length)

## Commands

```bash
pip install -e .                    # Install in dev mode
pytest tests/                       # Run tests
python scripts/demo.py              # Run demo generation
```

## Development Notes

- Phase 1: Single-token scalable power sampling (current)
- Phase 2: Batched scalable power sampling (planned)
- Phase 3: Self-distillation integration (planned)
