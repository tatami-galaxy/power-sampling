"""
Trace monitoring for GRPO training.

Tracks epistemic verbalization density, response length, solve rate,
and their correlations across training steps. Logs to TensorBoard.

Reuses epistemic marker definitions and counting from
scripts/analysis/epistemic_analysis.py (Kim et al., 2025).
"""

from collections import defaultdict

import numpy as np
from transformers import TrainerCallback

from scripts.analysis.epistemic_analysis import count_markers, word_count


class MonitoredReward:
    """Wraps a reward function to buffer per-completion trace metrics.

    Transparent to GRPOTrainer — returns the same rewards as the
    underlying function, but records per-completion statistics into
    an internal buffer that TraceMonitorCallback flushes on each
    logging step.
    """

    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
        self.buffer: list[dict] = []

    def __call__(self, completions, answer, difficulty=None, **kwargs):
        rewards = self.reward_fn(completions, answer, **kwargs)
        for i, (completion, reward) in enumerate(zip(completions, rewards)):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            markers = count_markers(text)
            wc = word_count(text)
            total = sum(markers.values())
            self.buffer.append({
                "reward": reward,
                "word_count": wc,
                "total_markers": total,
                "density_per_1k": (total / wc * 1000) if wc > 0 else 0.0,
                "difficulty": str(difficulty[i]) if difficulty is not None else None,
            })
        return rewards

    # Expose __name__ so TRL can label the reward function in logs.
    @property
    def __name__(self):
        return getattr(self.reward_fn, "__name__", "monitored_reward")

    def flush(self) -> list[dict]:
        data = self.buffer
        self.buffer = []
        return data


class TraceMonitorCallback(TrainerCallback):
    """Flushes the MonitoredReward buffer on each logging step and writes
    aggregate trace metrics to the trainer's log dict (→ TensorBoard).

    Metrics logged:
        monitor/epistemic_density       — mean epistemic markers per 1k words
        monitor/response_length         — mean word count
        monitor/solve_rate              — fraction of correct completions

        monitor/corr_solve_epistemic    — Pearson r across prompts
        monitor/corr_solve_length       — Pearson r across prompts
        monitor/corr_epistemic_length   — Pearson r across prompts

        monitor/within_density_diff     — mean(density|correct) - mean(density|incorrect)
        monitor/within_length_diff      — mean(length|correct) - mean(length|incorrect)

        monitor/difficulty/{bucket}/solve_rate
        monitor/difficulty/{bucket}/epistemic_density
        monitor/difficulty/{bucket}/response_length
    """

    def __init__(self, monitored_reward: MonitoredReward, num_generations: int):
        self.monitored_reward = monitored_reward
        self.num_generations = num_generations

    def on_log(self, args, state, control, logs=None, **kwargs):
        entries = self.monitored_reward.flush()
        if not entries or logs is None:
            return

        densities = [e["density_per_1k"] for e in entries]
        lengths = [e["word_count"] for e in entries]
        rewards = [e["reward"] for e in entries]

        logs["monitor/epistemic_density"] = float(np.mean(densities))
        logs["monitor/epistemic_density_std"] = float(np.std(densities))
        logs["monitor/response_length"] = float(np.mean(lengths))
        logs["monitor/response_length_std"] = float(np.std(lengths))
        logs["monitor/solve_rate"] = float(np.mean(rewards))

        # ----- Per-prompt group metrics -----
        G = self.num_generations
        num_prompts = len(entries) // G

        if num_prompts < 2:
            self._log_difficulty(entries, logs)
            return

        prompt_solve_rates = []
        prompt_densities = []
        prompt_lengths = []
        within_density_diffs = []
        within_length_diffs = []

        for p in range(num_prompts):
            group = entries[p * G : (p + 1) * G]
            g_rewards = [e["reward"] for e in group]
            g_densities = [e["density_per_1k"] for e in group]
            g_lengths = [e["word_count"] for e in group]

            prompt_solve_rates.append(np.mean(g_rewards))
            prompt_densities.append(np.mean(g_densities))
            prompt_lengths.append(np.mean(g_lengths))

            # Within-prompt: compare correct vs incorrect completions.
            # Only meaningful when the group has both correct and incorrect.
            num_correct = sum(1 for r in g_rewards if r > 0)
            if 0 < num_correct < len(g_rewards):
                correct_d = [d for d, r in zip(g_densities, g_rewards) if r > 0]
                incorrect_d = [d for d, r in zip(g_densities, g_rewards) if r == 0]
                within_density_diffs.append(np.mean(correct_d) - np.mean(incorrect_d))

                correct_l = [l for l, r in zip(g_lengths, g_rewards) if r > 0]
                incorrect_l = [l for l, r in zip(g_lengths, g_rewards) if r == 0]
                within_length_diffs.append(np.mean(correct_l) - np.mean(incorrect_l))

        # Cross-prompt correlations (need sufficient data + variance).
        sr = np.array(prompt_solve_rates)
        d = np.array(prompt_densities)
        l = np.array(prompt_lengths)

        if len(sr) >= 5 and np.std(sr) > 1e-8:
            if np.std(d) > 1e-8:
                logs["monitor/corr_solve_epistemic"] = float(np.corrcoef(sr, d)[0, 1])
            if np.std(l) > 1e-8:
                logs["monitor/corr_solve_length"] = float(np.corrcoef(sr, l)[0, 1])
        if len(d) >= 5 and np.std(d) > 1e-8 and np.std(l) > 1e-8:
            logs["monitor/corr_epistemic_length"] = float(np.corrcoef(d, l)[0, 1])

        # Within-prompt differences (positive = correct completions are MORE epistemic/longer).
        if within_density_diffs:
            logs["monitor/within_density_diff"] = float(np.mean(within_density_diffs))
        if within_length_diffs:
            logs["monitor/within_length_diff"] = float(np.mean(within_length_diffs))

        # ----- Per-difficulty bucket -----
        self._log_difficulty(entries, logs)

    @staticmethod
    def _log_difficulty(entries: list[dict], logs: dict):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            if e["difficulty"] is not None:
                buckets[e["difficulty"]].append(e)

        for bucket, group in buckets.items():
            # TensorBoard tag-safe: replace / with _ (e.g. "1/8" → "1_8")
            tag = bucket.replace("/", "_")
            logs[f"monitor/difficulty/{tag}/solve_rate"] = float(
                np.mean([e["reward"] for e in group])
            )
            logs[f"monitor/difficulty/{tag}/epistemic_density"] = float(
                np.mean([e["density_per_1k"] for e in group])
            )
            logs[f"monitor/difficulty/{tag}/response_length"] = float(
                np.mean([e["word_count"] for e in group])
            )
