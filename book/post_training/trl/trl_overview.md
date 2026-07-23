# TRL Examples

The TRL section collects practical, implementation-focused examples for post-training workflows. It complements the theory chapters with runnable code paths and highlights how to wire reward models, optimizers, and data for real training loops.

In this subsection you will find:

- **PPO trainer deep dive** — how `PPOTrainer`'s actor-critic loop implements GAE (including the backward recursion), the KL-penalized reward, and clipped policy/value losses.
- **DPO trainer deep dive** — how `DPOTrainer` turns preference pairs into the DPO loss: the concatenated chosen/rejected forward pass, reference-model strategies (including the LoRA adapter trick), and the implicit-reward metrics.
- **Standard RL with GRPO** — a practical walkthrough of `GRPOTrainer` for the single-turn (non-agentic) setting: config, reward functions, and rollout/update mechanics.
- **Multi-task reward training with GRPO** — mixing multiple tasks and reward functions in one run.
- **Agentic RL with GRPO** using TRL as an end-to-end example, with tool-calling, multi-turn rollouts, and reward design for agent behavior.
