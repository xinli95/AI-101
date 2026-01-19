# Post-training Overview

Post-training adapts a base model to target tasks, preferences, or constraints. This chapter focuses on the core methods and how they are implemented in practice:

- **Supervised fine-tuning (SFT):** learns from high-quality demonstrations to set behavior and style.
- **RLHF / RLVR:** optimizes for preferences or verifiable rewards; the PPO and GRPO sections cover policy updates and advantage estimation details.
- **Distillation:** transfers behavior from a larger or better model into a smaller or faster one.

It also includes **implementation examples with TRL**, including an agentic RL walkthrough.

Together, these methods turn a general model into a reliable, aligned, and efficient system.
