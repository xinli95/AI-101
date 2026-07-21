# Standard RL with GRPO in TRL: A Practical Walkthrough of `GRPOTrainer`

This tutorial is a hands-on tour of TRL's `GRPOTrainer` for the "regular" RL setting: one prompt in, one completion out, one reward computed on that completion. There are no tool calls, no multi-turn rollouts, and no environment interaction — just a policy, a group of sampled completions, and a reward function. This is the setting most RLVR pipelines (math, code, instruction following) actually run in, and it is the right place to start before moving on to the agentic case.

If you want the underlying math first (advantage computation, clipped objective, DAPO / Dr. GRPO / GSPO variants), read {ref}`grpo-family-tutorial` — this tutorial assumes that background and focuses on how those ideas map onto `GRPOTrainer`'s API. For multi-turn tool-use rollouts, see [Agentic RL with GRPO](trl_grpo_agent.md); for mixing several tasks/rewards in one run, see [Multi-task Reward Training with GRPO](trl_grpo_multitask_rewards.md).

Reference docs:

- https://huggingface.co/docs/trl/main/en/grpo_trainer

## 1. The shape of a "standard" GRPO rollout

Before touching code, it helps to be precise about what makes this the *non-agentic* case:

- Each training example is a single prompt.
- For each prompt, `GRPOTrainer` samples a **group of `G` completions** from the current policy in one shot (no intermediate tool calls or environment turns).
- Each completion is scored once by one or more reward functions.
- Advantages are computed **relative to the group**, and the policy is updated with a PPO-style clipped objective (optionally regularized toward a reference model via KL).

Everything below is about configuring and driving that loop.

## 2. Installation and requirements

```bash
pip install "trl[vllm]" transformers accelerate datasets
```

`vllm` is optional but strongly recommended once you go past toy models — generation is almost always the training-time bottleneck in GRPO because you sample `G` completions per prompt every step.

## 3. Dataset format

`GRPOTrainer` only requires a `prompt` column. Everything else in the dataset is passed through to your reward function(s) as keyword arguments, which is how ground-truth answers, difficulty labels, etc. reach the reward computation.

```python
from datasets import Dataset

dataset = Dataset.from_list([
    {"prompt": "What is 12 * 7? Answer with just the number.", "answer": "84"},
    {"prompt": "What is 9 + 16? Answer with just the number.", "answer": "25"},
])
```

`prompt` can be a plain string (as above) or a list of chat messages (`[{"role": "user", "content": "..."}]`) if you want the trainer to apply the tokenizer's chat template.

## 4. Writing reward functions

A reward function takes the batch of prompts and completions (plus any extra dataset columns as `**kwargs`) and returns one float per completion.

```python
import re

def correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        match = re.search(r"-?\d+", completion)
        guess = match.group(0) if match else None
        rewards.append(1.0 if guess == gt else -1.0)
    return rewards
```

A few things worth knowing:

- You can pass a **list** of reward functions to `reward_funcs`. Their outputs are combined (optionally weighted with `reward_weights`) before advantages are computed.
- A reward function may return `None` for individual samples; those samples are excluded from that reward's aggregation. This matters more for multi-task setups (see [Multi-task Reward Training with GRPO](trl_grpo_multitask_rewards.md)) but works the same way here.
- Reward functions can be `async def` — useful if a reward needs an I/O-bound call (an external verifier, an API-based judge). TRL runs concurrent async reward functions together.
- `trl.rewards` ships a few ready-made reward functions (e.g. `accuracy_reward`) if you don't need anything custom.

## 5. Core config knobs and how they map to the theory

`GRPOConfig` is where the group-relative-advantage math from {ref}`grpo-family-tutorial` becomes trainer arguments. The important ones for a standard (non-agentic) run:

| Config field | Role | Connects to |
| --- | --- | --- |
| `num_generations` | Group size $G$ sampled per prompt | Group-relative advantage $\hat A_i$ |
| `beta` | KL coefficient against the reference model | $\beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$ term; `0.0` disables the reference model entirely (saves memory) |
| `epsilon` / `epsilon_high` | Lower/upper PPO clip range | $\operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)$ |
| `loss_type` | Loss formulation: `"grpo"`, `"dr_grpo"`, `"dapo"`, `"bnpo"`, and other variants | Which advantage/normalization recipe from §9 of {ref}`grpo-family-tutorial` gets used |
| `scale_rewards` | `"group"` (default), `"batch"`, or `False`/`"none"` | Whether $\hat A_i$ divides by $\sigma_R$ at all — `False` is the Dr. GRPO-style fix for reward-scale bias |
| `num_iterations` | PPO epochs per collected batch (the $\mu$ in "sample once, update $\mu$ times") | How off-policy the clipped ratio $r_{i,t}(\theta)$ can get |
| `max_completion_length` / `max_prompt_length` | Truncation limits | Controls rollout cost and interacts with length-bias failure modes |
| `temperature`, `top_p`, `top_k`, `min_p` | Sampling parameters for rollout generation | Determines how diverse the group is (affects $\sigma_R$ and entropy) |

`loss_type="grpo"` is the vanilla, token-normalized-per-sequence formulation and is length-biased (see §3.2 and §5 of {ref}`grpo-family-tutorial`); `"dr_grpo"` and `"dapo"`-style options are there specifically to correct that. If you're seeing runaway completion lengths without reward gains, this is the first knob to try, alongside `scale_rewards=False`.

## 6. A complete minimal example

Putting it together for a small math task:

```python
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

dataset = Dataset.from_list([
    {"prompt": f"What is {a} + {b}? Answer with just the number.", "answer": str(a + b)}
    for a, b in [(3, 4), (12, 7), (9, 16), (21, 5), (8, 8)]
])

def correctness_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        match = re.search(r"-?\d+", completion)
        guess = match.group(0) if match else None
        rewards.append(1.0 if guess == gt else -1.0)
    return rewards

config = GRPOConfig(
    output_dir="grpo_arithmetic",
    num_generations=8,
    max_completion_length=64,
    max_prompt_length=128,
    beta=0.0,
    loss_type="dr_grpo",
    scale_rewards=False,
    temperature=1.0,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    log_completions=True,
    report_to="trackio",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=config,
    reward_funcs=correctness_reward,
    train_dataset=dataset,
)

trainer.train()
```

This is deliberately tiny (toy dataset, small model) so you can run it on a single GPU to confirm the loop works before scaling up to a real dataset like GSM8K or a math-verifier-based reward.

## 7. The reference model and the KL term

`beta` controls whether a frozen reference model is used to regularize the policy:

- `beta=0.0` (the common choice for RLVR-style training): no reference model is loaded at all, which roughly halves memory usage relative to PPO-style setups that always keep a reference/critic. There is no KL penalty.
- `beta>0`: a reference model is kept in memory (or loaded from the initial policy weights) and $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$ is estimated with a Schulman-style approximator and subtracted from the objective.

If you do use `beta>0` over a long run, `sync_ref_model=True` with `ref_model_sync_steps` periodically refreshes the reference model toward the current policy (via `ref_model_mixup_alpha`-weighted mixing) instead of anchoring forever to the initial checkpoint. This trades off "how far the policy is allowed to drift" against "how stale the anchor becomes."

## 8. Speeding up rollouts with vLLM

Generation dominates GRPO's wall-clock time because every step samples `num_generations` completions per prompt. TRL supports two vLLM integration modes:

**Colocate mode** — vLLM runs inside the same process as training, sharing the GPU(s):

```python
config = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,  # fraction of GPU memory reserved for vLLM
)
```

**Server mode** — vLLM runs as a separate process (or on separate GPUs/nodes) that the trainer queries over HTTP. Start the server first:

```bash
trl vllm-serve --model Qwen/Qwen2.5-0.5B-Instruct
```

then point the trainer at it:

```python
config = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",
    vllm_server_host="0.0.0.0",
    vllm_server_port=8000,
)
```

Server mode is the better choice when you want generation and training to use disjoint hardware (e.g. a dedicated inference GPU pool), or when scaling to multi-node training.

One subtlety worth knowing: vLLM's numerics don't exactly match the training model's forward pass, so the log-probabilities used for the importance-sampling ratio can be slightly off between "what generated the completion" and "what the trainer thinks generated it." TRL corrects for this with `vllm_importance_sampling_correction` (on by default when using vLLM), which is a practical, implementation-level cousin of the token-vs-sequence-level ratio discussion in the GSPO section of {ref}`grpo-family-tutorial`.

## 9. Rollout and update cadence

A few knobs control how many rollouts are collected before an update, and how many gradient updates are squeezed out of them:

- `num_generations`: group size $G$ per prompt.
- `generation_batch_size` / `steps_per_generation`: how many prompts' worth of completions are generated together before optimization starts (these two are mutually exclusive ways of expressing the same thing).
- `num_iterations`: how many optimizer passes ($\mu$) are run over one collected batch of rollouts before generating a fresh batch. `num_iterations=1` is on-policy-per-batch; higher values reuse rollouts across several PPO-clipped updates, which is more sample-efficient but pushes the "old" policy further from the current one, making clipping and the importance-sampling ratio matter more.

If training feels off-policy in a way that's causing instability (large clip fractions, KL spikes), the first things to check are `num_iterations` and how stale the vLLM-generated log-probs are relative to the trainer's own forward pass.

## 10. What to monitor and how it maps to failure modes

`GRPOTrainer` logs reward, KL, and clipping statistics; turn on `log_completions=True` to also see actual generations during training. Tie these back to the failure modes in §3 and §10 of {ref}`grpo-family-tutorial`:

- **Reward mean / std collapsing to near-zero within groups** → entropy collapse; the group has stopped producing diverse completions, so $\hat A_i$ becomes uninformative. Consider raising `temperature`, or checking `top_entropy_quantile` / `entropy_coef` if your TRL version exposes entropy regularization.
- **Completion length climbing without reward improving** → classic length bias. Try `loss_type="dr_grpo"` and/or `scale_rewards=False`.
- **Large clip fractions or KL spikes** → the policy moved too far between rollout collection and optimization; reduce `num_iterations`, shrink the learning rate, or double-check vLLM/training log-prob mismatch (`vllm_importance_sampling_correction`).
- **`mask_truncated_completions`**: if completions are hitting `max_completion_length` frequently, set this to `True` so truncated (and therefore not-actually-finished) completions don't pollute the loss and metrics with cut-off text.

## 11. Minimal run command

For a real run, launch through `accelerate` so you can target multiple GPUs and (optionally) DeepSpeed/FSDP:

```bash
accelerate launch --config_file accelerate_configs/zero2.yaml \
  train_grpo.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir grpo_arithmetic_qwen2.5-0.5b \
  --use_vllm True \
  --vllm_mode colocate \
  --num_generations 8 \
  --max_completion_length 256 \
  --loss_type dr_grpo \
  --scale_rewards False \
  --report_to trackio \
  --log_completions True \
  --num_train_epochs 3
```

(`train_grpo.py` here is your own script that builds the dataset, defines the reward function(s), and calls `GRPOTrainer(...).train()` as in §6.)

## 12. Quick mental model

- One prompt → one group of `G` completions, generated once, scored once — no tool calls, no multi-turn state.
- Advantages are computed **relative to the group**, not from a learned critic.
- `beta` decides if there's a reference model and a KL penalty at all; `beta=0.0` is common for RLVR.
- `loss_type` and `scale_rewards` are your levers for the length-bias and reward-scale pathologies described in the GRPO theory chapter — try `dr_grpo` + `scale_rewards=False` first if training looks unstable.
- vLLM (colocate or server mode) is what makes sampling `G` completions per prompt affordable at scale; it introduces a train/inference log-prob mismatch that TRL corrects for automatically.
