# Multi-task Reward Training with GRPO

This tutorial shows how to train with multiple tasks in a single GRPO run, how `None` rewards are handled, and how to route rewards efficiently when only some rewards apply to each sample.

## 1. Problem setup

We assume a dataset that mixes tasks and includes a `task` column to identify each sample:

```python
from datasets import Dataset

dataset = Dataset.from_list(
    [
        {"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
    ]
)
```

## 2. Simple multi-reward approach (with `None`)

The simplest approach is to provide multiple reward functions and let each one return `None` for irrelevant tasks. GRPOTrainer will ignore `None` values when aggregating.

```python
from trl import GRPOTrainer

def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            correct = check_math_solution(prompt, completion)
            rewards.append(1.0 if correct else -1.0)
        else:
            rewards.append(None)
    return rewards

def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "code":
            works = test_code_solution(prompt, completion)
            rewards.append(1.0 if works else -1.0)
        else:
            rewards.append(None)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

Pros:

- Very simple and explicit.
- Easy to add more task-specific rewards.

Tradeoff:

- Every reward function is called for every sample, even if most are irrelevant.

## 3. Efficient routing in a single reward function

If reward computation is expensive, route tasks inside a single reward function and only compute what is needed.

```python
def math_reward(prompts, completions):
    return [1.0 if any(ch.isdigit() for ch in c) else -1.0 for c in completions]

def code_reward(prompts, completions):
    return [1.0 if "def" in c else -1.0 for c in completions]

def routed_reward(prompts, completions, task, **kwargs):
    rewards = [0.0] * len(prompts)

    math_idxs = [i for i, t in enumerate(task) if t == "math"]
    code_idxs = [i for i, t in enumerate(task) if t == "code"]

    if math_idxs:
        math_prompts = [prompts[i] for i in math_idxs]
        math_comps = [completions[i] for i in math_idxs]
        math_rewards = math_reward(math_prompts, math_comps)
        for idx, r in zip(math_idxs, math_rewards):
            rewards[idx] = r

    if code_idxs:
        code_prompts = [prompts[i] for i in code_idxs]
        code_comps = [completions[i] for i in code_idxs]
        code_rewards = code_reward(code_prompts, code_comps)
        for idx, r in zip(code_idxs, code_rewards):
            rewards[idx] = r

    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=routed_reward,
    train_dataset=dataset,
)

trainer.train()
```

This avoids running all rewards for all samples while keeping a single GRPOTrainer instance.

## 4. Tips for multi-task reward design

- Keep rewards task-specific and avoid conflicting signals.
- Use clear task labels (e.g., `"math"`, `"code"`), and keep them consistent.
- If reward scales differ greatly, use `reward_weights` or normalization to balance them.
- If a reward is not applicable, return `None` (multi-reward case) or skip it in the routed function.

## 5. Minimal trainer configuration

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    reward_weights=[1.0, 1.0],
    reward_aggregation_method="sum_then_normalize",
    scale_rewards="group",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=config,
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)
```
