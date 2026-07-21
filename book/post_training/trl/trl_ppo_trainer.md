# PPO Trainer Deep Dive: GAE, Clipping, and the Backward Recursion in Practice

The theory chapters [PPO From Scratch](../ppo.md) and [From Return to Advantage: Full Derivation to GAE](../ppo_advantage.md) derive the clipped objective and the GAE recursion on paper. This tutorial closes the gap between that math and a real, running trainer: TRL's `PPOTrainer`. We walk through the actual training loop — rollout generation, turning a scalar reward-model score into a per-token reward, the **backward recursive GAE computation**, and the clipped policy/value losses — line by line, with a small runnable example that makes the backward recursion concrete.

This is the "full RLHF PPO" setting: a policy model, a separate value model (critic), a frozen reward model, and a frozen reference model, all in play at once. If you haven't read {doc}`../ppo_advantage`, at least skim §5–§8 there first — this tutorial assumes you know why $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$ holds and focuses on *how that equation becomes code*.

Reference implementation (read at TRL `v0.28.0`, the last tag before the legacy `trl.PPOTrainer` import path was removed):

- https://github.com/huggingface/trl/blob/v0.28.0/trl/experimental/ppo/ppo_trainer.py
- https://github.com/huggingface/trl/blob/v0.28.0/trl/experimental/ppo/ppo_config.py

```{note}
As of recent TRL releases, this trainer lives under `trl.experimental.ppo` — import it as `from trl.experimental.ppo import PPOTrainer, PPOConfig`. TRL's primary recommendation for RLVR-style training has shifted toward critic-free trainers like `GRPOTrainer` (see [Standard RL with GRPO in TRL](trl_grpo_trainer.md)); §16 below explains exactly which piece of *this* tutorial GRPO throws away and why.
```

## 1. The four models

Unlike `GRPOTrainer`, which only needs a policy (and optionally a reference model for KL), `PPOTrainer` is a full actor-critic setup and needs four models:

| Model | Role | Trainable? |
| --- | --- | --- |
| `model` (policy) | Generates responses; the thing you're actually optimizing | Yes |
| `value_model` | Predicts $V(s_t)$ at every token position; the critic used for GAE | Yes |
| `ref_model` | Frozen copy of the initial policy; anchors the KL penalty | No |
| `reward_model` | Frozen scalar reward model; scores full completions | No |

Internally, the policy and value model are packaged into one `nn.Module` so `accelerate`/DeepSpeed only has to wrap a single model:

```python
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits
```

This is packaging for distributed training convenience, not weight-tying — `value_model` is a separate model (typically `AutoModelForSequenceClassification` with a 1-dimensional score head) with its own backbone. Both the policy and value model get one joint optimizer step per update.

## 2. Dataset format

`PPOTrainer` expects a **pre-tokenized** dataset with an `input_ids` column (the prompt/query tokens), not raw text:

```python
queries = data["input_ids"].to(device)
```

You tokenize prompts up front (typically left-padded, since generation appends tokens on the right), unlike `GRPOTrainer`'s `prompt` text column that gets templated internally.

## 3. Rollout: generation, log-probs, reference log-probs

For each batch of queries, the trainer:

1. Generates responses from the current policy with sampling (`batch_generation`).
2. Computes the policy's log-prob of each *sampled* token (`selective_log_softmax` — it indexes the log-softmax at the actually-sampled token id, avoiding a full `[batch, seq, vocab]` softmax materialization where possible).
3. Runs the same tokens through the **reference model** to get reference log-probs at the same positions.
4. Truncates each response at the first stop token (usually EOS); anything after is treated as padding.

```python
query_response, logitss = batch_generation(unwrapped_model.policy, queries, ...)
response = query_response[:, context_length:]
logprob = selective_log_softmax(logitss, response)

ref_output = forward(ref_policy, query_response, pad_token_id)
ref_logprob = selective_log_softmax(ref_output.logits[:, context_length - 1 : -1], response)

postprocessed_response = truncate_response(stop_token_id, pad_token_id, response)
```

Both `logprob` and `ref_logprob` are `[batch, response_length]` — one log-probability per generated token, not one per sequence. This per-token granularity is exactly what the KL penalty in the next section needs.

## 4. From a reward-model score to a per-token reward

The reward model only scores a *complete* response — it returns one scalar per sequence, not one per token:

```python
_, score, _ = get_reward(reward_model, postprocessed_query_response, pad_token_id, context_length)
```

The value model, by contrast, is evaluated once per token (it needs $V(s_t)$ at every position for GAE):

```python
full_value, _, _ = get_reward(unwrapped_value_model, query_response, pad_token_id, context_length)
value = full_value[:, context_length - 1 : -1].squeeze(-1)  # [batch, response_length]
```

To turn the single scalar `score` into something GAE can consume, PPO builds a **dense, per-token reward signal**: a KL penalty at every generated token, plus the reward-model's scalar dropped in at the last valid token position.

```python
# Formula from http://joschu.net/blog/kl-approx.html for the k1/k3 estimators
logr = ref_logprobs - logprobs
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # k3
non_score_reward = -args.kl_coef * kl          # [batch, response_length], dense
rewards = non_score_reward.clone()

actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[actual_start, actual_end] += scores    # sparse: reward-model score at the last token only
```

The intuition: the reward model can only judge the *finished* response, so its signal is sparse (one value, at the end). But you want the policy discouraged from drifting off-distribution at *every* step, not just penalized at the end — so the KL term is added densely, token by token. This "dense KL + sparse terminal reward" pattern is the standard shape of the reward stream in RLHF PPO, and it's exactly what feeds into GAE next.

`missing_eos_penalty` (subtracted from `scores` before this step, if set) is a blunt but effective way to discourage responses that ran off the end of `max_new_tokens` without ever emitting an EOS token.

Optionally, the whole token-reward stream can be variance-normalized (mean preserved, since `shift_mean=False`) before advantages are computed:

```python
if args.whiten_rewards:
    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)
```

## 5. GAE: the backward recursion, implemented

This is the part you asked about specifically. Recall from {doc}`../ppo_advantage` §8 that the infinite-horizon GAE series

$$
\hat A_t = \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \cdots
$$

collapses into the one-step recursion

$$
\hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}, \qquad \hat A_T = 0.
$$

Here is that recursion as code, unmodified from the trainer:

```python
lastgaelam = 0
advantages_reversed = []
gen_length = responses.shape[1]
for t in reversed(range(gen_length)):
    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], axis=1)
returns = advantages + values
```

Mapping code to math, term by term:

- `delta` is exactly the TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
- `lastgaelam` is $\hat A_{t+1}$ carried forward from the *previous* loop iteration (previous in code order, but later in time) — it plays the role of the recursive term on the right-hand side.
- The loop runs `for t in reversed(range(gen_length))`: **you cannot compute $\hat A_t$ before you know $\hat A_{t+1}$**, so the only order that works is last-token-first. `lastgaelam` is the running accumulator that makes each step $O(1)$ instead of re-summing the whole future geometric series from scratch — this is why it's the standard "backward pass" every PPO implementation shares, from the original OpenAI baselines code through trlx to TRL.
- The boundary condition $\hat A_T = 0$ shows up as `nextvalues = ... else 0.0`: at the very last generated token there is no "next state" to bootstrap from, so the recursion is seeded with zero, matching the terminal condition in the theory chapter.
- `advantages_reversed.append(...)` builds the list from $t=T-1$ down to $t=0$; `advantages_reversed[::-1]` flips it back into chronological order before stacking into a tensor.
- `returns = advantages + values` recovers the value-function regression target ($\hat A_t = G_t^{\text{est}} - V(s_t) \Rightarrow G_t^{\text{est}} = \hat A_t + V(s_t)$) — this is what the value loss in §7 regresses toward, not the raw reward.

One subtlety: this all happens on a `[batch, gen_length]` tensor, so the "backward loop" is over **time**, not over the batch — every sequence in the batch gets its GAE computed in parallel via vectorized tensor ops on `values[:, t]` and `rewards[:, t]`. Sequences shorter than `gen_length` are handled uniformly because `values` was already zero-filled at padded positions (via `padding_mask_p1`) before this loop runs; the resulting advantages at those positions get explicitly masked to zero afterward anyway.

## 6. Making it concrete: backward recursion vs. the naive sum

To see that the backward loop really does compute the same thing as the infinite (here, finite-horizon) series, here's a small standalone script — no TRL, no torch, just Python floats for one sequence:

```python
gamma, lam = 1.0, 0.95
rewards = [0.0, 0.0, 0.0, 1.0]        # dense KL terms all 0 here for simplicity; terminal reward = 1.0
values  = [0.3, 0.5, 0.4, 0.2]        # critic's value estimate at each token
T = len(rewards)

# TD residuals, with V(s_T) = 0 (no next state after the last token)
deltas = []
for t in range(T):
    next_v = values[t + 1] if t < T - 1 else 0.0
    deltas.append(rewards[t] + gamma * next_v - values[t])

# --- naive: direct series sum, O(T^2) ---
naive_advantages = []
for t in range(T):
    a_t = sum((gamma * lam) ** l * deltas[t + l] for l in range(T - t))
    naive_advantages.append(a_t)

# --- backward recursion, O(T), exactly the TRL implementation ---
lastgaelam = 0.0
backward_advantages_reversed = []
for t in reversed(range(T)):
    lastgaelam = deltas[t] + gamma * lam * lastgaelam
    backward_advantages_reversed.append(lastgaelam)
backward_advantages = backward_advantages_reversed[::-1]

print("naive:   ", [round(a, 6) for a in naive_advantages])
print("backward:", [round(a, 6) for a in backward_advantages])
# naive:    [0.653397, 0.373575, 0.62, 0.8]
# backward: [0.653397, 0.373575, 0.62, 0.8]
```

Both loops produce identical numbers, but the naive version does $O(T^2)$ work (a nested sum per timestep) while the backward recursion does $O(T)$ — one multiply-add per timestep, because `lastgaelam` already *is* the tail sum from the previous step. That efficiency argument is the entire reason the "backward pass" exists: it is not backpropagation, it's just the standard trick for evaluating a linear recurrence in linear time instead of re-deriving each term from scratch.

## 7. Final advantage whitening

After GAE, advantages get one more normalization pass — mean-centered and scaled to unit variance (this time *with* mean shifted to zero, unlike the optional reward whitening in §4), masked so padded positions stay exactly zero:

```python
advantages = masked_whiten(advantages, ~padding_mask)
advantages = torch.masked_fill(advantages, padding_mask, 0)
```

This is a pure variance-reduction trick on top of GAE — it doesn't change the recursion, just rescales its output across the batch before it's used in the policy loss.

## 8. The clipped policy loss, implemented

This is the direct implementation of $L^{\text{CLIP}}$ from {doc}`../ppo` §7:

```python
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)                          # r_t(theta)
pg_losses = -mb_advantage * ratio                          # -[r_t * A_t]
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
pg_loss_max = torch.max(pg_losses, pg_losses2)              # note: max, not min
pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
```

Two things worth making explicit:

- `new_logprobs` comes from re-running the **current** policy on the stored rollout tokens (`mb_query_responses`); `mb_logprobs` is the log-prob stored at rollout time (i.e. $\pi_{\theta_{\text{old}}}$). Their difference exponentiated is exactly $r_t(\theta) = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$ from the theory chapter.
- The code takes `torch.max` of *negated* terms, while the theory chapter takes `min` of the (positive) objective. These are the same thing: $-\min(x, y) = \max(-x, -y)$. Since PPO trains by minimizing a loss (not maximizing an objective), every sign in the implementation is flipped relative to the paper's $L^{\text{CLIP}}$, which is a common source of confusion when reading trainer code against the math for the first time.

## 9. The clipped value loss

The original PPO paper's headline equation only clips the policy ratio, but essentially every practical PPO implementation (including this one) also clips the *value* update, mirroring the same trust-region idea for the critic:

```python
vpredclipped = torch.clamp(vpred, mb_values - args.cliprange_value, mb_values + args.cliprange_value)
vf_losses1 = torch.square(vpred - mb_return)
vf_losses2 = torch.square(vpredclipped - mb_return)
vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~padding_mask_p1[micro_batch_inds])
vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds])
```

`mb_values` are the value estimates *at rollout time* (before this update); `vpred` is the value model's current prediction after however many gradient steps have already happened in this PPO epoch. Clipping `vpred` to stay within `cliprange_value` of `mb_values` prevents the critic from moving too far in one update, just like `cliprange` does for the policy — again taking the `max` (of squared errors) so the more pessimistic estimate wins.

## 10. Combining losses, and the epoch/minibatch/microbatch nesting

```python
loss = pg_loss + args.vf_coef * vf_loss
accelerator.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

A single joint loss and a single optimizer step update both the policy and the value model together (they were packaged into one `PolicyAndValueWrapper`, remember). The surrounding loop structure is:

```python
for ppo_epoch_idx in range(args.num_ppo_epochs):        # reuse the same rollout batch, μ times
    b_inds = np.random.permutation(args.local_batch_size)  # fresh shuffle each epoch
    for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
            ...  # forward, loss, backward, optimizer.step()
```

`num_ppo_epochs` (default `4`) is the $\mu$ from the DAPO-style objective in {ref}`grpo-family-tutorial`'s §2.1 — how many gradient-update passes get squeezed out of one batch of rollouts before generating fresh ones. Every pass reshuffles which examples land in which minibatch, but always over the *same* rollout data collected once per outer `update` step. This is also exactly why the clipping matters: by epoch 3 or 4, $\pi_\theta$ has already drifted from the $\pi_{\theta_{\text{old}}}$ that generated the data, so the ratio $r_t(\theta)$ is no longer pinned at 1 and clipping is doing real work, not just sitting there as a formality.

## 11. What to monitor

The trainer logs a specific, well-chosen set of diagnostics every `update` step:

| Metric | What it's computing |
| --- | --- |
| `objective/kl` | $\sum_t \text{kl}_t$ per sequence, averaged over the batch — the actual KL term being penalized |
| `objective/non_score_reward` | The dense KL-penalty contribution to reward, summed per sequence |
| `objective/scores` | Mean raw reward-model score (before KL penalty) |
| `objective/rlhf_reward` | `non_score_reward + scores` — the combined signal actually being optimized |
| `policy/approxkl_avg` | $\tfrac12(\Delta\log\pi)^2$ — a cheap second-order KL approximation used only for monitoring policy drift within an epoch, *not* the same estimator used in the reward |
| `policy/clipfrac_avg` | Fraction of tokens where the clipped term was binding (`pg_losses2 > pg_losses`) |
| `val/clipfrac_avg` | Same idea, for the value loss |
| `val/ratio`, `val/ratio_var` | Mean/variance of $r_t(\theta)$ — should hover near 1 early in an epoch and drift as `num_ppo_epochs` progresses |
| `policy/entropy_avg` | Exact response-token entropy (via `logsumexp` over the logits) |

If `policy/clipfrac_avg` is consistently near 0, clipping isn't doing anything and you could likely raise `cliprange` or `num_ppo_epochs`. If it's very high alongside a growing `val/ratio_var`, the policy is moving too fast between epochs — lower the learning rate or `num_ppo_epochs`.

## 12. Config cheat sheet

| `PPOConfig` field | Default | What it controls |
| --- | --- | --- |
| `gamma` | `1.0` | Discount factor in $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$; `1.0` is common for short LM responses where there's no reason to discount within a single completion |
| `lam` | `0.95` | $\lambda$ in the GAE recursion — bias/variance tradeoff, see {doc}`../ppo_advantage` §7 |
| `kl_coef` | `0.05` | $\beta$-equivalent: weight of the dense per-token KL penalty added into the reward stream |
| `kl_estimator` | `"k1"` | `"k1"` (simple, unbiased) or `"k3"` (lower-variance, unbiased) KL estimator, per [Schulman's approximation note](http://joschu.net/blog/kl-approx.html) |
| `cliprange` | `0.2` | $\epsilon$ for the policy ratio clip |
| `cliprange_value` | `0.2` | Clip range for the value prediction, independent of `cliprange` |
| `vf_coef` | `0.1` | Weight of the value loss in the combined `loss = pg_loss + vf_coef * vf_loss` |
| `whiten_rewards` | `False` | Whether to variance-normalize the token-reward stream before GAE (§4) |
| `num_ppo_epochs` | `4` | How many optimizer passes ($\mu$) over one rollout batch before generating fresh rollouts |
| `missing_eos_penalty` | `None` | Score penalty applied if a response never emits EOS within `response_length` |
| `response_length` | `53` | Max generated tokens per response |

## 13. Minimal end-to-end example

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.ppo import PPOConfig, PPOTrainer

base_model = "EleutherAI/pythia-160m"

tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(base_model)
ref_policy = AutoModelForCausalLM.from_pretrained(base_model)
value_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
reward_model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)

def tokenize(example):
    return tokenizer(example["prompt"], truncation=True, max_length=128)

dataset = load_dataset("trl-lib/tldr", split="train").map(tokenize, remove_columns=["prompt"])

config = PPOConfig(
    output_dir="ppo_pythia_160m",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_ppo_epochs=4,
    response_length=53,
    kl_coef=0.05,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    missing_eos_penalty=1.0,
)

trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)

trainer.train()
```

In practice `reward_model` is a checkpoint you've already trained with `RewardTrainer`, not a randomly-initialized head — the toy example above is for tracing the mechanics, not for producing a useful policy.

## 14. Why TRL moved beyond this: PPO vs. GRPO

Everything in §5–§9 exists to solve one problem: **you need a critic to get a low-variance advantage estimate without a full Monte-Carlo rollout of the value function**. That critic is a second trainable model with its own loss (§9), its own clipping range, its own hyperparameters (`vf_coef`, `cliprange_value`) — and its own failure modes if it's miscalibrated, since a bad $V(s_t)$ silently corrupts every downstream advantage through the GAE recursion.

GRPO's entire pitch (see {ref}`grpo-family-tutorial` §2) is to delete that whole apparatus: sample a *group* of completions per prompt, and use the group's own reward mean/std as the baseline instead of a learned $V(s_t)$. No value model, no GAE backward loop, no value-loss clipping — at the cost of needing several completions per prompt instead of one. If you found §5–§9 here to be the most complex and fragile part of the trainer (most people do), that's precisely the part GRPO was designed to remove.

## 15. Quick mental model

- Four models: policy (trained), value model (trained), reference model (frozen, for KL), reward model (frozen, scores full completions).
- The reward model gives one number per sequence; the KL penalty gives one number per token. They're combined into a dense per-token reward stream *before* GAE ever runs.
- GAE's backward loop is just an efficient way to evaluate the linear recurrence $\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}$ — iterate from the last token to the first, carry `lastgaelam` forward as the accumulator, seed it with $\hat A_T = 0$.
- `advantages + values = returns`, and `returns` is the regression target for the value loss — not the raw reward.
- The policy loss clips the probability ratio; the value loss separately clips the value prediction — same trust-region idea, applied twice, with independent config knobs.
- `num_ppo_epochs` reuses one batch of rollouts for multiple gradient updates, which is exactly why the ratio-based clipping is load-bearing rather than cosmetic.
