# DPO Trainer Deep Dive: Preference Pairs, the Concatenated Forward, and the Implicit Reward in Practice

The theory chapter [DPO: Preference Optimization Without a Reward Model or RL Loop](../dpo.md) derives the DPO loss on paper. This tutorial closes the gap between that math and a real, running trainer: TRL's `DPOTrainer`. We walk through the data path — preference-pair tokenization, the **concatenated chosen+rejected forward pass**, sequence log-probabilities, the three ways the reference model can be provided — and then map the `dpo_loss` code line by line back to the boxed equation in the theory chapter.

The single most important structural fact, worth stating before any code: **`DPOTrainer` contains no generation loop.** Where [`PPOTrainer`](trl_ppo_trainer.md) spends most of its complexity on rollouts, rewards, and GAE, `DPOTrainer` is a subclass of the standard supervised `Trainer` whose only real customizations are a pairwise data collator, a bespoke forward pass, and a custom loss. This is §6.1 of the theory chapter ("RL objective, SFT mechanics") made concrete.

Reference implementation (read at TRL `v0.28.0`, same tag as the PPO tutorial):

- https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/dpo_trainer.py
- https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/dpo_config.py

## 1. Two models, not four

| Model | Role | Trainable? |
| --- | --- | --- |
| `model` (policy) | The model being aligned; produces $\log\pi_\theta$ | Yes |
| `ref_model` | Frozen anchor; produces $\log\pi_{\text{ref}}$ in the implicit reward | No |

Compare with `PPOTrainer`'s four models: the reward model is gone (the policy *is* the implicit reward model), and the value model is gone (there are no advantages to estimate — nothing is sampled, so nothing needs credit assignment).

If you pass `ref_model=None`, the trainer picks one of three strategies:

```python
if ref_model:
    self.ref_model = ref_model
elif self.is_peft_model or args.precompute_ref_log_probs:
    # The `model` with adapters turned off will be used as the reference model
    self.ref_model = None
else:
    self.ref_model = create_reference_model(model)
```

- **Explicit `ref_model`**: a separate frozen copy, held in memory, forward-passed every batch.
- **PEFT/LoRA trick**: when training with adapters, the base weights *are* the reference policy — so instead of a second model, the trainer just disables the adapter for the reference forward pass (`null_ref_context` below). Reference memory cost: zero.
- **`create_reference_model(model)`**: the fallback — clone the initial policy and freeze it.

The adapter trick in full:

```python
@contextmanager
def null_ref_context(self):
    with (
        self.accelerator.unwrap_model(self.model).disable_adapter()
        if self.is_peft_model and not self.ref_adapter_name
        else nullcontext()
    ):
        yield
```

There is also `precompute_ref_log_probs=True`: since the reference model is frozen *and the dataset is fixed* (no generation!), all reference log-probs can be computed once up front, cached as dataset columns, and the reference model discarded entirely before training starts. This option is only possible because DPO is offline — PPO could never precompute anything about data that hasn't been generated yet.

## 2. Dataset format: preference pairs

`DPOTrainer` expects three columns — `prompt`, `chosen`, `rejected` — as raw text (standard format) or as lists of chat messages (conversational format, templated internally):

```python
# standard
{"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}

# conversational
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "chosen": [{"role": "assistant", "content": "It is blue."}],
 "rejected": [{"role": "assistant", "content": "It is green."}]}
```

Tokenization keeps the three pieces separate (`tokenize_row`), so the prompt is stored once per pair:

```python
prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]
...
chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]
```

Note the contrast with `PPOTrainer`, which wants a pre-tokenized prompt-only dataset — there, responses don't exist until the policy generates them. Here the responses are the dataset.

## 3. The concatenated forward pass

The loss needs sequence log-probs for the chosen *and* rejected completion, from the policy *and* the reference — four quantities per pair. To get the policy-side two in **one** forward pass, `concatenated_inputs` stacks chosen and rejected along the batch dimension (docstring: *"We do this to avoid doing two forward passes, because it's faster for FSDP"*):

```python
# prompt is duplicated: same prompt for both completions
output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)

# completions are padded to a common length, then stacked: chosen first, rejected second
max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
output["completion_input_ids"] = torch.cat(
    (
        pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
        pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
    ),
)
```

A batch of $B$ preference pairs becomes a tensor of $2B$ sequences; rows `[:B]` are chosen, rows `[B:]` are rejected. `concatenated_forward` then glues prompt and completion into one sequence per row and builds a **loss mask** that zeroes out the prompt — the same prompt-masking idea as SFT ({doc}`../sft` §4), for the same reason:

```python
input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
# Mask the prompt but not the completion for the loss
loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)
```

## 4. From logits to one number per response

The DPO loss consumes the **sequence-level** log-probability $\log\pi(y\mid x)=\sum_t \log\pi(y_t\mid x,y_{<t})$. The implementation: index the log-softmax at each realized token (`selective_log_softmax`, the same helper `PPOTrainer` uses), zero the masked positions, and sum over time:

```python
labels = torch.roll(input_ids, shifts=-1, dims=1)      # next-token targets
loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

per_token_logps = selective_log_softmax(logits, labels)
per_token_logps[~loss_mask] = 0
...
all_logps = per_token_logps[:, 1:].sum(-1)             # [2B] — one scalar per sequence

output["chosen_logps"] = all_logps[:num_examples]      # first half of the stack
output["rejected_logps"] = all_logps[num_examples:]    # second half
```

Two details worth noticing:

- The `torch.roll` on `input_ids` is the standard shift-by-one alignment between position-$t$ logits and the token at position $t{+}1$ — the same causal-LM bookkeeping as SFT, just done manually because the loss isn't `CrossEntropyLoss`.
- The **sum** (not mean) over tokens matters: the theory's $\log\pi(y\mid x)$ is a sum, so longer responses have more terms and typically lower (more negative) log-probs. This is one root of DPO's length bias (§7.2 of the theory chapter). The IPO variant divides by length here — you can see it as a special case in the code: `if "ipo" in self.loss_type: all_logps = all_logps / loss_mask.sum(-1)`.

The reference model runs through the *same* `concatenated_forward` under `torch.no_grad()` (`compute_ref_log_probs`), producing `ref_chosen_logps` and `ref_rejected_logps`. Four numbers per pair, as required.

## 5. `dpo_loss`: the boxed equation, implemented

Recall the loss from the theory chapter:

$$
\mathcal{L}_{\text{DPO}}
=-\log\sigma\!\Big(
\underbrace{\beta\big(\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}
-\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)}_{\beta\,\cdot\,\texttt{logits}}
\Big).
$$

The default (`loss_type="sigmoid"`) path in `dpo_loss`:

```python
logratios = chosen_logps - rejected_logps               # log πθ(y_w) − log πθ(y_l)
ref_logratios = ref_chosen_logps - ref_rejected_logps   # log πref(y_w) − log πref(y_l)
logits = logratios - ref_logratios                      # the implicit-reward margin, pre-β

losses = (
    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
)
```

Mapping code to math:

- `logits` is the margin $\big(\log\pi_\theta(y_w)-\log\pi_\theta(y_l)\big)-\big(\log\pi_{\text{ref}}(y_w)-\log\pi_{\text{ref}}(y_l)\big)$ — algebraically identical to the difference of the two log-ratios in the equation, just grouped by model instead of by response. Grouping this way makes the "reference as anchor" reading direct: the loss asks whether the *policy's* preference gap has grown relative to the *reference's* gap.
- `F.logsigmoid(self.beta * logits)` is $\log\sigma(\beta\cdot\text{margin})$; the leading minus makes it a loss.
- With `label_smoothing=0` (the default) the second term vanishes and this is exactly $-\log\sigma(\beta z)$. With $\epsilon>0$ it becomes the *conservative DPO* loss $-(1-\epsilon)\log\sigma(\beta z)-\epsilon\log\sigma(-\beta z)$ — hedging against the possibility that a fraction $\epsilon$ of preference labels are flipped.

Immediately after, the trainer computes the **implicit rewards** — not used in the loss (note the `.detach()`), purely for logging:

```python
chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
```

This is $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ from the theory chapter, computed per response. Every `rewards/*` metric in the logs is derived from these two tensors.

## 6. The `loss_type` zoo

The same four log-probs feed a long menu of alternative losses — the variant papers from §7.3 of the theory chapter are mostly one-branch additions to this `if/elif` chain. A sampler:

| `loss_type` | Loss on the margin $z=$ `logits` | Idea |
| --- | --- | --- |
| `"sigmoid"` (default) | $-\log\sigma(\beta z)$ | Original DPO |
| `"ipo"` | $(z-\tfrac{1}{2\beta})^2$ | Squared loss toward a fixed target margin; resists overfitting saturated pairs; per-token-normalized log-probs |
| `"hinge"` | $\max(0,\,1-\beta z)$ | SLiC-style SVM margin |
| `"robust"` | debiased sigmoid loss | Provably unbiased under label-flip noise rate = `label_smoothing` |
| `"apo_zero"` / `"apo_down"` | separate terms per response | Anchors chosen/rejected likelihoods individually instead of only their gap |
| `"sft"` | $-\log\pi_\theta(y_w\mid x)$ | Plain NLL on chosen — for mixing, see below |

`loss_type` accepts a *list* with `loss_weights`, so `loss_type=["sigmoid", "sft"]` trains DPO plus an SFT term on the chosen response (the MPO/RPO recipe; `rpo_alpha` is the legacy spelling). This directly counteracts the "chosen log-prob falls during training" dynamic from §6.3 of the theory chapter by explicitly pulling $\log\pi_\theta(y_w)$ up while the pairwise term handles the margin.

## 7. What to monitor

| Metric | What it's computing | What to look for |
| --- | --- | --- |
| `rewards/accuracies` | Fraction of pairs where implicit reward ranks chosen above rejected | Should climb well above 0.5; plateauing near 0.5 means the model isn't learning the preferences |
| `rewards/margins` | Mean $\hat r_w-\hat r_l$ | Should grow steadily; exploding margins with falling `logps/chosen` = classic DPO overfit |
| `rewards/chosen`, `rewards/rejected` | Mean implicit rewards $\beta\log(\pi_\theta/\pi_{\text{ref}})$ | Both drifting strongly negative means the policy is bleeding probability mass away from *both* responses |
| `logps/chosen` | Raw $\log\pi_\theta(y_w)$ | The single best early-warning signal — if it falls fast, the model is "winning" the margin by trashing the rejected response and leaking mass to neither; consider adding an SFT term or lowering LR |
| `logps/rejected` | Raw $\log\pi_\theta(y_l)$ | Expected to fall; compare its slope against `logps/chosen` |

The failure mode these metrics triangulate is exactly the offline-training pathology from the theory chapter §6.3: the loss only constrains the margin, so nothing stops both log-probs from sinking together.

## 8. Config cheat sheet

| `DPOConfig` field | Default | What it controls |
| --- | --- | --- |
| `beta` | `0.1` | The $\beta$ from the derivation — implicit-reward scale / inverse KL-tolerance. Lower = policy may drift further from reference |
| `loss_type` | `"sigmoid"` | Which branch of §6's zoo; accepts a list plus `loss_weights` for mixtures |
| `label_smoothing` | `0.0` | Assumed preference-label noise rate (conservative/robust DPO) |
| `learning_rate` | `1e-6` | Note how small — 1–2 orders below typical SFT LRs; DPO is famously LR-sensitive |
| `max_length` / `max_prompt_length` | `1024` / `512` | Truncation of the concatenated sequence / the prompt part |
| `precompute_ref_log_probs` | `False` | Compute all reference log-probs once up front; frees the ref model during training |
| `reference_free` | `False` | Replace $\pi_{\text{ref}}$ with uniform (ratios drop out) — mostly for ablations; SimPO-style training uses the separate `CPOTrainer` family |
| `sync_ref_model` | `False` | TR-DPO: every `ref_model_sync_steps` (512) steps, update the reference toward the policy via `π_ref ← α·π_θ + (1−α)·π_ref` (`ref_model_mixup_alpha=0.6`) — a mild step back toward on-policy freshness |
| `rpo_alpha` / `ld_alpha` | `None` | Add-an-SFT-term (RPO) and length-desensitization (LD-DPO) knobs |

## 9. Minimal end-to-end example

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ~62k preference pairs in conversational format (prompt/chosen/rejected)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

config = DPOConfig(
    output_dir="qwen-0.5b-dpo",
    beta=0.1,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=1024,
    logging_steps=10,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,          # trainer clones & freezes the initial model as reference
    args=config,
    processing_class=tokenizer,
    train_dataset=dataset,
)

trainer.train()
```

Swap in `peft_config=LoraConfig(...)` and the reference model costs no extra memory at all (§1's adapter trick). Note what is *absent* compared to the PPO example: no reward model, no value model, no `response_length`, no KL coefficient — and the dataset carries the responses instead of the policy generating them.

## 10. Quick mental model

- Two models: policy (trained) and frozen reference — or just one, with LoRA adapters toggled off for the reference pass.
- No generation anywhere in the trainer: the dataset's chosen/rejected pairs are teacher-forced, exactly like SFT — this is the "SFT mechanics" half of the theory chapter's verdict.
- Chosen and rejected are stacked into one batch of $2B$ rows for a single forward pass; prompt tokens are loss-masked; per-token log-probs are summed into one scalar per response.
- The loss is `-logsigmoid(beta * ((policy margin) − (reference margin)))` — the boxed DPO equation with the ratios regrouped by model.
- `rewards/*` metrics are the implicit reward $\beta\log(\pi_\theta/\pi_{\text{ref}})$, detached — the "secretly a reward model" quantity, logged live.
- Watch `logps/chosen`: the margin-only loss is perfectly happy to sink both responses' probabilities, and that metric is where you catch it doing so.
