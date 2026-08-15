# TRL Overview

The TRL section collects practical, implementation-focused examples for post-training workflows. It complements the theory chapters with runnable code paths and highlights how to wire reward models, optimizers, and data for real training loops.

In this subsection you will find:

- **PPO trainer deep dive** — how `PPOTrainer`'s actor-critic loop implements GAE (including the backward recursion), the KL-penalized reward, and clipped policy/value losses.
- **DPO trainer deep dive** — how `DPOTrainer` turns preference pairs into the DPO loss: the concatenated chosen/rejected forward pass, reference-model strategies (including the LoRA adapter trick), and the implicit-reward metrics.
- **Standard RL with GRPO** — a practical walkthrough of `GRPOTrainer` for the single-turn (non-agentic) setting: config, reward functions, and rollout/update mechanics.
- **Multi-task reward training with GRPO** — mixing multiple tasks and reward functions in one run.
- **Agentic RL with GRPO** using TRL as an end-to-end example, with tool-calling, multi-turn rollouts, and reward design for agent behavior.

## Source-code map for post-training

The theory chapters explain *what* SFT, reward modeling, DPO, GRPO, and distillation optimize. TRL's trainer directory shows *where those objectives live in code*. As of 2026-08-14, the main trainer source tree is:

- [`trl/trainer/__init__.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/__init__.py): lazy exports for the public trainer classes.
- [`trl/trainer/base_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/base_config.py): `_BaseConfig`, a TRL-flavored subclass of `transformers.TrainingArguments`.
- [`trl/trainer/base_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/base_trainer.py): `_BaseTrainer`, a thin shared layer over `transformers.Trainer` for telemetry, tags, and model cards.
- `*_config.py`: dataclass config for one training method, for example `SFTConfig`, `DPOConfig`, `GRPOConfig`.
- `*_trainer.py`: the actual trainer implementation, for example `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`.
- [`callbacks.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/callbacks.py): shared callbacks such as `SyncRefModelCallback` and `LogCompletionsCallback`.
- [`model_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/model_config.py): CLI/model-loading config, especially useful in example scripts.
- [`utils.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py): shared helpers such as PEFT setup, quantization setup, dropout disabling, and distributed utilities.

The recurring design pattern is:

```text
algorithm idea
  -> Config dataclass: expose hyperparameters
  -> Trainer.__init__: load policy/reference/reward/teacher models and choose collator
  -> _prepare_dataset / data collator: normalize raw dataset into tensors and masks
  -> _prepare_inputs: for online methods, generate completions before loss
  -> compute_loss / _compute_loss: implement the actual objective
  -> log / callbacks: expose the diagnostics that reveal failure modes
```

So when reading any TRL trainer, do not start from every utility function. Start with the config fields, then follow the batch from dataset columns into the collator, then into `compute_loss`.

## 1. The trainer taxonomy

| Trainer | Post-training role | Offline or online? | Dataset shape | Extra models | Core objective |
| --- | --- | --- | --- | --- | --- |
| `SFTTrainer` | Teach target behavior by imitation | Offline | `prompt` + `completion`, or chat messages, or already-tokenized text | None | supervised next-token NLL on selected tokens |
| `RewardTrainer` | Train a reward model for RLHF/PPO-style pipelines | Offline | `chosen` / `rejected`, optionally with `prompt` and `margin` | None beyond the reward model itself | Bradley-Terry pairwise ranking loss |
| `DPOTrainer` | Preference optimization without reward-model training or rollout | Offline | `prompt`, `chosen`, `rejected` | frozen reference policy, or PEFT reference adapter/base model | pairwise log-ratio margin loss |
| `KTOTrainer` | Preference optimization from unpaired desirable/undesirable samples | Offline | `prompt`, `completion`, `label` | frozen reference policy | KTO loss plus KL-style reference term |
| `RLOOTrainer` | Critic-free online RL with a leave-one-out baseline | Online | `prompt`; reward columns passed through | reward function/model, optional reference policy | policy-gradient update using sampled completions and leave-one-out advantages |
| `GRPOTrainer` | Critic-free online RLVR / RLHF with group-relative advantages | Online | `prompt`; reward columns passed through | reward function/model, optional reference policy | PPO-style clipped loss with group-relative advantages |
| `DistillationTrainer` | On-policy knowledge distillation from a teacher model | Online | `prompt` | teacher causal LM | token-distribution matching by generalized JSD/KL |

```{note}
`PPOTrainer` is still important conceptually, and this book has a dedicated PPO deep dive. In recent TRL layouts, however, the legacy PPO implementation is outside the main `trl/trainer` directory, under an experimental PPO path. That is why the table above focuses on the trainers exported from `trl/trainer/__init__.py`.
```

## 2. Shared skeleton: what every trainer inherits

Most TRL trainers are specialized versions of Hugging Face `Trainer`, not completely separate training engines.

| File | Class / module | Why it matters |
| --- | --- | --- |
| `base_config.py` | `_BaseConfig(TrainingArguments)` | Gives all TRL configs the normal `Trainer` knobs: batch size, gradient accumulation, optimizer, scheduler, logging, checkpointing, FSDP/DeepSpeed, etc. |
| `base_trainer.py` | `_BaseTrainer(Trainer)` | Adds TRL tags, telemetry, model-card generation, and a common base class for method-specific trainers. |
| `callbacks.py` | `SyncRefModelCallback`, `LogCompletionsCallback`, `RichProgressCallback`, etc. | Handles cross-cutting behavior: reference-model synchronization, sampled-completion logging, progress display. |
| `utils.py` | `get_peft_config`, `get_quantization_config`, `disable_dropout_in_model`, etc. | Keeps PEFT, quantization, and distributed model preparation out of each trainer's core objective code. |

This is the first design principle: **TRL leaves generic training mechanics to `transformers.Trainer`, and each trainer mostly changes data preparation plus loss computation.**

## 3. `SFTTrainer`: supervised imitation as masked language modeling

Required source files:

- [`sft_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py): `SFTConfig`
- [`sft_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py): `SFTTrainer`, `DataCollatorForLanguageModeling`, `DataCollatorForVisionLanguageModeling`, `dft_loss`

What to look for:

- `SFTConfig`: sequence length, packing/padding behavior, loss type, assistant/completion-only masking, Liger/chunked-loss options.
- `SFTTrainer.__init__`: loads tokenizer/processor, decides whether the dataset is text-only or vision-language, picks the collator.
- `_prepare_dataset`: converts raw standard/conversational samples into tokenized fields and applies chat templates when needed.
- `DataCollatorForLanguageModeling`: pads batches, builds labels, and masks tokens that should not contribute to loss.
- `compute_loss`: mostly delegates to the normal causal-LM loss, with optional TRL-specific loss paths such as dynamic fine-tuning loss.

Connection to the SFT chapter:

```text
raw instruction example
  -> chat/template formatting
  -> input_ids + attention_mask
  -> labels with prompt tokens set to ignore_index
  -> causal LM cross-entropy on response/assistant tokens
```

SFT is therefore the cleanest trainer to read first. It teaches the core TRL pattern: dataset normalization, masking, then `Trainer`-style loss.

## 4. `RewardTrainer`: learning the scalar judge

Required source files:

- [`reward_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/reward_config.py): `RewardConfig`
- [`reward_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/reward_trainer.py): `RewardTrainer`, `DataCollatorForPreference`

What to look for:

- Dataset fields: usually `chosen` and `rejected`; processed datasets use `chosen_ids` and `rejected_ids`. A `prompt` can be prepended during preprocessing.
- `DataCollatorForPreference`: stacks the chosen batch and rejected batch into one tensor of size `2B`.
- `compute_loss`: splits model scores into `rewards_chosen` and `rewards_rejected`, then applies:

$$
\mathcal{L}_{\text{RM}}
=-\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)-m\right)
$$

where `margin` is optional. `center_rewards_coefficient` can additionally push reward outputs toward a mean-zero scale.

Connection to RLHF:

`RewardTrainer` trains the reward model used by classic PPO-style RLHF. It is not needed for DPO, where the policy/reference log-ratio creates an implicit reward, and it is often replaced by verifiers or callable reward functions in GRPO/RLVR.

## 5. `DPOTrainer`: offline preference optimization

Required source files:

- [`dpo_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_config.py): `DPOConfig`
- [`dpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py): `DPOTrainer`, `DataCollatorForPreference`, `DataCollatorForVisionPreference`

What to look for:

- `DPOConfig.beta`: scale on the log-ratio margin.
- `DPOConfig.loss_type`: supports the default DPO sigmoid loss plus variants such as IPO, hinge, robust, EXO pair, AOT, APO, DiscoPOP, and an SFT auxiliary term.
- Reference strategy in `DPOTrainer.__init__`: explicit `ref_model`, cloned frozen model, precomputed reference log-probs, or PEFT adapter/base-model reference.
- `DataCollatorForPreference`: builds a batch whose first half is prompt+chosen and second half is prompt+rejected, with a `completion_mask`.
- `_compute_loss`: obtains policy log-probs, obtains or reads reference log-probs, computes chosen/rejected log-ratios, then applies the selected pairwise loss.

Connection to the DPO chapter:

```text
policy chosen logp     policy rejected logp
reference chosen logp  reference rejected logp
        -> chosen log-ratio - rejected log-ratio
        -> beta-scaled pairwise loss
        -> implicit rewards for logging
```

The important code-level distinction from PPO/GRPO: **`DPOTrainer` does not sample completions during training**. It is offline and teacher-forced, like SFT, but the loss is pairwise and reference-regularized.

## 6. `KTOTrainer`: unpaired preference data

Required source files:

- [`kto_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/kto_config.py): `KTOConfig`
- [`kto_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py): `KTOTrainer`, `DataCollatorForUnpairedPreference`, `DataCollatorForVisionUnpairedPreference`

What to look for:

- Dataset fields: `prompt`, `completion`, and `label`, where `label=True` means desirable and `label=False` means undesirable.
- `_prepare_dataset`: can convert paired `chosen`/`rejected` data into unpaired preference examples.
- Collators: build `completion_mask` for the actual sample and, when needed, `KL_*` fields for mismatched completions used to estimate the KL term.
- Reference-model logic: similar to DPO, including PEFT reference handling and optional precomputed reference log-probs.

Design principle:

KTO exists for data regimes where you have thumbs-up/thumbs-down examples rather than clean pairwise comparisons for the same prompt. It still needs a reference policy because the method is about moving desirable examples up and undesirable examples down while controlling drift from the starting model.

## 7. `RLOOTrainer`: online RL without a critic

Required source files:

- [`rloo_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_config.py): `RLOOConfig`
- [`rloo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/rloo_trainer.py): `RLOOTrainer`

What to look for:

- Dataset field: `prompt`. Other columns are forwarded to reward functions.
- `reward_funcs`: can be callable functions, async functions, model IDs, or sequence-classification reward models.
- `_prepare_inputs`: generates completions online and buffers them across gradient accumulation / multiple iterations.
- `_calculate_rewards`: scores each sampled completion with one or more reward sources.
- Advantage computation: for each prompt, sample multiple completions and compare each completion against the reward mean of the *other* completions, hence leave-one-out.
- `compute_loss`: uses per-token log-probs, optional KL against a reference model, and policy-gradient style weighting.

Connection to GRPO:

RLOO and GRPO are siblings. Both remove PPO's learned value model. RLOO uses a leave-one-out baseline; GRPO uses group-relative normalization. If PPO is "actor plus critic", these trainers are "actor plus sampled peer baseline".

## 8. `GRPOTrainer`: group-relative online RL

Required source files:

- [`grpo_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py): `GRPOConfig`
- [`grpo_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py): `GRPOTrainer`

What to look for:

- `GRPOConfig.num_generations`: group size per prompt.
- `GRPOConfig.reward_weights`, `reward_aggregation`, `scale_rewards`: how multiple rewards become one advantage signal.
- `GRPOConfig.loss_type`: the GRPO-family loss variant, including modern normalization/clipping variants such as DAPO/Dr. GRPO-style behavior.
- `GRPOConfig.beta`: optional reference-model KL coefficient. `beta=0.0` avoids keeping a reference model.
- vLLM / continuous batching fields: generation is the bottleneck, so TRL puts a lot of engineering around fast rollouts.
- `_generate_and_score_completions`: the central online-RL data path: prompt batch -> generated completions -> reward matrix -> grouped advantages -> tensors for loss.
- `_compute_loss`: per-token log-probs, old log-probs for importance ratios, optional reference KL, clipping, masking, and loss normalization.
- Tool/environment paths: current `GRPOTrainer` also has hooks for tools and environments, which is why it can support simple agentic rollouts in addition to single-turn RLVR.

Connection to the GRPO chapter:

```text
one prompt
  -> G sampled completions
  -> rewards R_1 ... R_G
  -> group baseline / normalization
  -> one advantage per completion
  -> advantage broadcast over completion tokens
  -> PPO-style clipped policy update
```

The key design choice is that GRPO replaces PPO's critic with **within-prompt comparison**. This makes the trainer easier to scale for RLVR, but also makes group construction, reward variance, completion length, truncation masks, and rollout freshness much more important.

## 9. `DistillationTrainer`: on-policy teacher matching

Required source files:

- [`distillation_config.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/distillation_config.py): `DistillationConfig`
- [`distillation_trainer.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/distillation_trainer.py): `DistillationTrainer`

What to look for:

- `teacher_model_name_or_path` / `teacher_model`: how the teacher is loaded.
- `temperature`: softens teacher/student token distributions before divergence computation.
- `beta`: here it is not a KL penalty against a reference model. It selects the divergence family: `0.0` behaves like forward KL, `1.0` like reverse KL, intermediate values give generalized JSD.
- `_generate_and_score_completions`: name is shared with online RL trainers, but there is no reward scoring. The student generates completions from prompts.
- `_prepare_inputs`: buffers generated prompt/completion tensors across gradient accumulation.
- `_compute_loss`: runs student and teacher on the same generated tokens, obtains their hidden states/logit heads, and computes a chunked divergence loss over completion tokens.

Connection to the distillation chapter:

```text
student samples y ~ pi_student(. | x)
teacher gives soft next-token distribution over the same trajectory
student minimizes divergence to teacher distribution on generated completion tokens
```

This is different from SFT-style distillation on teacher-written answers. The student is trained on its own mistakes, and the teacher supplies the distributional correction.

## 10. How to choose the right trainer

| If you have... | Use... | Reason |
| --- | --- | --- |
| High-quality prompt/answer demonstrations | `SFTTrainer` | Direct imitation is the simplest and most stable first stage. |
| Chosen/rejected pairs and you want a scalar reward model | `RewardTrainer` | Produces a reward model for PPO/RLHF-style pipelines. |
| Chosen/rejected pairs and you want to update the policy directly | `DPOTrainer` | Avoids reward-model training and online rollouts. |
| Unpaired desirable/undesirable examples | `KTOTrainer` | Does not require paired responses for the same prompt. |
| Prompt-only tasks with callable/verifier rewards | `GRPOTrainer` | Best default for RLVR-style math/code/verifier training. |
| Prompt-only tasks with multiple samples and a leave-one-out baseline | `RLOOTrainer` | A critic-free online RL alternative close in spirit to GRPO. |
| A larger teacher model and a smaller student | `DistillationTrainer` | Matches the teacher's soft distribution on student-generated completions. |

## 11. A practical reading order

If you are connecting this source tree back to the earlier theory chapters, read in this order:

1. `SFTTrainer`: understand masking, collation, and causal-LM loss.
2. `RewardTrainer`: understand pairwise chosen/rejected batching.
3. `DPOTrainer`: reuse pairwise batching, replace reward-model scores with policy/reference log-ratios.
4. `GRPOTrainer`: switch from offline pairs to online generation and reward functions.
5. `RLOOTrainer`: compare its leave-one-out baseline to GRPO's group baseline.
6. `DistillationTrainer`: compare online generation with GRPO, but replace scalar reward with teacher distribution matching.
7. `KTOTrainer`: return to offline preference optimization, but remove the requirement that examples arrive as pairs.

That reading path makes the code feel much less sprawling: most trainers are variations on three questions:

- Where do the completions come from: dataset, policy rollout, or teacher/student interaction?
- What produces the training signal: labels, pairwise preference, scalar reward, group baseline, or teacher distribution?
- Which tokens are allowed to receive loss: all completion tokens, assistant-only tokens, pairwise completion masks, or generated rollout masks?
