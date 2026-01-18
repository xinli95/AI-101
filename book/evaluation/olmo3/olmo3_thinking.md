(olmo3-think)=

# OLMo 3 Think

> This tutorial explains how **OLMo 3 Think** is trained as a reasoning model.

## What is OLMo 3 Think?

**OLMo 3 Think** is a post-trained version of OLMo 3 designed to solve harder reasoning problems by:

- generating an **extended thinking trace** (a “thought trajectory”), and then
- producing a final answer.

The report frames OLMo 3 Think as a reasoning model trained using:

1. high-quality “thinking” data curation (**Dolci Think**), and
2. a three-stage recipe:
   - **Thinking SFT**
   - **Thinking DPO**
   - **Thinking RL** (RLVR with GRPO)

They also introduce **OlmoRL**, an RL training infrastructure designed for long reasoning traces and verifiable rewards.

## Why this pipeline matters

Many modern reasoning-model pipelines rely heavily on RLVR:

- either RL directly from the base model (“RL-Zero”), or
- a small/light SFT warmup, then RLVR

OLMo 3 Think differs in two important ways:

1. They use **both SFT and DPO before RLVR**
2. Their RLVR is **multi-objective**, mixing verifiable and non-verifiable domains

Empirically, the report claims this yields consistent gains across:

- SFT → DPO → RLVR

## Overview: the 3-stage recipe

### Stage 1: Thinking SFT (seed the format + capabilities)

**Goal:** teach the model to output:

- a reasoning trace
- followed by the final answer

This stage also improves skills across:

- math
- science
- coding
- instruction following
- chat + safety

The dataset:

- **[Dolci Think SFT](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B)**
- size: **~2.3M supervised examples**

#### How Dolci Think SFT is curated

The report describes a pipeline that looks like:

1. **Prompt sourcing**

   - gather prompts for each capability from many public datasets

2. **Re-generating completions**

   - for prompts that have incomplete or missing completions
   - generate new completions using a strong reasoning model (e.g., DeepSeek-R1 or QwQ-32B)
   - ensure each completion includes:
     - a reasoning trace
     - a final answer

3. **Correctness filtering**

   - verify outputs with domain-specific checks, e.g.:
     - synthetic test cases for code
     - verifiers for strict-format or precise instruction constraints

4. **Heuristic filtering**

   - remove bad/unsafe/low-quality examples:
     - unclear license
     - incomplete reasoning traces
     - excessive repetition
     - references to other model providers
     - other heuristic signals

5. **Topic filtering**
   - classify prompts by topic (OpenAI query taxonomy)
   - drop or downsample irrelevant topics (e.g., image generation requests)

#### How they choose the final SFT mixture

After data sources are curated and filtered, they select the mixture in a _midtraining-like_ way:

- evaluate many sources in parallel via lightweight SFT probes
- run centralized integration tests using the promising sources

An interesting note from the report:

> every data source helped at least one benchmark

### Stage 2: Thinking DPO (preference tuning before RL)

Preference tuning (DPO) is often viewed as an “alignment” step, but the report emphasizes it can also improve **capabilities** when used before RL in reasoning models.

OLMo 3 Think uses:

- **[Dolci Think DPO](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B)** (preference dataset)
- pairs are constructed using **Delta Learning**

Key dataset sizes:

- 7B: ~150K preference pairs
- 32B: ~200K preference pairs

**Breakdown of datasets in Dolci-Think-RL used for RL training**

| Category      | Prompt Dataset                                                                                          | # Prompts Used in Think RL | # Prompts Used in Instruct RL | Link                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------- | -------------------------: | ----------------------------: | ------------------------------------------------------------------------ |
| Precise IF    | [IF-RLVR](https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5)                           |                     30,186 |                        38,000 | https://github.com/allenai/IFBench                                       |
| Math          | [Open-Reasoner-Zero](https://huggingface.co/Open-Reasoner-Zero)                                         |                      3,000 |                        14,000 | https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero                 |
| Math          | [DAPO-Math](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)                            |                      2,584 |                         7,000 | https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed          |
| Math          | [AceReason-Math](https://huggingface.co/datasets/nvidia/AceReason-Math)                                 |                      6,602 |                             – | https://arxiv.org/abs/2505.16400                                         |
| Math          | [Polaris-Dataset](https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K)                  |                          – |                        14,000 | https://huggingface.co/POLARIS-Project                                   |
| Math          | [KlearReasoner-MathSub](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-MathSub-30K)           |                      3,000 |                         9,000 | https://github.com/suu990901/KlearReasoner                               |
| Math          | [OMEGA-train](https://huggingface.co/datasets/allenai/omega-problems)                                   |                     15,000 |                        20,000 | https://huggingface.co/datasets/allenai/omega-explorative                |
| Coding        | [AceCoder](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K)                                       |                      9,767 |                        20,000 | https://huggingface.co/collections/TIGER-Lab/acecoder                    |
| Coding        | [KlearReasoner-Code](https://huggingface.co/datasets/Kwai-Klear/KlearReasoner-CodeSub-15K)              |                      8,040 |                             – | https://github.com/suu990901/KlearReasoner                               |
| Coding        | [Nemotron Post-training Code](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) |                      2,303 |                             – | https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 |
| Coding        | [SYNTHETIC-2](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2)                               |                      3,000 |                             – | https://www.primeintellect.ai/blog/synthetic-2-release                   |
| General Chat  | [Tulu 3 SFT](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)                                |                      7,129 |                        18,955 | https://huggingface.co/collections/allenai/tulu-3-datasets               |
| General Chat  | [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M)                                  |                      7,129 |                        18,761 | https://huggingface.co/datasets/allenai/WildChat-4.8M                    |
| Multi-Subject | [Multi-Subject RLVR](https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR)                     |                      7,129 |                        12,234 | https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR            |
| **Total**     |                                                                                                         |                **104,869** |                   **171,950** |                                                                          |

#### Delta Learning intuition (why it helps)

Traditional synthetic preference data (UltraFeedback-like) depends on:

- a diverse pool of strong models
- an LLM judge to rank completions

But for reasoning traces, the pool of “open reasoning trace” models is limited.

**Delta Learning** solves this by focusing on the _gap_ between preferred and rejected answers:

- Chosen completion: strong model (e.g., Qwen-3-32B)
- Rejected completion: weak model (e.g., Qwen-3-0.6B)

The idea is:

> preference quality depends on the _delta_ between chosen and rejected responses more than either response alone

The report also notes an empirical detail:

- continuing SFT on extra synthetic completions can hurt
- but DPO using clear deltas improves and becomes a better RL starting point

#### Prompt reuse + filtering details

- Prompts are mostly reused from SFT
- Extra preference sources from earlier OLMo work may be included (e.g., UltraFeedback-like sets)
- Filtering is applied primarily to the **chosen** completion
- Rejected completions may remain unfiltered (since they are intentionally weaker)

#### Mixture selection under RL cost constraints

Reasoning DPO experiments are expensive, so they use a hierarchical strategy:

1. do mixture experiments using “non-thinking” outputs first (cheaper)
2. take the top 3 mixtures into full thinking-trace preference tuning

### Stage 3: Thinking RL (RLVR) using GRPO + mixed rewards

As the final stage, OLMo 3 Think runs RL training using:

- **GRPO** (Group Relative Policy Optimization)
- a mixture of:
  - verifiable reward domains
  - non-verifiable reward domains (LLM judge)

They frame this as RLVR:

- RL with verifiable rewards (plus judge rewards for non-verifiable tasks)

#### The RL dataset: Dolci Think RL

- **[Dolci Think RL](https://huggingface.co/datasets/allenai/Dolci-Think-RL-32B)**
- ~100K prompts
- domains include:
  - math
  - code
  - instruction following
  - general chat

## Rewards: verifiable vs non-verifiable

### Verifiable rewards (deterministic)

These rewards come from rule-based checks or test execution.

**Math**

- normalize answers
- check equivalence (e.g., via `sympy`)
- reward is often binary (correct / incorrect)

**Code**

- execute test cases
- reward can be:
  - binary (all tests pass)
  - or fractional (ratio of passing tests)

**Instruction following**

- verify constraint-specific rules
- similarly binary or fractional

### Non-verifiable rewards (LLM-judge)

General chat has no deterministic verifier.

So they use an **LLM judge**:

- judge model: Qwen-3-32B
- thinking mode off (judge produces a score directly)

Judging can be:

- reference-based (if ground truth exists)
- reference-free (otherwise)

## Why mix reward signals? (multi-objective RLVR)

Pure RLVR often optimizes for one verifiable domain (like math).

But the report emphasizes they mix domains to:

- prevent over-optimization (reward hacking)
- preserve general usefulness

A key observation they report:

- mixed reward training can have lower training reward
- but better downstream generalization

## GRPO: enhancements over “vanilla” GRPO

OLMo 3 Think uses GRPO as the base optimizer, but applies several enhancements inspired by recent work.

Key improvements listed:

1. **Zero Gradient Filtering**

   - drop prompts where all rollouts get identical reward (no learning signal)

2. **Active Sampling**

   - keep batch size constant by replacing filtered prompts

3. **Token-level normalization**

   - normalize GRPO loss by total tokens in batch (reduces length bias)

4. **No KL loss**

   - remove KL penalty for more flexible updates

5. **Asymmetric clipping**

   - set higher upper bound than lower bound for clipping

6. **No reward standard deviation normalization**
   - remove reward std term from the group advantage denominator

These changes preserve the overall GRPO structure (PPO-like), but adjust:

- advantage definition
- clipping
- normalization

## RL data curation details (code + chat)

### Code: create problems with usable test cases

Because test cases are required for verifiable rewards, they build a synthetic pipeline:

1. rewrite the problem + solution
2. generate test cases
3. execute test cases
4. keep problems where >80% test cases pass
5. remove failing remaining test cases

### Chat: rewrite + filter by “not too easy, not too hard”

They rewrite chat samples for clarity and extract reference answers.

Then:

- sample 8 responses per prompt
- compute F1 to reference
- drop prompts where F1 is outside [0.1, 0.9]

This aims to remove:

- trivial prompts (too easy)
- noisy / impossible prompts (too hard)

## Offline difficulty filtering before RL

Before RL training, they filter out prompts the policy already solves easily:

- generate 8 rollouts per prompt using the DPO checkpoint
- remove prompts with pass-rate > 62.5%

This improves RL sample efficiency by avoiding “already-solved” data.

They run this filtering for 7B first, then reuse results for 32B due to cost.

## RL experimentation strategy under high cost

RL is expensive, so they adopt proxy tuning strategies:

- short RL runs (~1K steps) to vet mixtures
- test algorithm changes in single-objective settings (math-only)
- mostly tune with 7B, then reuse for 32B

This is a recurring theme in OLMo 3:

> expensive stages require cheap proxy experiments.

## A practical takeaway you can reuse

If you want to train a reasoning model like OLMo 3 Think, the high-level playbook is:

1. **Seed reasoning format with SFT**
2. **Improve preference alignment + capability with DPO**
3. **Scale reasoning ability with RLVR**
4. Use a **multi-domain reward mix** to prevent narrow reward hacking
5. Invest in **RL infrastructure** (engine mismatch, stability, sampling)

---

## References

- OLMo 3 technical report: <https://arxiv.org/abs/2512.13961>
