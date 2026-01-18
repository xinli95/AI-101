(olmo3-midtraining)=

# OLMo 3 Midtraining

> This tutorial explains the **midtraining** stage in the OLMo 3 model development flow.

## Where midtraining sits in the full training pipeline

OLMo 3 uses a staged curriculum:

1. **Pretraining**: learn general language + broad capabilities from a huge natural corpus (trillions of tokens).
2. **Midtraining**: continue training the base model on a smaller but _more targeted_ mixture to boost specific skills and improve post-training behavior.
3. **Long-context training**: teach the model to operate reliably on longer sequences.

A key point: **the objective stays the same** across these phases (still next-token prediction / LM training). What changes is **the data** and **the token budget**.

## What makes midtraining different from pretraining?

Midtraining differs from pretraining mainly in two ways:

1. **More targeted data sources**
   - Instead of “mostly natural web text”, midtraining includes datasets chosen specifically to improve desired capabilities.
2. **Fewer total tokens**
   - Midtraining is much shorter than pretraining.
   - For OLMo 3, midtraining uses **~100B additional tokens**.

Intuitively:

- Pretraining builds the foundation.
- Midtraining _shapes_ the foundation toward the target capability profile.

## The midtraining corpus: Dolma 3 Dolmino Mix (100B tokens)

Midtraining uses a dedicated dataset mixture:

- **[Dolma 3 Dolmino Mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125)**
- Total size: **100B tokens**
- Goal: improve “key model capabilities” more efficiently than full pretraining-scale changes.

A practical way to interpret Dolmino Mix is:

> “A curated 100B token follow-up curriculum that nudges the base model toward stronger performance on important benchmarks without paying pretraining-level compute.”

## How they pick the midtraining mix: distributed feedback + integration tests

The report emphasizes that midtraining data selection is still empirical and noisy, but they build a system that supports **rapid iteration** with good signal.

They use a two-part iterative process:

### Part 1: Parallel (distributed) feedback via microannealing

Many candidate data sources are explored **in parallel** using **microannealing experiments**.

Microannealing = lightweight training runs that:

- start from a pretrained checkpoint,
- add a small targeted dataset (or ablate it),
- train for a short time,
- measure whether that data source helps.

Think of microannealing as:

> “Can this dataset move the needle if we ‘anneal’ it into the model a bit?”

This parallelism is crucial because the space of possible datasets is enormous, and you can’t do full 100B-token tests for every candidate.

### Part 2: Integration tests (centralized annealing runs)

Promising sources from microannealing don’t automatically become “final mix” members.

Instead, they are combined into a full **integration test**, where you build a candidate 100B-token mix including:

- all currently promising sources,
- plus required stabilizers / anchor data.

This validates that improvements survive _when datasets interact together_, not just in isolation.

The report states they ran **five rounds** of these integration tests while developing OLMo 3.

## How midtraining is evaluated (and why it differs from pretraining eval)

During midtraining, some benchmarks are already close to saturation from pretraining. So evaluation shifts to a set of tasks that still have headroom.

They primarily use:

- **Base Main**: benchmarks not yet saturated after pretraining

They also run **lightweight SFT experiments** using midtrained models to measure **post-trainability**.

### What is “post-trainability”?

Post-trainability is:

> “How well does the base model respond to later instruction tuning / supervised fine-tuning?”

Even if two midtraining mixtures look similar on pure base-model benchmarks, they may differ in how _easy_ it is to post-train them into a good assistant model.

So the evaluation includes:

- base metrics (capability / knowledge),
- plus quick post-training proxies (small SFT tests).

## Key design choices in the final midtraining mix

The final Dolmino Mix has a few important properties:

### 1) Include some pretraining data to avoid drift

Midtraining is more targeted and could over-specialize.

To stabilize the model, the midtraining mix includes **some pretraining data** to reduce distribution shift and prevent forgetting.

This is often called:

- “anchor data”
- “replay data”
- “anti-drift mixture components”

### 2) Include instruction + thinking (reasoning) data

The report says instruction data and “thinking” / reasoning data are broadly beneficial:

- improves performance across many benchmarks,
- lays early groundwork for later post-training.

However, there is a subtle engineering decision:

- they **avoid templates or special tokens** in midtraining instruction data.

Instead, they use **plain text formatting** that preserves the pretrained model’s natural output format.

This avoids confounding effects from introducing new chat formatting, which would complicate ablations and evaluation.

### 3) Expect domain tradeoffs (no free lunch)

Midtraining reveals clear tradeoffs:

- increase math/code ratio → math/code improves
- but other domains degrade

So the goal is not “maximize one metric” but to find a **balanced mixture** aligned with the target capability profile.

You can think of this as a constrained optimization problem:

- maximize important capabilities,
- while preventing regressions elsewhere.

## A surprising detail: the final midtraining model is a merge

The report notes that the final midtrained checkpoint is obtained by **merging two independently trained models** (different random seeds).

Empirically, this merge performs better than either model alone.

This is consistent with a broader observation in modern LLM training:

- different training stochasticity can lead to slightly different local optima,
- merging can average out quirks and yield a more robust model.

## Practical summary: how to copy this in your own model training

If you want an “OLMo 3 style” midtraining workflow, the recipe is:

1. **Start from a strong pretrained base**
2. **Run microannealing** per dataset (cheap, parallel)
3. **Run integration tests** periodically (expensive but realistic)
4. **Evaluate on non-saturated tasks**
5. **Measure post-trainability** using small SFT probes
6. **Keep anchor data** to prevent drift
7. **Accept domain tradeoffs** and pick a balanced mixture
8. **Consider checkpoint merging** for stability

---

## References

- OLMo 3 technical report: <https://arxiv.org/abs/2512.13961>
