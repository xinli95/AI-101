(olmo3-pretraining)=

# OLMo 3 Pretraining

> This tutorial summarizes the _pretraining_ part of the OLMo 3 “model flow” (data + evals + mixing) and explains the key ideas behind their empirical recipe.

## Why pretraining recipe development is hard

Pretraining is mostly an empirical optimization problem: you try many knobs (data mixture, filtering, objectives, schedules), run experiments, and keep what works.

Two things make this harder than it sounds:

- **Small-scale ≠ large-scale**: a change that helps a 30M–1B model might not help (or might even hurt) at 7B/32B scale.
- **Benchmarks can be noisy or saturated**: some tasks only become meaningful at certain scales, while others are already near-ceiling early.

OLMo 3 tackles this by (1) designing an evaluation suite that is _scale-aware_ and (2) building a data-mixing procedure that is cheap to iterate on but still predictive of full training.

## The evaluation backbone: OlmoBaseEval

OLMo 3’s base-model development uses **OlmoBaseEval**, a benchmark suite intended to guide pretraining and midtraining decisions. The technical report describes it as **43 tasks** (plus a held-out set to reduce overfitting on the dev suite).

### Design principle 1: Task clusters

Rather than tracking dozens of benchmarks independently, OlmoBaseEval groups tasks into **clusters** and aggregates scores within each cluster. The final clusters are:

- **[MC STEM](olmes/olmo3_base_stem_qa_mc.md)**
- **[MC Non-STEM](olmes/olmo3_base_nonstem_qa_mc.md)**
- **[GenQA](olmes/olmo3_base_gen.md)**
- **[Math](olmes/olmo3_base_math.md)**
- **[Code](olmes/olmo3_base_code.md)**
- **[Code Fill-in-the-Middle (FIM)](olmes/olmo3_base_code_fim.md)**

This clustering is motivated by the idea that if tasks behave similarly (e.g., they rank models similarly), you can treat them as a single capability signal.

### Design principle 2: Proxy metrics via scaling analysis

OLMo 3 explicitly studies _when_ tasks/metrics become informative across compute scales. The report emphasizes that some metrics show clear signal for **small-scale ablations**, and that these can act as **proxies** for larger-scale decisions.

A concrete example they discuss is using **bits-per-byte (BPB)** on “Easy” suites as a development-time signal, because it can be more stable earlier than large-scale pass@k style metrics.

### Design principle 3: Signal-to-noise ratio (SNR)

If a benchmark is too noisy, small improvements can be meaningless. OlmoBaseEval addresses this by either:

- removing very noisy tasks from the macro average, or
- evaluating on **more samples** for those tasks to reduce variance.

## Data strategy for pretraining

OLMo 3’s base model training is staged (pretraining → midtraining → long-context extension). For **pretraining**, they intentionally restrict themselves to **large, natural** data sources that have enough tokens to matter at the trillions-of-tokens scale.

Two explicit principles:

- A source is considered for **pretraining** only if it can yield enough tokens to impact capabilities at pretraining scale.
- **Structured “task data”** (QA pairs, chat templates, instruction traces) is _not_ used in pretraining; it is reserved for later stages (midtraining / long-context). This avoids confounding data ablations because small amounts of structured data can disproportionately move evaluation scores.

They also apply **topic and quality classification** before mixing. Using their **[WebOrganizer](https://huggingface.co/WebOrganizer/TopicClassifier)** tool, they partition the deduplicated corpus into **24 topics** (for example, Adult Content, Politics, and Science and Technology). The topic taxonomy is defined in **[formats.yaml](https://github.com/CodeCreator/WebOrganizer/blob/main/define_domains/taxonomies/formats.yaml)**. To speed up processing for the Dolma 3 pool, they distill the transformer-based models from **WebOrganizer** into a simpler **fastText** model and partition by **topic only** (not format). In parallel, they train a **fastText-based quality classifier** to assign each document a quality score. Following **[DCLM](https://arxiv.org/abs/2406.11794)**, they use **[OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)** and **[ELI5](https://huggingface.co/datasets/sentence-transformers/eli5)** as positive examples, plus **[UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)** and **[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)**, while negative examples are **30GB sampled from DCLM-RefinedWeb**.

## The pretraining corpus: Dolma 3 Mix (≈6T tokens)

OLMo 3 pretraining uses **Dolma 3 Mix**, a ~6T-token mixture sampled from a larger cleaned pool (~9T tokens). The report provides the following composition:

| Source                 | Type               | 9T pool tokens | 6T mix tokens (share) |
| ---------------------- | ------------------ | -------------: | --------------------: |
| Common Crawl           | Web pages          |          8.14T |         4.51T (76.1%) |
| olmOCR science PDFs    | Academic documents |           972B |          805B (13.6%) |
| Stack-Edu (Rebalanced) | GitHub code        |           137B |          409B (6.89%) |
| arXiv                  | Papers with LaTeX  |          21.4B |         50.8B (0.86%) |
| FineMath 3+            | Math web pages     |          34.1B |          152B (2.56%) |
| Wikipedia & Wikibooks  | Encyclopedic       |          3.69B |         2.51B (0.04%) |
| **Total**              |                    |      **9.31T** |      **5.93T (100%)** |

Even at a glance, you can see the design intent: keep the core distribution dominated by web text, then **meaningfully upweight** science PDFs and code, while still keeping the overall corpus “natural” enough to act as a general base model.

## How they choose the mixture: token-constrained “swarm” optimization

The report describes a **constrained data mixing** procedure with two parts:

1. **Base procedure** (for a fixed set of domains)
2. **Conditional mixing** (to update an existing mix when domains change)

### Base procedure (high level)

The core idea is: sample many candidate mixtures, train _tiny proxy models_ quickly, and learn a response surface that predicts evaluation performance from mixture weights.

A simplified version of the loop described in the report looks like this:

1. **Sample many mixtures** (a “swarm”).
2. **Train proxy models** for each mixture (small parameter count, small token budget).
3. **Evaluate** each proxy model on a _development_ suite.
4. Fit **per-task generalized linear models**: predict task performance from mixture weights.
5. **Optimize** the mixture under constraints using the learned predictors.

The report notes that step (4) uses a separate GLM per task, and step (5) enforces constraints like: keep a **~6T token budget** and avoid **over-repeating domains** ( avoiding repeating any domain more than roughly 4–7 times).

### Conditional mixing: updating mixes as data changes

In real pipelines, domains and filters change continuously. Re-running a full swarm every time is expensive.

**Conditional mixing** treats your _existing optimized mix_ as one **virtual domain** with frozen internal ratios, then re-runs the base procedure over:

- the virtual “old mix” domain, plus
- any new or modified domains.

This reduces the dimensionality of the search and makes iteration cheaper while reusing prior optimization work.

## Topic + quality classification (what “quality score” means)

Before mixing and upsampling, OLMo 3 first **labels each document** with:

- a **topic label** (one of 24 topics), and
- a **quality score** (used later for quality-aware upsampling).

This is done with their **[WebOrganizer](https://huggingface.co/WebOrganizer/TopicClassifier)** topic classifier, but they **distill** the original transformer classifiers into a faster model for large-scale processing.

### 1) Topic classification (24 topics)

They partition the deduplicated Dolma 3 pool into **24 broad topics** (examples given include _Adult Content_, _Politics_, _Science and Technology_).

Key choices:

- They **partition by topic, not by format** (so “PDF vs HTML” is not the primary bucket here).
- To scale to trillions of tokens, they distill the original transformer-based approach into a **fastText** topic classifier.

### 2) Quality classification (a fastText “good vs bad” signal)

They also train a **fastText-based quality classifier** that assigns each document a quality score.

Conceptually, it is trained like a _binary_ text classifier (high-quality vs low-quality), and then its output (e.g., probability/logit) can be used as a continuous **quality score**.

#### Positive training examples (high-quality)

Following the **[DCLM](https://arxiv.org/abs/2406.11794)** approach, they treat the following datasets as _positive_ examples:

- **[OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)**
- **[ELI5](https://huggingface.co/datasets/sentence-transformers/eli5)**

They also supplement positives with:

- **[UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)**
- **[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)**

The intuition is that these sources contain _more coherent_, _instruction-like_, and _information-dense_ documents that correlate with the properties we want the base model to see more often.

#### Negative training examples (low-quality)

Negative examples are sampled from:

- **DCLM-RefinedWeb** (30GB sample)

This gives the classifier a “what low-quality looks like” contrastive signal (e.g., spammy, low-information, noisy web text).

#### Why use fastText here?

fastText is extremely attractive for pipeline-scale filtering because it is:

- fast to train,
- fast to run over massive corpora,
- easy to distill into a lightweight model that still preserves useful ranking behavior.

### 3) How the quality score is used downstream: quality-aware upsampling

Once every document has a quality score, OLMo 3 uses **quality-aware upsampling** rather than a strict threshold filter.

Mixing chooses **how much** you draw from each domain/topic, but within a domain (e.g., Common Crawl) quality varies widely.

Quality-aware upsampling implements a **monotonically increasing curve**:

- low-quality documents are downsampled or dropped,
- high-quality documents are repeated more times,
- the overall token budget stays fixed.

This gives you the “benefits of filtering” (more high-quality text) while keeping a smoother control knob than a hard cutoff.

## Putting it together: the OLMo 3 base-model training curriculum

OLMo 3’s **base model training** is staged:

1. **Pretraining**: general capability foundation on Dolma 3 Mix (~6T tokens).
2. **Midtraining**: targeted, high-quality data to boost math/code/QA and improve post-trainability.
3. **Long-context extension**: train on long documents (including large-scale science PDFs) to reach long context windows.

The key takeaway is that OLMo 3 deliberately _separates concerns_:

- Pretraining stays “mostly natural” and large-scale.
- Structured / instruction-like data is delayed to midtraining, where it can be controlled and evaluated cleanly.
- Long-context ability is taught explicitly in its own phase rather than hoping it emerges.

## A practical mental model you can reuse

If you want to adapt the OLMo 3 approach for your own base model recipe, think in three layers:

1. **Evaluation design**: make a small suite that is stable at the scales you can afford, and aggregate tasks into clusters.
2. **Mixture optimization**: use cheap proxy runs + a learned predictor to search the mixture space under constraints.
3. **Quality shaping**: treat “which domain/topic?” and “which quality tier?” as separate knobs (mixing vs upsampling).

---

## References

- OLMo 3 technical report: <https://arxiv.org/abs/2512.13961>
- Olmo 3 and the Open LLM Renaissance: <https://substack.com/@cwolferesearch/p-179769076>
