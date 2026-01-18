(olmo3-longcontext)=

# OLMo 3 Long-Context Training

## What “long-context training” means in OLMo 3

After pretraining and midtraining, OLMo 3 runs a dedicated **long-context training** phase.

Important: the **training objective does not change**.

It is still standard language modeling (next-token prediction).  
What changes is:

1. **the dataset** (more long documents + synthetic long-context tasks), and
2. **the sequence length distribution** (many examples are extremely long).

In the OLMo 3 recipe:

- Midtraining trains for ~100B tokens
- Long-context training also trains for **~100B tokens**

## The long-context dataset: “Dolma 3 Longmino Mix”

Long-context training uses a mixture called:

- **[Dolma 3 Longmino Mix](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125)**

This mix contains:

- **Long context data** (long documents + synthetic long-context tasks)
- **Short context data** (re-used from midtraining)

The ratio described is:

- **34% long-context**
- **66% short-context**
- i.e., a **1 : 2** mix (long : short)

The goal of mixing in short-context data is to:

- stabilize training,
- reduce drift,
- keep short-context performance strong while extending context.

## Long context data source #1: long academic PDFs (with GZIP filtering)

A major source of long documents comes from the **academic PDF pretraining corpus**.

However, not all long documents are equally useful for long-context learning. Some are:

- too repetitive / redundant (e.g., boilerplate-heavy),
- or too random / noisy.

### The GZIP compressibility heuristic

OLMo 3 applies a heuristic **GZIP-based filter** on long documents:

- compute the compressibility score of each document via GZIP
- remove documents in:
  - the **top 20%** compressibility (most redundant)
  - the **bottom 20%** compressibility (least compressible / potentially noisy)

So the kept set is the “middle 60%” by compressibility.

**Interpretation:**

- very compressible → highly repetitive → low information per token
- very incompressible → too random / irregular → may be noisy or unstructured
- middle → better balance of structure + information

### Why this is interesting

The report notes something counterintuitive:

> This GZIP heuristic outperforms more sophisticated, model-based methods that use perplexity signals to identify “good” long-range dependency documents.

This is a great lesson for scaling pipelines:

- cheap heuristics can outperform expensive “smart” methods
- because they can be applied consistently at massive scale

## Long context data source #2: synthetic long-context extraction tasks (CLIPPER-style)

In addition to real long documents, OLMo 3 also trains on **synthetic long-context tasks**.

The motivation is:

- You want the model to learn “extract and aggregate information across long inputs”
- But you don’t want your synthetic generator LLM to _require_ long-context ability itself

This approach is inspired by **[CLIPPER: Compression enables long-context synthetic data generation](https://arxiv.org/abs/2502.14854)**.

### How synthetic tasks are generated (step-by-step)

Given a long document:

1. **Partition** it into several sections
2. For each section:
   1. identify common **noun phrases**
   2. for each noun phrase, extract **k = 8** text snippets from that section
3. Construct a prompt to a generator LLM (they use **OLMo 2 32B**) that includes:
   - the noun phrases
   - the extracted snippets
4. Ask the LLM to **synthesize an aggregation task**, for example:
   - write a summary
   - produce true/false claims
   - create a conversational explainer
   - or other information extraction / reasoning tasks

### Why this avoids the “generator must have long context” assumption

The generator never needs to read the _full_ long document.

Instead, it reads:

- _structured snippets_
- _key noun phrases_
- _section-level evidence_

So the generator can produce high-quality tasks without having perfect long-context attention.

### How the synthetic data is used for training

Training examples are constructed like this:

- **Input:** the full long document
- **Target:** the synthetic “aggregation task output”

This teaches the model:

- retrieve relevant parts from far away
- aggregate consistently
- avoid forgetting details across long spans

## Efficiency challenge: batching long sequences wastes compute

Long-context training has a huge practical problem:

- sequence lengths vary drastically

If your maximum context length is `S`, and you batch sequences of mixed lengths, you waste a lot of FLOPs on padding.

### Why padding is expensive

A batch tensor is shaped like:

$$
B \times S \times d
$$

Where:

- $B$ = batch size
- $S$ = context length
- $d$ = hidden dimension

If most examples have length $\ll S$, then most tokens in that tensor are:

- padding tokens
- computed on by the model
- but contribute no learning signal

## Document packing: reduce padding, keep GPU busy

OLMo 3 uses **document packing** to improve efficiency.

Instead of “1 document per row”, you pack multiple shorter documents into the same row until the row is near full.

### Key detail: block attention across packed documents

If you pack multiple documents into one sequence, you must prevent attention across document boundaries.

OLMo 3 does this by applying an **inter-document attention mask**, so that:

- tokens from doc A cannot attend to tokens from doc B

This keeps the packed example equivalent to “separate examples” while avoiding padding waste.

## RoPE context extension: why YaRN wins

To truly support long context, you also need a positional encoding strategy that works beyond the original trained length.

OLMo 3 experiments with multiple RoPE extension techniques, including:

- adjusted base frequency scaling
- position interpolation
- **YaRN**

The report highlights the winning configuration:

> Applying **YaRN only to full attention layers** yields the best overall performance.

### Why only apply YaRN to full attention layers?

OLMo 3 uses a mix of attention types, including:

- full attention layers
- SWA (sliding window attention) layers

They apply YaRN only where full attention is used, while leaving positional embeddings unchanged in SWA layers.

The intuition is:

- full attention layers need global positional generalization
- SWA layers already focus locally, and changing them may not help (or may harm)

### How it is evaluated

They report improvements on long-context focused evaluations like:

- advanced Needle-in-a-Haystack (NIH)
- RULER
- HELMET

## Model merging for long-context: merge adjacent checkpoints

Long-context training is expensive, so they do not run multiple full long-context runs with different random seeds.

Instead, they:

- take **three adjacent checkpoints** near the end of a single long-context training run
- **merge** them

This improves performance further, similar to the midtraining “merge two seeds” trick, but cheaper.

## Where OLMo 3 long-context performance lands

The report states OLMo 3 long-context capability is:

- **comparable to**
- or **slightly worse than**
  Qwen-2.5 (depending on benchmark)

The key takeaway is that OLMo 3 reaches a strong open long-context baseline using:

- curated long documents (PDFs + GZIP filtering)
- synthetic extraction tasks (CLIPPER-style)
- efficiency tricks (document packing)
- careful RoPE extension (YaRN)
- checkpoint merging

---

## Practical summary: the OLMo 3 long-context recipe (checklist)

If you want to copy this approach:

1. Collect long documents (e.g., PDFs)
2. Filter them using **GZIP compressibility** (drop extreme redundancy + extreme noise)
3. Create synthetic extraction tasks without assuming generator has long context
4. Mix long-context with short-context data (e.g., 1:2 ratio)
5. Use **document packing** + inter-document masks
6. Test RoPE extension methods; YaRN is a strong candidate
7. Apply YaRN selectively (full-attention only) if using hybrid attention
8. Merge adjacent checkpoints for a cheap robustness boost

---

## References

- OLMo 3 technical report: <https://arxiv.org/abs/2512.13961>
- CLIPPER: Compression enables long-context synthetic data generation: <https://arxiv.org/abs/2502.14854>
