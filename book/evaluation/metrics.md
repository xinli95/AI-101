# LLM Evaluation Metrics

This tutorial focuses **only** on metrics commonly used in **OLMES-style LLM evaluation**, including:

- **Exact Match (EM)**
- **F1 (token overlap / QA F1)**
- **Recall (substring recall)**
- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- **MC1 / MC2** (TruthfulQA-style multiple choice)
- **pass@k** (code generation)
- **Ranking metrics** (NDCG@k, MRR, Recall@k, MAP, win-rate)

---

## 1. Exact Match (EM)

### Definition

Exact Match (EM) is a **0/1** metric that checks whether the predicted answer exactly equals a reference answer _after normalization_ (typically lowercasing and whitespace cleanup).

For a single reference:

$$
\text{EM}(pred, ref) = \mathbb{1}[\text{norm}(pred) = \text{norm}(ref)]
$$

For multiple valid references $refs=\{ref_1,\dots,ref_m\}$, a common rule is **max over references**:

$$
\text{EM}(pred, refs) = \max_{j\in\{1..m\}} \mathbb{1}[\text{norm}(pred)=\text{norm}(ref_j)]
$$

### Worked example

References:

- `"New York City"`
- `"NYC"`

Prediction: `"nyc"`

After case normalization: `"NYC"` matches →

$$
\text{EM} = 1
$$

---

## 2. QA Token-overlap F1 (SQuAD-style F1)

### Definition

This metric measures **partial correctness** by token overlap between the predicted answer and the reference.

Tokenize (usually after normalization):

- prediction tokens: $P$
- reference tokens: $G$

Let $|P|$ be #pred tokens, $|G|$ be #gold tokens, and $|P\cap G|$ be # of overlapping tokens (with multiplicity if using bags/multisets).

$$
\text{Precision} = \frac{|P\cap G|}{|P|}, \quad \text{Recall} = \frac{|P\cap G|}{|G|}
$$

$$
\text{F1} = \frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

As with EM, for multiple references, many QA benchmarks use:

$$
\text{F1}(pred, refs) = \max_{ref\in refs} \text{F1}(pred, ref)
$$

### Worked example

Reference: `"the cat sat"` → tokens = `[the, cat, sat]`

Prediction: `"cat sat on"` → tokens = `[cat, sat, on]`

Overlap tokens = `[cat, sat]` → $|P\cap G|=2$

$$
\text{Precision} = 2/3, \quad \text{Recall} = 2/3
$$

$$
\text{F1} = \frac{2\cdot (2/3)\cdot (2/3)}{(2/3)+(2/3)} = 2/3 \approx 0.667
$$

---

## 3. Substring Recall ("did you mention the gold answer?")

### Definition

This is a very simple recall-style metric used in some QA evaluations:

- return **1** if _any_ gold reference string appears as a **substring** in the prediction
- otherwise return **0**

With case-insensitive matching:

$$
\text{Recall}(pred, refs) = \mathbb{1}[\exists ref\in refs:\ \text{lower}(ref) \subseteq \text{lower}(pred)]
$$

### Worked examples

**Example A (hit):**

- ref = `"Barack Obama"`
- pred = `"The answer is Barack Obama, the former president."`

Gold is contained → Recall = 1.

**Example B (miss):**

- ref = `"Barack Obama"`
- pred = `"The answer is Obama."`

Substring `"barack obama"` is not present → Recall = 0.

---

## 4. ROUGE (ROUGE-1 / ROUGE-2 / ROUGE-L)

ROUGE is widely used for **summarization-style outputs**.

- **ROUGE-1**: unigram overlap
- **ROUGE-2**: bigram overlap
- **ROUGE-L**: based on _Longest Common Subsequence (LCS)_

ROUGE is commonly reported as **Precision / Recall / F1**.

### 4.1 ROUGE-1 (unigram overlap)

Let:

- $U\_{pred}$: multiset of unigrams in prediction
- $U\_{ref}$: multiset of unigrams in reference

Overlap count is the clipped overlap (like BLEU-style clipping).

$$
\text{ROUGE-1 Precision} = \frac{|U_{pred}\cap U_{ref}|}{|U_{pred}|}
$$

$$
\text{ROUGE-1 Recall} = \frac{|U_{pred}\cap U_{ref}|}{|U_{ref}|}
$$

$$
\text{ROUGE-1 F1} = \frac{2PR}{P+R}
$$

#### Worked example (ROUGE-1 recall)

Reference: `"the cat sat on the mat"` (6 tokens)

Prediction: `"the cat sat"` (3 tokens)

Overlap unigrams = 3 (`the`, `cat`, `sat`).

$$
\text{ROUGE-1 Recall} = 3/6 = 0.5
$$

---

### 4.2 ROUGE-2 (bigram overlap)

ROUGE-2 uses overlapping **bigrams** instead of unigrams.

#### Worked example

Reference: `"the cat sat on the mat"`

Reference bigrams:

- the cat
- cat sat
- sat on
- on the
- the mat

Prediction: `"the cat sat"`

Prediction bigrams:

- the cat
- cat sat

Overlap bigrams = 2.

$$
\text{ROUGE-2 Recall} = 2/5 = 0.4
$$

---

### 4.3 ROUGE-L (LCS-based)

ROUGE-L uses the **Longest Common Subsequence (LCS)** length between prediction and reference.

Let $L$ be LCS length (in tokens). Then:

$$
P = \frac{L}{|pred|}, \quad R = \frac{L}{|ref|}, \quad F1=\frac{2PR}{P+R}
$$

#### Worked example

Reference tokens: `[the, cat, sat, on, the, mat]` (length 6)

Prediction tokens: `[the, cat, sat]` (length 3)

The LCS is `[the, cat, sat]` so $L=3$.

$$
P = 3/3 = 1, \quad R = 3/6 = 0.5, \quad F1 = \frac{2\cdot 1\cdot 0.5}{1+0.5}=0.667
$$

---

## 5. MC1 and MC2 (TruthfulQA-style multiple choice)

These metrics are designed for multiple-choice settings where:

- you have multiple answer options
- **some options are true/correct** (possibly more than one)
- the model assigns a **score** to each option (usually log-likelihood)

Let options be $1..n$.

- truth label: $label_i\in\{0,1\}$
- model score: $s_i$ (higher means more preferred)
- set of true answers: $T=\{i:label_i=1\}$

### 5.1 MC1 (top-1 correctness)

MC1 checks if the **single best** option is true:

$$
i^* = \arg\max_i s_i, \quad \text{MC1} = \mathbb{1}[i^*\in T]
$$

#### Worked example

| option | true? | score $s_i$ |
| ------ | ----: | ----------: |
| A      |     0 |        -2.0 |
| B      |     1 |        -0.4 |
| C      |     0 |        -1.1 |
| D      |     1 |        -0.7 |

The best score is option B (−0.4), and B is true →

$$
\text{MC1} = 1
$$

### 5.2 MC2 (probability mass on true options)

MC2 measures how much probability mass the model assigns to **all true answers**.

Step 1: convert scores to a probability distribution (softmax):

$$
p_i = \frac{e^{s_i}}{\sum_{j=1}^{n} e^{s_j}}
$$

Step 2: sum probabilities over true options:

$$
\text{MC2} = \sum_{i\in T} p_i
$$

#### Worked example (same table)

Compute unnormalized weights:

- A: $e^{-2.0}=0.135$
- B: $e^{-0.4}=0.670$ (true)
- C: $e^{-1.1}=0.333$
- D: $e^{-0.7}=0.497$ (true)

Total weight:

$$
Z = 0.135+0.670+0.333+0.497 = 1.635
$$

True mass:

$$
p_B + p_D = (0.670+0.497)/1.635 = 1.167/1.635 \approx 0.714
$$

So:

$$
\text{MC2} \approx 0.714
$$

### Interpretation

- **MC1** is strict: only top-1 matters (0/1).
- **MC2** is softer: gives partial credit if the model assigns high probability to true options, even if it is uncertain.

---

## 6. pass@k (code generation)

pass@k is used when an LLM generates **multiple candidate solutions** for the same coding problem.

Each candidate solution is evaluated by unit tests:

- pass ✅
- fail ❌

Let:

- $n$: number of generated solutions
- $c$: number of solutions that pass
- $k$: number of attempts allowed ("try up to k samples")

### Definition

pass@k is the probability that **at least one** of the $k$ attempts passes.

Assuming you sample $k$ solutions uniformly **without replacement**, then:

1. Probability that **all k fail**:

$$
P(\text{all fail}) = \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

2. Therefore, probability that **at least one passes**:

$$
\boxed{\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}}
$$

### Worked examples

Suppose $n=10$ total solutions and $c=2$ pass.

**pass@1**:

$$
\text{pass@1} = 1 - \frac{\binom{8}{1}}{\binom{10}{1}} = 1 - \frac{8}{10} = 0.2
$$

**pass@5**:

$$
\text{pass@5} = 1 - \frac{\binom{8}{5}}{\binom{10}{5}} = 1 - \frac{56}{252} \approx 0.778
$$

### Interpretation

- pass@1 = one-shot success rate
- pass@k grows with k because you get more chances
- a model can have low pass@1 but high pass@10 if it can sometimes produce a correct solution

---

## 7. Ranking / search metrics

Ranking metrics are used when the model outputs a **ranked list** of items (documents, answers, candidates).

### 7.1 NDCG@k (Normalized Discounted Cumulative Gain)

Given a ranked list with relevance labels $rel_i$ (e.g., graded relevance 0–3, where higher means more relevant), DCG@k is:

$$
DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}
$$

NDCG@k normalizes by the ideal ranking:

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

IDCG@k is computed by sorting the same relevance labels in **descending** order (best possible ranking) and applying the DCG formula to the top $k$.

**Worked example (DCG)**

Suppose $k=3$ and the relevance labels for the top 3 results are:

- $rel_1 = 3$
- $rel_2 = 2$
- $rel_3 = 0$

Then:

$$
DCG@3 = \frac{2^3 - 1}{\log_2(2)} + \frac{2^2 - 1}{\log_2(3)} + \frac{2^0 - 1}{\log_2(4)}
      = \frac{7}{1} + \frac{3}{1.585} + \frac{0}{2}
      \approx 7 + 1.893 + 0 = 8.893
$$

If the ideal ordering is already sorted by relevance, then $IDCG@3 = DCG@3$ and $NDCG@3 = 1.0$.

### 7.2 MRR (Mean Reciprocal Rank)

Reciprocal rank for one query is $1 / rank$ of the first relevant item. MRR averages across queries:

$$
MRR = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{rank_q}
$$

### 7.3 Recall@k

Fraction of relevant items retrieved in top $k$:

$$
Recall@k = \frac{|\,\text{relevant} \cap \text{top-}k\,|}{|\,\text{relevant}\,|}
$$

### 7.4 MAP (Mean Average Precision)

Average precision for one query is the mean of precision at ranks where a relevant item appears:

$$
P@i = \frac{\#\text{relevant items in top } i}{i}
$$

$$
AP = \frac{1}{|\text{relevant}|} \sum_{i=1}^{n} P@i \cdot \mathbb{1}[i \text{ is relevant}]
$$

MAP averages AP across queries.

**Worked example (single query)**

Suppose the ranked list relevance is:

- `[1, 0, 1, 0, 1]`

Precision at relevant ranks:

- $P@1 = 1/1 = 1.0$
- $P@3 = 2/3 \approx 0.667$
- $P@5 = 3/5 = 0.6$

Total relevant items $R = 3$.

$$
AP = \frac{1}{3}(1.0 + 0.667 + 0.6) \approx 0.756
$$

**Multiple queries (MAP)**

If you have $Q$ queries, compute $AP_q$ for each one and average:

$$
MAP = \frac{1}{Q} \sum_{q=1}^{Q} AP_q
$$

### 7.5 Win-rate (pairwise preference)

For pairwise human judgments, win-rate is:

$$
\text{win-rate} = \frac{\#\text{wins}}{\#\text{wins} + \#\text{losses}}
$$

This is common for head-to-head model comparisons or preference-based evals.

---

## 8. Summary cheat sheet

| Metric           | Typical use case           | Output range | Better | What it measures                      |
| ---------------- | -------------------------- | -----------: | -----: | ------------------------------------- |
| EM               | short-answer QA            |          0/1 |      ↑ | exact correctness                     |
| F1 (QA)          | QA partial credit          |         0..1 |      ↑ | token overlap quality                 |
| substring Recall | "did it mention the gold?" |          0/1 |      ↑ | containment of gold string            |
| ROUGE-1/2/L      | summarization              |         0..1 |      ↑ | overlap / sequence similarity         |
| MC1              | multi-choice               |          0/1 |      ↑ | top-1 chooses a true answer           |
| MC2              | multi-choice               |         0..1 |      ↑ | probability mass on true answers      |
| pass@k           | code generation            |         0..1 |      ↑ | chance of ≥1 passing among k attempts |
| NDCG@k           | ranking / search           |         0..1 |      ↑ | discounted gain vs ideal              |
| MRR              | ranking / search           |         0..1 |      ↑ | rank of first relevant item           |
| Recall@k         | ranking / search           |         0..1 |      ↑ | fraction of relevant in top k         |
| MAP              | ranking / search           |         0..1 |      ↑ | average precision across queries      |
| win-rate         | pairwise preference        |         0..1 |      ↑ | % wins in head-to-head comparisons    |
