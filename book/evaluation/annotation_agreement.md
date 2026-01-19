(inter-annotator-agreement)=

# Inter-Annotator Agreement

This note introduces **inter-annotator agreement (IAA)** statistics, focusing on:

- **Cohen’s kappa** (two annotators)
- **Fleiss’ kappa** (multiple annotators)
- **Krippendorff’s alpha** (multiple annotators, missing data, ordinal/continuous variants)

We will also use an example validation table (from the paper _How People Use ChatGPT_) to show how these metrics are reported and interpreted in practice.

---

## Why measure inter-annotator agreement?

When labels are produced by human raters (SMEs, crowd workers, trained annotators), agreement metrics help you answer:

- Are the **labeling guidelines clear** enough to produce consistent labels?
- Is the task **inherently ambiguous** (even experts disagree)?
- Can an LLM/classifier match **human labeling behavior** reliably?
- Which classes are confusing, and where do instructions need refinement?

Agreement is not only about “how often raters match” — we usually care about **agreement beyond chance**.

---

## Chance-corrected agreement: the core idea

Two raters might “agree” frequently even if the task is trivial or heavily imbalanced (e.g., nearly everything is labeled “No”).

Chance-corrected metrics adjust for the agreement you would expect if raters were **guessing with similar label frequencies**.

A common template is:

$$
\text{Agreement beyond chance} = \frac{\text{Observed} - \text{Expected}}{1 - \text{Expected}}
$$

Kappa statistics use this template directly; Krippendorff’s alpha uses the analogous form in terms of **disagreement**.

---

## Cohen’s kappa (two annotators)

### When to use

- Exactly **two** raters label the same set of items
- Labels are typically **nominal** (unordered categories), though there is a weighted variant for ordinal scales

### Definition

Let:

- $p_o$ = observed agreement rate
- $p_e$ = expected agreement rate by chance (based on each rater’s marginal label frequencies)

Cohen’s kappa is:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

### How to compute $p_o$ and $p_e$

Given a confusion matrix $M$ where $M_{ij}$ counts items labeled $i$ by rater A and $j$ by rater B:

Observed agreement:

$$
p_o = \frac{\sum_i M_{ii}}{N}
$$

Expected agreement:

- $p_i^A$ = fraction of labels assigned to class $i$ by rater A
- $p_i^B$ = fraction of labels assigned to class $i$ by rater B

$$
p_e = \sum_i p_i^A p_i^B
$$

### Interpretation

- $\kappa = 1$: perfect agreement
- $\kappa = 0$: agreement is no better than chance
- $\kappa < 0$: systematic disagreement (worse than chance)

---

## Weighted Cohen’s kappa (ordinal labels)

When labels are **ordered** (e.g., relevance grades 0–3), disagreement should depend on distance:

- 3 vs 2 is less severe than 3 vs 0

Weighted kappa adds a weight matrix $w_{ij}$ that encodes disagreement severity (often linear or quadratic).

This is commonly used for:

- relevance grading
- Likert-scale ratings
- quality tiers

---

## Fleiss’ kappa (multiple annotators)

### When to use

- **3+ raters**
- Each item is labeled by the **same number of raters** (or you can restrict to items that meet this requirement)
- Nominal labels (classic Fleiss’ kappa)

### What it measures

Fleiss’ kappa estimates agreement beyond chance across a group of raters.

One useful way to interpret it:

- sample two raters at random for the same item
- compute how often they agree, averaged over items
- chance-correct that agreement using the global label distribution

### Why it shows up in papers

Many annotation efforts use 3–5 annotators per item. Fleiss’ kappa gives a single summary number.

---

## Krippendorff’s alpha (multiple annotators, missingness-friendly)

### When to use

Krippendorff’s alpha is often preferred when:

- not all items are labeled by the same number of raters
- some ratings are missing
- labels are ordinal or continuous
- you want one method that generalizes well across setups

### Definition

Krippendorff’s alpha is:

$$
\alpha = 1 - \frac{D_o}{D_e}
$$

- $D_o$ = observed disagreement
- $D_e$ = expected disagreement by chance

### Key concept: a distance function

Alpha is defined via a **distance** (disagreement) function $\delta(c, c')$.

- Nominal: $\delta(c,c') = 0$ if same else $1$
- Ordinal: distance increases with how far apart labels are (often squared distance)
- Interval/ratio: distance can be squared numeric difference

### Observed disagreement $D_o$

For each item $i$ with $n_i$ ratings, compute disagreement over all unordered rater pairs:

$$
D_o = \frac{\sum_i \sum_{a<b} \delta(v_{ia}, v_{ib})}{\sum_i \binom{n_i}{2}}
$$

### Expected disagreement $D_e$

Let $p(c)$ be the overall fraction of ratings that are label $c$ across the entire dataset.

Then:

$$
D_e = \sum_c \sum_{c'} p(c) p(c') \delta(c,c')
$$

### Interpretation

- $\alpha = 1$: perfect agreement
- $\alpha \approx 0$: chance-level agreement
- $\alpha < 0$: systematic disagreement

### Why alpha is practical for real labeling pipelines

- Handles **unequal raters per item**
- Handles **missing labels**
- Naturally supports **ordinal** labels via distance weighting

---

## Code implementations (from scratch)

### Cohen's kappa

```python
import numpy as np

rater_a = ["yes", "no", "yes", "yes", "no"]
rater_b = ["yes", "no", "no", "yes", "no"]

def cohen_kappa_from_scratch(y1, y2, labels=None):
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    assert len(y1) == len(y2)

    if labels is None:
        labels = np.unique(np.concatenate([y1, y2]))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)

    # Confusion matrix
    M = np.zeros((K, K), dtype=int)
    for a, b in zip(y1, y2):
        M[label_to_idx[a], label_to_idx[b]] += 1

    N = M.sum()
    po = np.trace(M) / N

    pA = M.sum(axis=1) / N  # row marginals
    pB = M.sum(axis=0) / N  # col marginals
    pe = np.sum(pA * pB)

    if np.isclose(1 - pe, 0.0):
        return 1.0  # degenerate case

    kappa = (po - pe) / (1 - pe)
    return kappa, M, po, pe


kappa, M, po, pe = cohen_kappa_from_scratch(rater_a, rater_b)
print("Cohen kappa:", kappa)
print("Observed agreement po:", po)
print("Expected agreement pe:", pe)
print("Confusion matrix:\n", M)
```

### Weighted Cohen's kappa

```python
import numpy as np

# example: 0=bad, 1=ok, 2=good, 3=excellent
rater_a = [3, 2, 2, 1, 0, 3, 2]
rater_b = [3, 2, 1, 1, 0, 2, 2]

def weighted_cohen_kappa_from_scratch(y1, y2, labels=None, weight_type="quadratic"):
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    assert len(y1) == len(y2)

    if labels is None:
        labels = np.unique(np.concatenate([y1, y2]))

    labels = np.array(sorted(labels))  # important for ordinal
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)

    # Confusion matrix O
    O = np.zeros((K, K), dtype=float)
    for a, b in zip(y1, y2):
        O[label_to_idx[a], label_to_idx[b]] += 1

    N = O.sum()
    O = O / N  # normalize

    # Expected matrix E from marginals
    pA = O.sum(axis=1)
    pB = O.sum(axis=0)
    E = np.outer(pA, pB)

    # Weight matrix W: 0 means perfect match, 1 means worst mismatch
    W = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if weight_type == "linear":
                W[i, j] = abs(i - j) / (K - 1)
            elif weight_type == "quadratic":
                W[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
            else:
                raise ValueError("weight_type must be 'linear' or 'quadratic'")

    num = np.sum(W * O)
    den = np.sum(W * E)

    if np.isclose(den, 0.0):
        return 1.0

    kappa_w = 1.0 - num / den
    return kappa_w, O, E, W


kappa_w, O, E, W = weighted_cohen_kappa_from_scratch(
    rater_a, rater_b, weight_type="quadratic"
)
print("Weighted Cohen kappa (quadratic):", kappa_w)
```

### Fleiss' kappa

```python
import numpy as np

# rows = items, cols = classes, entry = # raters voting for that class
ratings = np.array([
    [0, 0, 3, 0],  # all 3 raters chose class 2
    [1, 2, 0, 0],  # disagreement
    [0, 3, 0, 0],
    [0, 1, 2, 0],
])

def fleiss_kappa(ratings_matrix):
    """
    ratings_matrix: shape (N_items, K_classes)
    Each row sums to n_raters (same for every item).
    """
    M = np.asarray(ratings_matrix, dtype=float)

    N, K = M.shape
    n = M.sum(axis=1)

    if not np.allclose(n, n[0]):
        raise ValueError("Each item must have the same number of raters.")

    n = int(n[0])  # number of raters per item

    # P_i = agreement for item i
    P_i = (np.sum(M * (M - 1), axis=1)) / (n * (n - 1))
    P_bar = np.mean(P_i)

    # p_j = overall proportion for class j
    p_j = np.sum(M, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)

    if np.isclose(1 - P_e, 0.0):
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa, P_bar, P_e, p_j, P_i


kappa_f, P_bar, P_e, p_j, P_i = fleiss_kappa(ratings)
print("Fleiss kappa:", kappa_f)
print("P_bar:", P_bar, "P_e:", P_e)
print("Class proportions:", p_j)
```

### Helpers: build a Fleiss matrix

```python
def to_fleiss_matrix(labels_by_item, classes=None):
    # labels_by_item: list of list, each inner list = labels from raters for one item
    if classes is None:
        classes = sorted(set(lab for row in labels_by_item for lab in row))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    N = len(labels_by_item)
    K = len(classes)
    M = np.zeros((N, K), dtype=int)

    for i, row in enumerate(labels_by_item):
        for lab in row:
            M[i, class_to_idx[lab]] += 1

    return M, classes


labels_by_item = [
    ["A", "A", "B"],
    ["B", "B", "B"],
    ["A", "C", "C"],
]

M, classes = to_fleiss_matrix(labels_by_item)
print("Classes:", classes)
print(M)

kappa_f, *_ = fleiss_kappa(M)
print("Fleiss kappa:", kappa_f)
```

---

## Common pitfalls (what you should watch for)

### 1) Prevalence / imbalance effects

For rare-label tasks (e.g., safety violations), it is common to see:

- high raw agreement
- but lower kappa/alpha than expected

Reason: chance agreement $p_e$ can become very large if most labels are the same.

Best practice: report alongside kappa/alpha:

- raw agreement
- label distribution
- confusion matrix / per-class agreement

### 2) Agreement is bounded by task clarity

If humans disagree heavily, model-vs-human agreement will usually be limited too.

A low kappa for a subjective task often indicates:

- unclear rubric
- intrinsically ambiguous task
- need to redesign labels or add decision rules and examples

---

## Worked example: how papers report kappa for “LLM-as-annotator” validation

Below is an example table (adapted from the paper _How People Use ChatGPT_) that validates LLM-generated labels against in-house human annotators on the WildChat corpus.

**How to read the reported columns:**

1. **Fleiss’ κ (human only)**  
   Agreement among humans → indicates task clarity and rubric quality.

2. **Fleiss’ κ (with model)**  
   Treat the model as an additional annotator. If this increases, the model behaves “human-like” and may stabilize labeling.

3. **Cohen’s κ (human vs. human)**  
   Mean pairwise agreement between humans.

4. **Cohen’s κ (model vs. plurality)**  
   Agreement between model and the human **plurality vote** (majority label). This is often the most operational “model matches consensus” metric.

### Example validation table

| Task                                        | nlabels | Fleiss’ κ (human only) | Fleiss’ κ (with model) | Cohen’s κ (human vs. human) | Cohen’s κ (model vs. plurality) |
| ------------------------------------------- | ------: | ---------------------: | ---------------------: | --------------------------: | ------------------------------: |
| Work Related (binary)                       |     149 |      0.66 [0.54, 0.76] |      0.68 [0.59, 0.77] |                        0.66 |               0.83 [0.72, 0.92] |
| Asking / Doing / Expressing (3-class)       |     149 |      0.60 [0.51, 0.68] |      0.63 [0.56, 0.70] |                        0.60 |               0.74 [0.64, 0.83] |
| Conversation Topic (coarse)                 |     149 |      0.46 [0.38, 0.53] |      0.48 [0.41, 0.54] |                        0.47 |               0.56 [0.46, 0.65] |
| IWA Classification                          |     100 |      0.34 [0.23, 0.45] |      0.47 [0.40, 0.53] |                        0.37 |                               — |
| GWA Classification                          |     100 |      0.33 [0.22, 0.44] |      0.47 [0.40, 0.54] |                        0.36 |                               — |
| Interaction Quality (3-class incl. unknown) |     149 |      0.13 [0.04, 0.22] |      0.10 [0.04, 0.17] |                        0.20 |               0.14 [0.01, 0.27] |

**Notes reported by the paper:**

- An item contributes only if all required raters provided a nonempty label.
- Confidence intervals are 95% percentile intervals from a nonparametric bootstrap with 2,000 resamples.
- “—” indicates cases where plurality is not defined (e.g., only two humans participated).

### Interpreting patterns in the table

- **High human κ + high model-vs-plurality κ (e.g., Work Related)**  
  The task is well-defined and the model matches human consensus well → good candidate for automation.

- **Moderate human κ + moderate model κ (e.g., Conversation Topic)**  
  The task has ambiguity; model performance is limited by label noise → improve rubric and add examples.

- **Low human κ + low model κ (e.g., Interaction Quality)**  
  Humans do not agree, so the model cannot learn/replicate a stable target. The bottleneck is the task definition, not only the model.

- **Fleiss κ improves when adding the model (e.g., IWA/GWA)**  
  The model behaves like a consistent rater relative to noisy humans, or aligns with the majority more reliably. This can be useful for pre-labeling, but you should still check for systematic bias.

---

## Practical guidance for building reliable labeling systems

1. Run a **pilot annotation** (a few hundred items), compute κ/α, and inspect confusions.
2. Improve the guideline with:
   - decision rules
   - counterexamples
   - explicit boundary cases
3. Re-run the pilot and verify agreement improves.
4. For production:
   - monitor agreement drift over time
   - calibrate raters periodically
   - adjudicate disagreements for gold data

---

## Summary

- **Cohen’s κ**: two raters, chance-corrected agreement.
- **Fleiss’ κ**: multiple raters (fixed raters-per-item), chance-corrected agreement.
- **Krippendorff’s α**: flexible, supports missingness and ordinal/continuous data via a distance function.
- In real evaluations, κ/α should be read together with:
  - raw agreement,
  - label distribution,
  - confusion patterns,
  - and qualitative “vibe checks” / manual audits.

## References

- How People Use ChatGPT: <https://www.nber.org/system/files/working_papers/w34255/w34255.pdf>
