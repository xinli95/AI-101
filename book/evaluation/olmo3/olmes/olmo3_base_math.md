# Math (`olmo3:base:math`) -- Tasks, Datasets, Metrics

This document summarizes the **math** task bundle, the Hugging Face datasets it uses, and the evaluation metrics. It is self-contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `olmo3:base:math`

This suite combines **math word problems** and **competition-style math**.

- **GSM8K (sampling, 8-shot)**
- **GSM-Symbolic (sampling, 8-shot)**
  - main
  - p1
  - p2
- **Minerva Math (sampling, 4-shot)**
  - algebra
  - counting_and_probability
  - geometry
  - intermediate_algebra
  - number_theory
  - prealgebra
  - precalculus

---

## 2) Datasets used (Hugging Face `dataset_path`)

| Task group | Dataset (HF `dataset_path` and link) |
|---|---|
| GSM8K | `gsm8k` -- [gsm8k](https://huggingface.co/datasets/gsm8k) |
| GSM-Symbolic | `apple/GSM-Symbolic` -- [apple/GSM-Symbolic](https://huggingface.co/datasets/apple/GSM-Symbolic) |
| Minerva Math | `EleutherAI/hendrycks_math` -- [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |

---

## 3) Metrics used and how they are calculated

These tasks typically **sample multiple generations** and then score them with pass@k or majority-vote metrics.

### Answer extraction
The model generates a solution. An answer string (usually a number) is extracted from the completion using task-specific normalization rules.

### Exact Match (EM)
- **Exact Match**: a prediction is correct if the extracted answer matches a gold answer after normalization.

### Pass@k
- **pass@k**: the probability that **at least one** of the *k* sampled generations is correct.
- In this suite, **pass@1** is the primary score for GSM8K and GSM-Symbolic configurations.

### Majority@k (Maj@k)
- **maj@k**: majority vote accuracy over *k* sampled generations.

### Aggregation to suite score
- **Per task:** metrics are averaged across questions.
- **Suite level:** metrics are **macro-averaged** across tasks.

---

## 4) Prompt structure (math tasks)

These tasks use **few-shot chain-of-thought (CoT)** prompts by default.

Typical pattern:
```
Question: ...
Answer: (reasoning steps...) ... final answer ...
```

- **GSM8K / GSM-Symbolic** use 8-shot demonstrations with step-by-step reasoning, then a final answer.
- **Minerva Math** uses a math-problem format (problem statement + solution), typically ending with a boxed final answer.

```{note}
Because the tasks are sampled, pass@k and maj@k are meaningful summaries of the model's ability to reach a correct solution across multiple attempts.
```
