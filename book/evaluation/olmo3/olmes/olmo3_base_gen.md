# Generative QA (`olmo3:base:gen`) -- Tasks, Datasets, Metrics

This document summarizes the **generative QA** task bundle, the Hugging Face datasets it uses, and the evaluation metrics. It is self-contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `olmo3:base:gen`

This suite mixes **ranked-classification (RC)** tasks (log-likelihood over choices or continuations) with **free-form generative QA** tasks.

- **HellaSwag (RC)**
- **Winogrande (RC)**
- **LAMBADA (next-word prediction)**
- **Basic Skills (RC)**
  - arithmetic
  - coding
  - common_knowledge
  - logical_reasoning
  - string_operations
  - pattern
- **DROP (generative QA)**
- **Jeopardy (generative QA)**
- **Natural Questions Open (generative QA)**
- **SQuAD (generative QA)**
- **CoQA (generative QA)**

---

## 2) Datasets used (Hugging Face `dataset_path`)

| Task | Dataset (HF `dataset_path` and link) |
|---|---|
| HellaSwag (RC) | `allenai/hellaswag` -- [allenai/hellaswag](https://huggingface.co/datasets/allenai/hellaswag) |
| Winogrande (RC) | `winogrande` -- [winogrande](https://huggingface.co/datasets/winogrande) |
| LAMBADA | `EleutherAI/lambada_openai` -- [EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) |
| Basic Skills | `allenai/basic-skills` -- [allenai/basic-skills](https://huggingface.co/datasets/allenai/basic-skills) |
| DROP | `EleutherAI/drop` -- [EleutherAI/drop](https://huggingface.co/datasets/EleutherAI/drop) |
| Jeopardy | `soldni/jeopardy` -- [soldni/jeopardy](https://huggingface.co/datasets/soldni/jeopardy) |
| Natural Questions Open | `google-research-datasets/nq_open` -- [google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open) |
| SQuAD | `allenai/squad` -- [allenai/squad](https://huggingface.co/datasets/allenai/squad) |
| CoQA | `EleutherAI/coqa` -- [EleutherAI/coqa](https://huggingface.co/datasets/EleutherAI/coqa) |

---

## 3) Metrics used and how they are calculated

This suite uses **two families of metrics**:

### A) Ranked-classification (RC) accuracy
Used by **HellaSwag**, **Winogrande**, and **Basic Skills**. The model is scored by comparing the likelihood of candidate continuations.

Key metrics:
- **`acc_raw`**: accuracy using raw log-likelihood of each choice
- **`acc_per_token`**, **`acc_per_char`**, **`acc_per_byte`**: length-normalized variants
- **`acc_uncond`** (if enabled): likelihood normalized by an unconditioned prompt

These are computed by selecting the most likely option and checking if it matches the gold label.

### B) Generative QA metrics
Used by **DROP**, **Jeopardy**, **Natural Questions**, **SQuAD**, and **CoQA**.

- **Exact Match (EM)**: whether the generated answer matches a gold answer after normalization.
- **F1**: token-level overlap between the generated answer and the gold answer set.

Some tasks use specialized answer normalization (e.g., DROP-style numeric handling). The primary score in this suite is a **macro-average across tasks**.

### C) LAMBADA greedy accuracy
For LAMBADA, the model is evaluated on the **final word** of a passage:
- **`greedy_acc`**: whether greedy decoding produces the correct final word.

---

## 4) Prompt structure (generative and RC tasks)

### Ranked-classification tasks
These tasks **score candidate continuations** instead of free-form generation.

- **HellaSwag (RC)**: context is shown; each candidate ending is scored by log-likelihood.
- **Winogrande (RC)**: the model substitutes each option into a sentence with a blank and scores the continuation.
- **Basic Skills (RC)**: short question prompt; choices are scored by likelihood.

### Generative QA tasks
These tasks **generate an answer string** given a context.

Typical structure:
```
Title/Context/Passage: ...
Question: ...
Answer:
```

- **SQuAD / CoQA / DROP** include a passage, then a question, then `Answer:`.
- **Natural Questions** uses the question alone (short-answer style).
- **Jeopardy** uses a category + question format.

```{note}
The suite score is a macro-average across tasks, so each task contributes equally even if dataset sizes differ.
```
