# STEM MC QA (`stem_qa_mc`) — Tasks, Datasets, Metrics

This document summarizes the **STEM multiple‑choice QA** task bundle (`stem_qa_mc`), the **Hugging Face datasets** it uses, and the **metrics** used to score it. It is self‑contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `stem_qa_mc`

`stem_qa_mc` is a **bundle of multiple‑choice STEM QA tasks**. It includes:

- **ARC (MC, xlarge)**
  - ARC‑Easy (MC)
  - ARC‑Challenge (MC)
- **MMLU STEM (MC)**
  - 18 STEM subjects (see list below)
- **MedMCQA (MC)**
- **MedQA‑English (MC)**
- **SciQ (MC, xlarge)**

### MMLU STEM subjects (MC)

- abstract_algebra
- astronomy
- college_biology
- college_chemistry
- college_computer_science
- college_mathematics
- college_physics
- computer_security
- conceptual_physics
- electrical_engineering
- elementary_mathematics
- high_school_biology
- high_school_chemistry
- high_school_computer_science
- high_school_mathematics
- high_school_physics
- high_school_statistics
- machine_learning

---

## 2) Datasets used (Hugging Face `dataset_path`)

| Task group | Dataset (HF `dataset_path` and link) |
|---|---|
| ARC‑Easy / ARC‑Challenge (MC) | `ai2_arc` — [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| MMLU STEM (MC) | `cais/mmlu` — [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| MedMCQA (MC) | `openlifescienceai/medmcqa` — [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| MedQA‑English (MC) | `davidheineman/medqa-en` — [davidheineman/medqa-en](https://huggingface.co/datasets/davidheineman/medqa-en) |
| SciQ (MC) | `sciq` — [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) |

---

## 3) Metrics used and how they are calculated

All tasks in `stem_qa_mc` are **multiple‑choice** evaluations. Each question has *N* answer options. The model is scored by computing **log‑likelihood** for each option and comparing them.

### Per‑choice scores
For each choice *i*:

- **`sum_logits_i`** = sum of log‑probabilities for the choice continuation
- **`num_tokens_i`**, **`num_chars_i`**, **`num_bytes_i`** = length of the choice continuation
- **`logits_per_token_i`** = `sum_logits_i / num_tokens_i`
- **`logits_per_char_i`** = `sum_logits_i / num_chars_i`
- **`bits_per_byte_i`** = `-log2(e) * (sum_logits_i / num_bytes_i)`

Lower **bits/byte** is better (it corresponds to higher likelihood per byte).

### Accuracy metrics (per question)
Let `gold` be the correct option index.

- **`acc_raw`**: 1 if `argmax(sum_logits)` == `gold`, else 0
- **`acc_per_token`**: 1 if `argmax(logits_per_token)` == `gold`, else 0
- **`acc_per_char`**: 1 if `argmax(logits_per_char)` == `gold`, else 0
- **`acc_per_byte`**: 1 if `argmin(bits_per_byte)` == `gold`, else 0

### Optional unconditioned accuracy
Some tasks can compute a length‑normalized, *unconditioned* variant:

- For each choice: `score_i = sum_logits_i - sum_logits_uncond_i`
- **`acc_uncond`**: 1 if `argmax(score_i)` == `gold`, else 0

### Correct‑choice log‑likelihood metrics
These are the log‑likelihood values for the **gold** option only:

- **`sum_logits_corr`** = `sum_logits_gold`
- **`logits_per_token_corr`** = `logits_per_token_gold`
- **`logits_per_char_corr`** = `logits_per_char_gold`
- **`bits_per_byte_corr`** = `bits_per_byte_gold`

### Aggregation to task and suite scores
- **Per task:** metrics are averaged across questions (mean).
- **For the `stem_qa_mc` bundle:** metrics are **macro‑averaged** across its tasks (each task contributes equally, regardless of size).

---

```{note}
Because multiple‑choice options can differ in length, the suite reports both raw accuracy and length‑normalized variants (per‑token/char/byte). The “primary” score for a specific task may differ (e.g., `acc_raw` vs `acc_per_char`), but the suite itself is aggregated by macro‑average across tasks.
```

---

## 4) Prompt structure (MC STEM tasks)

All tasks in `stem_qa_mc` use the shared multiple‑choice prompt builder. The **general shape** is:

```
Question: {question}
A. {choice_1}
B. {choice_2}
C. {choice_3}
D. {choice_4}
Answer:
```

Some tasks may have 5 choices (A–E). The final `Answer:` is the point where the model is scored on the likelihood of each choice label.

### Per‑task notes

- **ARC‑Easy / ARC‑Challenge (MC)**  
  Uses the standard MC layout shown above; includes 4–5 choices depending on the question.

- **MMLU STEM (MC)**  
  Adds a **subject‑specific intro line** before the question, e.g.  
  `The following are multiple choice questions (with answers) about {subject}.`

- **MedMCQA (MC)**  
  Standard MC layout with 4 options (A–D).

- **MedQA‑English (MC)**  
  Standard MC layout, often 4 options but can be more.

- **SciQ (MC)**  
  Standard MC layout, but the answer choices are **shuffled per example** (deterministic by index) so the correct answer is not always the same letter.

### Few‑shot formatting

When `num_shots > 0`, few‑shot examples are appended before the target question:

```
{example_1_question_and_choices}{example_1_answer}

{example_2_question_and_choices}{example_2_answer}

...

{current_question_and_choices}
Answer:
```

This keeps the target question in the same format while providing labeled examples above it.
