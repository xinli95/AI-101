# Non-STEM MC QA (`olmo3:base:nonstem_qa_mc`) -- Tasks, Datasets, Metrics

This document summarizes the **non-STEM multiple-choice QA** task bundle, the Hugging Face datasets it uses, and the evaluation metrics. It is self-contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `olmo3:base:nonstem_qa_mc`

This suite is a **bundle of multiple-choice QA tasks** focused on non-STEM domains plus MC conversions of open QA datasets.

- **MMLU Humanities (MC)**
  - formal_logic
  - high_school_european_history
  - high_school_us_history
  - high_school_world_history
  - international_law
  - jurisprudence
  - logical_fallacies
  - moral_disputes
  - moral_scenarios
  - philosophy
  - prehistory
  - professional_law
  - world_religions

- **MMLU Social Sciences (MC)**
  - econometrics
  - high_school_geography
  - high_school_government_and_politics
  - high_school_macroeconomics
  - high_school_microeconomics
  - high_school_psychology
  - human_sexuality
  - professional_psychology
  - public_relations
  - security_studies
  - sociology
  - us_foreign_policy

- **MMLU Other (MC)**
  - anatomy
  - business_ethics
  - clinical_knowledge
  - college_medicine
  - global_facts
  - human_aging
  - management
  - marketing
  - medical_genetics
  - miscellaneous
  - nutrition
  - professional_accounting
  - professional_medicine
  - virology

- **Commonsense QA (MC)**
- **PIQA (MC)**
- **SocialIQA (MC)**

- **Open-QA to MC conversions (gen2mc)**
  - CoQA (MC)
  - DROP (MC)
  - Jeopardy (MC)
  - Natural Questions Open (MC)
  - SQuAD (MC)

---

## 2) Datasets used (Hugging Face `dataset_path`)

| Task group | Dataset (HF `dataset_path` and link) |
|---|---|
| MMLU Humanities / Social Sciences / Other (MC) | `cais/mmlu` -- [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| Commonsense QA (MC) | `commonsense_qa` -- [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) |
| PIQA (MC) | `piqa` -- [nthngdy/piqa](https://huggingface.co/datasets/nthngdy/piqa) |
| SocialIQA (MC) | `social_i_qa` -- [social_i_qa](https://huggingface.co/datasets/social_i_qa) |
| CoQA gen2mc (MC) | `allenai/coqa-gen2mc` -- [allenai/coqa-gen2mc](https://huggingface.co/datasets/allenai/coqa-gen2mc) |
| DROP gen2mc (MC) | `allenai/drop-gen2mc` -- [allenai/drop-gen2mc](https://huggingface.co/datasets/allenai/drop-gen2mc) |
| Jeopardy gen2mc (MC) | `allenai/jeopardy-gen2mc` -- [allenai/jeopardy-gen2mc](https://huggingface.co/datasets/allenai/jeopardy-gen2mc) |
| Natural Questions gen2mc (MC) | `allenai/nq-gen2mc` -- [allenai/nq-gen2mc](https://huggingface.co/datasets/allenai/nq-gen2mc) |
| SQuAD gen2mc (MC) | `allenai/squad-gen2mc` -- [allenai/squad-gen2mc](https://huggingface.co/datasets/allenai/squad-gen2mc) |

---

## 3) Metrics used and how they are calculated

All tasks in this suite are **multiple-choice** evaluations. Each question has *N* answer options. The model is scored by computing **log-likelihood** for each option and comparing them.

### Per-choice scores
For each choice *i*:

- **`sum_logits_i`** = sum of log-probabilities for the choice continuation
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
Some tasks can compute a length-normalized, *unconditioned* variant:

- For each choice: `score_i = sum_logits_i - sum_logits_uncond_i`
- **`acc_uncond`**: 1 if `argmax(score_i)` == `gold`, else 0

### Correct-choice log-likelihood metrics
These are the log-likelihood values for the **gold** option only:

- **`sum_logits_corr`** = `sum_logits_gold`
- **`logits_per_token_corr`** = `logits_per_token_gold`
- **`logits_per_char_corr`** = `logits_per_char_gold`
- **`bits_per_byte_corr`** = `bits_per_byte_gold`

### Aggregation to suite score
- **Per task:** metrics are averaged across questions (mean).
- **Suite level:** metrics are **macro-averaged** across tasks (each task contributes equally).

---

## 4) Prompt structure (non-STEM MC tasks)

All tasks use a shared multiple-choice prompt pattern. The **general shape** is:

```
Question: {question}
A. {choice_1}
B. {choice_2}
C. {choice_3}
D. {choice_4}
Answer:
```

### Variants you should expect

- **PIQA** uses a goal-oriented prefix:
  `Goal: {goal}` then two options (A/B).
- **SocialIQA** uses a short context plus a question.
- **Gen2MC datasets (CoQA/DROP/Jeopardy/NQ/SQuAD)** embed a passage or context, then a question, then MC options.

### Few-shot formatting
When `num_shots > 0`, few-shot examples are appended before the target question:

```
{example_1_question_and_choices}{example_1_answer}

{example_2_question_and_choices}{example_2_answer}

...

{current_question_and_choices}
Answer:
```

```{note}
Because choices can differ in length, the suite reports both raw accuracy and length-normalized variants (per-token/char/byte). The suite score is a macro-average across tasks.
```
