# Code FIM (`olmo3:base:code_fim`) -- Tasks, Datasets, Metrics

This document summarizes the **code infilling (FIM)** task bundle, the Hugging Face dataset it uses, and the evaluation metrics. It is self-contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `olmo3:base:code_fim`

This suite evaluates **fill-in-the-middle (FIM)** code completion:

- **HumanEval FIM (single-line)**
- **HumanEval FIM (multi-line)**
- **HumanEval FIM (random span)**

---

## 2) Dataset used (Hugging Face `dataset_path`)

| Task group | Dataset (HF `dataset_path` and link) |
|---|---|
| HumanEval FIM variants | `loubnabnl/humaneval_infilling` -- [loubnabnl/humaneval_infilling](https://huggingface.co/datasets/loubnabnl/humaneval_infilling) |

---

## 3) Metrics used and how they are calculated

These tasks use **execution-based correctness**.

### Pass@k
- **pass@k**: the probability that **at least one** of the *k* generated infills passes all unit tests.
- In this suite, **pass@1** is the primary score.

The model's infilled code is executed with the provided tests; a completion is correct if all tests pass.

---

## 4) Prompt structure (FIM)

FIM prompts split code into **prefix** and **suffix**, and the model must generate the missing middle.

Typical pattern (token names may vary by model):
```
<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>
```

- **Single-line / multi-line / random** variants differ only in the size and location of the missing span.

```{note}
The evaluation is based on test execution, so any correct infill that passes the tests is accepted.
```
