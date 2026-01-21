# Code (`olmo3:base:code`) -- Tasks, Datasets, Metrics

This document summarizes the **code generation** task bundle, the Hugging Face datasets it uses, and the evaluation metrics. It is self-contained and focuses on the evaluation ideas rather than code.

---

## 1) Tasks covered in `olmo3:base:code`

This suite evaluates **code generation with execution-based testing**.

- **BigCodeBench**
- **HumanEval (Codex format)**
- **DeepSeek LeetCode**
- **DS-1000**
- **MBPP**
- **MultiPL-E HumanEval (6 languages)**
- **MultiPL-E MBPP (6 languages)**

MultiPL-E uses the following 6-language subset:
- cpp, java, js, php, rs, sh

---

## 2) Datasets used (Hugging Face `dataset_path`)

| Task | Dataset (HF `dataset_path` and link) |
|---|---|
| BigCodeBench | `bigcode/bigcodebench` -- [bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench) |
| HumanEval | `openai_humaneval` -- [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| DeepSeek LeetCode | `davidheineman/deepseek-leetcode` -- [davidheineman/deepseek-leetcode](https://huggingface.co/datasets/davidheineman/deepseek-leetcode) |
| DS-1000 | `xlangai/DS-1000` -- [xlangai/DS-1000](https://huggingface.co/datasets/xlangai/DS-1000) |
| MBPP | `google-research-datasets/mbpp` -- [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| MultiPL-E (HumanEval + MBPP) | `nuprl/MultiPL-E` -- [nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E) |

---

## 3) Metrics used and how they are calculated

Code tasks use **execution-based correctness** rather than exact string match.

### Pass@k
- **pass@k**: the probability that **at least one** of the *k* generated samples passes all unit tests.
- In this suite, **pass@1** is the primary score for each task.

### How pass@k is computed
1. Generate *k* code samples for a prompt.
2. Run each sample against the task's test suite.
3. Mark success if **any** sample passes all tests.

Metrics are averaged across problems, and then **macro-averaged** across tasks for the suite score.

---

## 4) Prompt structure (code tasks)

These tasks provide a **problem description and/or code stub**, and the model must generate a valid completion.

Common pattern:
```
# Problem description / function signature
# ...

# Write the function here
```

- **HumanEval / MBPP**: small Python functions with tests.
- **BigCodeBench**: larger programmatic tasks, often with hidden tests.
- **DS-1000**: data-science tasks with library-specific expectations.
- **DeepSeek LeetCode**: LeetCode-style prompts.
- **MultiPL-E**: same task translated into multiple languages; the model must follow language-specific syntax.

```{note}
Because evaluation is test-based, formatting differences are acceptable as long as the code executes and passes all tests.
```
