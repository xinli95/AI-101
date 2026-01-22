# OLMo3 Adapt Suite (`olmo3:adapt`) -- Capability-Oriented Overview

This document summarizes the `olmo3:adapt` task suite, organized by **capability area**. The suite is used to evaluate both **OLMo-3 Think** and **OLMo-3 Instruct** models. It focuses on what is evaluated and which datasets are used.

---

## Knowledge

**Tasks**

- **MMLU (CoT)**: broad academic knowledge with short reasoning and a final answer letter.
- **PopQA**: factual recall with short answers.
- **SimpleQA**: short-form factual QA.

**Datasets (HF links)**

- MMLU: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- PopQA: [akariasai/PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- SimpleQA: [lighteval/SimpleQA](https://huggingface.co/datasets/lighteval/SimpleQA)

**Primary metrics (typical)**

- Exact Match (EM) or EM with light normalization (for MMLU-style answer extraction).
- SimpleQA also reports F1 in addition to EM.

---

## Reasoning

**Tasks**

- **BBH (CoT)**: multi-step reasoning across diverse logic tasks.
- **GPQA**: graduate-level science reasoning (MC with extracted answer letter).
- **ZebraLogic**: logic-grid reasoning with structured outputs.
- **AGI-Eval (English, CoT)**: exam-style reasoning across multiple subjects.

**Datasets (HF links)**

- BBH: [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh)
- GPQA: [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
- ZebraLogic: private dataset (`allenai/ZebraLogicBench-private`) -- [allenai/ZebraLogicBench-private](https://huggingface.co/datasets/allenai/ZebraLogicBench-private)\n+- AGI-Eval: packaged local data (AGIEval v1) -- [ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval)

**Primary metrics (typical)**

- Exact Match with task-specific answer extraction.
- ZebraLogic uses a specialized structured-equality metric.

---

## Math

**Tasks**

- **Minerva Math**: competition-style math problems.
- **GSM8K (adapt)**: grade-school word problems, boxed numeric answers.
- **OMEGA**: math generalization across exploratory/compositional/transformative splits.
- **AIME 2024/2025**: contest math with integer answers.

**Datasets (HF links)**

- Minerva Math: [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- GSM8K: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- OMEGA: `allenai/omega-{broad_category}` -- [OMEGA (arXiv:2506.18880 datasets)](https://huggingface.co/datasets?other=arxiv:2506.18880)
- AIME 2024/2025: [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

**Primary metrics (typical)**

- Exact Match or Exact Match (flexible) after answer extraction.

---

## Coding

**Tasks**

- **HumanEval+**: function completion with unit tests.
- **MBPP+**: short programming problems with tests.
- **LiveCodeBench (code generation)**: execution-based evaluation on recent coding problems.

**Datasets (HF links)**

- HumanEval+: [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus)
- MBPP+: [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus)
- LiveCodeBench: [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite)

**Primary metrics (typical)**

- **pass@1** (execution-based success on unit tests).

---

## Chat / Instruction Following

**Tasks**

- **AlpacaEval v3**: pairwise preference evaluation (LLM-as-judge).
- **IFEval**: instruction-following compliance on verifiable constraints.

**Datasets (HF links)**

- AlpacaEval: [tatsu-lab/alpaca_eval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
- IFEval: [HuggingFaceH4/ifeval](https://huggingface.co/datasets/HuggingFaceH4/ifeval)

**Primary metrics (typical)**

- AlpacaEval: win rate / length-controlled win rate.
- IFEval: instruction-level and prompt-level accuracy (strict and loose).

---

```{note}
The suite aggregates results across these tasks; each task contributes equally to capability-level and overall summaries when macro-averaged.
```

---

## Prompt structure (by task)

Below are **representative prompt shapes** for each task in `olmo3:adapt`. These are illustrative rather than exact strings.

### Knowledge

**MMLU (CoT)**

```
[Subject intro]
Question: ...
A. ...
B. ...
C. ...
D. ...
(Reasoning...)
Therefore, the answer is (X)
```

**PopQA**

```
Q: {question} A:
```

**SimpleQA**

```
Question: {question}
Answer:
```

### Reasoning

**BBH (CoT)**

```
[Task description]
Q: ...
A: (reasoning...) final answer
```

**GPQA**

```
Answer the following multiple choice question...
Let's think step by step:
...
The correct answer is: (X)
```

**ZebraLogic**

```
[Logic puzzle statement]
[Required output format]
```

**AGI-Eval (English, CoT)**

```
[Exam-style prompt]
Question: ...
Answer: (reasoning...) final answer
```

### Math

**Minerva Math**

```
Problem: ...
Solution: (reasoning...)
\boxed{answer}
```

**GSM8K (adapt)**

```
Solve the following grade school math word problem:
{question}

Show your work and conclude with "Therefore, the final answer is \boxed{answer}."
```

**OMEGA**

```
[Problem statement]
Present the answer in LaTeX: \boxed{...}
```

**AIME 2024/2025**

```
{problem}

Present the answer in LaTeX: \boxed{Your answer}
```

### Coding

**HumanEval+**

```
# Problem description
def func(...):
    ...
# Complete the function
```

**MBPP+**

```
# Problem statement
# Function signature
# Examples/tests
```

**LiveCodeBench (code generation)**

```
Problem: ...
Input/Output specs...
Constraints...
```

### Chat / Instruction Following

**AlpacaEval v3**

```
Instruction: ...
(Assistant response)
```

**IFEval**

```
[Instruction(s) with verifiable constraints]
```
