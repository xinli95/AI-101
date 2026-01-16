---
title: OpenJudge Built-in Graders
---

# OpenJudge Built-in Graders

This tutorial is based on the official OpenJudge documentation page:

- OpenJudge Built-in Graders Overview: https://modelscope.github.io/OpenJudge/built_in_graders/overview/

OpenJudge provides **50+ pre-built graders** to evaluate AI responses across multiple dimensions (general quality, agent behaviors, multimodal, code & math, text matching, and format validation).

---

## 1. What is a “Grader” in OpenJudge?

In OpenJudge, a **Grader** is a standardized evaluation module that produces a structured score output (with explanations/metadata when available).

Key design features:

- **Unified API design**: graders follow a consistent interface with an `aevaluate()` method
- Standard return object includes:
  - `score`
  - `reason`
  - `metadata`

This makes it easy to compose graders into pipelines and compare results across tasks.

---

## 2. Two Implementation Types of Built-in Graders

OpenJudge organizes graders into two implementation styles:

### 2.1 LLM-Based graders

Best for:

- subjective evaluation
- nuanced response quality judgments
- safety or hallucination detection

### 2.2 Code-Based graders

Best for:

- deterministic evaluation
- zero-cost (no judge-model calls)
- high-speed batch scoring

---

## 3. Full List of Pre-built Graders (From the Official Page)

Below is the complete list of graders shown on the Built-in Graders Overview page, grouped by category.

---

### 3.1 General Graders

These graders evaluate **fundamental response quality** such as relevance, hallucination, harmfulness, instruction following, and correctness.

| Grader                       | What it evaluates                                                | Type      | Score Range |
| ---------------------------- | ---------------------------------------------------------------- | --------- | ----------- |
| `RelevanceGrader`            | How relevant a response is to the user query                     | LLM-Based | 1–5         |
| `HallucinationGrader`        | Whether response contains fabricated info unsupported by context | LLM-Based | 1–5         |
| `HarmfulnessGrader`          | Harmful, offensive, inappropriate content                        | LLM-Based | 1–5         |
| `InstructionFollowingGrader` | Whether response follows given instructions                      | LLM-Based | 1–5         |
| `CorrectnessGrader`          | Whether response matches reference answer                        | LLM-Based | 1–5         |

---

#### Example prompt (RelevanceGrader)

```python
# English Prompt
RELEVANCE_PROMPT_EN = textwrap.dedent(
    """
You are a professional data annotator responsible for evaluating how relevant the model response is to the user's query. Your task is to score according to the following criteria:

<Scoring Criteria>
A highly relevant response should:
- Directly address the user's question or request.
- Provide information that is on-topic and pertinent to the query.
- Include sufficient detail to satisfy the user's information needs.
- Stay focused without drifting to unrelated topics.
- For multi-turn conversations, maintain context awareness from previous exchanges.

Points should be deducted for:
- Completely off-topic or unrelated responses.
- Vague or superficial answers that lack specific information.
- Partial responses that omit key information requested.
- Responses that acknowledge the query but fail to provide useful content.
- Generic statements that don't specifically address the question.
</Scoring Criteria>

<Guidance>
- Carefully read the query (or conversation history) and model response.
- Determine if the response directly addresses what the user is asking.
- Check if the information provided is complete, partial, or missing.
- Assess whether the response stays on-topic or includes irrelevant content.
- For conversations, consider whether the response maintains context from earlier turns.
- The score should reflect how well the response aligns with the user's information needs.
</Guidance>

<Reminder>
The goal is to evaluate relevance to the query, not overall quality.
A score of 5 means the response is highly relevant and comprehensive.
A score of 1 means the response is completely irrelevant to the query.
</Reminder>
<query>
{query}
</query>

<response>
{response}
</response>

Additional context (ignore if empty):
<context>
{context}
</context>

The following is the correct response for your reference (ignore if empty):
<reference_response>
{reference_response}
</reference_response>

# Output Instructions
**Note**: If a reference response is provided, you may use it as a baseline for comparison to better assess the quality and relevance of the evaluated response.

Provide your evaluation in the following structured JSON format:
{
    "score": <integer between 1 and 5, where 5 means highly relevant and 1 means completely irrelevant>,
    "reason": "<brief explanation for the assigned score, specifically mentioning how the response addresses or fails to address the query>"
}

Scoring Scale:
- 5: Perfectly relevant: the response completely fulfills the user's search intent, accurately answering the question or providing the required information.
- 4: Highly relevant: the response largely meets the search requirements, possibly lacking some details or having minor inaccuracies, but still a high-quality and directly relevant result.
- 3: Partially relevant: the response has some connection to the query but does not fully meet the requirements; the user may need to further filter or supplement the information.
- 2: Weakly relevant: the response has only a weak connection to the query, possibly covering the same topic but deviating from the core intent, and has low practical value.
- 1: Irrelevant: the response is completely unrelated to the query, or contains misleading or incorrect information.

JSON:
"""
).strip()
```

### 3.2 Agent Graders

Agent graders evaluate **an AI agent’s lifecycle**, not only final answers.

OpenJudge’s agent evaluation can cover:

- actions and alignment
- tool usage and tool-call correctness
- memory write/retrieval behavior
- planning feasibility
- reflection quality
- trajectory-level reasoning

#### 3.2.1 Action Graders

| Grader                      | What it evaluates                      | Type       | Score Range |
| --------------------------- | -------------------------------------- | ---------- | ----------- |
| `ActionAlignmentGrader`     | Whether agent actions align with goals | LLM-Based  | {0, 1}      |
| `ActionLoopDetectionGrader` | Detects repetitive action loops        | Code-Based | {0, 1}      |

#### 3.2.2 Tool Graders

| Grader                        | What it evaluates                              | Type       | Score Range |
| ----------------------------- | ---------------------------------------------- | ---------- | ----------- |
| `ToolSelectionGrader`         | Whether tool choice is appropriate             | LLM-Based  | 1–5         |
| `ToolCallAccuracyGrader`      | Whether tool call is correct                   | LLM-Based  | 1–5         |
| `ToolCallSequenceMatchGrader` | Whether tool call sequence matches expectation | Code-Based | {0, 1}      |
| `ToolCallSuccessGrader`       | Whether tool calls succeeded                   | LLM-Based  | {0, 1}      |
| `ToolParameterCheckGrader`    | Whether tool parameters are valid              | LLM-Based  | {0, 1}      |

#### 3.2.3 Memory Graders

| Grader                               | What it evaluates                    | Type      | Score Range |
| ------------------------------------ | ------------------------------------ | --------- | ----------- |
| `MemoryAccuracyGrader`               | Accuracy of stored memories          | LLM-Based | {0, 1}      |
| `MemoryDetailPreservationGrader`     | Whether key details are preserved    | LLM-Based | {0, 1}      |
| `MemoryRetrievalEffectivenessGrader` | Quality of memory retrieval behavior | LLM-Based | {0, 1}      |

#### 3.2.4 Plan & Reflection Graders

| Grader                                 | What it evaluates                | Type      | Score Range |
| -------------------------------------- | -------------------------------- | --------- | ----------- |
| `PlanFeasibilityGrader`                | Whether plans are executable     | LLM-Based | {0, 1}      |
| `ReflectionAccuracyGrader`             | Whether reflections are accurate | LLM-Based | {0, 1}      |
| `ReflectionOutcomeUnderstandingGrader` | Understanding of outcomes        | LLM-Based | {0, 1}      |
| `ReflectionProgressAwarenessGrader`    | Awareness of task progress       | LLM-Based | {0, 1}      |

#### 3.2.5 Observation Graders

| Grader                             | What it evaluates                           | Type       | Score Range |
| ---------------------------------- | ------------------------------------------- | ---------- | ----------- |
| `ObservationInformationGainGrader` | Measures information gain from observations | Code-Based | [0, 1]      |

#### 3.2.6 Trajectory Graders

| Grader                          | What it evaluates                   | Type      | Score Range |
| ------------------------------- | ----------------------------------- | --------- | ----------- |
| `TrajectoryComprehensiveGrader` | Comprehensive trajectory evaluation | LLM-Based | {0, 1}      |

---

### 3.3 Text Graders

Text graders are **fast, algorithm-based** grading utilities, useful for similarity, string matching, and numeric comparisons.

| Grader                 | What it evaluates                                           | Type       | Score Range |
| ---------------------- | ----------------------------------------------------------- | ---------- | ----------- |
| `SimilarityGrader`     | Text similarity with 15+ algorithms (BLEU, ROUGE, F1, etc.) | Code-Based | [0, 1]      |
| `StringMatchGrader`    | String matching (exact/prefix/suffix/regex, etc.)           | Code-Based | {0, 1}      |
| `NumberAccuracyGrader` | Numeric comparison with tolerance                           | Code-Based | {0, 1}      |

---

### 3.4 Code Graders

These graders evaluate code correctness from execution, syntax constraints, and style.

| Grader                  | What it evaluates                             | Type       | Score Range |
| ----------------------- | --------------------------------------------- | ---------- | ----------- |
| `CodeExecutionGrader`   | Executes code against test cases              | Code-Based | [0, 1]      |
| `SyntaxCheckGrader`     | Validates Python syntax using AST             | Code-Based | {0, 1}      |
| `CodeStyleGrader`       | Checks indentation and naming conventions     | Code-Based | [0, 1]      |
| `PatchSimilarityGrader` | Compares code patches using `SequenceMatcher` | Code-Based | [0, 1]      |

---

### 3.5 Math Graders

These graders verify mathematical expressions and computations.

| Grader                       | What it evaluates                              | Type       | Score Range |
| ---------------------------- | ---------------------------------------------- | ---------- | ----------- |
| `MathExpressionVerifyGrader` | Verifies math expressions (LaTeX & plain text) | Code-Based | {0, 1}      |

---

### 3.6 Format Graders

Format graders validate structural constraints and penalize invalid formatting.

| Grader                          | What it evaluates                    | Type       | Score Range  |
| ------------------------------- | ------------------------------------ | ---------- | ------------ |
| `JsonValidatorGrader`           | Validates JSON syntax                | Code-Based | {0, 1}       |
| `JsonMatchGrader`               | Deep comparison of JSON structures   | Code-Based | {0, 1}       |
| `LengthPenaltyGrader`           | Penalizes too short/long responses   | Code-Based | ≤0 (penalty) |
| `NgramRepetitionPenaltyGrader`  | Penalizes repetitive n-grams         | Code-Based | ≤0 (penalty) |
| `ReasoningFormatGrader`         | Checks `<think>` and `<answer>` tags | Code-Based | {0, 1}       |
| `ReasoningToolCallFormatGrader` | Validates tool call format with JSON | Code-Based | {0, 1}       |

---

### 3.7 Multimodal Graders

Multimodal graders evaluate **vision-language alignment** and image-related outputs.

| Grader                   | What it evaluates                 | Type      | Score Range |
| ------------------------ | --------------------------------- | --------- | ----------- |
| `ImageCoherenceGrader`   | Image-text coherence              | LLM-Based | {0, 1}      |
| `ImageHelpfulnessGrader` | Whether images help understanding | LLM-Based | {0, 1}      |
| `TextToImageGrader`      | Text-to-image generation quality  | LLM-Based | {0, 1}      |
| `ImageEditingGrader`     | Image editing quality             | LLM-Based | {0, 1}      |

---

## 4. How to Choose the Right Built-in Grader

A quick selection guide:

### If you are evaluating “answer quality”

Start with:

- `RelevanceGrader`
- `InstructionFollowingGrader`
- `CorrectnessGrader`
- `HallucinationGrader`
- `HarmfulnessGrader`

### If you are evaluating an “agent system”

Use the Agent graders for:

- tool correctness
- action alignment
- trajectory quality
- memory behavior

### If you are evaluating deterministic tasks

Use code-based graders for speed and reproducibility:

- `SimilarityGrader`
- `StringMatchGrader`
- `CodeExecutionGrader`
- `JsonValidatorGrader`

---

## 5. Next Steps (Suggested by OpenJudge)

After reviewing built-in graders, the docs recommend exploring:

- Running graders at scale with evaluation tasks
- Creating custom graders when built-ins don’t cover your requirements

---

## 6. Summary

OpenJudge’s built-in graders provide a robust library spanning:

- **General** response quality (relevance, hallucination, safety, instruction following, correctness)
- **Agent** lifecycle evaluation (actions, tools, memory, planning, reflection, trajectory)
- **Text** similarity and matching
- **Code & math** correctness checks
- **Format** validation and penalties
- **Multimodal** coherence/helpfulness/generation/editing quality

All graders share a unified interface and are designed for reliable evaluation workflows.
