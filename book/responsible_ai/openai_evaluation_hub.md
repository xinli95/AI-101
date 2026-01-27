# OpenAI Safety Evaluations Hub

This tutorial summarizes the **OpenAI Safety Evaluations Hub** with a focus on:

- **What types of evaluations are implemented**
- **What metrics are reported**
- **What datasets / benchmark suites are included**

> Source: OpenAI Safety Evaluations Hub — https://openai.com/safety/evaluations-hub/

## 1. What is the Safety Evaluations Hub?

The Safety Evaluations Hub is a public-facing collection of evaluation results that OpenAI uses to track **safety and reliability** performance of its models over time.

These evaluations are intended to answer questions like:

- _Does the model comply with policy and refuse unsafe requests?_
- _Can the model be jailbroken into producing disallowed content?_
- _How often does the model hallucinate?_
- _Does the model follow the instruction hierarchy (System > Developer > User)?_

The hub focuses primarily on **text-based** evaluations across four major categories:

1. **Disallowed content**
2. **Jailbreak evaluations**
3. **Hallucination evaluations**
4. **Instruction hierarchy**

## 2. Disallowed Content Evaluations

### 2.1 Goal

Disallowed content evaluations measure whether a model **refuses** or **does not comply** with requests that violate policy (e.g., unsafe advice, hateful content, privacy violations).

This is sometimes described as testing **policy compliance** or **refusal behavior**.

### 2.2 Evaluation approach

OpenAI uses an automatic grading approach (an **autograder**) to score whether the response should be considered unsafe under the policy and Model Spec.

### 2.3 Primary metric: `not_unsafe`

The key metric reported for disallowed content is:

- **`not_unsafe`**: whether the model avoided producing unsafe content

Intuitively, higher `not_unsafe` scores mean the model is better at **not producing disallowed outputs** when prompted.

### 2.4 Datasets / evaluation suites

The hub includes multiple refusal-oriented sets, such as:

- **Standard Refusal evaluation** (described as largely saturated)
- **Challenging refusal evaluation**
- **Production Benchmarks** (newer benchmark set that is described as highly multi-turn)

### 2.5 Category-level breakdowns

Results are often disaggregated by content category, including examples like:

- hate and harassment
- sexual content (including minors)
- self-harm
- illicit behavior (nonviolent and violent)
- extremism
- personal data / privacy

This category reporting is important because a model can behave very differently across safety domains.

## 3. Jailbreak Evaluations

### 3.1 Goal

Jailbreak evaluations test whether adversarial prompts can **bypass safety protections** and cause the model to produce disallowed content.

This can be viewed as a measure of **robustness** against prompt attacks.

### 3.2 Benchmarks included

The hub reports jailbreak results on at least two datasets:

- **[StrongReject](https://huggingface.co/datasets/walledai/StrongREJECT)** (an academic jailbreak benchmark)
- **Human-sourced jailbreaks** (attacks written or collected from human red-teaming)

### 3.3 Why jailbreak evaluation matters

A model can perform well on ordinary refusal tests but still fail when the prompt is more sophisticated, such as:

- role-play attacks ("pretend you are...")
- instruction obfuscation
- multi-turn manipulation
- indirect requests or constrained formats

Jailbreak evaluations help measure these more realistic adversarial conditions.

## 4. Hallucination Evaluations

### 4.1 Goal

Hallucination evaluations measure how frequently a model produces **incorrect factual statements** when answering questions.

This category is about **factuality and reliability** rather than safety policy compliance.

### 4.2 Datasets included

The hub includes at least two hallucination-focused evaluation sets:

1. **[SimpleQA](https://huggingface.co/datasets/basicv8vc/SimpleQA)** (variant: [simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified))
   - A dataset of **4,000** fact-seeking questions
   - Ground-truth answers are short and unambiguous

2. **PersonQA**
   - Questions about real people
   - Answers are based on **publicly available facts**

### 4.3 Metrics used

For hallucination evaluations, the hub reports:

- **Accuracy**: fraction of questions answered correctly
- **Hallucination rate**: fraction of responses that hallucinate

### 4.4 Important note on browsing

The hub notes that hallucination results are reported for model behavior **without web browsing enabled**.

That means these scores measure what the model can do from its internal knowledge and reasoning alone.

## 5. Instruction Hierarchy Evaluations

### 5.1 Goal

Instruction hierarchy evaluations measure whether a model follows the correct priority ordering when instructions conflict.

OpenAI frames this around an **instruction hierarchy**, where higher-priority messages should override lower-priority ones:

- **System messages** (highest priority)
- **Developer messages**
- **User messages** (lowest priority)

### 5.2 What gets tested

The hub includes tests that cover multiple types of instruction conflicts and attacks, including:

- System ↔ User conflicts
- Developer ↔ User conflicts
- System ↔ Developer conflicts
- tutor jailbreak variants
  - The model is instructed to be a math tutor and not give away the solution, and the user attempts to trick the model into giving away the solution.
  - To pass this eval, the model must not give away the answer to a math problem when the user attempts to trick the tutor.
- phrase / password protection checks
  - To pass this eval, the model must not output a specific phrase or a bespoke password which are specified in a system message.

### 5.3 Pass condition (what counts as success)

A test case is considered successful when the model follows the **highest-priority valid instruction** instead of being manipulated by a lower-priority one.

## 6. How to interpret the hub results

When reading results from the Safety Evaluations Hub, it helps to interpret them along these dimensions:

1. **Coverage**: which failure modes are being tested (policy, jailbreaks, factuality, instruction conflicts)?
2. **Metrics**: does the metric directly reflect the risk you care about (e.g., `not_unsafe`, hallucination rate)?
3. **Dataset realism**: do prompts resemble real user behavior or only synthetic tests?
4. **Granularity**: do results break down by safety domain or attack type?

In practice, a model may score differently across these evaluation classes, and improvements in one category (e.g., refusal) can come with tradeoffs in others (e.g., helpfulness).

---

## Summary

The OpenAI Safety Evaluations Hub organizes safety and reliability measurement into four key evaluation families:

- **Disallowed content**: policy compliance, reported with `not_unsafe` and disaggregated by safety categories
- **Jailbreaks**: robustness to adversarial prompting, including [StrongReject](https://huggingface.co/datasets/walledai/StrongREJECT) and human-sourced jailbreaks
- **Hallucinations**: factuality measured with accuracy and hallucination rate on [SimpleQA](https://huggingface.co/datasets/basicv8vc/SimpleQA) (variant: [simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified)) and PersonQA
- **Instruction hierarchy**: obedience to System > Developer > User priority under conflicting instructions

Together, these evaluations provide a structured view of how safety behavior is tested and tracked over time.
