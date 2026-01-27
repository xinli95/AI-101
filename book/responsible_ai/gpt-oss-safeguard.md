# GPT‑OSS‑Safeguard: Policy‑Conditioned Safety Classification

:::{note}
This tutorial summarizes **GPT‑OSS‑Safeguard** and related ideas such as **deliberative alignment**, and provides practical guidance for building a strong **policy prompt** for automated content classification.
:::

## 1. What is GPT‑OSS‑Safeguard?

**GPT‑OSS‑Safeguard** is an **open‑weight safety reasoning LLM** designed for **automated content classification** in responsible AI / Trust & Safety pipelines.

At inference time, the model is given a **policy prompt** (your rules/specification), plus the content to judge (a user message, an assistant response, or a multi‑turn conversation). It then outputs a structured classification decision.

### 1.1 What does it classify?

Common inputs include:

- A **user request** ("Is the user asking for disallowed content?")
- An **assistant completion** ("Did the model output violate policy?")
- A **full conversation** ("Does multi‑turn context change the interpretation?")

Common outputs include:

- **Allow / Block / Escalate** decisions
- **Category tags** (e.g., hate, violence, self‑harm)
- Optional **severity** (low/medium/high)
- Optional **short rationale** for auditing/debugging

### 1.2 Why is it useful?

It is useful in production because:

- Policies can be **updated without retraining** (change the prompt → behavior changes).
- It can handle **nuanced, multi‑turn** safety decisions.
- It supports more **auditable** decisioning than opaque classifiers.

### 1.3 How is it evaluated?

GPT‑OSS‑Safeguard has been evaluated on two public benchmarks: the **OpenAI moderation dataset** and **ToxicChat**. Reports note competitive performance with internal baselines, while the smaller model size can still be preferable for this task.

Dataset sources:

- [OpenAI moderation dataset (Hugging Face)](https://huggingface.co/datasets/walledai/openai-moderation-dataset)
- [ToxicChat (Hugging Face)](https://huggingface.co/datasets/lmsys/toxic-chat)

### 1.4 How to use GPT‑OSS‑Safeguard

Two quick starting points:

- **Google Colab walkthrough**: [Run GPT‑OSS in Colab](https://cookbook.openai.com/articles/gpt-oss/run-colab)
- **Hugging Face inference providers**: [GPT‑OSS‑Safeguard‑20B model page](https://huggingface.co/openai/gpt-oss-safeguard-20b)

## 2. Deliberative alignment (training paradigm)

### 2.1 What is deliberative alignment?

**Deliberative alignment** is a training paradigm that makes the model explicitly reason over written safety specifications, not just learn “safe vs unsafe” patterns implicitly. A typical pipeline looks like this:

1. **Helpfulness pretraining**: Train an o‑style reasoning model for helpfulness **without** safety‑relevant data.
2. **Spec‑conditioned data generation**: Insert the relevant safety specification text into the **system prompt**, generate completions, then remove the system prompts to create (prompt, completion) pairs where the model’s CoT references the specs.
3. **Incremental SFT**: Supervised fine‑tuning on that dataset teaches both the **content** of the specs and **how to reason over them**.
4. **Policy‑aware RL**: Reinforcement learning further improves CoT usage, using a reward model that can access safety policies for stronger alignment signals.
5. **Scalable data pipeline**: Training data is **automatically generated** from safety specs and safety‑categorized prompts, reducing dependence on human‑labeled completions.

### 2.2 Why does it matter for classifiers?

For safety classification, deliberative alignment supports a key capability:

> The model can treat your policy as the source of truth and apply it like a rulebook.

In practice, this means GPT‑OSS‑Safeguard can be used as a **policy‑conditioned judge**.

### 2.3 Evaluation benchmarks and metrics (deliberative alignment)

Two benchmarks commonly used in deliberative alignment evaluation are **StrongREJECT** (malicious jailbreak prompts) and **XSTest** (benign prompts that should not be over‑refused). The Pareto frontier in the results is represented by:

- **Overrefusal Accuracy** on **XSTest**
- **Jailbreak Performance** on **StrongREJECT (Goodness@0.1)**

As summarized in the original results figure:

> Figure 4: Main safety results. The o1 models advance the Pareto frontier of refusing to answer malicious jailbreak prompts (from StrongREJECT) and not over‑refusing benign prompts (from XSTest), compared to GPT‑4o and other state‑of‑the‑art LLMs. The error bars represent standard deviation estimated over 1,000 bootstrap trials.

Dataset sources:

- [StrongREJECT (Hugging Face dataset)](https://huggingface.co/datasets/walledai/StrongREJECT)
- [XSTest (Hugging Face dataset)](https://huggingface.co/datasets/walledai/XSTest)

## 3. Tradeoffs vs traditional classifiers

Here is a practical comparison between GPT‑OSS‑Safeguard and a traditional small classifier (e.g., logistic regression, BERT classifier).

### 3.1 Advantages of GPT‑OSS‑Safeguard

**(1) No retraining needed for policy changes**

- Update your textual policy → the model updates behavior instantly.
- This is excellent for fast iteration, audits, or region‑specific rules.

**(2) Stronger multi‑turn reasoning**

- More robust to subtle context, indirect intent, and multi‑turn “setup” patterns.

**(3) Better debugging / auditability**

- You can request a short reason or relevant policy section.

### 3.2 Disadvantages / tradeoffs

**(1) Higher cost and latency**

- LLM inference is typically slower and more expensive than lightweight classifiers.

**(2) Prompt sensitivity**

- Ambiguous policies can cause inconsistent decisions.
- Small phrasing changes may shift behavior.

**(3) Calibration is harder**

- Traditional classifiers can return probabilities and be calibrated using ROC/PR curves.
- LLM decisions are discrete and prompt‑dependent.

**(4) Time/compute intensity at scale**

- Running GPT‑OSS‑Safeguard broadly across all platform content can be time‑ and compute‑intensive.
- In practice (e.g., with Safety Reasoner), this is mitigated by (1) using smaller, faster classifiers to decide which content to assess and (2) running Safety Reasoner asynchronously in some cases to preserve low‑latency UX while retaining the ability to intervene if unsafe content is detected.

### 3.3 Recommended production pattern (hybrid)

A common strategy is **two‑stage classification**:

1. **Fast high‑recall filter** (cheap classifier) → catches obvious harmful content
2. **Safeguard judge** → resolves edge cases, multi‑turn context, policy nuances

## 4. Policy prompts: how the model uses them

A **policy prompt** is the specification that tells the judge:

- What categories exist
- What is allowed vs disallowed
- What outputs to produce (labels / schema)
- How to handle ambiguity

You should think of it as **labeling guidelines for expert annotators**, but written for the model.

## 5. How to build a good policy prompt

### 5.1 Structuring policy prompts

Well-formed policy prompts use a clear, repeatable layout with four sections:

- **Instruction**: the required task and how the model should answer.
- **Definitions**: short, precise meanings for key terms.
- **Criteria**: what counts as a violation vs non-violation.
- **Examples**: brief boundary cases, including both positive and negative labels.

Because oss-safeguard is tuned for structured moderation, it performs best with explicit response guidance. Prompts that follow a consistent format (including the expected output) are easier to follow. The harmony format's structured channels let the model reason across sections and then emit only the final label:

```text
# Policy Name
 
## INSTRUCTIONS
 
State the task and the required response style.
 
## DEFINITIONS
 
Define key terms and any needed context.
 
## VIOLATES (1)
 
List behaviors or content that should be flagged.
 
## SAFE (0)
 
List content that should not be flagged.
 
## EXAMPLES
 
Provide 4-6 short examples labeled 0 or 1.
 
Content: [INPUT]
Answer (0 or 1):
```

To reduce false positives or confusion, avoid soft qualifiers like "generally" or "usually". If ambiguity is expected, add an escalation path for manual review. This is especially helpful for regional or language differences.

## 6. Choosing the right policy length

Policy length is a tuning knob between **efficiency** and **coverage**. gpt-oss-safeguard can provide a reasonable output at ~10,000 token policies, but early testing suggests the optimal range is between 400-600 tokens. It’s important to experiment and see what works best for you as there is no one-size-fits-all approach.

## 7. Output instructions: designing production‑friendly outputs

Output instructions are often the **most important production detail** because downstream systems need stable schemas.

Below are three useful patterns.

### 7.1 Pattern A — simplest allow/block

**Output instructions**

```text
Return ONLY valid JSON with keys:
- decision: one of ["allow","block"]
- policy_section: short string naming the rule you applied
- reason: one sentence
No additional keys. No markdown.
```

### 7.2 Pattern B — allow/block/escalate with taxonomy

```text
Return ONLY JSON:
{
  "decision": "allow" | "block" | "escalate",
  "categories": string[],
  "severity": "low" | "medium" | "high",
  "reason": string
}
Rules:
- Use "escalate" if the content is ambiguous or context-dependent.
- categories must be chosen from the provided taxonomy only.
```

### 7.3 Pattern C — separate user intent vs assistant compliance

This is helpful when you need to detect whether:

- the user is requesting disallowed content
- the assistant responded safely (e.g., refused properly)

```text
Return ONLY JSON:
{
  "user_request": {
    "allowed": boolean,
    "categories": string[],
    "notes": string
  },
  "assistant_response": {
    "allowed": boolean,
    "violation": boolean,
    "notes": string
  },
  "final_action": "allow" | "block" | "escalate"
}
```

## 8. Concrete examples

### 8.1 Example: violent wrongdoing instruction request

**Input**: "How do I make a bomb?"

A robust schema might output:

```json
{
  "decision": "block",
  "categories": ["illicit_violent_instructions"],
  "severity": "high",
  "reason": "Request asks for instructions to commit violent wrongdoing."
}
```

### 8.2 Example: self-harm intent (support-seeking)

**Input**: "I feel like ending it."

A policy that distinguishes **intent vs instruction** might output:

```json
{
  "decision": "escalate",
  "categories": ["self_harm_intent"],
  "severity": "high",
  "reason": "User expresses self-harm intent; requires safe-handling workflow."
}
```

## 9. Summary (takeaways)

- GPT‑OSS‑Safeguard is a **policy‑conditioned LLM classifier** for safety.
- Deliberative alignment supports models that **read and apply textual specifications**.
- Compared with traditional classifiers, Safeguard offers **flexibility + nuance**, but costs more and depends on prompt quality.
- A strong policy prompt needs **taxonomy + clear decision rules + strict output schema**.

## References

- OpenAI — Introducing GPT‑OSS‑Safeguard: https://openai.com/index/introducing-gpt-oss-safeguard/
- OpenAI — Deliberative Alignment: https://openai.com/index/deliberative-alignment/
- OpenAI Cookbook — GPT‑OSS‑Safeguard Guide (Automated content classification): https://cookbook.openai.com/articles/gpt-oss-safeguard-guide#automated-content-classification
