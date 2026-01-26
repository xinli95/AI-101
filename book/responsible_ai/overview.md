# Overview

This section provides a practical view of **responsible AI** across three layers:

1. **Evaluation and transparency** (how safety performance is measured and reported)
2. **Moderation and enforcement** (how content is classified at runtime)
3. **Policy‑conditioned safety** (how LLMs can follow explicit rules)

## What’s included

- **OpenAI Safety Evaluations Hub**: a structured view of safety and reliability evaluations, including refusal behavior, jailbreak robustness, hallucinations, and instruction hierarchy testing. See [OpenAI Safety Evaluations Hub](./openai_evaluation_hub.md).
- **OpenAI Moderation API**: how moderation works, what outputs it provides, and the full taxonomy used for content classification. See [OpenAI Moderation API](./openai_moderation.md).
- **GPT‑OSS‑Safeguard**: an open‑weight safety classifier with policy prompts, deliberative alignment framing, and production tradeoffs. See [GPT‑OSS‑Safeguard](./gpt-oss-safeguard.md).
- **Anthropic safety approach**: a case study on user well‑being risks and layered mitigations in product and model behavior. See [Anthropic — Protecting the Well‑Being of Users](./anthropic.md).

## How to read this section

- Start with **evaluations** to understand what safety looks like when measured.
- Move to **moderation** to see how those safety categories become actionable controls.
- Use **GPT‑OSS‑Safeguard** for policy‑conditioned classification and prompts.
- Compare with **Anthropic’s approach** for an alternative product‑safety lens.
