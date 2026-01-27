# Overview

This section frames **responsible AI** around two main aspects:

1. **Safety evaluation of LLMs** (how safety performance is measured and reported)
2. **The moderation layer** (how content is classified and enforced at runtime)

## What’s included

- **OpenAI Safety Evaluations Hub**: a structured view of safety and reliability evaluations, including refusal behavior, jailbreak robustness, hallucinations, and instruction hierarchy testing. See [OpenAI Safety Evaluations Hub](./openai_evaluation_hub.md).
- **OpenAI Moderation API**: a traditional classifier‑based moderation layer with a published taxonomy and outputs. See [OpenAI Moderation API](./openai_moderation.md).
- **GPT‑OSS‑Safeguard**: a generative moderation layer that uses policy prompts and structured outputs. See [GPT‑OSS‑Safeguard](./gpt-oss-safeguard.md).
- **Anthropic safety approach**: an enterprise example with a product‑safety lens and layered mitigations. See [Anthropic — Protecting the Well‑Being of Users](./anthropic.md).

## How to read this section

- Start with **evaluations** to understand how model safety is measured.
- Then read **moderation** to see how those categories become runtime controls (classifier or generative).
- Use **GPT‑OSS‑Safeguard** if you want policy‑prompted, structured moderation.
- Compare with **Anthropic’s approach** for an enterprise‑focused example.
