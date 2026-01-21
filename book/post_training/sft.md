# Supervised Fine-tuning

## Overview

Supervised Fine-Tuning (SFT) adapts a pretrained large language model (LLM) to follow instructions, match a desired style, or specialize in a domain.
This tutorial covers:

- What SFT is optimizing (theory)
- How SFT is implemented in practice (concatenation + causal masking + loss masking)
- What catastrophic forgetting is and why it happens
- Practical mitigation strategies

---

## 1. What problem is SFT solving?

A pretrained autoregressive LLM is typically trained with next-token prediction on large-scale text:

$$
\max_{\theta} \; \mathbb{E}_{x \sim D_{\text{pretrain}}}\left[\sum_{t}\log p_{\theta}(x_t \mid x_{<t})\right]
$$

Pretraining gives broad language competence, but it does not guarantee reliable instruction-following.

SFT uses supervised examples of prompt-response pairs:

- Prompt $x$: instruction / user message / context
- Response $y$: the ideal assistant output

The SFT objective is to maximize the conditional likelihood of responses given prompts:

$$
\max_{\theta}\; \mathbb{E}_{(x,y)\sim D_{\text{SFT}}}\left[\log p_{\theta}(y \mid x)\right]
$$

Equivalently, minimize negative log-likelihood:

$$
\min_{\theta}\; \mathbb{E}_{(x,y)\sim D_{\text{SFT}}}\left[-\log p_{\theta}(y \mid x)\right]
$$

---

## 2. Token-level SFT loss (teacher forcing)

Let the response tokens be $y=(y_1,\dots,y_T)$. For an autoregressive model:

$$
p_{\theta}(y \mid x) = \prod_{t=1}^{T} p_{\theta}(y_t \mid x, y_{<t})
$$

So the standard cross-entropy / negative log-likelihood loss is:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log p_{\theta}(y_t \mid x, y_{<t})
$$

In training, we feed the ground-truth previous response tokens $y_{<t}$ rather than the model's sampled tokens. This is called **teacher forcing**.

---

## 3. How SFT is implemented for decoder-only Transformers

Most modern LLMs are decoder-only Transformers (causal language models). SFT is commonly implemented by concatenating prompt and response into a single token sequence.

### 3.1 Sequence construction

Given a prompt $x$ and response $y$, we build a single sequence:

$$
s = [\text{BOS},\; x,\; y,\; \text{EOS}]
$$

In tokens:

```
<BOS>  prompt_tokens  response_tokens  <EOS>
```

For chat models, $x$ and $y$ are usually wrapped with a chat template (e.g., System/User/Assistant markers). The mechanics remain the same.

### 3.2 Causal attention mask

Even though the full sequence is provided, the model uses a **causal mask** so position $t$ can only attend to positions $\le t$.
That keeps the factorization:

$$
p_{\theta}(s_t \mid s_{<t})
$$

---

## 4. Loss masking: compute loss only on the response

A key practical detail: we usually compute loss only for tokens in the response.

### 4.1 Why not compute loss on prompt tokens?

At inference time, the prompt is given as input context; we do not want the model to learn to "generate" the prompt.
So prompt tokens are treated as conditioning context.

### 4.2 Masked loss

Let $m_t \in \{0,1\}$ indicate whether position $t$ is part of the response. Then:

$$
\mathcal{L}_{\text{masked}}(\theta) = -\sum_{t} m_t\,\log p_{\theta}(s_t \mid s_{<t})
$$

This means:

- prompt tokens: $m_t=0$ (no gradient)
- response tokens: $m_t=1$ (contribute to loss)

---

## 5. Why training is parallel but inference is sequential

### 5.1 Training (SFT)

During training, the full prompt+response sequence is known. The model can compute logits for **all positions** in a single forward pass (batched matrix ops).
Even though dependencies are left-to-right, computation is parallelizable across positions.

### 5.2 Inference (decoding)

During inference, future response tokens are unknown. So generation must proceed token-by-token:

1. Start with prompt tokens
2. Predict next token
3. Append it
4. Repeat

This is inherently sequential.

---

# Catastrophic Forgetting in SFT

## 6. What is catastrophic forgetting?

**Catastrophic forgetting** means that after fine-tuning on a new dataset, the model improves on the new distribution but degrades on previously learned capabilities.
In LLMs, "previous capabilities" can include:

- general instruction following
- reasoning (math/logic)
- coding
- factual knowledge and breadth
- conversational naturalness
- safety/refusal behaviors (if applicable)

---

## 7. Why does forgetting happen?

SFT optimizes only the new dataset objective:

$$
\min_{\theta} \; \mathbb{E}_{(x,y)\sim D_{\text{SFT}}}\left[-\log p_{\theta}(y \mid x)\right]
$$

If $D_{\text{SFT}}$ is narrow or distribution-shifted relative to the model's pretraining/instruction distribution, gradients can "pull" parameters toward the new behavior and overwrite representations that supported old behaviors.

Common triggers:

- Narrow single-domain dataset
- Highly templated outputs / style bias
- Too many steps or too large learning rate
- Small dataset leading to overfitting + style collapse

---

## 8. Forgetting vs classic overfitting

They are related but not identical:

- **Overfitting**: memorization / spurious correlations, train improves but test degrades on the _same task distribution_.
- **Forgetting**: new task improves but _other tasks/capabilities_ degrade due to interference/overwriting.

In practice, narrow SFT can cause both at once.

---

# Mitigating Catastrophic Forgetting

## 9. Data replay / mixture training (most practical)

Mix your new dataset with a replay dataset representing capabilities you want to preserve:

$$
D_{\text{mix}} = \alpha D_{\text{new}} + (1-\alpha) D_{\text{replay}}
$$

If the new dataset is narrow, a common starting point is $\alpha \in [0.3, 0.7]$.

Replay sources:

- a subset of high-quality instruction tuning data
- a curated "keep skills" set (math/code/general QA)

---

## 10. Reduce update aggressiveness

If forgetting is severe, your fine-tuning is likely too aggressive.
Common fixes:

- lower learning rate
- fewer epochs / fewer steps
- early stopping
- mild weight decay

---

## 11. Parameter-efficient fine-tuning (LoRA / QLoRA)

LoRA-style methods adapt behavior by learning low-rank updates while keeping most base weights fixed.
This often preserves general capabilities better than full fine-tuning.

---

## 12. Regularization toward the base model

### 12.1 L2 regularization to initial weights

Let $\theta_0$ be pretrained weights. Add:

$$
\mathcal{L}(\theta)=\mathcal{L}_{\text{SFT}}(\theta)+\lambda\|\theta-\theta_0\|^2
$$

### 12.2 KL regularization on outputs

Penalize deviation in output distributions from the base model:

$$
\mathcal{L}(\theta)=\mathcal{L}_{\text{SFT}}(\theta)+\beta\,\mathrm{KL}\big(p_{\theta}(\cdot\mid x)\,\|\,p_{\theta_0}(\cdot\mid x)\big)
$$

---

## 13. Multi-task balancing and batching

If you fine-tune across multiple tasks, use explicit multi-task sampling:

- tag samples by task
- balance proportions
- ensure batches contain diverse tasks

This reduces specialization collapse.

---

## 14. Freeze parts of the model

Freezing lower layers (embeddings, early transformer blocks) can help preserve general language representations.
Fine-tune only higher layers to reduce overwriting.

---

## 15. Improve dataset diversity and quality

Many forgetting issues are data-driven.
To reduce drift:

- increase prompt diversity
- avoid overly repetitive templates
- include multiple styles and lengths
- remove low-quality or inconsistent labels

---

# Summary

SFT trains an LLM to maximize $\log p_{\theta}(y\mid x)$ using teacher forcing on concatenated prompt+response sequences, while computing loss only on response tokens.
Because SFT focuses on a potentially narrow distribution, it can cause catastrophic forgetting.
The most practical mitigations are replay/mixed training, gentler optimization (LR/steps), and parameter-efficient tuning (LoRA/QLoRA).
