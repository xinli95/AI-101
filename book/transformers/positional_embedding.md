# Positional Embeddings and Context Extension

Transformers are permutation-invariant by default: without extra signals, they do not know token order.
So we must inject **positional information**.

Modern LLMs typically use **Rotary Position Embeddings (RoPE)** rather than learned absolute position embeddings.
RoPE is lightweight, works naturally with attention, and generalizes better to longer contexts.

---

## Why Position Matters

Self-attention computes similarity using dot products (e.g., $QK^\top$).
If we feed the same set of tokens in a different order, vanilla attention would produce the same result.
Position encodings break this symmetry so the model can learn order-sensitive patterns.

---

## A Quick Tour of Positional Encoding Options

There are three broad families:

- **Absolute**: add a position vector $P[t]$ to token embedding (classic Transformer)

- **Relative**: make attention scores depend on relative distance (e.g., T5-style biases)

- **Rotary (RoPE)**: rotate query/key vectors by position-dependent angles (LLaMA-style)

This chapter focuses on **RoPE** because it is the most common choice in modern decoder-only LLMs.

---

## RoPE: Rotary Position Embeddings

RoPE injects position by applying a **rotation** to each query and key vector, dimension by dimension.
Crucially, it makes the attention score depend on **relative position**.

```{note}

RoPE is applied to **queries and keys**, not to values.

```

### 1. View Each Pair of Dimensions as a 2D Plane

Assume a head dimension $d$ is even. Group it into pairs:

$$

(x_0, x_1), (x_2, x_3), \dots, (x_{d-2}, x_{d-1})


$$

Each pair is treated like a 2D vector that can be rotated.

### 2. Position-Dependent Rotation Angles

For pair index $i \in \{0, \dots, d/2 - 1\}$, define a frequency:

$$

\omega_i = \theta^{-\frac{2i}{d}}


$$

where $\theta$ is a base constant (often $10{,}000$). For position $t$, the angle is:

$$

\phi_{t,i} = t \cdot \omega_i


$$

### 3. Apply Rotation to Queries/Keys

For a 2D pair $(u, v)$, rotation by angle $\phi$ is:

$$

\begin{pmatrix}

u' \\

v'

\end{pmatrix}

=

\begin{pmatrix}

\cos\phi & -\sin\phi \\

\sin\phi & \cos\phi

\end{pmatrix}

\begin{pmatrix}

u \\

v

\end{pmatrix}


$$

Apply this rotation to every pair of dimensions in $q_t$ and $k_t$:

$$

\tilde{q}_t = \mathrm{RoPE}(q_t, t), \quad \tilde{k}_t = \mathrm{RoPE}(k_t, t)


$$

Then attention uses rotated vectors:

$$

\mathrm{score}(t, s) = \frac{\tilde{q}_t \cdot \tilde{k}_s}{\sqrt{d}}


$$

### 4. Why RoPE Encodes Relative Position

A key property (sketch): dot products of rotated vectors depend on the **difference** in angles.
Because angles are proportional to position, the attention score becomes a function of **relative distance** $(t - s)$.

```{admonition} Intuition

:class: tip

RoPE turns absolute positions into rotations, so relative distances show up as relative rotation differences.

```

---

## Minimal RoPE Implementation (Conceptual)

This code mirrors the common implementation pattern used in many open-source LLMs.

```python

import torch


def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):

    assert head_dim % 2 == 0

    i = torch.arange(0, head_dim, 2, device=device).float()  # 0,2,4,...

    inv_freq = 1.0 / (theta ** (i / head_dim))

    t = torch.arange(max_seq_len, device=device).float()

    freqs = torch.einsum('t,f->tf', t, inv_freq)  # (T, head_dim/2)

    cos = freqs.cos()

    sin = freqs.sin()

    return cos, sin


def apply_rope(x, cos, sin):

    # x: (..., T, head_dim)

    # cos/sin: (T, head_dim/2)

    x1 = x[..., ::2]  # even dims

    x2 = x[..., 1::2] # odd dims

    # rotate: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)

    x_rotated_even = x1 * cos - x2 * sin

    x_rotated_odd  = x1 * sin + x2 * cos

    # interleave back

    out = torch.empty_like(x)

    out[..., ::2] = x_rotated_even

    out[..., 1::2] = x_rotated_odd

    return out

```

In practice, models precompute `cos/sin` once and reuse them, and often store them in higher precision.

---

## The Context Length Problem

RoPE is defined for any position index, but models are trained with a maximum context length (e.g., 4K, 8K).
When you push beyond the training range, performance can degrade.

Why does this happen?

- The model only saw certain position distributions during training

- RoPE angles grow with position, changing attention patterns

- Long-range dependencies become harder to represent

So we want **context extension**: methods that let the model handle longer sequences than it was trained on.

---

## Context Extension Methods (Overview)

Several strategies exist:

- **RoPE scaling / interpolation**: modify how angles grow with position

- **NTK-aware scaling**: scale frequencies in a way that preserves attention behavior

- **YaRN**: a practical recipe that combines scaling and smoothing for long context

- **Long-context fine-tuning**: continue training with longer sequences

This section focuses on **YaRN** because it is widely used as a practical, training-free (or low-cost) extension technique.

---

## YaRN (Yet another RoPE eNhancement)

YaRN is a method to extend RoPE to longer contexts by **rescaling** the positional frequencies and
**smoothing** the transition between short-range and long-range behavior.

### The Goal

We want a model trained on context length $L$ to work at a longer length $L' = sL$ (scale factor $s > 1$),
without severely breaking attention patterns.

### High-Level Idea

RoPE uses angles:

$$

\phi_{t,i} = t \cdot \omega_i


$$

If we naively extend to larger $t$, angles grow too fast.
YaRN effectively **slows down** the angle growth beyond the training range by modifying the effective frequencies.

A simplified mental model is:

$$

\phi'_{t,i} \approx \frac{t}{s} \cdot \omega_i \quad \text{(scaled positions)}


$$

But YaRN is not just a constant scaling; it uses a **ramp / smoothing** so that:

- Short-context behavior stays close to original

- Long-context behavior is gradually adjusted

```{admonition} Practical takeaway

:class: important

YaRN is an inference-time (or minimal fine-tuning) trick to make RoPE-based models behave better at longer contexts.

```

### What YaRN Changes in Code (Conceptually)

Implementations vary, but conceptually YaRN modifies one or more of:

- the frequency schedule ($\omega_i$)

- the position index mapping ($t \mapsto f(t)$)

- a blending/ramp function that transitions from original RoPE to scaled RoPE

At the end of the day, you still compute `cos/sin`, but with adjusted angles.

---

## How to Think About Context Extension in Practice

### 1. Training-free vs training-based

- **Training-free** (e.g., RoPE scaling / YaRN): quick to try, may not match a true long-context-trained model

- **Long-context fine-tuning**: best quality, costs compute and data

### 2. What to evaluate

When extending context, you should evaluate:

- Long-range retrieval / needle-in-a-haystack

- Perplexity vs context length

- Generation quality at long prompts

- Attention stability (e.g., does it collapse to local?)

### 3. KV cache becomes the bottleneck

Longer context increases KV cache size linearly. Context extension often pairs well with:

- GQA/MQA

- KV cache quantization

- MLA-style KV compression

---

## Summary

- Transformers need positional information; modern LLMs commonly use **RoPE**.

- RoPE rotates query/key vectors with position-dependent angles so attention depends on **relative position**.

- Models trained at length $L$ often degrade beyond $L$.

- Context extension methods modify RoPE behavior; **YaRN** is a practical approach combining scaling and smoothing.
