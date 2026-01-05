# Feed-Forward Network (FFN) and SwiGLU

In a Transformer block, attention mixes information **across tokens**, but it is mostly linear in nature (dot-products + weighted sums).
The **Feed-Forward Network (FFN)** provides the main **nonlinearity** and **feature transformation** applied **independently at each token position**.

```{note}

A Transformer block typically alternates:

- (1) Attention: token-to-token communication

- (2) FFN: per-token nonlinear transformation

```

---

## Classic Transformer FFN

Given hidden states $X \in \mathbb{R}^{T \times d}$, the classic FFN is:

$$

\mathrm{FFN}(X) = \phi(XW_1 + b_1)W_2 + b_2


$$

where:

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$ expands the dimension

- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$ projects back

- $\phi$ is a nonlinearity (ReLU in the original paper, later GELU)

### Why is $d_{\text{ff}}$ larger than $d$?

The FFN is like a per-token MLP with a "hidden" width. Expanding to a larger $d_{\text{ff}}$ increases capacity.

A common choice is:

$$

d_{\text{ff}} \approx 4d


$$

---

## Gated FFNs: GLU Family

Many modern LLMs use **gated** feed-forward layers that tend to work better than plain ReLU/GELU MLPs.

### GLU (Gated Linear Unit)

A GLU-style FFN splits into two branches:

- One branch produces "features"

- Another branch produces "gates" that modulate those features

A generic GLU form is:

$$

\mathrm{GLU}(X) = (XW_a) \odot \sigma(XW_g)


$$

where $\odot$ is elementwise multiplication.

Gating helps the model learn **conditional computation**: which features to pass through and which to suppress.

---

## SwiGLU (Swish-Gated Linear Unit)

**SwiGLU** is a popular gated FFN variant used in many modern LLMs (e.g., PaLM-style architectures, LLaMA family).

### Definition

SwiGLU uses the **SiLU / Swish** nonlinearity as the gate activation.

Let:

- $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$

- $W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$

- $W_3 \in \mathbb{R}^{d_{\text{ff}} \times d}$

Then:

$$

\mathrm{SwiGLU}(X) = \left(\mathrm{SiLU}(XW_1) \odot (XW_2)\right) W_3


$$

where:

$$

\mathrm{SiLU}(u) = u \cdot \sigma(u)


$$

### Intuition

- $(XW_2)$ is the "value" branch (features)

- $\mathrm{SiLU}(XW_1)$ is the "gate" branch

- Elementwise multiplication selectively passes features

```{admonition} Why this is called "SwiGLU"

:class: tip

It is a GLU-style gated FFN where the gate activation is **Swish/SiLU**.

```

---

## Parameter Count and the "4d" Rule of Thumb

Classic FFN uses two matrices: $W_1$ and $W_2$. SwiGLU uses three: $W_1, W_2, W_3$.

If you kept the same $d_{\text{ff}} = 4d$, SwiGLU would have more parameters.

In practice, many models adjust $d_{\text{ff}}$ downward so total compute stays similar.

A common heuristic:

$$

d_{\text{ff}} \approx \frac{8}{3} d


$$

for SwiGLU, which roughly matches the parameter/compute budget of a $4d$ GELU FFN.

---

## Where FFN Sits in the Transformer Block

In a modern **Pre-Norm** decoder block (typical LLM), you often see:

$$

x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))


$$

$$

x \leftarrow x + \mathrm{FFN}(\mathrm{Norm}(x))


$$

So FFN is always paired with:

- Normalization (often RMSNorm)

- Residual connection

---

## PyTorch Implementation (Minimal)

Below is a minimal SwiGLU FFN module.

```python

import torch

import torch.nn as nn

import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):

        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate branch

        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # value branch

        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # project back


    def forward(self, x):

        gate = F.silu(self.w1(x))

        val  = self.w2(x)

        return self.w3(gate * val)

```

### Shape check

If $x$ has shape $(B, T, d)$, then:

- `w1(x)`, `w2(x)` have shape $(B, T, d_{\text{ff}})$

- after gating and `w3`, output returns to $(B, T, d)$

---

## Practical Notes

### 1. Activation choice

Why SiLU/Swish?

- Smooth, non-saturating behavior

- Often trains more stably than ReLU

- Empirically strong in large-scale LLMs

### 2. Bias terms

Many LLM implementations set FFN linear layers to `bias=False` for efficiency and because normalization reduces the need.

### 3. Efficiency

FFN typically dominates compute in Transformers (often more FLOPs than attention), especially at shorter context lengths.

---

## Summary

- The FFN provides **per-token nonlinearity and feature transformation**.

- Classic FFN is a 2-layer MLP (ReLU/GELU).

- Modern LLMs often use **gated FFNs**, especially **SwiGLU**.

- SwiGLU computes: $(\mathrm{SiLU}(XW_1) \odot XW_2) W_3$.

- FFN is used with **Pre-Norm + residual connections** inside each Transformer block.
