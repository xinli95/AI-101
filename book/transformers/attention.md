# Attention

Attention is the core operation that lets Transformers **mix information across tokens** in a sequence.
Intuitively, each token decides **what to look at** (other tokens) and **how much to copy** from them.

```{note}

In decoder-only LLMs, attention is usually **causal**: token $t$ may attend to tokens $\le t$ but not future tokens.

```

---

## Scaled Dot-Product Attention

Start with three matrices derived from the current hidden states:

- Queries: $Q \in \mathbb{R}^{T \times d_k}$

- Keys: $K \in \mathbb{R}^{T \times d_k}$

- Values: $V \in \mathbb{R}^{T \times d_v}$

where $T$ is sequence length.

The attention weights are:

$$

A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) \in \mathbb{R}^{T \times T}


$$

where $M$ is a mask (e.g., causal mask uses $-\infty$ above the diagonal).

The output is:

$$

\mathrm{Attn}(Q,K,V) = A V \in \mathbb{R}^{T \times d_v}


$$

### Why the $\sqrt{d_k}$ scaling?

Without scaling, dot products grow with dimension and can push softmax into saturated regimes,
making gradients small. The scaling keeps logits in a reasonable range.

---

## Multi-Head Attention (MHA)

Instead of one attention computation, Transformers use **multiple heads** in parallel.

### Shapes

Let the model dimension be $d_{\text{model}}$ and the number of heads be $H$.
Usually $d_k = d_v = d_{\text{model}}/H$.

The model learns head-specific projections:

$$

Q_h = X W^Q_h, \quad K_h = X W^K_h, \quad V_h = X W^V_h


$$

for $h \in \{1,\dots,H\}$, where $X \in \mathbb{R}^{T \times d_{\text{model}}}$.

Each head computes:

$$

O_h = \mathrm{Attn}(Q_h, K_h, V_h)


$$

Then we concatenate heads and project back:

$$

\mathrm{MHA}(X) = \mathrm{Concat}(O_1,\dots,O_H) W^O


$$

### Why multiple heads?

A helpful mental model:

- Each head can specialize in a different pattern (local syntax, long-range dependency, delimiter matching, etc.)

- The concatenation lets the model combine these "views".

---

## Causal Attention and the KV Cache

During autoregressive decoding, we generate one token at a time.
Recomputing $K$ and $V$ for all previous tokens at every step would be wasteful.

### KV cache idea

At step $t$, we compute $q_t$ for the new token, but reuse cached keys and values:

- Cache: $K_{\le t}$ and $V_{\le t}$

- New attention: $\mathrm{Attn}(q_t, K_{\le t}, V_{\le t})$

This makes decoding much faster, but it introduces a major bottleneck:

```{admonition} KV cache is a memory problem

:class: important

The KV cache grows linearly with sequence length and is multiplied by the number of attention heads.

```

---

## Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

MHA stores separate $K$ and $V$ per head. This is expressive, but expensive for the KV cache.

### Multi-Query Attention (MQA)

MQA shares **one** key head and **one** value head across all query heads:

- Many query heads: $H$ different $Q_h$

- One shared key/value: $K_{\text{shared}}, V_{\text{shared}}$

This can reduce KV cache size by roughly a factor of $H$, often with a small quality tradeoff.

### Grouped-Query Attention (GQA)

GQA is a compromise between MHA and MQA.

Let there be $H$ query heads but only $G$ key/value heads, where $1 \le G \le H$.
Each key/value head is shared by a **group** of query heads.

You can think of it as:

- **MHA:** $G = H$ (no sharing)

- **MQA:** $G = 1$ (maximum sharing)

- **GQA:** $1 < G < H$ (group sharing)

### Why GQA is popular

- Much smaller KV cache than MHA

- Usually better quality than MQA

- Widely adopted in modern open-weight LLMs for efficient inference

---

## Multi-Head Latent Attention (MLA)

Multi-Head Latent Attention (MLA) is another approach to reduce inference cost,
introduced in DeepSeek-V2 and used in later DeepSeek models.

### Problem MLA targets

Even with GQA/MQA, the KV cache can dominate memory at long context lengths.
MLA compresses the information stored for keys/values into a **latent** representation.

### High-level idea

Instead of caching full per-token keys and values (large tensors), MLA:

1. Projects token representations into a **smaller latent vector** (dimension $r \ll d_{\text{model}}$)

2. Caches this latent vector

3. Reconstructs (or partially reconstructs) key/value information as needed for attention

Conceptually:

$$

X_t \xrightarrow{\;W^{\text{down}}\;} c_t \in \mathbb{R}^{r} \quad \text{(cache)}


$$

$$

c_t \xrightarrow{\;W^{K}_{\text{up}},\;W^{V}_{\text{up}}\;} K_t, V_t \quad \text{(used for attention)}


$$

### Why this helps

- Cache stores $c_t$ of size $r$ instead of storing full $K_t$ and $V_t$

- Significant KV memory reduction at long context lengths

- Can be combined with other tricks (quantized KV cache, long-context attention variants)

```{admonition} Intuition

:class: tip

GQA/MQA reduce KV cache by **sharing** keys/values across heads.

MLA reduces KV cache by **compressing** keys/values into a smaller latent space.

```

---

## Putting It Together: When to Use What

Attention variants are mostly about an accuracy–efficiency tradeoff:

- **MHA:** most expressive, highest KV cost

- **GQA:** strong tradeoff, very common in modern LLMs

- **MQA:** fastest/smallest KV, may lose some quality

- **MLA:** compresses KV cache further via latent representations

A practical way to remember:

- If KV cache is not your bottleneck: use **MHA**

- If decoding memory bandwidth is bottleneck: consider **GQA/MQA**

- If long-context KV cache dominates: consider **MLA-style compression**

---

## Minimal PyTorch Skeletons

These snippets are **illustrative** and omit many performance details.

### Multi-Head Attention (MHA) skeleton

```python

import torch

import torch.nn as nn


class MHA(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model

        self.n_heads = n_heads

        self.d_head = d_model // n_heads


        self.Wq = nn.Linear(d_model, d_model, bias=False)

        self.Wk = nn.Linear(d_model, d_model, bias=False)

        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.Wo = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x, mask=None):

        B, T, D = x.shape

        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)


        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        if mask is not None:

            att = att + mask

        att = att.softmax(dim=-1)

        out = att @ v


        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.Wo(out)

```

### GQA idea in one sentence

In GQA, you keep $H$ query heads but project keys/values to only $G$ heads and broadcast them to groups.

---

## Summary

- **Scaled dot-product attention** computes weighted sums of values based on query–key similarity.

- **Multi-head attention** runs multiple attentions in parallel for richer pattern matching.

- **KV cache** makes decoding fast but introduces a major memory bottleneck.

- **GQA/MQA** reduce KV cache by sharing keys/values across heads.

- **MLA** reduces KV cache by compressing key/value information into a smaller latent representation.
