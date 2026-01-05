# Embedding

After tokenization, the model receives an **input sequence of token IDs** (integers).
Neural networks cannot directly operate on discrete IDs, so the first learnable step is to map each token ID to a **continuous vector** using an **embedding layer**.

```{note}

Throughout this section, assume a vocabulary size of $V$, an embedding dimension of $d$, and a sequence of length $T$.

```

## From Token IDs to Embedding Vectors

Let the tokenized input be:

$$

x = [x_1, x_2, \dots, x_T], \quad x_t \in \{0, 1, \dots, V-1\}


$$

An embedding layer is a trainable lookup table (a matrix):

$$

E \in \mathbb{R}^{V \times d}


$$

The embedding vector for token $x_t$ is the $x_t$-th row of $E$:

$$

\mathrm{embed}(x_t) = E[x_t] \in \mathbb{R}^{d}


$$

So the embedded sequence becomes:

$$

H^{(0)} = [E[x_1]; E[x_2]; \dots; E[x_T]] \in \mathbb{R}^{T \times d}


$$

This $H^{(0)}$ is the first continuous representation the Transformer works with.

### Embedding is Equivalent to a Linear Layer on One-Hot Inputs

It can be helpful to see embeddings as a special case of a linear layer.

Let $\mathrm{onehot}(x_t) \in \mathbb{R}^{V}$ be a one-hot vector with a 1 at position $x_t$. Then:

$$

\mathrm{onehot}(x_t)^\top E = E[x_t]


$$

So embedding lookup is equivalent to multiplying a one-hot vector by $E$, just implemented efficiently.

### Practical View in PyTorch

In PyTorch, embeddings are implemented with `torch.nn.Embedding`:

```python

import torch

import torch.nn as nn


V = 50000   # vocab size

d = 4096    # embedding dim


embed = nn.Embedding(V, d)


x = torch.tensor([10, 20, 30])  # token IDs, shape (T,)

h0 = embed(x)                   # shape (T, d)

print(h0.shape)

```

```{admonition} Common confusion

:class: tip

The embedding layer does **not** know anything about token meaning by itself.
It is just a matrix initialized randomly and learned through training.

```

## Positional Information and the Full Input Representation

Token embeddings alone do not contain word order. Transformers add position information using either:

- **Positional embeddings** (learned vectors added to token embeddings), or

- **Rotary Position Embeddings (RoPE)** (a rotation applied inside attention).

A common additive representation is:

$$

H^{(0)}_t = E[x_t] + P[t]


$$

where $P[t] \in \mathbb{R}^d$ is a positional embedding for position $t$.

---

## Tying Embedding in the Final Output Layer

Many language models use **weight tying** between:

- The **input embedding matrix** $E \in \mathbb{R}^{V \times d}$, and

- The **output projection matrix** $W_{\text{out}} \in \mathbb{R}^{V \times d}$ (or its transpose).

### Output Layer in a Language Model

At each position $t$, the Transformer produces a hidden state:

$$

h_t \in \mathbb{R}^{d}


$$

To predict the next token, we compute logits over the vocabulary:

$$

z_t = W_{\text{out}} h_t + b, \quad z_t \in \mathbb{R}^{V}


$$

Then apply softmax:

$$

p(x_{t+1} = i \mid x_{\le t}) = \frac{\exp(z_{t,i})}{\sum_{j=0}^{V-1}\exp(z_{t,j})}


$$

### What Weight Tying Does

With weight tying, we set:

$$

W_{\text{out}} = E


$$

so the logits become:

$$

z_t = E h_t + b


$$

Interpretation: the score for token $i$ is the dot product between its embedding vector $E[i]$ and the hidden state $h_t$ (plus bias).

### Why Weight Tying Helps

- **Fewer parameters:** reduces model size by reusing the same matrix.

- **Regularization:** encourages a consistent geometry for input and output tokens.

- **Empirical gains:** often improves perplexity for the same parameter budget.

### PyTorch Example

Here is the typical pattern used in many Transformer implementations:

```python

import torch

import torch.nn as nn


class TinyLM(nn.Module):

    def __init__(self, V, d):

        super().__init__()

        self.embed = nn.Embedding(V, d)

        self.lm_head = nn.Linear(d, V, bias=False)


        # Weight tying: output projection shares weights with input embedding

        self.lm_head.weight = self.embed.weight


    def forward(self, x, h):

        # x: token ids, h: hidden states from transformer, shape (T, d)

        logits = self.lm_head(h)

        return logits

```

```{note}

Some implementations tie weights by assigning references (as above).
Others copy once at init and enforce tying during training.
The key idea is: they share the *same underlying parameters*.

```

### A Useful Geometric View

With weight tying, the model predicts the next token by measuring similarity between $h_t$ and each token embedding.

- If $h_t$ is close (in dot-product sense) to $E[i]$, token $i$ gets a higher logit.

- Training shapes the embedding space so tokens that appear in similar contexts have compatible vectors.

---

## Summary

- **Tokenization** produces integers; **embedding** turns them into vectors.

- The embedding matrix $E$ is a learnable lookup table of shape $V \times d$.

- The output layer maps hidden states back to vocabulary logits.

- **Weight tying** sets the output projection weights equal to the input embedding weights, reducing parameters and often improving generalization.
