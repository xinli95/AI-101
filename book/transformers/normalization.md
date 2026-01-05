# Normalization

Deep neural networks are sensitive to the **scale and distribution** of hidden activations.
Without normalization, training can become unstable, slow, or even diverge.

In Transformer-based models, **normalization layers** are applied before every
attention block and feed-forward block to stabilize optimization and improve gradient flow.

```{note}

Modern large language models (LLMs) typically use **RMS Normalization (RMSNorm)**,
but it is best understood in relation to **Batch Normalization** and **Layer Normalization**.

```

---

## Why Normalization Is Needed

Consider a hidden state vector at some layer:

$$

h = (h_1, h_2, \dots, h_d) \in \mathbb{R}^d


$$

As depth increases:

- The magnitude of activations can drift

- Gradients can explode or vanish

- Training becomes highly sensitive to learning rate

Normalization addresses this by **rescaling activations to a controlled range**.

---

## Batch Normalization (BatchNorm)

Batch Normalization was one of the earliest and most influential normalization techniques.

### Definition

For a batch of activations $x$ with batch dimension $B$:

$$

\mu_B = \frac{1}{B} \sum_{i=1}^B x_i, \quad
\sigma_B^2 = \frac{1}{B} \sum_{i=1}^B (x_i - \mu_B)^2


$$

BatchNorm normalizes each feature:

$$

\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}


$$

Then applies a learned affine transform:

$$

y_i = \gamma \hat{x}_i + \beta


$$

### Key Properties

- Normalizes **across the batch dimension**

- Introduces dependency between examples in the same batch

- Uses running averages during inference

### Why BatchNorm Is a Poor Fit for Transformers

- Sequence lengths vary

- Autoregressive decoding uses batch size 1

- Batch statistics change between training and inference

As a result, BatchNorm is **rarely used** in modern language models.

---

## Layer Normalization (LayerNorm)

Layer Normalization was proposed to address BatchNorm’s limitations in sequence models.

### Definition

LayerNorm normalizes **within a single token’s hidden state**:

$$

\mu = \frac{1}{d} \sum_{j=1}^d h_j, \quad
\sigma^2 = \frac{1}{d} \sum_{j=1}^d (h_j - \mu)^2


$$

The normalized output is:

$$

\hat{h}_j = \frac{h_j - \mu}{\sqrt{\sigma^2 + \epsilon}}


$$

With learnable parameters:

$$

y_j = \gamma_j \hat{h}_j + \beta_j


$$

### Properties

- Independent of batch size

- Works naturally with variable-length sequences

- Used in early Transformers (e.g., BERT, GPT-2)

### Cost of LayerNorm

LayerNorm requires computing both mean and variance,
which adds non-trivial overhead at large scale.

---

## RMS Normalization (RMSNorm)

RMSNorm is a simplified alternative to LayerNorm that removes the mean-centering step.

### Definition

RMSNorm computes the root-mean-square (RMS) of the hidden state:

$$

\mathrm{RMS}(h) = \sqrt{\frac{1}{d} \sum_{j=1}^d h_j^2}


$$

Normalization is performed as:

$$

\hat{h}_j = \frac{h_j}{\mathrm{RMS}(h) + \epsilon}


$$

With a learnable scale parameter:

$$

y_j = g_j \hat{h}_j


$$

### Key Differences from LayerNorm

- No mean subtraction

- No bias term

- Fewer operations

### Why RMSNorm Works

- Transformer activations are often approximately zero-mean already

- Scaling, not centering, is the dominant stabilizing factor

- Improved numerical efficiency

### Usage in Modern LLMs

RMSNorm is used in many modern architectures, including LLaMA, Mistral, Qwen, and DeepSeek.

---

## Pre-Norm vs Post-Norm Transformers

Normalization placement also matters.

### Post-Norm (Original Transformer)

$$

x \rightarrow \text{Attention} \rightarrow \text{Add} \rightarrow \text{Norm}


$$

### Pre-Norm (Modern LLMs)

$$

x \rightarrow \text{Norm} \rightarrow \text{Attention} \rightarrow \text{Add}


$$

### Why Pre-Norm Is Preferred

- Better gradient flow in deep networks

- More stable training

- Enables very deep Transformers

---

## PyTorch Examples

### LayerNorm

```python

import torch

import torch.nn as nn


ln = nn.LayerNorm(normalized_shape=4096)

x = torch.randn(2, 10, 4096)

y = ln(x)

```

### RMSNorm

```python

class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-6):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(d))

        self.eps = eps


    def forward(self, x):

        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()

        return x / (rms + self.eps) * self.weight

```

---

## Summary

- **BatchNorm** normalizes across batches and is unsuitable for Transformers.

- **LayerNorm** normalizes across hidden dimensions and works well for sequence models.

- **RMSNorm** simplifies LayerNorm by removing mean-centering.

- Modern LLMs overwhelmingly use **Pre-Norm + RMSNorm** for stability and efficiency.
