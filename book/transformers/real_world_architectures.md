# Transformer Anatomy: Llama 3 and Gemma 4

The earlier sections explained the individual parts of a Transformer.
This section answers two more practical questions:

1. **If we implement a Llama-like model from scratch, what classes do we need to write?**
2. **When Hugging Face loads a model, what files and modules are actually involved?**

We will use Llama 3 / Llama 3.2 as the clean reference model, then compare it with Gemma 4.

```{note}
This page is intentionally concrete. The goal is to connect formulas, PyTorch modules, Hugging Face model files, and real checkpoint names.
```

---

## What Does "Implement From Scratch" Mean?

There are three different levels of "from scratch":

| Level | What you implement | What you still reuse |
|---|---|---|
| Educational forward pass | Model classes, tensor shapes, attention, MLP, norms | PyTorch, tokenizer, maybe downloaded weights |
| Full inference stack | Forward pass plus generation loop, sampling, KV cache | PyTorch kernels and tokenizer library |
| Full pretraining system | Model, data pipeline, distributed training, optimizer, checkpointing | GPU libraries and distributed runtimes |

Raschka's Llama and Gemma notebooks are closest to the first level. They are excellent for learning because they show the model as ordinary PyTorch modules instead of hiding it behind `AutoModelForCausalLM`.

For a Llama-style model, the minimal class list is:

```text
RMSNorm
RoPE utilities
GroupedQueryAttention
FeedForward / SwiGLU
TransformerBlock
LlamaModel
optional: weight-loading helper
optional: generation loop
```

That is the full architecture. Everything else is engineering.

---

## Minimal Llama 3.2 Configuration

Start with a config dictionary. This is the from-scratch version of Hugging Face's `config.json`.

```python
import torch

LLAMA32_1B_CONFIG = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}
```

Each field immediately creates a tensor shape:

| Config field | Used by | Shape consequence |
|---|---|---|
| `vocab_size` | token embedding and output head | `(vocab_size, emb_dim)` and `(emb_dim, vocab_size)` |
| `emb_dim` | residual stream | every token vector has width 2048 |
| `n_layers` | block stack | build 16 Transformer blocks |
| `n_heads` | query heads | 32 query heads |
| `n_kv_groups` | key/value heads | 8 KV heads, shared across query groups |
| `hidden_dim` | MLP | expand 2048 -> 8192 -> 2048 |
| `context_length` | masks and RoPE cache | precompute position info up to 131K tokens |

The attention head dimension is:

$$
d_h = \frac{2048}{32} = 64
$$

The GQA group size is:

$$
\text{group size} = \frac{32}{8} = 4
$$

So every key/value head is reused by 4 query heads.

---

## Step 1: RMSNorm

Llama uses RMSNorm instead of LayerNorm. The implementation is small:

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x_float / (rms + self.eps)
        return (x_norm * self.weight.float()).to(dtype=x.dtype)
```

Input and output shape are the same:

```text
[batch, seq_len, emb_dim] -> [batch, seq_len, emb_dim]
```

RMSNorm does not mix tokens. It rescales each token vector independently.

---

## Step 2: RoPE Utilities

RoPE is not a layer with trainable parameters. It is a deterministic rotation applied to query and key vectors.

At model initialization, precompute cosine and sine tables:

```python
def compute_rope_params(head_dim, theta_base, context_length, device=None):
    assert head_dim % 2 == 0

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(context_length, device=device).float()
    freqs = torch.outer(positions, inv_freq)

    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin
```

Then apply the rotation inside attention:

```python
def apply_rope(x, cos, sin):
    # x: [batch, heads, seq_len, head_dim]
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_even * cos - x_odd * sin
    x_rotated[..., 1::2] = x_even * sin + x_odd * cos
    return x_rotated
```

The key point:

```text
RoPE changes Q and K.
RoPE does not change V.
```

---

## Step 3: Grouped-Query Attention

The attention module owns four projections:

```text
q_proj: hidden -> all query heads
k_proj: hidden -> fewer key heads
v_proj: hidden -> fewer value heads
o_proj: all query heads -> hidden
```

For Llama 3.2 1B:

| Projection | Output width |
|---|---:|
| `q_proj` | `32 * 64 = 2048` |
| `k_proj` | `8 * 64 = 512` |
| `v_proj` | `8 * 64 = 512` |
| `o_proj` | `2048` |

That is why GQA saves KV-cache memory: keys and values are narrower than queries.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg["n_heads"]
        self.num_kv_heads = cfg["n_kv_groups"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.group_size = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            cfg["emb_dim"], self.num_heads * self.head_dim, bias=False, dtype=cfg["dtype"]
        )
        self.k_proj = nn.Linear(
            cfg["emb_dim"], self.num_kv_heads * self.head_dim, bias=False, dtype=cfg["dtype"]
        )
        self.v_proj = nn.Linear(
            cfg["emb_dim"], self.num_kv_heads * self.head_dim, bias=False, dtype=cfg["dtype"]
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, cfg["emb_dim"], bias=False, dtype=cfg["dtype"]
        )

    def forward(self, x, mask, cos, sin):
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        context = weights @ v

        context = context.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(context)
```

Shape flow:

```text
x:       [B, T, 2048]
q:       [B, 32, T, 64]
k/v:     [B,  8, T, 64]
k/v rep: [B, 32, T, 64]
context: [B, 32, T, 64]
output:  [B, T, 2048]
```

In production inference, you usually do not recompute all previous `k` and `v`.
You append new keys and values to a KV cache.
The educational version can skip that at first.

---

## Step 4: SwiGLU Feed-Forward Network

The Llama MLP has three matrices, not two:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.up_proj = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=cfg["dtype"])
        self.down_proj = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

Shape flow:

```text
x:          [B, T, 2048]
gate_proj:  [B, T, 8192]
up_proj:    [B, T, 8192]
multiply:   [B, T, 8192]
down_proj:  [B, T, 2048]
```

Attention mixes information across tokens.
The MLP transforms each token independently.

---

## Step 5: One Transformer Block

A Llama block is Pre-Norm:

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = GroupedQueryAttention(cfg)
        self.mlp = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask, cos, sin)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
```

The block preserves shape:

```text
[B, T, emb_dim] -> [B, T, emb_dim]
```

This is why blocks can be stacked in a `ModuleList`.

---

## Step 6: The Full Llama Model

The complete model is just:

```text
embedding table
N Transformer blocks
final RMSNorm
LM head
```

```python
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.norm = RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.lm_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids):
        b, t = input_ids.shape

        x = self.embed_tokens(input_ids)
        mask = torch.triu(
            torch.ones(t, t, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask, self.cos, self.sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```

You can test a randomly initialized model before loading real weights:

```python
cfg = LLAMA32_1B_CONFIG
model = Llama3Model(cfg)

input_ids = torch.randint(0, cfg["vocab_size"], (2, 16))
logits = model(input_ids)

print(logits.shape)
# torch.Size([2, 16, 128256])
```

At this point you have implemented the Llama architecture.
It will not produce useful text until you load trained weights.

---

## Loading a Model

When you write:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

Hugging Face does several things.

### 1. It Downloads the Model Configuration

The most important file is:

```text
config.json
```

This file says which architecture to instantiate and with what hyperparameters.
For Llama, the relevant fields look like:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "vocab_size": 128256,
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": true
}
```

`model_type` selects the implementation family.
For example, `"llama"` maps to Llama config and model classes inside `transformers`.

### 2. It Instantiates the Python Module

For causal language modeling, the top-level module is:

```text
LlamaForCausalLM
```

Inside it:

```text
LlamaForCausalLM
  lm_head
  model: LlamaModel
    embed_tokens
    layers: ModuleList[LlamaDecoderLayer]
    norm
```

Inside one decoder layer:

```text
LlamaDecoderLayer
  self_attn: LlamaAttention
    q_proj
    k_proj
    v_proj
    o_proj
  mlp: LlamaMLP
    gate_proj
    up_proj
    down_proj
  input_layernorm
  post_attention_layernorm
```

This is the same module tree we wrote from scratch.

You can inspect it:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    device_map="auto",
    dtype="auto",
)

print(model)
print(model.model.layers[0])
```

Or inspect only the config without downloading weights:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")

print(cfg.model_type)
print(cfg.hidden_size)
print(cfg.num_hidden_layers)
print(cfg.num_attention_heads, cfg.num_key_value_heads)
```

### 3. It Downloads the Weights

The learned tensors are stored in one or more files:

```text
model.safetensors
```

or, for larger models:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

The tensor names must match the module names.
For example:

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```

Loading weights is basically:

```text
create empty module tree from config
read tensors from safetensors
copy each tensor into the matching parameter name
```

Raschka's notebook makes this explicit with a helper like:

```python
from safetensors.torch import load_file

state_dict = load_file("model.safetensors")
load_weights_into_llama(model, cfg, state_dict)
```

Hugging Face does the same job automatically inside `from_pretrained()`.

### 4. It Loads the Tokenizer

The model cannot consume raw strings.
It consumes token IDs.

Tokenizer files commonly include:

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

Some older tokenizers use files such as:

```text
vocab.json
merges.txt
tokenizer.model
```

The tokenizer must agree with the model's `vocab_size` and special tokens.
If the tokenizer and model checkpoint do not match, the model will receive the wrong token IDs.

### 5. It Optionally Loads Generation and Chat Metadata

Other useful files:

```text
generation_config.json
chat_template.jinja
processor_config.json
preprocessor_config.json
```

For plain text completion, these are convenient but not always essential.
For instruction-tuned or multimodal models, they matter more because prompts must be formatted exactly as the model saw during training.

---

## Minimum Requirements for Different Tasks

| Task | Required pieces |
|---|---|
| Random forward pass | config + architecture code |
| Use pretrained model on token IDs | config + architecture code + weights |
| Generate from text | config + architecture code + weights + tokenizer + generation loop |
| Chat with an instruct model | all of the above + chat template / special tokens |
| Load a multimodal model | all of the above + processor + modality encoders |

This is the most useful mental model:

```text
config decides the shape
code defines the computation
weights store learned tensors
tokenizer maps text to IDs
generation code turns logits into new tokens
```

---

## How Generation Works

Once the model returns logits, generation is just a loop:

```python
def generate(model, input_ids, max_new_tokens, eos_token_id=None):
    model.eval()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if eos_token_id is not None and next_token_id.item() == eos_token_id:
            break

    return input_ids
```

This greedy version is deliberately simple.
Production `generate()` adds:

- KV cache
- temperature
- top-k / top-p sampling
- repetition penalties
- stopping criteria
- batch handling
- streaming

But the core is still:

```text
forward -> last-token logits -> choose next token -> append -> repeat
```

---

## How Llama Maps to Hugging Face `transformers`

The from-scratch names and Hugging Face names are nearly one-to-one:

| From-scratch class or field | Hugging Face Llama module or field |
|---|---|
| `Llama3Model` | `LlamaForCausalLM` + inner `LlamaModel` |
| `embed_tokens` | `model.embed_tokens` |
| `TransformerBlock` | `model.layers[i]` |
| `GroupedQueryAttention` | `model.layers[i].self_attn` |
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | same names in HF |
| `FeedForward` | `model.layers[i].mlp` |
| `gate_proj`, `up_proj`, `down_proj` | same names in HF |
| `RMSNorm` before attention | `input_layernorm` |
| `RMSNorm` before MLP | `post_attention_layernorm` |
| `norm` | `model.norm` |
| `lm_head` | `lm_head` |
| `n_heads` | `num_attention_heads` |
| `n_kv_groups` | `num_key_value_heads` |
| `emb_dim` | `hidden_size` |
| `hidden_dim` | `intermediate_size` |
| `rope_base` | `rope_theta` |

This is why learning from scratch helps.
Once you can write the small version, the Hugging Face model printout stops being mysterious.

---

## Gemma 4: Same Spine, More Design Choices

Gemma 4 is more complex than Llama, but it is not a different species.
It is still:

```text
tokens -> embeddings -> repeated decoder blocks -> final norm -> logits
```

The interesting differences are inside the blocks.

### Dense Gemma 4 Config

Raschka's Gemma 4 notebook uses dense E2B and E4B configs like this:

```python
GEMMA4_E2B_CONFIG = {
    "vocab_size": 262_144,
    "vocab_size_per_layer_input": 262_144,
    "emb_dim": 1536,
    "hidden_dim": 4 * 1536,
    "n_layers": 35,
    "n_heads": 8,
    "head_dim": 256,
    "n_kv_heads": 1,
    "global_head_dim": 512,
    "context_length": 131_072,
    "sliding_window": 512,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
    "hidden_size_per_layer_input": 256,
    "num_kv_shared_layers": 20,
    "use_double_wide_mlp": True,
    "rope_local_base": 10_000.0,
    "rope_global_base": 1_000_000.0,
    "rope_global_type": "proportional",
    "rope_global_partial_rotary_factor": 0.25,
    "layer_norm_eps": 1e-6,
    "final_logit_softcap": 30.0,
    "tie_word_embeddings": True,
    "dtype": torch.bfloat16,
}
```

The big architectural differences from Llama are:

| Design choice | Llama 3 / 3.2 | Gemma 4 E2B / E4B |
|---|---|---|
| Attention layers | full causal attention in every layer | mostly sliding attention, periodic full attention |
| KV sharing | GQA | MQA/GQA plus optional cross-layer KV sharing |
| RoPE | one main RoPE recipe | separate local and global RoPE recipes |
| Normalization | two RMSNorms per block | RMSNorm before and after attention/MLP outputs |
| MLP activation | SiLU / SwiGLU | GELU with tanh approximation |
| Extra embeddings | token embedding only | token embedding plus per-layer embeddings |
| Output head | may or may not be tied | tied to token embedding in dense configs |

### Gemma 4 Block Structure

A simplified Gemma 4 dense block looks like:

```python
class Gemma4DenseBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_type = cfg["layer_types"][layer_idx]
        self.att = Gemma4Attention(cfg, layer_idx)
        self.mlp = Gemma4FeedForward(cfg, layer_idx)

        self.input_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_attention_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.pre_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])
        self.post_feedforward_layernorm = Gemma4RMSNorm(cfg["emb_dim"], eps=cfg["layer_norm_eps"])

    def forward(self, x, mask_local, mask_global, cos_local, sin_local, cos_global, sin_global):
        mask = mask_local if self.layer_type == "sliding_attention" else mask_global
        cos = cos_local if self.layer_type == "sliding_attention" else cos_global
        sin = sin_local if self.layer_type == "sliding_attention" else sin_global

        residual = x
        x = self.input_layernorm(x)
        x = self.att(x, mask, cos, sin)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x
```

This is still the same residual pattern, but Gemma controls scale more aggressively.

### Sliding Attention vs Full Attention

A full causal mask blocks future tokens:

```text
token t can attend to tokens <= t
```

A sliding-window mask also blocks tokens too far in the past:

```text
token t can attend to roughly tokens [t - window, ..., t]
```

For Gemma 4 E2B:

```python
layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * 7
```

That means:

```text
4 local layers
1 global layer
repeat 7 times
= 35 layers
```

Most layers are cheap local layers.
Periodic full layers let information travel globally.

### Per-Layer Embeddings

In Llama, token identity enters once:

```text
input_ids -> embed_tokens -> residual stream
```

Gemma 4 adds per-layer embedding input:

```text
input_ids
  -> embed_tokens_per_layer
  -> reshape into one small vector per layer
  -> feed layer i a layer-specific token signal
```

Conceptually:

```text
main residual stream:        [B, T, emb_dim]
per-layer token signal:      [B, T, n_layers, ple_dim]
signal used by layer i:      [B, T, ple_dim]
```

This gives each layer a compact token-specific side input.

### Gemma 4 in Hugging Face

Gemma 4 is multimodal in Hugging Face, so the top-level load path often uses a processor:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
```

The text Transformer is usually inside a nested text config:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("google/gemma-4-E2B")
text_cfg = cfg.text_config

print(text_cfg.hidden_size)
print(text_cfg.num_hidden_layers)
print(text_cfg.num_attention_heads)
print(text_cfg.num_key_value_heads)
print(text_cfg.layer_types[:10])
print(text_cfg.rope_parameters)
```

For Gemma 4, loading may need more than a tokenizer:

```text
config.json
model weights
tokenizer files
processor files
image/audio preprocessing files
chat template
```

That is the main practical difference from a text-only Llama checkpoint.

---

## Hugging Face Anatomy

Hugging Face does not treat a model as one giant Python file.
A model family is a small package under:

```text
src/transformers/models/<model_name>/
```

The package separates five concerns:

```text
configuration: what shapes and architectural choices exist?
modeling: how tensors flow through PyTorch modules?
tokenization / processing: how raw user inputs become tensors?
auto registration: how AutoConfig / AutoModel discover the classes?
task heads: which task-specific wrappers sit on top of the base model?
```

This separation is the main design principle. The same base architecture can support many tasks, many loading paths, and many input modalities.

### Llama Directory: A Minimal Text-Only Package

In `transformers` v5.14.0, the Llama package contains:

```text
models/llama/
  __init__.py
  configuration_llama.py
  modeling_llama.py
  tokenization_llama.py
```

Each file has a clear job:

| File | Main responsibility |
|---|---|
| `configuration_llama.py` | Defines `LlamaConfig`, validates architectural hyperparameters |
| `modeling_llama.py` | Defines PyTorch modules and task heads |
| `tokenization_llama.py` | Defines tokenizer wrappers |
| `__init__.py` | Exposes public classes through lazy imports |

The key classes are:

```text
LlamaConfig
LlamaRMSNorm
LlamaRotaryEmbedding
LlamaMLP
LlamaAttention
LlamaDecoderLayer
LlamaPreTrainedModel
LlamaModel
LlamaForCausalLM
LlamaForSequenceClassification
LlamaForQuestionAnswering
LlamaForTokenClassification
LlamaTokenizer
```

The hierarchy is:

```text
PreTrainedConfig
  -> LlamaConfig

PreTrainedModel
  -> LlamaPreTrainedModel
       -> LlamaModel
       -> LlamaForCausalLM
       -> LlamaForSequenceClassification
       -> LlamaForQuestionAnswering
       -> LlamaForTokenClassification
```

The important split is:

```text
LlamaModel = base decoder stack
LlamaForCausalLM = LlamaModel + lm_head + generation support
```

So when you call:

```python
AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
```

you get the base model. When you call:

```python
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

you get the base model plus a language-modeling head.

### What `LlamaConfig` Is Responsible For

`LlamaConfig` is not just a dictionary.
It is the contract between checkpoint files and Python code.

It defines fields such as:

```text
model_type = "llama"
vocab_size
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
hidden_act
max_position_embeddings
rms_norm_eps
use_cache
tie_word_embeddings
rope_parameters
attention_bias
attention_dropout
mlp_bias
head_dim
```

It also performs architecture checks, such as ensuring `hidden_size` is divisible by `num_attention_heads`.

The `model_type` string is crucial:

```text
config.json says "model_type": "llama"
AutoConfig maps "llama" -> LlamaConfig
AutoModelForCausalLM maps LlamaConfig -> LlamaForCausalLM
```

This is how `Auto*` classes avoid hard-coding every model in user code.

### What `LlamaPreTrainedModel` Adds

`LlamaPreTrainedModel` is the model-family base class.
It inherits from `PreTrainedModel` and stores integration metadata:

```text
base_model_prefix = "model"
supports_gradient_checkpointing = True
_no_split_modules = ["LlamaDecoderLayer"]
_skip_keys_device_placement = ["past_key_values"]
_supports_flash_attn = True
_supports_sdpa = True
_supports_flex_attn = True
_supports_attention_backend = True
```

These fields are not the mathematical architecture.
They tell the library how to load, shard, compile, checkpoint, and run the model efficiently.

Examples:

| Field | Why it exists |
|---|---|
| `base_model_prefix` | tells HF where the base model lives inside a task wrapper |
| `_no_split_modules` | prevents device mapping from splitting one decoder layer across boundaries |
| `_skip_keys_device_placement` | avoids moving cache objects as if they were ordinary tensors |
| `_supports_flash_attn` / `_supports_sdpa` | lets attention dispatch choose optimized kernels |

This is the difference between an educational implementation and a production library implementation.

### What `LlamaModel.forward()` Handles

Our from-scratch model only accepted `input_ids`.
The HF model must support a much wider API:

```text
input_ids
attention_mask
position_ids
past_key_values
inputs_embeds
use_cache
extra generation / tracing kwargs
```

The base model forward pass does roughly:

```text
validate exactly one of input_ids or inputs_embeds
embed token IDs if needed
create or reuse DynamicCache
create position IDs
create causal mask
compute RoPE position embeddings
run decoder layers
apply final RMSNorm
return BaseModelOutputWithPast
```

This is why HF model code can look more complicated than a teaching implementation.
Much of the extra code is for masks, cache, output format, device mapping, and generation compatibility.

### What `LlamaForCausalLM` Adds

`LlamaForCausalLM` wraps the base model:

```text
LlamaForCausalLM
  model: LlamaModel
  lm_head: Linear(hidden_size, vocab_size)
```

Its forward pass does:

```text
call LlamaModel
take last hidden states
compute logits with lm_head
optionally compute next-token loss if labels are provided
return CausalLMOutputWithPast
```

It also inherits `GenerationMixin`, which gives the model `.generate()`.

So the design is:

```text
base model = representation computation
task head = logits/loss for a task
GenerationMixin = autoregressive decoding utilities
```

---

## Gemma4 Unified

Gemma4 Unified is larger because it must handle text, image, video, and audio.
In v5.14.0, the package contains:

```text
models/gemma4_unified/
  __init__.py
  configuration_gemma4_unified.py
  feature_extraction_gemma4_unified.py
  image_processing_gemma4_unified.py
  modeling_gemma4_unified.py
  modular_gemma4_unified.py
  processing_gemma4_unified.py
  video_processing_gemma4_unified.py
```

The extra files are not decorative.
They exist because multimodal models have more input contracts:

| File | Main responsibility |
|---|---|
| `configuration_gemma4_unified.py` | text, vision, audio, and unified config classes |
| `modeling_gemma4_unified.py` | text decoder, multimodal embedding merge, task heads |
| `processing_gemma4_unified.py` | combines tokenizer, image processor, video processor, audio feature extractor |
| `image_processing_gemma4_unified.py` | image resize, patching, padding, position IDs |
| `video_processing_gemma4_unified.py` | frame/video patch processing |
| `feature_extraction_gemma4_unified.py` | audio feature extraction |
| `modular_gemma4_unified.py` | modular source that reuses/adapts existing Gemma and Llama components |
| `__init__.py` | lazy public exports |

### Nested Configs

Llama has one config:

```text
LlamaConfig
```

Gemma4 Unified has several:

```text
Gemma4UnifiedTextConfig
Gemma4UnifiedVisionConfig
Gemma4UnifiedAudioConfig
Gemma4UnifiedConfig
```

The top-level config owns the sub-configs:

```text
Gemma4UnifiedConfig
  text_config
  vision_config
  audio_config
  image/audio/video special token IDs
```

This lets the model be partially constructed.
For example, if `vision_config` is missing, the vision embedder does not need to be initialized.

### Text-Only vs Unified Classes

Gemma4 Unified has both text-only and multimodal model classes:

```text
Gemma4UnifiedTextModel
Gemma4UnifiedForCausalLM
Gemma4UnifiedModel
Gemma4UnifiedForConditionalGeneration
```

Their roles are:

| Class | Role |
|---|---|
| `Gemma4UnifiedTextModel` | decoder-only text Transformer |
| `Gemma4UnifiedForCausalLM` | text Transformer plus LM head |
| `Gemma4UnifiedModel` | multimodal wrapper that merges text/image/video/audio embeddings |
| `Gemma4UnifiedForConditionalGeneration` | multimodal wrapper plus LM head and generation support |

A useful way to see the composition:

```text
Gemma4UnifiedForConditionalGeneration
  model: Gemma4UnifiedModel
    language_model: AutoModel.from_config(text_config)
    embed_vision: Gemma4UnifiedVisionEmbedder
    embed_audio: Gemma4UnifiedMultimodalEmbedder
  lm_head
```

The interesting line is:

```text
language_model = AutoModel.from_config(config.text_config)
```

That means the unified model delegates the text decoder to the AutoModel system.
This keeps the multimodal wrapper focused on input fusion instead of duplicating the whole decoder.

### Multimodal Forward Pass

For text-only Llama, `input_ids` are embedded and passed to decoder layers.

For Gemma4 Unified, the forward pass has an extra step:

```text
embed text token IDs
find placeholder tokens for image/video/audio
turn images/videos/audio into soft embeddings
scatter soft embeddings into the text embedding sequence
pass the merged sequence into the language model
compute logits
```

The sequence still becomes a single stream of embeddings:

```text
[text token, text token, image placeholder, text token]
                |
                replaced with image soft token embedding
```

So even multimodal generation eventually reduces to:

```text
merged embeddings -> decoder-only language model -> logits
```

### Processor Classes Are Part of the Model Contract

For a multimodal checkpoint, `AutoProcessor` is as important as `AutoTokenizer`.

`Gemma4UnifiedProcessor` combines:

```text
tokenizer
image processor
video processor
audio feature extractor
chat template logic
```

It must produce tensors whose names match the model forward signature:

```text
input_ids
attention_mask
pixel_values
pixel_values_videos
input_features
input_features_mask
image_position_ids
video_position_ids
mm_token_type_ids
```

This is a general HF principle:

```text
processor output keys should line up with model.forward() argument names
```

That is why multimodal models need more files than text-only models.

---

## Design Checklist

If you were implementing a new model family in Hugging Face style, the design checklist would look like this.

### 1. Start With the Config

Define a `PreTrainedConfig` subclass:

```python
class MyModelConfig(PreTrainedConfig):
    model_type = "my_model"

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
```

The config should contain every architecture choice needed to instantiate empty modules.
It should not contain learned tensors.

For multimodal models, define sub-configs:

```text
MyModelTextConfig
MyModelVisionConfig
MyModelAudioConfig
MyModelConfig
```

### 2. Write the Base Modules

These are ordinary `nn.Module` classes:

```text
RMSNorm / LayerNorm
RotaryEmbedding or positional embedding
Attention
MLP
DecoderLayer or EncoderLayer
```

They should be small enough that one layer can be inspected independently.

### 3. Write the Base Model

The base model owns the stack:

```text
MyModel
  embed_tokens
  layers
  norm
  rotary_emb
```

It returns hidden states, not task logits.

For decoder-only LMs, the base model should handle:

```text
input_ids or inputs_embeds
attention_mask
position_ids
past_key_values
use_cache
```

### 4. Write Task Heads

Task heads wrap the base model:

```text
MyModelForCausalLM
MyModelForSequenceClassification
MyModelForQuestionAnswering
MyModelForTokenClassification
```

Each task head should:

```text
call the base model
transform hidden states into task-specific logits
compute loss if labels are provided
return a typed ModelOutput
```

For language models, inherit `GenerationMixin` so `.generate()` works.

### 5. Define the Pretrained Base Class

Create a family-specific base class:

```python
class MyModelPreTrainedModel(PreTrainedModel):
    config_class = MyModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
```

This class is where production behavior lives:

```text
weight initialization
device-map hints
attention backend support
gradient checkpointing support
cache placement rules
which modules cannot be split
```

### 6. Define Tokenizer / Processor Classes

For text-only models, define or reuse tokenizer classes.

For multimodal models, define processors:

```text
ImageProcessor
VideoProcessor
FeatureExtractor
Processor
```

The processor should return a `BatchFeature` or dict whose keys match `model.forward()`.

### 7. Expose Classes Lazily

The model package `__init__.py` should expose public classes without importing heavy dependencies immediately.
In v5.14.0, Llama and Gemma4 Unified use `_LazyModule` plus an import-structure helper.

This keeps:

```text
import transformers
```

fast and lightweight.

### 8. Register With Auto Classes

To make this work:

```python
AutoConfig.from_pretrained(...)
AutoModel.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
AutoProcessor.from_pretrained(...)
```

the new config, model, tokenizer, and processor classes must be discoverable by the library's auto mappings.

Conceptually:

```text
model_type string -> Config class
Config class -> Model class
Config class -> task-specific model class
model repo files -> tokenizer / processor class
```

The user-facing result is simple:

```python
model = AutoModelForCausalLM.from_pretrained(repo_id)
```

but this only works because the package has declared the mapping between config and implementation.

### 9. Return Standard Output Objects

HF models usually return typed output containers:

```text
BaseModelOutputWithPast
CausalLMOutputWithPast
ModelOutput subclasses for multimodal extras
```

This keeps downstream code consistent:

```python
outputs = model(**inputs)
outputs.logits
outputs.past_key_values
outputs.hidden_states
outputs.attentions
```

Gemma4 Unified defines custom outputs because it may also return:

```text
image_hidden_states
audio_hidden_states
shared_kv_states
```

### 10. Keep the Educational Core Visible

Even in production code, the architecture should still be readable:

```text
config -> modules -> base model -> task head -> processor/tokenizer -> auto API
```

Llama is the clean version of this pattern.
Gemma4 Unified is the same pattern extended to multiple modalities.

---

## A Good Reading Order for Any New Model

When you open a Hugging Face model repo, read it in this order:

1. `config.json`: what architecture and shapes?
2. `tokenizer_config.json` / `tokenizer.json`: what tokenization?
3. `model.safetensors.index.json`: how are weights named and sharded?
4. model class printout: what modules were instantiated?
5. first block: what does one layer contain?
6. attention module: MHA, MQA, GQA, sliding, or hybrid?
7. MLP module: GELU, SwiGLU, MoE, or something else?
8. generation config and chat template: how should prompts be formatted?

For Llama, this process reveals a clean decoder-only model.
For Gemma 4, the same process reveals the same Transformer spine plus multimodal processing, hybrid attention, per-layer embeddings, and more normalization.

---

## References

- Raschka, *LLMs from Scratch*, [Converting GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama)
- Raschka, *LLMs from Scratch*, [Gemma 4](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/17_gemma4)
- Hugging Face Transformers, [Llama model documentation](https://huggingface.co/docs/transformers/model_doc/llama)
- Hugging Face Transformers, [Gemma 4 model documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- Hugging Face Transformers v5.14.0 source, [Llama implementation](https://github.com/huggingface/transformers/tree/v5.14.0/src/transformers/models/llama)
- Hugging Face Transformers v5.14.0 source, [Gemma4 Unified implementation](https://github.com/huggingface/transformers/tree/v5.14.0/src/transformers/models/gemma4_unified)
- Meta Llama, [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Google, [Gemma 4 E2B model card](https://huggingface.co/google/gemma-4-E2B)
