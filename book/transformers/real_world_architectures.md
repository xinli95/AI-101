# Real-World Transformer Architectures: Llama 3 and Gemma 4

The previous sections introduced the main pieces of a modern decoder-only Transformer:

- tokenization and embeddings
- causal self-attention
- KV cache, MQA, and GQA
- RMSNorm and Pre-Norm residual blocks
- gated feed-forward networks
- RoPE and long-context scaling

This section connects those pieces to real Hugging Face `transformers` models. We will use two examples:

- **Llama 3 / Llama 3.2**, a clean reference architecture for modern decoder-only LLMs
- **Gemma 4**, a more advanced descendant that keeps the same core loop but adds hybrid attention, per-layer embeddings, and multimodal wrappers

```{note}
The point here is not to memorize every number in every checkpoint. The useful skill is learning how to read a model config and infer the architecture it describes.
```

---

## The Hugging Face Mental Model

In Hugging Face, a model repository usually contains:

- `config.json`: architecture hyperparameters
- tokenizer files: how text becomes token IDs
- checkpoint files: learned tensors, often in `safetensors`
- generation and chat-template metadata

The architecture is mostly encoded in the config. For decoder-only language models, the key fields are usually:

| Concept | Typical config field | Meaning |
|---|---:|---|
| Vocabulary size | `vocab_size` | Number of token IDs the embedding table supports |
| Width | `hidden_size` | Residual-stream dimension $d_{\text{model}}$ |
| Depth | `num_hidden_layers` | Number of Transformer blocks |
| Attention heads | `num_attention_heads` | Number of query heads |
| KV heads | `num_key_value_heads` | Number of key/value heads for MHA, MQA, or GQA |
| FFN width | `intermediate_size` | Hidden dimension inside the feed-forward network |
| Positional system | `rope_theta`, `rope_scaling`, `rope_parameters` | RoPE base and scaling recipe |
| Normalization | `rms_norm_eps` | RMSNorm epsilon |
| Cache | `use_cache` | Whether generation returns reusable KV states |
| Embedding tying | `tie_word_embeddings` | Whether input embedding and output head share weights |

You can inspect these fields without loading model weights:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")

print(cfg.model_type)
print(cfg.hidden_size)
print(cfg.num_hidden_layers)
print(cfg.num_attention_heads, cfg.num_key_value_heads)
```

For models with nested sub-configs, such as Gemma 4, the text decoder lives inside `text_config`:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("google/gemma-4-E2B")
text_cfg = cfg.text_config

print(cfg.model_type)
print(text_cfg.hidden_size)
print(text_cfg.layer_types[:10])
print(text_cfg.rope_parameters)
```

---

## Llama 3 as the Clean Decoder-Only Reference

Llama 3 is a causal language model: it predicts the next token from the tokens before it.
The high-level computation is:

```text
input_ids
  -> token embeddings
  -> repeated decoder blocks
       -> RMSNorm
       -> causal grouped-query attention with RoPE
       -> residual add
       -> RMSNorm
       -> SwiGLU feed-forward network
       -> residual add
  -> final RMSNorm
  -> LM head
  -> logits over vocabulary
```

At training time, the model receives a sequence and predicts the next token at every position.
At generation time, it repeatedly feeds back the newly generated token and reuses the KV cache.

### A Llama-Style Block

Ignoring implementation details such as tensor parallelism and fused kernels, a Llama-style block looks like:

$$
x \leftarrow x + \mathrm{GQA}(\mathrm{RMSNorm}(x))
$$

$$
x \leftarrow x + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x))
$$

The attention sublayer is causal and uses RoPE on queries and keys:

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

$$
\tilde{Q} = \mathrm{RoPE}(Q),\quad \tilde{K} = \mathrm{RoPE}(K)
$$

$$
\mathrm{Attention}(\tilde{Q}, \tilde{K}, V)
= \mathrm{softmax}\left(\frac{\tilde{Q}\tilde{K}^{\top}}{\sqrt{d_h}} + M_{\text{causal}}\right)V
$$

The FFN uses a gated structure:

$$
\mathrm{SwiGLU}(x)
= W_{\text{down}}\left(\mathrm{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right)
$$

This is why Hugging Face Llama checkpoints contain names such as:

```text
model.embed_tokens
model.layers.0.input_layernorm
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj
model.layers.0.post_attention_layernorm
model.layers.0.mlp.gate_proj
model.layers.0.mlp.up_proj
model.layers.0.mlp.down_proj
model.norm
lm_head
```

Each name maps directly to one concept from the earlier chapter.

### Llama 3.2 1B: A Compact Example

The Raschka `standalone-llama32.ipynb` implementation is useful because it shows the architecture in compact PyTorch. A representative Llama 3.2 1B config is:

| Field | Value | Architectural meaning |
|---|---:|---|
| `vocab_size` | 128,256 | Token embedding rows and LM-head output classes |
| `context_length` / `max_position_embeddings` | 131,072 | Long-context target |
| `emb_dim` / `hidden_size` | 2,048 | Residual-stream width |
| `n_layers` / `num_hidden_layers` | 16 | Number of decoder blocks |
| `n_heads` / `num_attention_heads` | 32 | Query heads |
| `n_kv_groups` / `num_key_value_heads` | 8 | Key/value heads for GQA |
| `hidden_dim` / `intermediate_size` | 8,192 | FFN expansion dimension |
| `rope_base` / `rope_theta` | 500,000 | RoPE frequency base |
| `dtype` | `bfloat16` | Common inference/training weight dtype |

The head dimension is:

$$
d_h = \frac{2048}{32} = 64
$$

Because there are 32 query heads and 8 KV heads, each KV head is shared by:

$$
\frac{32}{8} = 4
$$

query heads. This is exactly GQA. Compared with full MHA, the KV cache stores 8 key heads and 8 value heads instead of 32 and 32, reducing attention-cache memory by about 4x for this part of the model.

### Llama 3 8B: The Same Pattern at Larger Width

Llama 3 8B uses the same conceptual block:

| Field | Typical value | Meaning |
|---|---:|---|
| `hidden_size` | 4,096 | Wider residual stream |
| `num_hidden_layers` | 32 | Deeper stack |
| `num_attention_heads` | 32 | Query heads |
| `num_key_value_heads` | 8 | GQA with 4 query heads per KV head |
| `intermediate_size` | 14,336 | SwiGLU FFN dimension |
| `vocab_size` | 128,256 | Llama 3 tokenizer vocabulary |
| `max_position_embeddings` | 8,192 | Original Llama 3 context length |
| `rope_theta` | 500,000 | RoPE base |

The architecture is not fundamentally different from the 1B version. The larger model mostly increases width and depth.

---

## Running Llama Through `transformers`

The simplest inference path is the pipeline API:

```python
import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

pipe("The key idea behind grouped-query attention is", max_new_tokens=80)
```

For a more explicit path:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

inputs = tokenizer("The key idea behind grouped-query attention is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=80, use_cache=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```{admonition} Practical note
:class: important

Some Meta Llama repositories are gated. If loading fails with an access error, the architecture can still be studied via `AutoConfig`, public model cards, or from-scratch educational implementations; loading official weights requires accepting the model license and authenticating with Hugging Face.
```

### From Scratch vs Hugging Face

Raschka's educational implementation and Hugging Face's production implementation describe the same computation at different levels of engineering:

| Educational implementation | Hugging Face implementation |
|---|---|
| Plain `nn.Linear` projections | Same tensors, often wrapped with optimized attention paths |
| Manual causal mask | Generated internally from `attention_mask` and cache position |
| Hand-written GQA repeat | Kernel-aware GQA inside attention implementation |
| Precomputed RoPE `cos`/`sin` buffers | RoPE utilities driven by config |
| Simple generation loop | `generate()` with sampling, beam search, cache classes, stopping criteria |
| Direct `.pth` loading | `from_pretrained()` loading sharded `safetensors` |

The important lesson: `transformers` does not hide the architecture. It packages the same modules behind a stable API.

---

## Gemma 4: Same Transformer Spine, More Modern Engineering

Gemma 4 is still built around decoder-only Transformer language modeling, but it adds several techniques aimed at long context, efficiency, and multimodal use.

The full Gemma 4 checkpoint can wrap text, image, audio, or video handling. For understanding the Transformer architecture, focus on the **text decoder**:

```text
input_ids
  -> token embeddings
  -> optional per-layer embedding signals
  -> repeated decoder blocks
       -> local sliding attention or global full attention
       -> extra RMSNorm points around attention and FFN outputs
       -> gated GELU feed-forward network
       -> optional per-layer embedding residual
  -> final RMSNorm
  -> tied LM head
  -> soft-capped logits
```

### Gemma 4 E2B and E4B at a Glance

The dense E2B and E4B models are good teaching examples because they are relatively small but architecturally rich.

| Field | Gemma 4 E2B | Gemma 4 E4B | Meaning |
|---|---:|---:|---|
| `vocab_size` | 262,144 | 262,144 | Main token vocabulary |
| `hidden_size` | 1,536 | 2,560 | Residual-stream width |
| `num_hidden_layers` | 35 | 42 | Decoder depth |
| `num_attention_heads` | 8 | 8 | Query heads |
| `num_key_value_heads` | 1 | 2 | MQA/GQA-style KV sharing |
| `head_dim` | 256 | 256 | Local attention head dimension |
| `global_head_dim` | 512 | 512 | Full-attention head dimension |
| `intermediate_size` | 6,144 | 10,240 | Gated FFN width |
| `sliding_window` | 512 | 512 | Local attention window |
| `max_position_embeddings` | 131,072 | 131,072 | Text context length |
| `hidden_size_per_layer_input` | 256 | 256 | Per-layer embedding width |
| `num_kv_shared_layers` | 20 | 18 | Later layers sharing KV projections |
| `tie_word_embeddings` | `True` | `True` | Input embedding and LM head share weights |

The "E" models are efficient-parameter variants. They include **Per-Layer Embeddings (PLE)**: every decoder layer receives an extra low-dimensional token signal instead of relying only on the first input embedding.

### Hybrid Attention: Local Plus Global

Llama 3 uses the same full causal attention pattern at each layer. Gemma 4 alternates between:

- **sliding attention**, where each token mostly attends to a recent local window
- **full attention**, where tokens can attend to the full causal prefix

For E2B, the layer pattern is:

```python
layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * 7
```

For E4B, the pattern is:

```python
layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 7
```

This gives most layers the speed and memory behavior of local attention, while periodic global layers maintain long-range communication. It is a practical answer to the long-context problem: not every layer needs full quadratic attention.

### Two RoPE Regimes

Gemma 4 uses separate RoPE settings for local and global attention:

| Layer type | RoPE behavior |
|---|---|
| Sliding attention | Default RoPE, usually with `rope_theta = 10_000` |
| Full attention | Proportional RoPE, usually with `rope_theta = 1_000_000` and partial rotary factor `0.25` |

This is why the Gemma 4 config stores nested `rope_parameters`, for example:

```python
{
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10_000.0,
    },
    "full_attention": {
        "rope_type": "proportional",
        "rope_theta": 1_000_000.0,
        "partial_rotary_factor": 0.25,
    },
}
```

The local layers preserve short-range behavior. The global layers use a long-context-friendly positional recipe.

### Per-Layer Embeddings

In a classic decoder-only Transformer, token identity enters once:

```text
input_ids -> token embedding -> residual stream
```

Gemma 4 adds a second path:

```text
input_ids
  -> packed per-layer embedding table
  -> reshape to [batch, seq, num_layers, ple_dim]
  -> feed layer i a small auxiliary vector
```

It also projects the normal input embedding into the same per-layer space. The two signals are combined and normalized before being used inside each layer.

Conceptually, this gives every layer its own compact reminder of token identity:

$$
p_i = \frac{p^{\text{token}}_i + p^{\text{context}}_i}{\sqrt{2}}
$$

Then layer $i$ can add a gated residual derived from $p_i$.

```{admonition} Why PLE matters
:class: tip

PLE increases useful capacity without making the main residual stream as wide as a conventional model with the same total parameter count. That is why Gemma 4 E2B can have more total embedding parameters than its "effective" parameter count suggests.
```

### More Normalization Points

Llama-style blocks typically use RMSNorm before attention and before the FFN.
Gemma 4 uses additional RMSNorms around sublayer outputs:

```text
x_attn = attention(input_layernorm(x))
x = x + post_attention_layernorm(x_attn)

x_ffn = mlp(pre_feedforward_layernorm(x))
x = x + post_feedforward_layernorm(x_ffn)
```

This is still residual learning, but with more explicit scale control.

### Gemma 4 Feed-Forward Network

Gemma 4 uses a gated FFN like other modern LLMs, but its activation is GELU with tanh approximation rather than Llama's SiLU/SwiGLU:

$$
\mathrm{GatedGELU}(x)
= W_{\text{down}}\left(\mathrm{GELU}(xW_{\text{gate}}) \odot xW_{\text{up}}\right)
$$

The shape logic is the same as SwiGLU:

- `gate_proj`: creates the gate branch
- `up_proj`: creates the value branch
- elementwise multiply
- `down_proj`: returns to `hidden_size`

---

## Running Gemma 4 Through `transformers`

For multimodal Gemma 4, Hugging Face recommends loading a processor with the model:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain sliding-window attention in one sentence."},
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=80)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
```

For text-only inspection, the config is often enough:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("google/gemma-4-E2B")
text = cfg.text_config

print(text.hidden_size)
print(text.num_hidden_layers)
print(text.num_attention_heads, text.num_key_value_heads)
print(text.sliding_window)
print(text.layer_types)
```

---

## Llama 3 vs Gemma 4

| Design choice | Llama 3 / Llama 3.2 | Gemma 4 E2B / E4B |
|---|---|---|
| Core architecture | Decoder-only causal LM | Decoder-only text decoder inside multimodal wrapper |
| Attention pattern | Full causal attention in each layer | Mixture of sliding-window and full attention layers |
| KV sharing | GQA | MQA/GQA plus optional cross-layer KV sharing |
| Positional encoding | RoPE, with long-context scaling in Llama 3.2 | Separate local/global RoPE settings |
| Normalization | Pre-Norm RMSNorm | RMSNorm before and after major sublayers |
| FFN | SwiGLU with SiLU | Gated GELU |
| Embeddings | Token embedding, sometimes tied in smaller Llama 3.2 | Token embedding plus Per-Layer Embeddings, tied output head |
| Long-context strategy | RoPE scaling plus GQA cache savings | Sliding/global attention, proportional RoPE, KV sharing |
| HF API | `AutoModelForCausalLM`, `AutoTokenizer` | `AutoProcessor`, multimodal model classes, nested `text_config` |

The common spine is still:

```text
tokens -> embeddings -> many residual attention/MLP blocks -> logits
```

The differences are engineering choices around memory, context length, modalities, and training stability.

---

## How to Read a New Transformer Config

When you see a new Hugging Face model, inspect it in this order:

1. `model_type`: identifies the architecture family.
2. `architectures`: tells you which model class is expected.
3. `vocab_size`: tells you the tokenizer/output space.
4. `hidden_size` and `num_hidden_layers`: width and depth.
5. `num_attention_heads` and `num_key_value_heads`: MHA, MQA, or GQA.
6. `intermediate_size` and `hidden_act`: FFN type and width.
7. `max_position_embeddings`, `rope_theta`, and `rope_parameters`: context and position strategy.
8. `sliding_window` or `layer_types`: whether attention is full, local, or hybrid.
9. `tie_word_embeddings`: whether input and output embeddings are shared.
10. model-specific fields such as Gemma 4's `hidden_size_per_layer_input` or `num_kv_shared_layers`.

This turns a large model from a mysterious object into a stack of familiar parts.

---

## References

- Raschka, *LLMs from Scratch*, [Converting GPT to Llama](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama)
- Raschka, *LLMs from Scratch*, [Gemma 4](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/17_gemma4)
- Hugging Face Transformers, [Llama model documentation](https://huggingface.co/docs/transformers/model_doc/llama)
- Hugging Face Transformers, [Gemma 4 model documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- Meta Llama, [Llama 3.2 model card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Google, [Gemma 4 E2B model card](https://huggingface.co/google/gemma-4-E2B)
