# Summary

The Transformer chapter built the model from the bottom up:

- Tokenizers turn text into token IDs.
- Embedding layers turn token IDs into vectors.
- Attention mixes information across token positions.
- RMSNorm and residual connections stabilize deep stacks.
- Feed-forward networks provide per-token nonlinear computation.
- RoPE injects position information into attention.
- Real architectures such as Llama 3 and Gemma 4 combine these same pieces with engineering choices for speed, memory, long context, and deployment.

When reading a new Hugging Face model, start with `config.json`: the fields for width, depth, attention heads, KV heads, FFN size, normalization, RoPE, and cache behavior usually tell you the architecture before you ever load the weights.
