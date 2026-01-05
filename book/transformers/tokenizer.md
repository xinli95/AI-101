# Tokenizer

This tutorial explains **Byte Pair Encoding (BPE) over bytes** in two stages:

1. Implementing a **toy BPE tokenizer from scratch** to understand the core algorithm.

2. Understanding how **real-world tokenizers** implement the same idea using `tiktoken` and Hugging Face `transformers`.

---

## Part 1: Implementing BPE From Scratch

In this section, we build a **minimal, educational BPE tokenizer** that operates on bytes.
The goal is _conceptual clarity_, not performance.

### 1. What Does "Over Bytes" Mean?

All text is first converted into **UTF-8 bytes**.
This guarantees:

- No unknown tokens

- Full reversibility

- Language and symbol agnosticism

```python

def to_bytes(text):

    return list(text.encode('utf-8'))

```

### 2. Initialize Tokens

Each byte starts as its own token.
We represent tokens as **tuples of integers** so they are hashable.

```python

def bytes_to_tokens(byte_seq):

    return [(b,) for b in byte_seq]

```

### 3. Count Adjacent Token Pairs

BPE repeatedly finds the **most frequent adjacent token pair**.

```python

from collections import Counter


def get_pair_frequencies(tokens):

    pairs = Counter()

    for i in range(len(tokens) - 1):

        pairs[(tokens[i], tokens[i + 1])] += 1

    return pairs

```

### 4. Merge a Token Pair

When a pair is merged, it becomes a **new token** formed by concatenating the bytes.

```python

def merge_pair(tokens, pair_to_merge):

    merged = []

    i = 0

    while i < len(tokens):

        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_merge:

            merged.append(tokens[i] + tokens[i+1])

            i += 2

        else:

            merged.append(tokens[i])

            i += 1

    return merged

```

### 5. Training the BPE Model

The training loop repeatedly:

1. Counts adjacent pairs

2. Selects the most frequent

3. Merges it

```python

def train_bpe(corpus, num_merges=50):

    corpus_tokens = [bytes_to_tokens(to_bytes(text)) for text in corpus]

    merges = []


    for _ in range(num_merges):

        pair_freqs = Counter()

        for tokens in corpus_tokens:

            pair_freqs.update(get_pair_frequencies(tokens))


        if not pair_freqs:

            break


        best_pair = pair_freqs.most_common(1)[0][0]

        merges.append(best_pair)


        corpus_tokens = [merge_pair(tokens, best_pair) for tokens in corpus_tokens]


    return merges

```

### 6. Encoding and Decoding

At inference time, merges are applied **greedily in order**.

```python

def encode(text, merges):

    tokens = bytes_to_tokens(to_bytes(text))

    for pair in merges:

        tokens = merge_pair(tokens, pair)

    return tokens


def decode(tokens):

    byte_seq = []

    for token in tokens:

        byte_seq.extend(token)

    return bytes(byte_seq).decode('utf-8', errors='replace')

```

This completes a **fully reversible BPE tokenizer**.

---

## Part 2: Real-World BPE Tokenizers

Modern LLM tokenizers implement the _same algorithm_, with engineering improvements.

### 1. `tiktoken` (OpenAI)

`tiktoken` is OpenAI’s production tokenizer, optimized for speed and exact model compatibility.

```python

import tiktoken


enc = tiktoken.encoding_for_model('gpt-4o-mini')

tokens = enc.encode('Hello 🤯')

```

#### Key Files (Conceptual)

- **`encoder.json`**: token → integer ID mapping

- **`merges.txt`**: ordered list of BPE merges

- **Regex pre-tokenizer**: splits text before byte conversion

Despite optimizations, the core idea is still:

```text

UTF-8 bytes → BPE merges → token IDs

```

### 2. Hugging Face `transformers`

Hugging Face uses the `tokenizers` Rust library under the hood.

```python

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('gpt2')

tokenizer('Hello 🤯')

```

#### Relevant Files

- **`vocab.json`**: maps subword tokens to IDs

- **`merges.txt`**: BPE merge rules

- **`tokenization_gpt2.py`**: Python wrapper logic

These components directly correspond to the toy implementation:

- Byte tuples → token strings

- Merge loop → Rust-optimized BPE

- Greedy application → deterministic encoding

---
