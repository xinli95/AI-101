(olmo3-instruct)=

# OLMo 3 Instruct

> This tutorial explains the **OLMo 3 Instruct** post-training pipeline. It parallels OLMo 3 Think (SFT → DPO → RLVR), but optimizes for _everyday assistant usage_: **multi-turn chat**, **conciseness**, and **tool use**, instead of long reasoning traces.

## Why Instruct models exist (even when Think models are strong)

Reasoning models (like OLMo 3 Think) can be very powerful, but many real-world user queries are:

- simple information seeking
- advice seeking
- conversational
- tool-based tasks (search, retrieval, function calling)

These do **not** require long “thinking trajectories”.

OLMo 3 Instruct is built to be:

- **fast at inference**
- **concise by default**
- **strong at multi-turn dialogue**
- **capable of tool use / function calling**

A core design goal from the report:

> Everyday chat settings often do not require inference-time scaling via long thoughts.

So Instruct models avoid the overhead of generating long reasoning traces unless needed.

## High-level pipeline: same stages, different target behavior

OLMo 3 Instruct uses the same three-stage post-training recipe as Think:

1. **Instruct SFT**
2. **Instruct DPO**
3. **Instruct RLVR** (with GRPO)

But the emphasis shifts:

- Think: maximize reasoning via explicit thought traces
- Instruct: maximize usability (chat + tools + conciseness)

## Evaluation focus for Instruct models

OLMo 3 Instruct evaluation includes:

- most of the Think benchmark suite (general capability + QA + math + code)
- plus additional tool-use metrics such as:
  - Berkeley Function Calling Leaderboard
  - LitQA2
  - SimpleQA

A notable qualitative result described:

- Instruct models benefit significantly from tool usage behavior learned in post-training
- At 7B scale, OLMo 3 Instruct can outperform Qwen-3 (thinking disabled) on some tasks
- At 32B scale, that gap is smaller / disappears

## Stage 1: Dolci Instruct SFT (multi-turn chat + agentic tool use)

OLMo 3 Instruct starts by building a new SFT dataset:

- **[Dolci Instruct SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)**

It builds on the prior OLMo 2 instruct data but introduces key improvements:

### Key changes vs earlier instruct datasets

1. **Remove reasoning traces**

   - any existing “chain-of-thought style” traces are stripped out
   - to encourage short, direct answers

2. **Upgrade synthetic generations to newer models**

   - use strong modern generators (e.g., GPT-4.1 instead of older GPT-3.5 era models)
   - generally yields more coherent and instruction-following completions

3. **Add extensive supervised function calling**
   - a major focus for OLMo 3 Instruct is realistic tool usage

### Function calling data: two strategies

The report describes two complementary ways of generating function-calling data.

#### (A) Real trajectories (tool use in executable environments)

They build datasets like:

- ScienceQA
- WebSearchQA

These are created by using a strong model (GPT-4.1 or GPT-5) equipped with tools that connect to:

- the web
- a corpus of papers

via **MCP servers**.

This produces _realistic trajectories_ containing:

- multiple interaction steps
- real tool outputs
- tool errors and edge cases

#### (B) Simulated interactions (large-scale synthetic tool use)

Real tool trajectories are expensive to collect at scale.

To increase tool diversity and volume, they also create simulated tool-use data:

- start with a pool of tools + API specs from public sources
- prompt a pool of generator LLMs (GPT-4o, GPT-4.1, GPT-5) to synthesize:
  - user requests
  - tool calls
  - tool outputs
  - assistant follow-ups

This provides broad coverage over:

- many tool schemas
- multi-turn agent loops
- multiple tool calls per conversation

**Details of function calling datasets. Multi-turn refers to multiple user turns per trajectory and multi-step refers to multiple environment interactions per user request**

| Dataset                   | Env. interactions | # Trajectories | # Unique functions | % Multi-turn | % Multi-step |
| ------------------------- | ----------------: | -------------: | -----------------: | -----------: | -----------: |
| Science QA (Real, MCP)    |             22.6K |              8 |                  – |        42.3% |            – |
| Web Search QA (Real, MCP) |              6.6K |              3 |                  – |        76.1% |            – |
| SimFC (Simulated)         |              200K |          42.6K |              42.3% |        23.8% |            – |

### Unified tool formatting is crucial

A major lesson emphasized:

> a unified format for tool calling data is necessary for strong tool performance

Their unified format includes:

- tool specs included in the system prompt
- tool calls wrapped in **XML tags** in assistant messages
- tool outputs emitted by a special **environment role** (with dedicated special tokens)

This consistency matters because tool learning is format-sensitive.

### Mixture tuning for Instruct SFT

They tune the data mix similarly to Think:

- start with a base mixture of ~100K supervised examples
- add/ablate domains on top of the OLMo 2 base
- evaluate impact per domain

### Starting point: Instruct is trained from Think SFT

A notable finding:

> training Instruct on top of the Think SFT model improves benchmark performance and does not increase average response length

So the Instruct pipeline does **not** start from the raw base model; it starts from:

- **OLMo 3 Think SFT checkpoint**

This provides stronger capabilities while still allowing Instruct SFT to reshape output behavior into short/direct responses.

## Stage 2: Instruct DPO (preferences for chat helpfulness + conciseness + tools)

Instruct DPO expands the Delta Learning idea to focus on general chat quality.

They use three main types of preference pairs:

### (1) Delta Learning pairs (Qwen-3, thinking OFF)

Same core contrastive idea as Think DPO, but here:

- both chosen and rejected completions come from Qwen-3
- thinking mode is turned OFF
- to emphasize direct assistant answers

### (2) Delta-maximized GPT-judged pairs (modern UltraFeedback-style)

They generate completions from a pool of diverse models, score them with GPT-4.1, and pick:

- best completion = chosen
- worst completion = rejected

Key twist:

> modern model pools often have small deltas (everyone is “pretty good”), so you must ensure at least one weak model exists in the pool

This is called **Delta Maximization**:

- enforce a big gap between best and worst completions
- otherwise preference tuning can be noisy or harmful

### (3) Multi-turn preference pairs

They generate multi-turn context by:

- self-talk / synthetic context expansion around an existing prompt

Then preferences differ only in the **final assistant response**, using large gaps like:

- GPT-3.5 vs GPT-4.1
- Qwen-3-0.6B vs Qwen-3-32B

This teaches:

- stable “assistant personality”
- turn-level helpfulness
- conversational continuity

### Length control: fight judge verbosity bias

LLM judges often prefer longer outputs.

To encourage conciseness:

- they filter preference pairs so chosen/rejected length differs by **≤ 100 tokens**

This may reduce some benchmark metrics, but improves:

- usability
- vibe tests
- downstream RL stability (it becomes a better starting point)

## Stage 3: Instruct RLVR (GRPO) with small modifications

Instruct RL is “the same stack” as Think RLVR, but with a few targeted changes.

### Key differences vs Think RL

1. **Less challenging math/code tasks**

   - remove the hardest problems (since the goal is general usability, not maximal reasoning)

2. **No offline difficulty filtering**

   - Think RL used difficulty filtering to avoid trivial prompts
   - Instruct RL drops this step because it is not focused on hard reasoning trajectories

3. **Response length cap**
   - maximum response length is capped at **8K tokens**
   - to prevent overly long answers in everyday chat

### RL mixture

Instruct RL is trained on a mix of:

- general chat
- math
- code

Rewards still include:

- verifiable rewards for math/code
- LLM-judge rewards for chat

Final model choice is based on:

- average performance
- response length analysis
- vibe tests

---

## Practical summary: how OLMo 3 Instruct differs from OLMo 3 Think

**Same structure**:

- SFT → DPO → RLVR (GRPO)

**Different target behavior**:

- no explicit thought trajectories
- more multi-turn conversation and assistant “polish”
- much stronger function calling behaviors (with unified tool format)
- explicit conciseness pressure (length filters + response cap)

---

## References

- OLMo 3 technical report: <https://arxiv.org/abs/2512.13961>
