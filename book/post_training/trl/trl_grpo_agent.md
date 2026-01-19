# Agentic RL with GRPO: Tools, Rewards, and Rollouts

This tutorial explains how to implement an agentic RL setup with GRPO in TRL. We use the BioGRID example to make the ideas concrete: a model learns to query a database via tools, gather evidence, and answer with a constrained format. The focus is on *how to build the agent loop*, *how to design reward functions*, and *how GRPO rollouts differ from standard RL rollouts*.

Reference script:

- https://github.com/xinli95/trl/blob/main/examples/scripts/grpo_agent.py

## 1. What “agentic” means in this setting

In agentic RL, the model is not limited to a single response. It can:

- Decide to call a tool (e.g., a SQL query function).
- Observe the tool result.
- Continue generation in a new turn, possibly calling tools again.

This produces multi-turn rollouts that interleave assistant messages and tool outputs. The sequence is learned end-to-end with GRPO so that good tool use and good final answers are reinforced together.

## 2. Implementation overview (BioGRID example)

The BioGRID script provides a minimal agentic RL loop:

1. Build a local SQLite database from BioGRID.
2. Format questions into a tool-aware prompt.
3. Provide the database query tool to `GRPOTrainer`.
4. Train with reward functions that score correctness, tool usage, and structure.

This gives a complete “agentic” loop: the model must retrieve evidence through the tool to answer correctly.

## 3. Tool-calling and rollout mechanics

When `tools` are passed to `GRPOTrainer`, rollouts are no longer single-step. Instead:

- The model generates a response that may include tool calls.
- The trainer executes the tool calls and appends tool results to the conversation.
- The model continues generation with the updated context.
- This repeats until the model stops calling tools or a max-iteration limit is reached.

This is the core difference from standard RL rollouts, where a single assistant completion is scored once. In agentic GRPO, a *rollout is a multi-turn interaction* and rewards can depend on both intermediate tool behavior and the final answer.

## 4. Designing reward functions for agentic behavior

Agentic RL needs rewards that capture more than correctness. In the BioGRID example, the rewards are implemented directly in the script as three functions:

**Correctness reward (`correctness_reward`)**

- Extracts the final answer from the last assistant message.
- Requires the strict format `*yes*` or `*no*` (case-insensitive).
- Penalizes missing or malformed final answers.
- Rewards correct answers and penalizes incorrect ones.

```python
def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        raw = completion[-1]["content"].lower()
        match = re.search(r"\*(yes|no)\*", raw)
        guess = match.group(1) if match else None
        reward = 0.0
        if guess is None:
            reward -= 0.5
        elif guess == ans.lower():
            reward += 0.6
        else:
            reward -= 1.0
        rewards.append(reward)
    return rewards
```

**Structure reward (`structure_reward`)**

- Checks whether a tool call is followed by a tool response.
- Gives a small positive reward for the expected tool-call sequence.
- Penalizes tool calls that do not receive a tool response.

```python
def structure_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        has_call = False
        has_response = False
        has_other = False
        for turn in completion:
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            else:
                content = turn.get("content")
                if content and content.strip() not in ["", "<think>"]:
                    has_other = True
        if has_call and has_response:
            reward = 0.1 if has_other else 0.05
        elif has_call and not has_response:
            reward = -0.15
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards
```

**Query quality and evidence reward (`query_reward`)**

- Parses all SQL tool calls and tool outputs from the conversation.
- Penalizes too many queries and low-information queries like `LIMIT 1`.
- Rewards use of `WHERE` clauses (more selective queries).
- Penalizes tool errors or missing tool calls.
- Compares query results to the final `*yes*`/`*no*` answer and rewards alignment with evidence.

```python
def query_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        reward = 0.0
        sql_queries = []
        tool_results = []
        for turn in completion:
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    sql = call["function"]["arguments"].get("sql_command", "").lower()
                    sql_queries.append(sql)
            if turn.get("role") == "tool" and turn.get("content"):
                tool_results.append(turn["content"])
        if len(sql_queries) > 3:
            reward -= 1.5
        where_count = 0
        for q in sql_queries:
            if "limit 1" in q:
                reward -= 1.0
            if " where " not in q:
                reward -= 0.5
            else:
                where_count += 1
        reward += min(where_count, 3) * 0.4
        combined_results = []
        error_detected = False
        for res in tool_results:
            if isinstance(res, dict) and "error" in res:
                error_detected = True
            elif isinstance(res, list):
                combined_results.extend(res)
        if error_detected:
            reward -= 2.0
        elif len(sql_queries) == 0:
            reward -= 1.5
        else:
            has_hits = len(combined_results) > 0
            correct_answer = ans.lower()
            if (has_hits and correct_answer == "yes") or (not has_hits and correct_answer == "no"):
                reward += 2.0
            else:
                reward -= 1.5
        rewards.append(reward)
    return rewards
```

## 5. Tool function used by the agent

The tool is a read-only SQL query against a local SQLite database. The timeout prevents long-running queries from stalling rollout.

```python
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def query_biogrid(sql_command: str) -> list[tuple]:
    with timeout(5):
        conn = sqlite3.connect("file:biogrid.db?mode=ro", uri=True)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_command)
            results = cursor.fetchall()
        finally:
            conn.close()
    return results
```

Together, these rewards teach the model to *query effectively* and *use evidence correctly*, not just to output the right label.

## 6. Prompt formatting for tool-aware training

The model is trained on prompts that explicitly describe the database, show how gene aliases appear, and enforce a strict final answer format. This prompt is injected as the single user message for each training example.

```python
def format_example(example):
    question = example["question"]
    preamble = textwrap.dedent("""\
    You have access to the BioGRID SQLite database.
    Use SQL queries to retrieve only the information needed to answer the question.

    Genes may appear in the database in columns `Alt_IDs_Interactor_A` `Alt_IDs_Interactor_B`, `Aliases_Interactor_A` and `Aliases_Interactor_B`,
    and each entry can contain multiple gene names or synonyms separated by '|', for example:
    'entrez gene/locuslink:JNKK(gene name synonym)|entrez gene/locuslink:MAPKK4(gene name synonym)|...'
    So a gene like 'JNKK' or 'MAPKK4' may appear inside one of these strings.

    If the database schema is unclear or you are unsure about column names:
    - First inspect the schema with `PRAGMA table_info(interactions);`
    - Or preview a few rows with `SELECT * FROM interactions LIMIT 1;`

    Otherwise, directly query the required data.

    Final answer must be enclosed in stars, e.g. *Yes* or *No*.
    Facts:
    - The NCBI Taxonomy identifier for humans is taxid:9606.
    """)
    content = f"{preamble}\nQuestion: {question}"
    prompt = [{"role": "user", "content": content}]
    return {"prompt": prompt}
```

This formatting is critical: it tells the model *what tools exist*, *how to use them*, and *what the answer format must be*. Without it, tool calling and reward alignment become unstable.

## 7. Reward aggregation and normalization

In GRPO, multiple reward functions can be combined and normalized. The trainer supports:

- Weighted reward functions (so some signals matter more).
- Two aggregation strategies:
  - Sum rewards then normalize.
  - Normalize each reward then sum.
- Optional scaling across group or batch, or no scaling at all.

This means your reward functions can be simple and task-focused while the trainer handles the stability and balancing logic.

## 8. Handling missing rewards (`None`)

Agentic setups often include rewards that are only relevant to some samples. GRPO supports this by allowing reward functions to return `None` for specific samples. Those values are ignored in aggregation, and training continues without crashing.

This is useful for multi-task or partially-labeled datasets where not every reward applies to every sample.

## 9. How rollouts differ from “normal” RL

**Standard RL rollout**

- Single assistant completion.
- Reward depends only on the final output.

**Agentic GRPO rollout**

- Multi-turn conversation with tool calls.
- Reward can depend on *how* the model got the answer, not just the answer.
- Tool outputs become part of the context, so the policy is trained on action–observation sequences.

In practice, this makes GRPO suitable for tasks where *reasoning via tools* is required, not just direct response generation.

## 10. Minimal run command

```bash
python examples/scripts/grpo_agent.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --output_dir grpo_biogrid_qwen_3g-1.7b \
  --push_to_hub True \
  --use_vllm True \
  --vllm_mode colocate \
  --max_completion_length 1024 \
  --report_to trackio \
  --log_completions True \
  --max_steps 400
```

## 11. Quick mental model

- The model chooses when to call tools.
- Tool results become new context turns.
- Rewards score correctness and tool behavior.
- GRPO optimizes the entire interaction, not just the last answer.
