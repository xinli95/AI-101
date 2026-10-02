# Post-training Overview

Post-training adapts a pretrained model to target tasks, preferences, or constraints. Before choosing a training method, identify what is missing: instructions, facts, consistent behavior, product-specific judgment, task skill, or serving efficiency.

## A mental model: the progression of intelligence

Lin Qiao's talk at Sequoia Capital's *Own Your Intelligence* event offers a useful analogy: adapting a model resembles learning from instructions and reference material, then from examples, preferences, and practice. The progression below summarizes the talk and its method-selection slide; the technical distinctions that follow clarify how to use the analogy.

- **Prompting — give clear instructions.** Use the model as-is, with instructions and a few examples, to test an idea and establish a baseline.
- **RAG — consult reference material.** Retrieve relevant information at inference time so the model can work with your private or changing facts.
- **Supervised fine-tuning (SFT) — study worked examples.** Like learning from literature and demonstrations, the model learns what a good response looks like. This can shape structure, style, and tool-use behavior.
- **Preference tuning — develop taste and judgment.** Learn which of several plausible answers better matches your product's preferences. DPO and classic RLHF are two ways to use preference feedback.
- **Reinforcement learning (RL) — practice a specialty with feedback.** Like professional training, repeated attempts and rewards can improve performance on a domain-specific task. The useful prerequisite is a reliable way to score attempts.

**Distillation adds an efficiency dimension:** a student learns from a stronger teacher, often to deliver useful behavior with a smaller, faster, or cheaper model.

Source: Lin Qiao, [*When (and How) to Post-Train Your Own AI Models* — official transcript](https://sequoiacap.com/podcast/when-%28and-how%29-to-post-train-your-own-ai-models). Video anchors: [progression and learning analogy, approximately 5:40–7:15](https://www.youtube.com/watch?v=yAvJ7b_FxUA&t=340s) and [method selection, approximately 8:30 onward](https://www.youtube.com/watch?v=yAvJ7b_FxUA&t=510s).

## Choose the method based on the problem

| What is missing? | Method to consider | What you supply | What it helps change |
| --- | --- | --- | --- |
| A clear task specification or an initial baseline | **Prompting** | Instructions, constraints, and in-context examples | How the existing model responds to this request |
| Private, current, or frequently changing facts | **RAG** | Retrieved documents or records | The evidence available when answering |
| Consistent output format or behavior | **[SFT](sft.md)** | High-quality prompt–response demonstrations | Learned response patterns, style, and tool-use habits |
| Product-specific taste: comparing answers is easier than specifying the ideal answer | **[DPO](dpo.md) / [RLHF](rlhf.md)** | Preference comparisons | Which responses the model tends to favor |
| Skill on a task whose attempts can be scored reliably | **RL / reinforcement fine-tuning (RFT)**; [RLVR](rlvr.md) for verifiable rewards | Training tasks, sampled attempts, and a reward or verifier | Strategies that achieve higher task reward |
| Lower latency or cost while retaining useful quality | **[Distillation](distillation.md)** | Teacher outputs or distributions | Behavior transferred to a student model |

## How to read this framework

**These are complementary choices, not a required ladder.** Prompting and RAG ordinarily leave model weights unchanged; SFT, preference tuning, and RL update trainable parameters. A tuned model can still use RAG. Distillation can use SFT on teacher-generated examples, so it describes a transfer goal rather than a mutually exclusive training algorithm.

**Demonstrations, comparisons, and rewards are different learning signals.** Standard SFT imitates target responses; it does not require explicit correct-versus-incorrect pairs. DPO learns directly from preferred/rejected response pairs without a separate reward model or an RL rollout loop. Classic RLHF learns a reward model from preferences and then uses RL to optimize the policy. Thus, preference tuning and RL overlap; the distinction in the table is about the problem and feedback available. See the [DPO paper](https://arxiv.org/abs/2305.18290) for the algorithmic distinction.

**Implementation choices and feedback quality matter.** LoRA is a parameter-efficient way to tune a model, including with SFT; it is not a separate learning objective. RAG depends on retrieving useful evidence, and RL depends on rewards that reflect actual success. For code, compilation alone is weaker evidence than passing meaningful correctness tests. Neither fine-tuning nor a professional-training analogy guarantees expertise.

For example, a support assistant might retrieve the latest product policy with RAG, learn a consistent escalation response through SFT, and learn the team's preferred tone through DPO. RL becomes relevant if it can practice support workflows in an environment with a trustworthy success signal. A teacher can then provide training data for a cheaper student, whose quality must be checked again.

Start with a prompting baseline and a representative [evaluation set](../evaluation/overview.md). Inspect failures, choose the method that addresses the observed gap, and compare quality, latency, and cost on held-out cases after each change. There is no need to use every method.

## Where to go next

- **[SFT](sft.md):** demonstration learning, loss masking, and catastrophic forgetting.
- **[RLHF](rlhf.md) and [DPO](dpo.md):** preference learning, including [PPO](ppo.md) and [advantage estimation](ppo_advantage.md).
- **[RLVR](rlvr.md):** verifiable rewards, [GRPO](grpo.md), and [agentic RL](agentic_rl.md).
- **[Distillation](distillation.md):** transferring teacher behavior to a student.
- **[TRL implementation examples](trl/trl_overview.md):** practical trainers and an agentic RL walkthrough.
