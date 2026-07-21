(agentic-rl-landscape)=

# Agentic RL: Environments, Reward Design, and Credit Assignment at the Frontier

Everything in {ref}`grpo-family-tutorial` and the {doc}`trl/trl_grpo_trainer` walkthrough shares one assumption: one prompt in, one completion out, one reward computed once. That is the setting most RLVR pipelines run in, and it's a good place to have started — but it is a special case. **Agentic RL** is what happens when you remove that assumption: the model interacts with an environment over many turns, calling tools, observing results, and revising its plan, before any reward is known. This chapter is a landscape survey of that harder setting — what makes it different, where the environments and reward signals come from, why credit assignment is currently considered the field's central open problem, and which frameworks people actually use to train this way in 2026.

This is a fast-moving research area; treat the specific numbers and paper claims below as reported by their authors, not as settled fact, and expect this landscape to keep shifting.

## 1. From single-turn RLVR to agentic RL

Formally, standard LLM-RL (including GRPO as covered so far) treats generation as what [*The Landscape of Agentic Reinforcement Learning for LLMs*](https://arxiv.org/abs/2509.02547) calls a "degenerate single-step" decision process: one state (the prompt), one action (the full completion), one reward. Agentic RL instead treats the interaction as a **temporally extended, partially observable Markov decision process (POMDP)**: the policy acts, the environment responds with a new (partially observed) state, and this repeats until the episode ends — at which point a reward (or several, along the way) becomes available.

| | Single-turn RLVR (GRPO as covered so far) | Agentic RL |
| --- | --- | --- |
| Decision structure | One state → one action → one reward | POMDP: many state → action → observation steps per episode |
| Typical episode length | One completion (hundreds to low thousands of tokens) | 100+ turns, 100K–1M tokens per episode, per the credit-assignment survey below |
| What produces the "next state" | Nothing — generation ends | Tool outputs, environment feedback, other agents |
| Where reward comes from | A verifier or reward model scoring the one completion | Sparse task-completion signals, optionally mixed with step-level progress signals |

The same survey frames "agentic" capability along several axes that RL is used to train: **planning**, **tool use**, **memory**, **reasoning**, **multi-agent coordination**, **self-improvement**, and **perception**. Not every agentic RL project touches all of these — the BioGRID tool-use example in {doc}`trl/trl_grpo_agent` mainly exercises tool use and reasoning — but they're the recurring vocabulary you'll see across the literature.

## 2. Environments: the new bottleneck

In single-turn RLVR, the "environment" is trivial: a prompt and a verifier. In agentic RL, the environment *is* the training signal's substrate — how many turns an episode can run, what partial credit is even observable, what failure modes are possible. As a result, a large fraction of agentic RL research effort now goes into building environments, not just algorithms, and a "Gym for agents" pattern (a `reset()`/`step()` API in the spirit of `Gymnasium`, adapted to text/tool actions) has become the standard abstraction for exposing them to a trainer.

A rough map of the environment landscape, by task domain:

| Domain | Representative environments / benchmarks |
| --- | --- |
| Web / browser agents | WebArena, VisualWebArena, Mind2Web, WebRL, AssistantBench, WorkArena — often unified behind **BrowserGym** (ServiceNow), a single Gym wrapper over most of these |
| OS / desktop / mobile | OSWorld, AndroidWorld, Windows Agent Arena |
| Coding / software engineering | SWE-bench (and SWE-bench Verified / Live / Pro), SWE-Gym, R2E-Gym, Multi-SWE-Bench, SWE-rebench |
| Tool use / API agents | ToolBench, AppWorld, τ-bench / τ²-bench, ToolSandbox, MINT-Bench, AgentBoard |
| Scientific / research agents | ScienceWorld, DiscoveryWorld, MLE-Bench, PaperBench, DeepResearch Bench |
| Games / embodied / text worlds | TextWorld, ALFWorld, Crafter, Voyager (Minecraft) |
| Safety / adversarial | AgentDojo, InjecAgent, ST-WebAgentBench, ToolEmu |
| Cross-domain "gyms" for RL training specifically | **AgentGym** / AgentGym-RL (14 environments spanning web, tools, games, embodied tasks), **GEM** ("General Experience Maker"), **verifiers** (trainer-agnostic environment + reward-verifier protocol), RAGEN's built-in environment suite |

Two things are worth internalizing from this list. First, task benchmarks (SWE-bench, WebArena, τ-bench, ...) and **training** gyms (AgentGym, GEM, verifiers, RAGEN) are different things even when they wrap the same underlying tasks — a benchmark just needs a scoring function, a training gym needs a fast, resettable, parallelizable `step()` loop that a rollout worker can hit thousands of times an hour. Second, the field is converging on a small number of wrapper standards (BrowserGym for web, `verifiers`/AgentGym as general-purpose adapters) precisely so that a new trainer doesn't need bespoke integration code for every task suite — the same motivation behind Gymnasium's original `reset`/`step`/`reward` contract in classic RL.

## 3. Reward design: from one scalar to a reward *stream*

The survey above formalizes agentic reward roughly as: a **task-completion reward** at the end of the episode, plus an optional **step-level progress reward** along the way, and zero otherwise. Both halves are active research topics.

**Outcome, process, and verifiable rewards.** Outcome rewards (did the PR pass the tests? did the purchase complete?) are the least effort to define and the hardest to learn from over long horizons, for reasons covered in §4. Process rewards attempt to give credit for good intermediate behavior — the `structure_reward` and `query_reward` functions in {doc}`trl/trl_grpo_agent` are exactly this, hand-written for one domain (rewarding a tool-call/tool-response pair, penalizing zero-evidence queries). At scale, people try to learn this signal instead of hand-writing it: **process reward models (PRMs)** trained to score partial trajectories. Two examples referenced in the credit-assignment literature: *ProgRM*, which extracts intermediate milestones automatically from demonstrations to produce dense progress estimates, and *AgentPRM*, a process-level reward model scoring trajectories to guide optimization.

**Reward hacking is worse, not better, in agentic settings.** A tool-using agent has more surface area to exploit a proxy reward than a single-turn completion does. Documented failure modes include verifier gaming (crafting outputs that satisfy a rule-based check without doing the task), rubric exploitation in open-ended grading (sycophancy, self-praise, padding responses to appear more thorough), and PRMs themselves being gamed once they become the optimization target — the classic Goodhart's-law problem, now one level removed. One emerging mitigation direction is **agentic verification**: rather than trusting a passive LLM judge reading a transcript, have the verifier actively probe environment state — e.g. execute a check command to confirm a side effect actually happened — which is harder to fool than a text-only judge. This is the same instinct behind why RLVR uses executable/rule-based verifiers over learned reward models wherever possible; agentic settings just make the stakes higher because the "proxy" is now mediating a multi-step interaction instead of one static answer.

## 4. Credit assignment: the field's central open problem

Recall from {ref}`grpo-family-tutorial` that GRPO computes one advantage $\hat A_i$ per *completion* and broadcasts it uniformly to every token in that completion: $\hat A_{i,t} = \hat A_i$ for all $t$. That's a defensible approximation when a completion is one coherent response. It stops being defensible when "one completion" is actually a 100-turn trajectory interleaving dozens of tool calls, only some of which mattered for the final outcome.

[*From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models*](https://arxiv.org/abs/2604.09459) — a 2026 survey covering roughly 47 credit-assignment methods published 2024–early 2026 — frames the problem along two axes:

- **Granularity**: token-level, segment-level, step-level, turn-level, and multi-agent credit assignment, in increasing order of how much structure you assume the trajectory has.
- **Methodology family**: Monte Carlo (sample-based trajectory evaluation), temporal-difference (bootstrapped value estimates, à la the GAE machinery in {doc}`trl/trl_ppo_trainer`), model-based (using a learned environment model to propagate credit), game-theoretic (Shapley-value-style attribution, mostly for multi-agent settings), and information-theoretic (mutual-information / causal attribution).

The same survey's diagnosis of *why* this is hard: agentic trajectories have stochastic transitions, partial observability, and — at 100+ turns / 100K–1M tokens — episode-level outcome reward becomes almost uninformative. If a 60-turn agent fails, "the episode got reward 0" doesn't tell you whether turn 3's tool call was wrong, turn 40's plan was wrong, or the final answer was just mis-formatted. This is the long-horizon analogue of the length-bias and entropy-collapse pathologies already covered for single-turn GRPO in {ref}`grpo-family-tutorial` §3 — except here the failure mode isn't "the objective quietly rewards being verbose," it's "the objective has almost no gradient signal to work with at all."

### Two concrete approaches

**GiGPO — hierarchical, still critic-free.** [*Group-in-Group Policy Optimization for LLM Agent Training*](https://arxiv.org/abs/2505.10978) extends GRPO's core idea (group-relative advantages, no learned critic) to the multi-turn setting with a **two-level grouping** scheme:

- **Episode-level groups**, computed exactly like vanilla GRPO — a group of full trajectories for the same task, normalized against each other, capturing overall task success.
- **Step-level groups**, formed by collecting the different actions taken from the *same* "anchor state" across different trajectories in the batch (e.g. every trajectory that happened to reach an identical intermediate game state or tool-call context), then computing a relative advantage *within that anchor group* — a group-relative baseline for a single step, reusing rollouts you already collected rather than sampling more.

Both levels stay critic-free — no value model, no GAE backward pass — which is the entire appeal of the GRPO family in the first place (see {doc}`trl/trl_ppo_trainer` §14 for what that machinery costs). The authors report double-digit success-rate improvements over single-level baselines on ALFWorld and WebShop at the 1.5B scale; treat that as a reported result from one paper, not a universal constant.

**StarPO / RAGEN — turn-level RL and its failure mode.** [*RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*](https://arxiv.org/abs/2504.20073) trains agents with **StarPO** ("State-Thinking-Actions-Reward Policy Optimization"), a trajectory-level objective for interactive, stochastic environments. Its most cited finding isn't the algorithm itself but a failure mode it exposes: an **"Echo Trap"**, where reward variance collapses and gradients spike partway through training — the agent converges onto a narrow set of shallow, repetitive strategies rather than genuinely reasoning. The fix, **StarPO-S**, stabilizes training with trajectory filtering (drop uninformative/degenerate rollouts before they pollute the batch), reintroducing a critic, and decoupled clipping ranges — notably, walking back *toward* some of the PPO-style machinery that critic-free methods try to avoid, which is a useful reminder that "no critic" is a spectrum, not a binary choice, once episodes get long and stochastic enough.

Between these two families sits a general lesson: **structural** credit assignment (GiGPO's anchor-state grouping, turn-level advantage decomposition) and **learned dense reward** (PRMs) are two different answers to the same problem, and current research doesn't clearly favor one — PRMs give a denser signal but inherit reward-hacking risk; structural methods stay hack-resistant (they're still grounded in verifiable outcome reward) but need the trajectory structure they exploit (shared states, turn boundaries) to actually exist in the task.

## 5. Popular frameworks: training the policy

| Framework | What differentiates it |
| --- | --- |
| [verl](https://github.com/volcengine/verl) (ByteDance) | The most widely adopted general-purpose LLM-RL library at this point; broad algorithm support (PPO, GRPO, and others), large contributor base, common substrate other agentic-RL frameworks fork or extend |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray + vLLM + DeepSpeed based; strong reward-model/critic support, async rollout + training colocation |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) (Berkeley/NovaSky) | Full-stack, modular; ships an agent layer plus a "gymnasium" of tool-use tasks (math, code, search, SQL) specifically for long-horizon agent training |
| [AReaL](https://github.com/inclusionAI/AReaL) (Ant Group / Tsinghua) | Fully asynchronous — decouples generation and training entirely so rollout workers never idle waiting on a training step; reports large throughput gains over synchronous systems |
| [slime](https://github.com/THUDM/slime) (THUDM/Zhipu) | Also fully async/decoupled, with GPUs explicitly partitioned between SGLang-based rollout engines and training engines |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) | Purpose-built for multi-turn, stochastic-environment agent RL; ships StarPO/StarPO-S and a diagnostic suite of built-in environments — more of a research testbed than a production trainer |
| [verl-agent](https://github.com/langfengQ/verl-agent) | verl extended for long-horizon LLM/VLM agents; the reference implementation of GiGPO |
| [rLLM](https://github.com/agentica-project/rllm) (Agentica / Berkeley) | Built on a heavily modified verl fork; the training system behind DeepSWE, DeepCoder, and DeepScaleR |
| [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT) | Decouples rollout and training onto independently scalable devices sharing one experience buffer; explicitly designed to tolerate delayed rewards, stragglers, and environment/agent failures via timeout/retry/skip |
| [Agent Lightning](https://github.com/microsoft/agent-lightning) (Microsoft) | A different design point: decouples RL *training* from agent *execution* entirely, so an agent built in LangChain, AutoGen, the OpenAI Agents SDK, or from scratch can be trained with near-zero code changes, rather than requiring the agent loop to live inside the RL framework |
| [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | Fully async, designed to scale to 1000+ GPUs; integrates with the `verifiers` environment protocol |
| [OpenPipe ART](https://github.com/openpipe/art) | Deliberately lightweight — a single-GPU-friendly GRPO trainer aimed at "on-the-job" training of existing agents rather than large-scale research infrastructure |

The design axis that best explains *why* there are this many frameworks is **how tightly rollout generation is coupled to training**: colocated (simplest, what {doc}`trl/trl_grpo_trainer` §8 calls vLLM "colocate mode"), synchronous-but-separate, or fully asynchronous/decoupled (AReaL, slime, Trinity-RFT, Agent Lightning) so that slow, long-horizon agentic rollouts never stall the training GPUs. That coupling question barely matters for single-turn RLVR, where a rollout takes milliseconds to seconds — it becomes the dominant systems problem once a rollout is a 100-turn, tool-calling episode that might take minutes and might hang on a flaky external API.

## 6. Grounding this in the book: what {doc}`trl/trl_grpo_agent` already does

The BioGRID example in {doc}`trl/trl_grpo_agent` is a small, concrete instance of almost everything in this chapter, worth re-reading with this vocabulary in hand:

- Its environment is a hand-rolled SQL-query tool against a local SQLite database — a minimal, single-domain version of the "training gym" pattern in §2, without the generality of AgentGym or `verifiers`.
- Its `correctness_reward` is the outcome reward; `structure_reward` and `query_reward` are exactly the **step-level progress reward** described in §3 — but *hand-designed*, not learned (no PRM) and not structurally derived (no GiGPO-style anchor-state grouping).
- It uses plain `GRPOTrainer`, meaning credit assignment is still the vanilla GRPO broadcast: one advantage per full multi-turn rollout, spread evenly across every token including tool calls and tool responses. That's a reasonable choice at the scale of that example (a handful of tool calls per episode) — but §4 is precisely the argument for why this stops scaling once episodes grow from a handful of turns to dozens or hundreds.

If you wanted to push that example further, the natural next steps mirror this chapter's structure: replace the hand-written `structure_reward`/`query_reward` with a learned process reward model if the domain got too varied to hand-write rules for, or move from `GRPOTrainer`'s flat broadcast to a GiGPO-style step-level advantage if episodes got long enough that "one advantage for the whole trajectory" stopped giving useful gradient to early tool calls.

## 7. Open problems, as of mid-2026

Pulling from the survey's own forward-looking discussion:

- **Credit assignment remains "a critical and largely unsolved problem."** Every technique in §4 is a partial answer; none is a general solution, and the survey explicitly frames granular credit assignment as the field's key bottleneck for advancing agentic systems.
- **Reward hacking and trustworthiness.** As agents get more tool access, failure modes escalate from "exploits a scoring rubric" toward genuinely concerning behavior — alignment faking, deception, sabotage-like actions have been observed in tool-access RL settings, not just academic curiosities.
- **Scaling the environments, not just the models.** Building enough diverse, high-quality, resettable environments to train on is now considered as much of a bottleneck as compute — echoing how the field's central resource question shifted, in prior eras, from "more parameters" to "more (and better) data."
- **The mechanistic debate.** Does RL teach agents genuinely new capabilities, or mostly elicit/amplify behaviors already latent in the base model? This question — first raised for math reasoning RL (see {ref}`grpo-family-tutorial` §7.3) — recurs for agentic RL with even less consensus, since agentic tasks are harder to probe for "did the base model already have this skill."
- **Deployment-side architecture.** Guardrails, human-in-the-loop verification, hierarchical orchestration, and inter-agent communication protocols are increasingly treated as part of the RL story, not a separate concern layered on afterward.

## 8. Quick mental model

- Agentic RL = single-turn RLVR's POMDP, extended over many turns of environment interaction before reward is known — the same algorithms (PPO, GRPO, and descendants) apply, but every assumption about "one completion, one reward" needs revisiting.
- The environment is now part of the research contribution, not just a data source — hence the explosion of "Gym for agents" projects (BrowserGym, AgentGym, `verifiers`, GEM, SWE-Gym, τ-bench, ...) each standardizing a `reset`/`step`/reward contract for a task domain.
- Reward design has to produce a *stream*, not a scalar: outcome reward at the end, optionally supplemented by process/step rewards that are either hand-written (like {doc}`trl/trl_grpo_agent`'s reward functions) or learned (PRMs) — with reward hacking risk rising in both cases as the interaction gets longer.
- Credit assignment — figuring out which of a hundred actions in a trajectory actually earned the final reward — is the field's hardest unsolved problem; GiGPO's step-level anchor-state groups and StarPO/RAGEN's turn-level objective (plus its "Echo Trap" cautionary tale) are two of the most concrete current answers.
- Framework choice increasingly comes down to one systems question: how decoupled is rollout generation from training? Tightly coupled is simpler; fully async (AReaL, slime, Trinity-RFT, Agent Lightning) is what long-horizon, tool-calling rollouts eventually demand.
