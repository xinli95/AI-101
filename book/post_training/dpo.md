# DPO: Preference Optimization Without a Reward Model or RL Loop

**Direct Preference Optimization (DPO)** trains a language model directly on human preference pairs — for each prompt, a *chosen* (preferred) response $y_w$ and a *rejected* response $y_l$ — without ever training a reward model and without ever running a reinforcement-learning loop. The paper's tagline captures the core insight: *"Your language model is secretly a reward model."*

This chapter answers three questions:

1. **Where does the DPO loss come from?** We derive it in three steps from the same KL-regularized RL objective that RLHF-with-PPO optimizes.
2. **Is DPO reinforcement learning, or is it a variant of SFT?** Short answer: its *objective* is derived from RL, but its *training loop* is mechanically supervised learning — and the honest label is "neither, exactly." Section 6 makes this precise.
3. **What does DPO trade away** relative to a full RLHF pipeline, and when does that matter?

Prerequisites: the SFT loss from [Supervised Fine-tuning](sft.md), and it helps to have seen the KL-penalized reward in [PPO From Scratch](ppo.md), but the derivation here is self-contained.

---

## 1. The Pipeline DPO Replaces

Classic RLHF (the InstructGPT recipe) turns preference data into a better policy in two stages:

**Stage A — fit a reward model.** Collect preference triples $(x, y_w, y_l)$ and assume preferences follow the **Bradley–Terry model**: the probability that humans prefer $y_w$ over $y_l$ is a logistic function of the difference in an underlying (unobserved) reward $r^*$:

$$
p^*(y_w \succ y_l \mid x)=\sigma\big(r^*(x,y_w)-r^*(x,y_l)\big),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

A parameterized reward model $r_\phi$ is fit by maximum likelihood on this model — i.e., binary classification of which response was preferred:

$$
\mathcal{L}_R(\phi)=-\mathbb{E}_{(x,y_w,y_l)\sim D}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big].
$$

**Stage B — RL against the reward model.** Optimize the policy to maximize reward while staying close to a reference policy $\pi_{\text{ref}}$ (usually the SFT checkpoint), with a KL penalty controlled by $\beta$:

$$
\max_{\pi_\theta}\;
\mathbb{E}_{x\sim D,\; y\sim \pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]
-\beta\,\mathbb{D}_{\text{KL}}\big[\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big].
$$

Stage B is where PPO, GAE, the value model, rollout generation, and everything in [PPO From Scratch](ppo.md) comes in. It is expensive (multiple models in memory, generation inside the training loop) and finicky to tune.

DPO's claim: for this *specific* objective, Stage A and Stage B can be collapsed into a single supervised loss on the preference data. The derivation takes three steps.

---

## 2. Step 1: The KL-Regularized Objective Has a Closed-Form Optimum

Fix any reward function $r$ and ask: which policy $\pi$ maximizes the Stage B objective? Rewrite it (dividing by $\beta$ and flipping max to min):

$$
\min_{\pi}\;\mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi(\cdot\mid x)}
\left[\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}-\frac{1}{\beta}r(x,y)\right].
$$

Now absorb the reward term into the log by defining a candidate distribution

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right),
\qquad
Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\left(\frac{1}{\beta}r(x,y)\right),
$$

where $Z(x)$ is the **partition function** — the normalizer that makes $\pi^*$ a valid probability distribution. Substituting, the objective becomes

$$
\min_{\pi}\;\mathbb{E}_{x}\left[
\mathbb{D}_{\text{KL}}\big[\pi(\cdot\mid x)\,\|\,\pi^*(\cdot\mid x)\big]
-\log Z(x)\right].
$$

$\log Z(x)$ does not depend on $\pi$, and a KL divergence is minimized (at zero) exactly when the two distributions are equal. So the optimal policy is $\pi^*$:

> **The optimum of KL-regularized reward maximization is the reference policy, reweighted by exponentiated reward.** Higher-reward responses get their probability multiplied up by $e^{r/\beta}$; $\beta$ controls how aggressive the reweighting is.

This is exact, but useless as-is: computing $Z(x)$ requires summing over all possible responses $y$ — astronomically intractable for text.

---

## 3. Step 2: Invert the Relationship — Express Reward in Terms of Policy

Take the log of the optimal-policy expression and solve for the reward:

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x).
$$

Read this as a change of variables: *any* reward function can be equivalently described by the policy it induces (up to the $Z$ term). The intractable $Z(x)$ is still sitting there — but notice it depends only on $x$, **not on $y$**. That is the pivot for the whole method.

---

## 4. Step 3: Plug Into Bradley–Terry — the Partition Function Cancels

The Bradley–Terry probability depends only on the **difference** of rewards for two responses to the *same prompt*:

$$
p^*(y_w \succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big).
$$

Substitute the reward-in-terms-of-policy expression for both responses. The $\beta\log Z(x)$ terms are identical and cancel:

$$
p^*(y_w \succ y_l\mid x)
=\sigma\!\left(
\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right).
$$

The preference probability is now written purely in terms of policies — no reward model, no partition function. So instead of fitting a reward model $r_\phi$ by maximum likelihood (Stage A) and then extracting its optimal policy with RL (Stage B), parameterize the policy $\pi_\theta$ directly and fit the *same* maximum-likelihood objective:

$$
\boxed{\;
\mathcal{L}_{\text{DPO}}(\theta)
=-\mathbb{E}_{(x,y_w,y_l)\sim D}
\left[\log\sigma\!\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right)\right].
\;}
$$

That is the entire algorithm. Each $\log\pi(y\mid x)$ is the **sequence log-probability** — the sum of per-token log-probs over the response, exactly the quantity SFT computes with teacher forcing. Training requires four forward passes' worth of log-probs per example (policy and reference, on chosen and rejected), one `log-sigmoid`, and a backward pass. No sampling, no reward model, no critic.

The quantity

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

is called the **implicit reward**: it is the reward function for which the current $\pi_\theta$ would be the optimal KL-regularized policy. This is the precise sense of "your language model is secretly a reward model" — and it is what TRL logs as `rewards/chosen` and `rewards/rejected` (see [DPO Trainer Deep Dive](trl/trl_dpo_trainer.md)).

---

## 5. What the Gradient Does

Differentiating the DPO loss gives (writing $\hat r_w=\hat r_\theta(x,y_w)$, $\hat r_l=\hat r_\theta(x,y_l)$):

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}
=-\beta\,\mathbb{E}_{(x,y_w,y_l)\sim D}
\Big[
\underbrace{\sigma\big(\hat r_l-\hat r_w\big)}_{\text{example weight}}
\big(
\underbrace{\nabla_\theta\log\pi_\theta(y_w\mid x)}_{\text{push }y_w\text{ up}}
-\underbrace{\nabla_\theta\log\pi_\theta(y_l\mid x)}_{\text{push }y_l\text{ down}}
\big)
\Big].
$$

Three observations:

- **It is contrastive.** Every update simultaneously increases the log-probability of the chosen response and decreases the log-probability of the rejected one. SFT only ever does the first half.
- **Examples are weighted by how wrong the model currently is.** The weight $\sigma(\hat r_l-\hat r_w)$ is large when the implicit reward ranks the rejected response *above* the chosen one, and near zero once the pair is already ordered correctly with a comfortable margin. Hard pairs dominate the gradient; solved pairs fade out. (This adaptive weight is exactly what a naive "maximize $\log\pi(y_w)-\log\pi(y_l)$" loss lacks, and without it that naive loss degenerates.)
- **The reference model sets the anchor.** Because the implicit rewards are log-ratios against $\pi_{\text{ref}}$, the loss cares about how far the policy has *moved from the reference* on each response, not about raw probabilities — this is the KL regularization of Stage B, surviving in supervised form.

---

## 6. So Is DPO RL? Is It SFT? (Placing It Precisely)

A natural reaction to the loss above: "this is just supervised fine-tuning with a fancier loss — where did the RL go?" That intuition is half right, and it is worth being exact about which half.

### 6.1 The training loop is mechanically supervised learning

Run down what happens per training step:

| | SFT | DPO | RLHF with PPO |
| --- | --- | --- | --- |
| Data per example | $(x, y)$ | $(x, y_w, y_l)$ | $x$ only (responses are generated) |
| Responses come from | fixed dataset | fixed dataset | **sampled from current policy** |
| Generation inside the loop | no | no | yes (rollouts every step) |
| Models needed | policy | policy + frozen reference | policy + value + reward + reference |
| Loss | NLL on $y$ | log-sigmoid of implicit-reward margin | clipped surrogate + value loss |
| Pushes probability down on specific responses | no | yes ($y_l$) | yes (negative advantages) |
| Learning signal | "reproduce this" | "prefer this over that" | scalar reward on model's own behavior |

In the SFT/DPO columns, the training loop is identical in shape: iterate over a fixed dataset, teacher-force full sequences through the model, compute a differentiable loss on given tokens, backprop. In TRL, `DPOTrainer` is literally a subclass of the standard supervised `Trainer` — no generation loop exists in it. In this mechanical sense, "DPO is an SFT variant with a contrastive, reference-anchored loss" is a fair description.

### 6.2 But the objective is RL-derived — and that is not just trivia

DPO is not an arbitrary supervised loss that happens to work. Sections 2–4 show it is the *exact* maximum-likelihood solution to the same KL-regularized reward-maximization problem RLHF solves: under the Bradley–Terry assumption, with infinite data and a policy class expressive enough to realize the optimum, DPO's optimum and RLHF's optimum are the same policy. The derivation is also what supplies the two ingredients plain SFT lacks — the reference-ratio anchoring and the $\sigma(\hat r_l-\hat r_w)$ weighting — neither of which you would likely guess without going through the RL objective.

### 6.3 What genuinely separates it from RL: no interaction

The defining feature of RL is missing: **DPO never trains on its own samples.** The Stage B objective takes an expectation over $y\sim\pi_\theta$ — the policy being optimized. DPO's loss replaces that with an expectation over a *fixed offline dataset*, typically generated by some other model (or an earlier checkpoint) and labeled once. There is no exploration, no feedback on what the current policy actually generates. In RL vocabulary, DPO is a purely **offline** method, and this has real consequences:

- **Distribution shift.** The preference pairs may look nothing like what $\pi_\theta$ generates mid-training; the loss keeps grading the model on responses it would never produce, while its actual failure modes go unexamined.
- **The margin is all that matters — so both probabilities can fall.** The loss only constrains the *gap* $\log\pi(y_w)-\log\pi(y_l)$ (relative to the reference gap). A well-documented dynamic is that $\log\pi_\theta(y_w)$ and $\log\pi_\theta(y_l)$ **both decrease** during training, with the rejected one falling faster. Probability mass leaks to sequences outside both responses — where the offline loss exerts no control at all. (This is why TRL logs `logps/chosen` and why watching it matters.)
- **No credit assignment within a sequence.** The loss sees one number per response (its total log-prob). PPO's per-token advantages can localize *which part* of a response was good; DPO cannot.

A useful summary: **DPO keeps RL's objective but SFT's data flow.** It sits between the two columns of the table, and the standard label for this family — *offline preference optimization* — is more informative than forcing it into either bucket.

The same lens sorts the whole post-training landscape by one question — *does the model train on its own generations?* Offline/off-policy: SFT, off-policy distillation, DPO. Online/on-policy: RLHF-PPO, [GRPO/RLVR](grpo.md), on-policy distillation ([Knowledge Distillation](distillation.md)). Hybrids exist for DPO too — *iterated DPO* regenerates pairs from the current policy and relabels them every round, buying back some on-policy signal while keeping the simple loss.

---

## 7. Practical Notes

### 7.1 The role of $\beta$

$\beta$ is the KL-penalty strength inherited from Stage B (typical range $0.01$–$0.5$; $0.1$ is the common default). Small $\beta$ lets the policy stray far from the reference (aggressive preference-fitting, higher risk of degeneration); large $\beta$ pins it close (safer, weaker effect). Note it enters the loss *inside* the sigmoid, scaling the implicit-reward margin — it is not a separately-added penalty term.

### 7.2 Where the pairs come from matters more than the loss

DPO inherits everything wrong with its dataset: label noise, position/length artifacts, and staleness relative to the current policy. The best-known artifact is **length bias** — human raters mildly prefer longer answers, so DPO-trained models drift verbose. Some mitigations live in the loss (length-normalized variants), most live in the data.

### 7.3 The variant zoo, in one paragraph

DPO's weaknesses each spawned a variant: **IPO** replaces the log-sigmoid with a squared loss to resist overfitting on saturated pairs; **SimPO** drops the reference model and length-normalizes the log-probs; **KTO** works from *unpaired* good/bad labels instead of pairs; **ORPO** folds a preference odds-ratio penalty directly into SFT, removing the reference model and the separate stage. TRL implements most of these as `loss_type` options in the same trainer — see the [DPO Trainer Deep Dive](trl/trl_dpo_trainer.md).

### 7.4 When to reach for DPO vs. online RL

DPO wins on cost and simplicity: two models instead of four, no generation in the loop, a loss you can debug like any classifier. Frontier-scale post-training pipelines, however, have largely converged on online RL (PPO/GRPO-family) for the final alignment stages, precisely because of the offline limitations in §6.3 — with DPO often kept as a cheap early stage or applied where preference data is abundant and static. For a fixed preference dataset and a modest compute budget, DPO remains the strongest default.

---

## 8. Summary

- DPO collapses RLHF's two stages (reward modeling + RL) into one supervised loss, by (1) writing down the closed-form optimum of KL-regularized reward maximization, (2) inverting it to express reward as a policy log-ratio, and (3) substituting into Bradley–Terry, where the intractable partition function cancels.
- The loss trains the policy as an implicit reward model: $\hat r_\theta=\beta\log(\pi_\theta/\pi_{\text{ref}})$, fit by logistic regression on preference pairs.
- The gradient is contrastive (up on chosen, down on rejected) and adaptively weighted (hard pairs dominate) — the two ingredients plain SFT lacks.
- Classification: **RL objective, SFT mechanics** — an *offline* preference-optimization method. It never trains on its own generations, which is both its efficiency advantage and the root of its failure modes (distribution shift, falling chosen log-probs, no per-token credit).
