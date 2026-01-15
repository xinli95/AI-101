(grpo-family-tutorial)=

# GRPO and Its Improvements: Vanilla GRPO -> DAPO, Dr. GRPO, and GSPO

This tutorial is a self-contained guide to **Group Relative Policy Optimization (GRPO)** and several influential follow-up methods that address practical and theoretical issues in vanilla GRPO: **DAPO**, **Dr. GRPO**, and **GSPO**. We also discuss a critical but often under-emphasized factor: **base model capability and selection**.

> Context: GRPO is widely used in **RLVR** (reinforcement learning with verifiable rewards) for reasoning-style training (e.g., math/code), where the reward comes from a verifier rather than a learned reward model. {cite}`wolfe_grpo_tricks`

---

## 1. Setup: RL for LLM completions

We consider prompts/questions $q$ drawn from a dataset $D$. For each prompt, we sample **a group of $G$ completions** (responses) from the current policy (an LLM) $\pi_\theta$:

- prompt: $q \sim D$
- group of completions: $\{o_i\}_{i=1}^G \sim \pi_\theta(\cdot \mid q)$
- each completion has tokens $o_{i,1:|o_i|}$

A reward is computed for each completion:

$$
R_i = R(q, o_i)
$$

For RLVR, $R_i$ is often a correctness indicator (0/1) from a verifier.

GRPO's defining idea is: **compute advantages within the group**, avoiding a learned critic/value model (unlike PPO).

---

## 2. Vanilla GRPO: what it is

### 2.1 GRPO as "PPO without a critic, using group normalization"

In PPO, you estimate advantages $A_t$ with a critic and often GAE. In GRPO, you instead compute relative advantages from the group's rewards.

A common (and highly used) GRPO recipe:

1. For each prompt $q$, sample $G$ completions.
2. Compute rewards $R_i$.
3. Normalize rewards in the group to produce an advantage-like signal $\hat A_{i,t}$ for tokens in completion $i$.
4. Apply a PPO-like clipped objective using an "old" policy $\pi_{\theta_{\text{old}}}$.

DAPO's paper writes a GRPO-style clipped objective (plus KL penalty) with **token-level importance ratio**:

$$
J_{\text{GRPO}}(\theta)
=
\mathbb{E}_{(q,\cdot)\sim D, \{o_i\}\sim \pi_{\theta_{\text{old}}}}
\left[
\frac{1}{G}\sum_{i=1}^G
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\Big(
 r_{i,t}(\theta)\hat A_{i,t},
\ \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat A_{i,t}
\Big)
\ -\ \beta D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})
\right]
$$

where the **token-level ratio** is:

$$
r_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}
$$

> Practical note: many implementations treat $\hat A_{i,t}$ as constant across tokens in a completion (derived from $R_i$), then possibly apply per-token weighting/normalization.

### 2.2 Group-relative advantage (typical form)

A widely used GRPO advantage is computed from group rewards:

$$
\hat A_i
=
\frac{R_i - \mu_R}{\sigma_R + \delta}
\quad\text{where}\quad
\mu_R=\frac{1}{G}\sum_{j=1}^G R_j,\
\sigma_R=\sqrt{\frac{1}{G}\sum_{j=1}^G (R_j-\mu_R)^2}
$$

and then $\hat A_{i,t}=\hat A_i$ for all tokens $t$ in completion $i$.

Intuition:

- If a completion scores above the group mean, increase its probability.
- If it scores below the mean, decrease its probability.
- Normalization is meant to stabilize scaling without a critic.

---

## 3. Why vanilla GRPO is tricky: issues and biases

Vanilla GRPO can be deceptively simple: several failure modes show up especially at scale. A recent overview summarizes common training pathologies observed with vanilla GRPO, including **entropy collapse**, **reward noise**, and **training instability/divergence**.

Below are the main issues emphasized by the DAPO / Dr. GRPO / GSPO line of work.

### 3.1 Entropy collapse and poor exploration

Empirically, vanilla GRPO training can suffer from:

- **entropy collapse**: next-token distribution becomes overly peaked,
- group samples become too similar,
- within-group normalization becomes less informative,
- exploration diminishes and learning stalls.

This is one key motivation for DAPO's interventions.

### 3.2 Length / token aggregation bias (verbosity bias)

Vanilla GRPO often aggregates token losses in a way that interacts badly with variable sequence lengths. Dr. GRPO's authors highlight an **optimization bias** in GRPO that can **artificially increase response length**, especially for incorrect outputs, wasting tokens and harming efficiency.

At a high level, if your loss/advantage is normalized per-token (or coupled to length), the optimizer can find "cheap" ways to change objective value by manipulating length rather than improving correctness.

### 3.3 Token-level importance sampling + clipping instability

GSPO argues that GRPO's instability stems from a **fundamental misapplication/invalidation of importance sampling weights** when using **token-level ratios** for long sequences, leading to high variance that grows with completion length and interacts badly with clipping, sometimes causing collapse.

This motivates GSPO's move from **token-level** to **sequence-level** likelihood ratios.

---

## 4. DAPO: practical "GRPO++" improvements for scaling RL

**DAPO** ("Decoupled Clip and Dynamic sAmpling Policy Optimization") is presented as an open-source system and algorithmic package to make GRPO-style RL training stable and effective at scale, particularly for long-CoT reasoning.

### 4.1 What DAPO claims is wrong with vanilla GRPO (empirically)

The DAPO discussion report that vanilla GRPO often shows:

- **Entropy collapse**
- **Noisy rewards** (reward does not steadily improve)
- **Training instability / divergence**, and unstable response length dynamics

### 4.2 What DAPO changes (high-level)

DAPO proposes "four key changes" to vanilla GRPO to address these issues. While implementations differ, the recurring themes across DAPO-style recipes are:

1. **Decouple clipping behavior** to avoid overly suppressing exploration (related to entropy collapse under clipping).
2. **Dynamic sampling / filtering** strategies so training focuses on informative rollouts and avoids pathological long outputs.
3. **Token-level loss design choices** that improve stability when optimizing long sequences.
4. **Overlong handling** (filtering or soft penalties) to prevent the model from exploiting length. (Many secondary writeups list "overlong filtering / punishment" among DAPO's practical tricks.)

---

## 5. Dr. GRPO: correcting token aggregation bias (length bias) and simplifying advantage normalization

**Dr. GRPO** is introduced in _Understanding R1-Zero-Like Training: A Critical Perspective_ as a method to remove an identified optimization bias in GRPO.

### 5.1 The core diagnosis

The paper identifies an optimization bias in GRPO that tends to **inflate response length**, especially for incorrect outputs, reducing token-efficiency and potentially harming learning dynamics.

### 5.2 The core fix

1. **Normalize sequence loss by a fixed constant** (e.g., `MAX_TOKENS`) rather than by the actual sequence length $|o_i|$.
   - This aims to remove incentives tied to length normalization.
2. **Remove the reward standard deviation from the advantage denominator**, i.e., use mean-centering without dividing by $\sigma$.
   - This changes scaling/robustness properties of the group-relative signal.

The same section reports that these changes mitigate the bias and can yield better per-token performance with fewer average tokens.

### 5.3 Practical implication

If you are seeing:

- outputs getting longer without accuracy gains,
- unstable training due to reward scaling,

Dr. GRPO-style changes are a minimal intervention to test first.

---

## 6. GSPO: move from token-level ratios to sequence-level ratios for stability

**GSPO (Group Sequence Policy Optimization)** proposes a more structural change: define importance ratios at the **sequence level**, not token level, and apply clipping/rewarding at the sequence level.

### 6.1 The GSPO diagnosis of GRPO

GSPO claims GRPO instability comes from invalid importance sampling usage with token-level ratios; variance accumulates with response length and is amplified by clipping, potentially leading to collapse.

### 6.2 The GSPO key idea

Instead of:

$$
r_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}
$$

GSPO uses a **sequence likelihood ratio** (conceptually):

$$
r_i^{\text{seq}}(\theta)=
\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}
$$

and then does **sequence-level clipping and optimization**.

GSPO reports improved stability/efficiency, and highlights effectiveness for large-scale RL training including MoE models, contributing to improvements in Qwen3.

### 6.3 What to take from GSPO if you are implementing RL infrastructure

- If your responses are long (reasoning traces), token-level ratios can become numerically/variance problematic.
- Sequence-level objectives may simplify some stability engineering and reduce collapse risk.
- GSPO is a more "algorithmic" change than DAPO's "bag of tricks," and targets a specific theoretical failure mode.

---

## 7. Base model capability and selection: the "hidden lever"

A major message from the Dr. GRPO / "Understanding R1-Zero-like training" line of work is: **base model choice can dominate perceived RL gains**.

### 7.1 The absolute requirement: the base must sometimes succeed under sampling

If a base policy cannot sample any correct trajectory among multiple rollouts, RL cannot improve it because there is no positive reward signal to reinforce.

Practical rule:

- Before RL, measure **Pass@G** (e.g., Pass@8): for each prompt, sample $G$ completions and compute how often at least one is correct.
- If Pass@G is near zero, RLVR will be extremely difficult.

### 7.2 Prompt template / formatting matters a lot for base models

Base models may be sensitive to prompt templates (and some "base" models behave SFT-like depending on pretraining data). Template choice can inflate or deflate "pre-RL" baseline performance, changing the apparent benefit of RL.

Implication:

- When comparing RL methods, control for prompt formatting and base model instruction-following properties, otherwise the comparison can be misleading.

### 7.3 "Aha moments" may not be purely created by RL

The same discussion reports evidence that some base models already exhibit self-reflection patterns (e.g., "wait", "aha") even before RL, suggesting RL may amplify behaviors that already exist rather than invent them from scratch.

---

## 8. A Unified Mathematical View: PPO, GRPO, Dr. GRPO, GSPO

All methods discussed so far can be expressed under a **single policy-gradient-with-importance-weighting template**. The differences lie in **(1) how advantages are computed** and **(2) how importance ratios are defined and constrained**.

### 8.1 Generic clipped policy objective

A generic objective covering PPO / GRPO-style methods is:

$$
\mathcal{L}(\theta)
=
\mathbb{E}\Big[
\min\big(
r(\theta)\,\hat A,
\ \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\,\hat A
\big)
\Big]
-
\beta\,D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})
$$

The methods differ in how $r(\theta)$ and $\hat A$ are defined:

| Method       | Advantage $\hat A$             | Importance Ratio $r(\theta)$ | Key Structural Choice                |
| ------------ | ------------------------------ | ---------------------------- | ------------------------------------ |
| PPO          | GAE from critic                | Token-level                  | Learned value function               |
| Vanilla GRPO | Group-relative reward          | Token-level                  | No critic, group normalization       |
| Dr. GRPO     | Group-relative (mean-centered) | Token-level                  | Length-bias correction               |
| GSPO         | Group-relative reward          | **Sequence-level**           | Variance control via sequence ratios |
| DAPO         | Group-relative reward          | Token-level                  | Stability & exploration engineering  |

This table highlights that **GRPO-family methods do not remove PPO’s core structure**—they replace the _critic-based advantage_ with _group-relative signals_, and then fix the resulting pathologies.

---

## 9. Pseudocode-Level Training Loops

Below are **conceptual pseudocode sketches** (not framework-specific) to clarify what tensors are stored and how updates differ.

---

### 9.1 Vanilla GRPO (baseline)

**Data collection**

1. Sample prompt $q$
2. Sample $G$ completions $\{o_i\}_{i=1}^G$ from $\pi_{\theta_{\text{old}}}$
3. Store:
   - token ids
   - logprobs $\log \pi_{\theta_{\text{old}}}(o_{i,t})$
   - completion lengths
4. Compute rewards $R_i$

**Advantage computation**

- Compute group mean (and optionally std):
  $$
  \hat A_i = \frac{R_i - \mu_R}{\sigma_R + \delta}
  $$
- Broadcast to tokens: $\hat A_{i,t} = \hat A_i$

**Optimization**

- Recompute logprobs under $\pi_\theta$
- Compute token-level ratios
- Apply clipped objective + KL penalty

---

### 9.2 Dr. GRPO (length-bias correction)

Same as vanilla GRPO, except:

- **Sequence loss normalization**
  - Normalize loss by a fixed constant (e.g. `MAX_TOKENS`) instead of $|o_i|$
- **Advantage normalization**
  - Use mean-centering only:
    $$
    \hat A_i = R_i - \mu_R
    $$

This removes incentives to exploit length and stabilizes scaling.

---

### 9.3 GSPO (sequence-level ratios)

**Data collection**

- Same as GRPO, but store **full sequence log-likelihood**:
  $$
  \log \pi_{\theta_{\text{old}}}(o_i \mid q)
  $$

**Advantage computation**

- Same group-relative advantage $\hat A_i$

**Optimization**

- Compute **sequence-level ratio**:
  $$
  r_i^{\text{seq}}(\theta)
  =
  \exp\left(
  \log \pi_\theta(o_i \mid q) - \log \pi_{\theta_{\text{old}}}(o_i \mid q)
  \right)
  $$
- Apply **sequence-level clipping**:
  $$
  \min\big(
  r_i^{\text{seq}} \hat A_i,\;
  \text{clip}(r_i^{\text{seq}}, 1-\epsilon, 1+\epsilon)\hat A_i
  \big)
  $$

Tokens inherit the same scalar weight; variance does **not** grow with sequence length.

---

### 9.4 DAPO (engineering-heavy variant)

DAPO follows GRPO structurally but adds:

- dynamic sampling / filtering
- entropy-preserving modifications
- overlong suppression
- careful clipping / KL scheduling

DAPO is best viewed as a **system-level stabilization recipe**, not a single equation change.

---

## 10. What to Monitor During Training (Critical Checklist)

GRPO-family methods are **fragile without proper diagnostics**. The following metrics should be monitored continuously.

---

### 10.1 Reward and signal quality

- Mean reward per batch
- Reward variance within groups
- Fraction of groups with at least one positive reward
- Pass@G (e.g. Pass@8, Pass@16)

> If Pass@G ≈ 0 early, RL has no signal to amplify.

---

### 10.2 Policy change & stability

- $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})$
- $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\theta_{\text{old}}})$
- Ratio statistics:
  - mean / max of $r$
  - fraction of clipped samples

Sudden KL spikes often precede collapse.

---

### 10.3 Entropy & exploration

- Token entropy (mean, trend)
- Top-k / nucleus mass concentration
- Fraction of identical completions within a group

Entropy collapse → group-relative advantages become meaningless.

---

### 10.4 Length dynamics (very important)

- Mean / median completion length
- Length conditioned on correctness
- Length over training steps

Red flags:

- Incorrect answers getting longer
- Length growing without reward improvement

Dr. GRPO-style fixes target exactly this pathology.

---

### 10.5 Numerical & variance signals

- Gradient norm
- Advantage magnitude distribution
- Ratio variance (token vs sequence)

GSPO is especially useful when ratio variance scales with length.

---

## 11. Final Takeaways

- **GRPO** replaces the critic with _group-relative advantages_ but introduces new failure modes.
- **DAPO** stabilizes training via practical, system-level interventions.
- **Dr. GRPO** corrects a subtle but harmful **length-based optimization bias**.
- **GSPO** fixes a deeper variance issue by redefining importance sampling at the **sequence level**.
- **Base model capability** and **prompt templates** often dominate RL gains—RL amplifies existing ability, it rarely creates it from nothing.

Understanding these methods as **variants within one unified policy optimization framework** makes it much easier to reason about when each fix is necessary—and when it is not.
