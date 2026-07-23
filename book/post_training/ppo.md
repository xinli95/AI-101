# PPO From Scratch: From Expected Return to the Clipped Objective

This tutorial builds **Proximal Policy Optimization (PPO)** step by step:

1. Start from the objective: maximize **expected total rewards** as a **weighted average over rollouts**.
2. Derive the policy gradient using the **log-derivative trick**:
   $\nabla_\theta J(\theta)=\mathbb{E}[\nabla_\theta \log p_\theta(\tau)\,R(\tau)].$
3. Expand $\log p_\theta(\tau)$ into a sum of $\log \pi_\theta(a_t\mid s_t)$.
4. Switch from **trajectory-level** expectations to **per-timestep** expectations — the notation used in PPO papers and the form that matches how the algorithm is actually implemented.
5. Derive PPO’s **surrogate objective** using the **probability ratio**
   $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},$
   understand why it is called a *surrogate* and why it is only valid **locally**,
   and then arrive at the **clipped** PPO objective.

---

## 1. Objective as a Weighted Average Over Rollouts

Consider a stochastic policy $\pi_\theta(a\mid s)$. A rollout (trajectory) is

$$
\tau = (s_0,a_0,s_1,a_1,\dots,s_T).
$$

Define the (discounted) return of a rollout:

$$
R(\tau) = \sum_{t=0}^{T} \gamma^t\, r(s_t,a_t).
$$

The probability of a rollout under policy $\pi_\theta$ in an MDP with dynamics $P$ is

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T}\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t).
$$

The RL objective is expected return under rollouts induced by the policy:

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[R(\tau)]
=\sum_{\tau} p_\theta(\tau)\,R(\tau).
$$

That last expression is exactly a **weighted average over all rollouts**: each rollout’s reward $R(\tau)$ weighted by its probability $p_\theta(\tau)$.

---

## 2. Derivation: The Policy Gradient via the Log-Derivative Trick

Start from the definition

$$
J(\theta)=\sum_{\tau} p_\theta(\tau)\,R(\tau).
$$

Differentiate with respect to $\theta$:

$$
\nabla_\theta J(\theta)
= \sum_{\tau} \nabla_\theta p_\theta(\tau)\,R(\tau).
$$

Use the **log-derivative trick**:

$$
\nabla_\theta p_\theta(\tau)
= p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau).
$$

Plug it in:

$$
\nabla_\theta J(\theta)
= \sum_{\tau} p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,R(\tau)
= \mathbb{E}_{\tau\sim p_\theta}\left[\nabla_\theta \log p_\theta(\tau)\,R(\tau)\right].
$$

Now approximate the expectation with a Monte Carlo sample average. If you sample $N$ rollouts $\{\tau_i\}_{i=1}^N$ from $\pi_\theta$:

$$
\nabla_\theta J(\theta)
\approx \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau_i)\,R(\tau_i).
$$

This is the classic **REINFORCE**-style gradient at the **trajectory** level.

---

## 3. Expand $\log p_\theta(\tau)$ into a Sum of $\log \pi_\theta$ Terms

Recall

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{T}\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t).
$$

Take logs:

$$
\log p_\theta(\tau)
=\log\rho_0(s_0)+\sum_{t=0}^{T}\log\pi_\theta(a_t\mid s_t)+\sum_{t=0}^{T}\log P(s_{t+1}\mid s_t,a_t).
$$

The environment terms $\rho_0$ and $P$ do **not** depend on $\theta$, so their gradients vanish:

$$
\nabla_\theta \log p_\theta(\tau)=\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t\mid s_t).
$$

Substitute back into the policy gradient:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[\left(\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t\mid s_t)\right)R(\tau)\right].
$$

### 3.1 Advantage Form (Better Credit Assignment + Lower Variance)

In practice, we replace the rollout-level return $R(\tau)$ with a **time-dependent** signal such as an **advantage**:

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\right].
$$

Intuition: $A^\pi(s_t,a_t)$ measures how much better (or worse) action $a_t$ is compared with the policy’s baseline behavior at state $s_t$.

### 3.2 What Exactly Is “the Policy Gradient” a Gradient *of*?

A common point of confusion: is the policy gradient the derivative of the **expected return**, or the derivative of the **policy network**? The answer is: conceptually the former, computationally the latter — and the whole point of the derivation above is to convert one into the other.

- **Conceptually**, the policy gradient is $\nabla_\theta J(\theta)$: the derivative of the expected return with respect to the policy network’s parameters $\theta$. That is the quantity we actually want, because gradient ascent on $J(\theta)$ is what improves the policy.
- **The obstacle** is that $J(\theta)$ runs through the environment — the dynamics $P(s_{t+1}\mid s_t,a_t)$ are an unknown black box, so we cannot write $J(\theta)$ as a differentiable computation graph and call backprop on it.
- **The log-derivative trick is the workaround.** It re-expresses $\nabla_\theta J(\theta)$ so that the only thing being differentiated is $\log\pi_\theta(a_t\mid s_t)$ — an output of the policy network itself, which autodiff handles trivially. The environment-dependent quantities ($R(\tau)$ or $A_t$) survive only as **fixed scalar weights** on those gradients; no gradient ever flows through them.

So when you read $\mathbb{E}\left[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t\right]$, read it as: *the derivative of expected return, rewritten so it can be computed purely from derivatives of the network’s own log-probabilities, weighted by how good each sampled action turned out to be.* If $A_t>0$, the update pushes the network to make $a_t$ more likely in $s_t$; if $A_t<0$, less likely.

---

## 4. From Log-Probability to PPO’s Ratio $r_t(\theta)$

PPO updates a new policy $\pi_\theta$ using trajectories generated by an **older** policy $\pi_{\theta_{\text{old}}}$. That means we are optimizing $\pi_\theta$ with off-policy data. The standard fix is **importance sampling**: reweight each sample by how likely the new policy would have produced it compared to the old one.

Define the per-step probability ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}.
$$

Interpretation:

- if $r_t(\theta)>1$, the new policy assigns higher probability to the sampled action than the old policy did, so its contribution is upweighted,
- if $r_t(\theta)<1$, the new policy is less likely to take that action, so its contribution is downweighted.

Connect it directly to **log-probabilities**:

$$
r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)\right).
$$

So the chain is literally:

- compute current log-prob $\log\pi_\theta(a_t\mid s_t)$,
- subtract stored old log-prob $\log\pi_{\theta_{\text{old}}}(a_t\mid s_t)$,
- exponentiate to get $r_t(\theta)$.

---

## 5. The Unclipped Surrogate Objective (Importance-Weighted Policy Gradient)

We want to improve $\pi_\theta$ while using samples from $\pi_{\theta_{\text{old}}}$.

### 5.1 A Change of Notation First: From Trajectories to Timesteps

Before deriving anything new, we need to switch notation — otherwise a sum is going to seem to vanish without explanation.

So far every expectation has been over **whole trajectories**, with a sum over timesteps inside:

$$
\mathbb{E}_{\tau\sim p_\theta}\left[\sum_{t=0}^{T} f(s_t,a_t)\right].
$$

From here on, PPO papers (and this tutorial) instead write expectations over **individual timesteps**:

$$
\mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}\left[f(s_t,a_t)\right],
$$

where $t\sim \pi_{\theta_{\text{old}}}$ means: *generate trajectories with $\pi_{\theta_{\text{old}}}$, then sample a single timestep — i.e., a state–action pair $(s_t,a_t)$ — uniformly from all the collected timesteps.*

These two views are equivalent up to a constant. By linearity of expectation, summing over the $T{+}1$ steps of a trajectory and then averaging over trajectories is the same as averaging over individual steps, scaled by the trajectory length:

$$
\mathbb{E}_{\tau}\left[\sum_{t=0}^{T} f(s_t,a_t)\right]
=(T{+}1)\;\mathbb{E}_{t}\left[f(s_t,a_t)\right].
$$

The constant $(T{+}1)$ scales the gradient uniformly, so it changes only the step size, not the direction — it gets absorbed into the learning rate. That is why the $\sum_{t=0}^{T}$ disappears from the formulas: it has been folded into the per-timestep expectation, not dropped.

This notation is also exactly how PPO is implemented in code:

1. Roll out $\pi_{\theta_{\text{old}}}$ and collect many trajectories.
2. **Flatten** them into one buffer of independent samples $(s_t, a_t, A_t, \log\pi_{\theta_{\text{old}}}(a_t\mid s_t))$ — the trajectory structure is no longer needed once the advantages are computed.
3. Sample random mini-batches of timesteps from the buffer and average the per-sample loss.

Step 3 is literally a Monte Carlo estimate of $\mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}[\cdot]$.

### 5.2 Importance Sampling: Correcting for Old-Policy Data

In per-timestep notation, the policy gradient from Section 3 reads

$$
\nabla_\theta J(\theta)
\propto \mathbb{E}_{t\sim \pi_\theta}\left[\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\right].
$$

Note the subscript: the samples must come from the **current** policy $\pi_\theta$. But our buffer was filled by $\pi_{\theta_{\text{old}}}$. Importance sampling fixes the mismatch: reweight each sample by the ratio of how likely the current policy is to take that action versus the old one:

$$
\nabla_\theta J(\theta)
\approx \mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}\left[
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\,\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)
\right].
$$

This is the theoretical justification for reusing old-policy data. (It is an $\approx$, not an $=$: the ratio corrects for the mismatch in **action** probabilities, but the **states** in the buffer were still visited by $\pi_{\theta_{\text{old}}}$, and no reweighting fixes that. This residual error is small only while $\pi_\theta$ stays close to $\pi_{\theta_{\text{old}}}$ — remember this, it returns in Section 5.3 and motivates clipping.)

Now define the **surrogate objective**

$$
L^{\text{PG}}(\theta)=\mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}\left[r_t(\theta)\,A_t\right],
$$

where $A_t$ is an advantage estimate and $r_t(\theta)$ is the ratio from Section 4. The connection to $\nabla_\theta J(\theta)$ comes from the log-derivative trick,

$$
\nabla_\theta \pi_\theta(a_t\mid s_t)=\pi_\theta(a_t\mid s_t)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t),
$$

so differentiating $L^{\text{PG}}$ gives

$$
\nabla_\theta L^{\text{PG}}(\theta)
= \mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}\left[
r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A_t
\right],
$$

which matches the importance-sampled form of $\nabla_\theta J(\theta)$ above.

### 5.3 Why Is $L^{\text{PG}}$ Called a “Surrogate”?

Because $L^{\text{PG}}(\theta)$ and the true objective $J(\theta)$ are **two different functions** that merely share a gradient at one point. It is *not* because $A_t$ is an estimate. Two things make it a stand-in rather than the real thing:

**1. Only the gradients match — the values do not.** $J(\theta)$ is the actual expected return; its value means something. $L^{\text{PG}}(\theta)$ was reverse-engineered purely so that autodiff produces the right gradient; its value means nothing. To see this concretely: before any update, $\theta=\theta_{\text{old}}$, so every ratio is $r_t=1$ and $L^{\text{PG}}=\mathbb{E}[A_t]$ — which is roughly $0$ if advantages are normalized — regardless of whether the policy is terrible or excellent. We never care what number $L^{\text{PG}}$ evaluates to; we only feed it to the optimizer to get $\nabla_\theta L^{\text{PG}} \approx \nabla_\theta J$ at the current parameters.

**2. The match is only local.** The gradients agree at $\theta=\theta_{\text{old}}$, and approximately agree nearby. But as $\theta$ moves away, the approximation degrades — recall from Section 5.2 that the importance ratio never corrects the state distribution, which still belongs to $\pi_{\theta_{\text{old}}}$. So $L^{\text{PG}}$ is a trustworthy proxy for $J$ only in a neighborhood of $\theta_{\text{old}}$. Outside that neighborhood, pushing $L^{\text{PG}}$ up can push the true return $J$ *down*.

This second point is the punchline that sets up the rest of the tutorial: an optimizer given $L^{\text{PG}}$ will happily chase it far beyond the region where it means anything. Something must confine the update to the neighborhood where the surrogate is valid.

---

## 6. Why PPO Modifies the Surrogate: Too-Big Updates

Section 5.3 ended with the problem: $L^{\text{PG}}$ is only trustworthy near $\theta_{\text{old}}$, but nothing in it enforces staying near. If you maximize $\mathbb{E}[r_t(\theta)A_t]$ directly — especially over multiple epochs on the same buffer — the ratio $r_t(\theta)$ can drift far from $1$. This causes:

- unstable training,
- destructive updates from noisy advantage estimates,
- collapsed policies (especially in high-dimensional action spaces).

TRPO addresses this by explicitly constraining the KL divergence between new and old policies; PPO instead uses a simpler, effective heuristic that acts directly on the ratio: **clipping**.

---

## 7. PPO Clipped Objective

Define the clipped ratio:

$$
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon).
$$

PPO’s clipped objective is:

$$
L^{\text{CLIP}}(\theta)
=\mathbb{E}_{t\sim \pi_{\theta_{\text{old}}}}\Big[
\min\big(
r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,A_t
\big)
\Big].
$$

### 7.1 What the Clipping Does (Key Intuition)

There are two parts:

- **Clipping the ratio around 1.** The ratio $r_t(\theta)$ is centered at 1 because $\pi_\theta=\pi_{\theta_{\text{old}}}$ implies no change. Clipping to $[1-\epsilon,\,1+\epsilon]$ prevents $r_t(\theta)$ from drifting too far, limiting how much the new policy can change its action probabilities in a single update.
- **Taking the min.** We compare the unclipped term $r_t(\theta)A_t$ with the clipped term and keep the smaller one. This makes the objective conservative: if the policy tries to improve too aggressively, the clipped term takes over and stops extra gain.

Concretely:

- If $A_t>0$, the objective uses $\min(r_t, \operatorname{clip}(r_t))A_t$, so once $r_t(\theta)>1+\epsilon$, further increases no longer help.
- If $A_t<0$, the same min means once $r_t(\theta)<1-\epsilon$, further decreases no longer help.

---

## 8. One Clean Derivation Chain (From Expected Return → Log-Prob → PPO)

Here is the full chain in one place.

### Step 1: Rollout objective

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[R(\tau)].
$$

### Step 2: Differentiate using the log-derivative trick

$$
\nabla J(\theta)=\mathbb{E}_{\tau\sim p_\theta}[\nabla\log p_\theta(\tau)\,R(\tau)].
$$

### Step 3: Expand $\log p_\theta(\tau)$ (environment terms vanish)

$$
\nabla\log p_\theta(\tau)=\sum_t \nabla\log \pi_\theta(a_t\mid s_t).
$$

### Step 4: Replace returns with advantages (variance reduction + credit assignment)

$$
\nabla J(\theta)=\mathbb{E}\left[\sum_t \nabla\log \pi_\theta(a_t\mid s_t)\,A_t\right].
$$

### Step 5: Switch from trajectory sums to per-timestep expectations

Flatten trajectories into individual timesteps; the constant trajectory-length factor is absorbed into the learning rate:

$$
\nabla J(\theta)\propto\mathbb{E}_{t}\left[\nabla\log \pi_\theta(a_t\mid s_t)\,A_t\right].
$$

### Step 6: Correct for old-policy data with the importance ratio

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}
=\exp(\log\pi_\theta-\log\pi_{\text{old}}).
$$

### Step 7: Unclipped surrogate objective (gradient matches Step 6, but only near $\theta_{\text{old}}$)

$$
L^{\text{PG}}(\theta)=\mathbb{E}_{t\sim \pi_{\text{old}}}[r_t(\theta)A_t].
$$

### Step 8: PPO clipped objective (keep the update inside the trust region where the surrogate is valid)

$$
L^{\text{CLIP}}(\theta)
=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\;\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right].
$$

That is PPO’s core idea: keep the benefits of a policy-gradient update while preventing overly large policy changes.
