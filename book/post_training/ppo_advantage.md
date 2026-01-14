# From Return to Advantage: Full Derivation to GAE

This tutorial walks step by step through the **refinement of policy gradient estimators**, starting from the trajectory-level return used in REINFORCE and ending with **Generalized Advantage Estimation (GAE)**, as used in PPO.

The goal is to explain **why and how** we replace the full return with the advantage, and how GAE naturally emerges from this refinement.

---

## 1. REINFORCE: Policy Gradient with Trajectory Return

We start from the policy optimization objective:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]
$$

where a trajectory (rollout) is:

$$
\tau = (s_0, a_0, s_1, a_1, \dots, s_T)
$$

and the total discounted return is:

$$
R(\tau) = \sum_{t=0}^{T} \gamma^t r_t
$$

Using the log-derivative trick, the policy gradient is:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau}\left[
\nabla_\theta \log p_\theta(\tau) \; R(\tau)
\right]
$$

Expanding the trajectory probability:

$$
\log p_\theta(\tau)
=
\sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \text{(env terms)}
$$

Environment terms do not depend on $\theta$, so:

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

Thus:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau}\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\; R(\tau)
\right]
$$

**Problem:** every action is credited with the same full-trajectory return → very high variance and poor credit assignment.

---

## 2. Reward-to-Go: Replacing $R(\tau)$ with $G_t$

Define the **reward-to-go** from time step $t$:

$$
G_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l}
$$

Then the policy gradient can be rewritten as:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau}\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\; G_t
\right]
$$

### Why this is valid

Split the full return:

$$
R(\tau) = \underbrace{\sum_{i < t} \gamma^i r_i}_{\text{past}} + \gamma^t G_t
$$

The past rewards do **not depend on $a_t$**, and therefore vanish in expectation because:

$$
\mathbb{E}_{a_t \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)] = 0
$$

So only future rewards matter for action $a_t$.

---

## 3. Baseline Trick: Subtract Any $b(s_t)$ Without Bias

We now introduce a baseline:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\big(G_t - b(s_t)\big)
\right]
$$

### Proof of unbiasedness

Because:

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}
[\nabla_\theta \log \pi_\theta(a|s)]
=
\nabla_\theta \sum_a \pi_\theta(a|s)
=
\nabla_\theta 1
=
0
$$

we have:

$$
\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)\; b(s_t)] = 0
$$

Thus, subtracting **any function of the state** does not change the expected gradient.

**Purpose:** reduce variance.

---

## 4. Choosing the Baseline: Value Function → Advantage

Define:

$$
V^\pi(s_t) = \mathbb{E}[G_t \mid s_t]
$$

$$
Q^\pi(s_t,a_t) = \mathbb{E}[G_t \mid s_t,a_t]
$$

Then the **advantage function** is:

$$
A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)
$$

Now observe:

$$
\mathbb{E}[G_t - V^\pi(s_t) \mid s_t,a_t]
=
A^\pi(s_t,a_t)
$$

So the policy gradient becomes:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}\left[
\sum_t
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\; A^\pi(s_t,a_t)
\right]
$$

This is the **advantage policy gradient**.

---

## 5. Temporal-Difference Residual as Advantage Estimator

In practice, we approximate $V^\pi$ with $V_\phi$.

Define the TD error:

$$
\delta_t
=
r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

If $V_\phi = V^\pi$, then:

$$
\mathbb{E}[\delta_t \mid s_t,a_t]
=
Q^\pi(s_t,a_t) - V^\pi(s_t)
=
A^\pi(s_t,a_t)
$$

So the TD residual is an **unbiased advantage estimator** when the value function is correct.

Note that this is exactly the 1-step advantage estimator:
$$
\delta_t = \hat A_t^{(1)}.
$$

---

## 6. $k$-Step Advantage Estimators

Define the $k$-step bootstrapped return:

$$
G_t^{(k)}
=
\sum_{l=0}^{k-1} \gamma^l r_{t+l}
+
\gamma^k V_\phi(s_{t+k})
$$

The corresponding advantage estimator is:

$$
\hat A_t^{(k)} = G_t^{(k)} - V_\phi(s_t)
$$

### Expressing $k$-step advantage via TD residuals

Using telescoping sums:

$$
\hat A_t^{(k)}
=
\sum_{l=0}^{k-1} \gamma^l \delta_{t+l}
$$

This reveals a continuum of advantage estimators:

- $k=1$: pure TD
- $k \to \infty$: Monte Carlo return

---

## 7. Generalized Advantage Estimation (GAE)

GAE forms an **exponentially weighted average** of all $k$-step advantages:

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}
=
(1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} \hat A_t^{(k)}
$$

Substitute the TD-residual form and rearrange sums:

$$
\hat A_t^{\text{GAE}}
=
\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

This is the canonical GAE formula.

### The role of $\lambda$

$\lambda \in [0,1]$ controls the bias-variance tradeoff by tuning how much weight to place on longer-horizon returns:

- $\lambda=0$ recovers the 1-step estimator $\hat A_t^{(1)}=\delta_t$ (low variance, higher bias),
- $\lambda \to 1$ approaches Monte Carlo-style advantages (lower bias, higher variance),
- intermediate values interpolate smoothly between these extremes.

---

## 8. Backward Recursive Form (Used in PPO)

From the series definition:

$$
\hat A_t
=
\delta_t + \gamma\lambda \delta_{t+1}
+ (\gamma\lambda)^2 \delta_{t+2} + \cdots
$$

we get the recursion:

$$
\hat A_t
=
\delta_t + \gamma\lambda \hat A_{t+1}
$$

with $\hat A_T = 0$ at terminal states (or masked).

This explains why GAE is computed **backward in time**.

---

## 9. Summary: Why Advantage and GAE

- Replacing $R(\tau)$ with $G_t$ improves credit assignment
- Subtracting $V(s_t)$ reduces variance
- TD residuals approximate advantages efficiently
- GAE interpolates between low-variance TD and low-bias Monte Carlo
- PPO uses GAE for stable, sample-efficient policy optimization

This completes the full derivation from **trajectory return** to **GAE-based advantage estimation**.
