# Knowledge Distillation

## From Supervised Fine-Tuning to On-Policy Distillation

## 1. Motivation

Large language models (LLMs) are often trained in multiple stages:

- pretraining on large corpora,
- supervised fine-tuning (SFT),
- and sometimes reinforcement learning–based alignment.

**Distillation** plays a central role in this pipeline. It allows a _student_ model to learn from a stronger _teacher_ model, transferring capabilities while reducing cost, latency, or model size.

This tutorial introduces:

- off-policy distillation (including SFT),
- on-policy distillation,
- and the role of forward vs. reverse KL divergence.

---

## 2. What Is Distillation?

At a high level, distillation trains a student policy $ \pi\_\theta $ to imitate a teacher policy $ \pi_T $.

The key questions are:

1. **Where do training trajectories come from?**
2. **What loss is optimized?**

Different answers lead to very different training dynamics.

---

## 3. Supervised Fine-Tuning (SFT) as Distillation

### 3.1 Next-token prediction loss

In SFT, the student is trained on a dataset of prompt–response pairs:

$$
(x, y_1, y_2, \dots, y_T)
$$

The objective is the negative log-likelihood (NLL):

$$
\mathcal{L}_{\text{SFT}}
=
-\sum_{t=1}^T \log \pi_\theta(y_t \mid x, y_{<t})
$$

This is often called:

- next-token prediction,
- cross-entropy loss,
- teacher forcing.

---

### 3.2 SFT as forward KL divergence

SFT can be interpreted as minimizing **forward KL divergence**.

Define a teacher distribution that is _one-hot_:

$$
\pi_T(\cdot \mid s_t) = \delta(\cdot = y_t)
$$

Then:

$$
\mathrm{KL}(\pi_T \,\|\, \pi_\theta)
=
-\log \pi_\theta(y_t) + \text{const}
$$

Thus:

> **SFT is forward KL minimization where the teacher distribution is one-hot.**

---

## 4. Off-Policy Distillation

### 4.1 Definition

Off-policy distillation generalizes SFT by allowing the teacher to provide a _soft_ distribution over tokens.

Training data consists of fixed teacher-generated trajectories:

$$
y \sim \pi_T(\cdot \mid x)
$$

---

### 4.2 Objective

The student minimizes forward KL:

$$
\mathcal{L}_{\text{off-policy}}
=
\mathbb{E}_{y \sim \pi_T}
\sum_t
\mathrm{KL}\big(
\pi_T(\cdot \mid s_t)
\;\|\;
\pi_\theta(\cdot \mid s_t)
\big)
$$

This includes:

- classical knowledge distillation,
- logit matching,
- temperature-scaled distillation.

---

### 4.3 Limitations

Off-policy distillation suffers from **distributional shift**:

- training conditions on teacher trajectories,
- inference runs on student trajectories.

Errors can compound in long-horizon generation.

---

## 5. On-Policy Distillation

### 5.1 Key idea

On-policy distillation flips the data source:

> The student trains on **its own generated trajectories**, while the teacher provides feedback.

Formally:

$$
y \sim \pi_\theta(\cdot \mid x)
$$

---

### 5.2 Why forward KL no longer works

Forward KL requires expectations over teacher-generated trajectories.
But in on-policy training:

- only student-induced states are available,
- teacher trajectories are inaccessible under those states.

This motivates a different objective.

---

## 6. Reverse KL Divergence

### 6.1 Definition

Reverse KL is defined as:

$$
\mathrm{KL}(\pi_\theta \,\|\, \pi_T)
=
\mathbb{E}_{a \sim \pi_\theta}
\left[
\log \pi_\theta(a) - \log \pi_T(a)
\right]
$$

Key properties:

- mode-seeking,
- penalizes actions the teacher dislikes,
- naturally on-policy.

---

### 6.2 On-policy distillation objective

The canonical on-policy distillation loss is:

$$
\mathcal{L}_{\text{on-policy}}
=
\mathbb{E}_{y \sim \pi_\theta}
\sum_t
\mathrm{KL}
\big(
\pi_\theta(\cdot \mid s_t)
\;\|\;
\pi_T(\cdot \mid s_t)
\big)
$$

This loss:

- evaluates the teacher on student states,
- avoids distribution mismatch,
- provides dense supervision.

---

## 7. Comparison Summary

| Method                  | Trajectory source | Teacher signal    | Objective        |
| ----------------------- | ----------------- | ----------------- | ---------------- |
| SFT                     | Teacher / human   | One-hot token     | NLL (forward KL) |
| Off-policy distillation | Teacher           | Soft distribution | Forward KL       |
| On-policy distillation  | Student           | Soft distribution | Reverse KL       |
| RLHF / PPO              | Student           | Scalar reward     | Policy gradient  |

---

## 8. Practical Takeaways

- **SFT is distillation**: it is forward KL with a degenerate teacher.
- **Off-policy distillation** improves supervision but not distribution shift.
- **On-policy distillation** aligns training with inference behavior.
- **Reverse KL** is essential for stable on-policy optimization.

---

## 9. Final Perspective

Distillation methods form a continuum:

- from pure imitation (SFT),
- to teacher-guided optimization (on-policy distillation),
- to reward-based learning (RL).

Understanding the objective functions clarifies _why_ each method behaves the way it does—and _when_ to use them.

---

## References

- https://thinkingmachines.ai/blog/on-policy-distillation/
- https://huggingfaceh4-on-policy-distillation.hf.space/
