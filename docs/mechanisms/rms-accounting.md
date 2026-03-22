# RMS Sensitivity Accounting

Per-example sensitivity accounting that produces tighter privacy
bounds than worst-case DP, based on the stochastic f-MIP framework of
[Leemann et al. (2023)](https://arxiv.org/abs/2306.07273).
Standard DP-SGD accounting assumes every example has worst-case
sensitivity (the clipping bound $C$). RMS accounting uses the
*observed* per-example sensitivities to produce a tighter privacy
guarantee that reflects the average-case attacker.

## Setup

Consider DP-SGD with $N$ training examples, batch sampling rate $q$,
noise multiplier $\sigma$, and clipping bound $C$. At each step $t$,
example $i$ in the batch has clipped gradient $g_i^{(t)}$ with
sensitivity $s_i^{(t)} = \|g_i^{(t)}\| / C \in [0, 1]$.

Standard accounting uses $s = 1$ for all examples (worst case after
clipping). The stochastic f-MIP notion averages over which example the
attacker targets.

## Definitions

### Trade-off functions

A **trade-off function** $T: [0,1] \to [0,1]$ maps the false positive
rate $\alpha$ of a hypothesis test to the minimum achievable false
negative rate $\beta$. It is convex, non-increasing, and satisfies
$T(\alpha) \le 1 - \alpha$. The Gaussian trade-off is
$G_\mu(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - \mu)$.

### Stochastic composition of trade-off functions

**Definition** (Leemann et al., Def. 4.1). Let $h: \mathcal{X} \to
\mathcal{F}$ map each data point to a trade-off function, and let
$\mathcal{D}$ be a distribution on $\mathcal{X}$. The **stochastic
composition** is

$$\left(\bigotimes_{x \sim \mathcal{D}} h(x)\right)(\alpha) = \min_{\bar{\alpha}:\, \mathbb{E}[\bar{\alpha}(x)] = \alpha}\; \mathbb{E}_{x \sim \mathcal{D}}\!\left[h(x)\!\left(\bar{\alpha}(x)\right)\right].$$

The attacker allocates per-example FPR budgets $\bar{\alpha}(x)$
subject to a global average FPR of $\alpha$, and minimises the global
FNR. This is the optimal strategy for an attacker who targets a
*random* example $x \sim \mathcal{D}$.

### f-Membership Inference Privacy

**Definition** (Leemann et al., Def. 4.2). An algorithm $\mathcal{A}$
is **$f$-MIP** with respect to data distribution $\mathcal{D}$ if

$$\bigotimes_{x' \sim \mathcal{D}} \mathrm{Test}(A_0;\, A_1(x')) \ge f,$$

where $A_0 = \mathcal{A}(X \cup \{x\})$ with $x \sim \mathcal{D}$ is
the output when the target is a fresh sample, and
$A_1(x') = \mathcal{A}(X \cup \{x'\})$ is the output when the target
$x'$ is in the training set.

## Key result: stochastic f-MIP in hockey-stick divergence

The hockey-stick divergence $\delta(\varepsilon)$ of a mechanism
with trade-off function $T$ is

$$\delta(\varepsilon) = \max_{\alpha \in [0,1]}\;\left[1 - T(\alpha) - e^\varepsilon \alpha\right].$$

**Theorem 1** (Stochastic hockey-stick identity). *The hockey-stick
divergence of the stochastic composition equals the expectation of
per-example hockey-stick divergences:*

$$\delta_{\mathrm{stoch}}(\varepsilon) = \mathbb{E}_{x \sim \mathcal{D}}\!\left[\delta_x(\varepsilon)\right],$$

*where $\delta_x(\varepsilon) = \max_\alpha\,[1 - T_x(\alpha) - e^\varepsilon\alpha]$ is the hockey-stick divergence for example $x$.*

**Proof.** Starting from the definitions,

$$\delta_{\mathrm{stoch}}(\varepsilon)
= \max_\alpha \left[1 - \min_{\bar{\alpha}:\,\mathbb{E}[\bar{\alpha}]=\alpha} \mathbb{E}\!\left[T_x\!\left(\bar{\alpha}(x)\right)\right] - e^\varepsilon \alpha\right].$$

Rewriting $1 - \min = \max(1 - \cdot)$ and substituting $e^\varepsilon\alpha = e^\varepsilon\mathbb{E}[\bar{\alpha}(x)]$:

$$= \max_\alpha \max_{\bar{\alpha}:\,\mathbb{E}[\bar{\alpha}]=\alpha}\; \mathbb{E}\!\left[1 - T_x\!\left(\bar{\alpha}(x)\right) - e^\varepsilon \bar{\alpha}(x)\right].$$

Since the outer $\max_\alpha$ and the constraint $\mathbb{E}[\bar{\alpha}] = \alpha$ together allow $\bar{\alpha}$ to range over *all* measurable functions $\bar{\alpha}: \mathcal{X} \to [0,1]$, the joint optimisation collapses:

$$= \max_{\bar{\alpha}: \mathcal{X}\to[0,1]}\; \mathbb{E}_{x}\!\left[1 - T_x\!\left(\bar{\alpha}(x)\right) - e^\varepsilon \bar{\alpha}(x)\right].$$

The integrand depends on $\bar{\alpha}(x)$ only through $x$, so the
maximum can be taken pointwise:

$$= \mathbb{E}_{x}\!\left[\max_{\alpha_x \in [0,1]}\;\left(1 - T_x(\alpha_x) - e^\varepsilon \alpha_x\right)\right] = \mathbb{E}_{x}\!\left[\delta_x(\varepsilon)\right]. \qquad \square$$

**Remark.** The FPR-allocation optimisation that makes the trade-off
function definition complex vanishes entirely in $\delta(\varepsilon)$
space. This is the reason PLD-based accounting is the natural home for
stochastic f-MIP.

## Single-step stochastic f-MIP for DP-SGD

For one step of DP-SGD with the Gaussian mechanism, example $i$ has
sensitivity $s_i = \|g_i\| / C$ and the per-example hockey-stick
divergence is $\delta_i(\varepsilon) = \delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/s_i)$,
the standard Gaussian mechanism divergence with effective noise
multiplier $\sigma / s_i$.

By Theorem 1, the single-step stochastic f-MIP is

$$\delta_{\mathrm{stoch}}^{(1)}(\varepsilon) = \frac{1}{N}\sum_{i=1}^N \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_i}\right).$$

This is **exact** — no approximation. Note the sum is over all $N$
training examples, not just the batch.

## RMS approximation

### RMS sensitivity

Define the **RMS sensitivity**:

$$s_{\mathrm{rms}} = \sqrt{\mathbb{E}_i[s_i^2]} = \sqrt{\frac{1}{N}\sum_{i=1}^N s_i^2}.$$

### Curvature of $\delta$ in $s^2$ (Corrected Analysis)

Define $\varphi(u) = \delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/\!\sqrt{u})$ for $u = s^2$.
For the Gaussian mechanism:

$$\varphi(u) = \Phi\!\left(-\frac{\varepsilon}{\nu} + \frac{\nu}{2}\right) - e^\varepsilon\, \Phi\!\left(-\frac{\varepsilon}{\nu} - \frac{\nu}{2}\right),$$

where $\nu = \sqrt{u}/\sigma$.

**$\varphi$ is NOT globally convex (or concave) in $u$.** Numerical
evaluation shows that the curvature depends on $\varepsilon$:

- At **small $\varepsilon$** (e.g., $\varepsilon \lesssim 0.7$ for
  typical parameters), $\varphi$ is **concave** in $u$:
  $\mathbb{E}[\varphi(u_i)] \le \varphi(\mathbb{E}[u_i])$.
- At **large $\varepsilon$** (e.g., $\varepsilon \gtrsim 0.7$),
  $\varphi$ is **convex** in $u$:
  $\mathbb{E}[\varphi(u_i)] \ge \varphi(\mathbb{E}[u_i])$.

The crossover point depends on $\sigma$ and the sensitivity
distribution.

> **Erratum.** A previous version of this document stated a Lemma
> claiming global convexity of $\varphi$ in $u = s^2$ and used it to
> derive Theorem 2 ($\delta_{\mathrm{stoch}} \le \delta_{\mathrm{RMS}}$
> for all $\varepsilon$). Both the Lemma and the resulting Theorem were
> **incorrect**. The function $\varphi$ changes curvature with
> $\varepsilon$, and the Jensen inequality does not yield a uniform
> bound in either direction.

### RMS vs mixture: relationship (Theorem 2, corrected)

**Theorem 2** (RMS is optimistic in the tail). *The single-step
stochastic f-MIP $\delta_{\mathrm{stoch}}^{(t)}$ and the RMS
approximation $\delta_{\mathrm{RMS}}^{(t)}$ satisfy:*

- *At small $\varepsilon$ (concave regime):*
  $\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) \le \delta_{\mathrm{RMS}}^{(t)}(\varepsilon)$
  *(RMS is conservative).*
- *At large $\varepsilon$ (convex regime):*
  $\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) \ge \delta_{\mathrm{RMS}}^{(t)}(\varepsilon)$
  *(RMS is optimistic).*

*In particular, the RMS $\varepsilon$ at a small target $\delta$ (the
operationally relevant regime) is a **lower bound** on the exact
stochastic f-MIP $\varepsilon$:*

$$\varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{stoch}} \le \varepsilon_{\mathrm{DP}}.$$

**Explanation.** Computing $\varepsilon$ at target $\delta$ requires
finding the $\varepsilon^*$ where $\delta(\varepsilon^*) = \delta_{\mathrm{target}}$.
For small $\delta_{\mathrm{target}}$, the relevant $\varepsilon^*$ is
large — squarely in the convex regime where
$\delta_{\mathrm{stoch}} \ge \delta_{\mathrm{RMS}}$. The mixture PLD
captures the heavy right tail of the sensitivity distribution, which
dominates privacy loss at large $\varepsilon$. The single-Gaussian RMS
PLD has lighter tails, leading to optimistically low $\varepsilon$.

Under composition (convolution of PLDs over $T$ steps), the heavier
per-step tails compound, **widening** the gap between mixture and RMS.

**Remark.** The RMS approximation *underestimates* privacy loss
(reports a smaller $\varepsilon$), making it an **optimistic** (unsafe)
approximation of stochastic f-MIP. It should be interpreted as a lower
bound, not a privacy guarantee. Use `--accounting mixture` for a
correct per-step stochastic f-MIP accounting.

### Batch RMS as sample estimate

The RMS implementation uses the **batch mean** as a sample estimate of
the population $\mathbb{E}_i[s_i^2]$:

$$(\hat{s}_{\mathrm{rms}}^{(t)})^2 = \frac{1}{|B_t|}\sum_{i \in B_t} \left(s_i^{(t)}\right)^2.$$

Since Poisson sampling includes each example independently with
probability $q$ (independently of gradient norms), this is an
**unbiased estimator** of $\mathbb{E}_i[s_i^2]$.

### Tightness relative to worst-case DP

**Theorem 3** (RMS gives lower $\varepsilon$ than worst-case DP).
*If the sensitivity distribution is non-degenerate (not all $s_i = 1$),
then $s_{\mathrm{rms}} < 1$, and for all $\varepsilon \ge 0$:*

$$\delta_{\mathrm{RMS}}(\varepsilon) = \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}}\right) < \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \sigma\right) = \delta_{\mathrm{DP}}(\varepsilon).$$

**Proof.** $s_{\mathrm{rms}} < 1$ when any $s_i < 1$, which gives
$\sigma/s_{\mathrm{rms}} > \sigma$, and
$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma')$ is strictly
decreasing in $\sigma'$. $\square$

**Remark.** This establishes that
$\varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{DP}}$. However,
per corrected Theorem 2, $\varepsilon_{\mathrm{RMS}}$ also
underestimates the exact stochastic f-MIP. The full ordering is:

$$\varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{stoch}} \le \varepsilon_{\mathrm{DP}}.$$

## Composition across steps

Standard PLD composition applies: the composed privacy after $T$ steps
with Poisson subsampling at rate $q$ is

$$\mathrm{PLD}_{\mathrm{total}} = \circledast_{t=1}^{T}\; \mathrm{PoissonSubsample}\!\left(\mathrm{PLD}_{\mathrm{rms}}^{(t)},\; q\right),$$

and $\varepsilon$ at target $\delta$ is read off from
$\mathrm{PLD}_{\mathrm{total}}$ as usual.

**Corollary.** *The RMS-composed $\varepsilon$ at any target $\delta$
satisfies*

$$\varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{mixture}} \le \varepsilon_{\mathrm{DP}}.$$

The right inequality follows from Theorem 3. The left inequality
follows from the per-step tail analysis (corrected Theorem 2)
propagated through composition: the mixture PLD has heavier tails
per step, and PLD convolution compounds this across steps.

## Summary of guarantees

| Quantity | Relation | Computational cost |
|---|---|---|
| $\varepsilon_{\mathrm{RMS}}$ (`--accounting rms`) | optimistic lower bound | $O(T \cdot G\log G)$ — same as standard |
| $\varepsilon_{\mathrm{mixture}}$ (`--accounting mixture`) | exact per-step f-MIP | $O(T \cdot B \cdot G\log G)$ — ~100 bins/step |
| $\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}}$ (exact multi-step f-MIP) | tightest (infeasible) | $O(N \cdot T \cdot G\log G)$ — infeasible |
| $\varepsilon_{\mathrm{DP}}$ (`--accounting standard`) | worst-case upper bound | $O(T \cdot G\log G)$ — same as standard |

Ordering: $\varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{stoch}}^{\mathrm{exact}} \le \varepsilon_{\mathrm{mixture}} \le \varepsilon_{\mathrm{DP}}$.

where $G$ is the PLD grid size and $B$ is the number of sensitivity bins.

**RMS vs mixture:** The RMS PLD (single Gaussian with $s_{\mathrm{rms}}$)
has lighter tails than the mixture PLD. At the large-$\varepsilon$
regime that determines $\varepsilon$ at small target $\delta$, the
mixture's heavier tails yield higher (correct) $\delta$, while RMS
underestimates. This gap compounds under composition. **Use mixture
for correct accounting; treat RMS as an optimistic estimate only.**

Both share the same multi-step approximation: they average per step
then compose, rather than composing per-example then averaging.
The mixture $\varepsilon$ may therefore slightly overestimate the
exact multi-step f-MIP (due to cross-trajectory terms in composition),
but this is a conservative direction.

When all examples are clipped ($s_i = 1$), both reduce to standard DP.

## Interpretation

The stochastic f-MIP accounting (both RMS and mixture modes) answers
the question: *how private is a randomly chosen training example, on
average?* This is a weaker guarantee than worst-case DP (which protects
the *worst* example) but a stronger guarantee than empirical auditing
(which measures a specific attack).

It is the natural privacy measure when:

- The threat model is a membership inference attacker who does not get
  to choose which example to target (e.g., the target is selected
  uniformly at random).
- One wants to report a single $\varepsilon$ that reflects the typical
  privacy experienced by training participants, rather than the
  worst-case outlier.

**Important:** The mixture mode gives the correct per-step stochastic
f-MIP. The RMS mode underestimates privacy loss (optimistic) and
should not be used as a privacy guarantee. It may still be useful as a
fast diagnostic or lower bound on $\varepsilon$.

## Usage

```bash
# Exact per-step stochastic f-MIP (recommended)
python train_causal_lm.py --accounting mixture ...

# RMS lower bound (fast, optimistic — NOT a valid privacy guarantee)
python train_causal_lm.py --accounting rms ...
```

Both modes add negligible overhead. Calibration still uses standard
worst-case DP (the noise level is unchanged); the stochastic f-MIP
$\varepsilon$ is reported alongside as a post-hoc measure.
