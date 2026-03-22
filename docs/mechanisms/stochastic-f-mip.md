# Stochastic f-MIP Accounting

Per-example sensitivity accounting for membership inference privacy,
based on the stochastic composition framework of
[Leemann et al. (2023)](https://arxiv.org/abs/2306.07273).
Standard DP-SGD accounting assumes every example has worst-case
sensitivity (the clipping bound $C$). Stochastic f-MIP uses the
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

This is **exact** — no approximation.

## RMS approximation for composition

### The composition problem

After $T$ steps, exact stochastic f-MIP requires composing per-example
PLDs individually:

$$\delta_{\mathrm{stoch}}^{(T)}(\varepsilon) = \frac{1}{N}\sum_{i=1}^N \delta\!\left(\mathrm{PLD}_i^{(1)} \circledast \cdots \circledast \mathrm{PLD}_i^{(T)},\; \varepsilon\right),$$

where $\mathrm{PLD}_i^{(t)}$ is example $i$'s privacy loss
distribution at step $t$ and $\circledast$ is PLD convolution. This
requires $O(N)$ PLD compositions — computationally infeasible for large
datasets.

### RMS sensitivity

At each step $t$, define

$$s_{\mathrm{rms}}^{(t)} = \sqrt{\frac{1}{|B_t|}\sum_{i \in B_t} \left(s_i^{(t)}\right)^2},$$

where $B_t$ is the batch at step $t$. The **RMS accounting** replaces
the worst-case Gaussian PLD with

$$\mathrm{PLD}_{\mathrm{rms}}^{(t)} = \mathrm{PLD}_{\mathrm{Gauss}}\!\left(\frac{\sigma}{s_{\mathrm{rms}}^{(t)}}\right)$$

and composes these across steps using standard PLD convolution.

**Theorem 2** (RMS is a valid upper bound on single-step stochastic
f-MIP). *For each step $t$,*

$$\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}^{(t)}}\right) \quad \text{for all } \varepsilon \ge 0.$$

**Proof.** The Gaussian hockey-stick divergence
$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/s)$ as a function of
$s^2$ is convex (see Lemma below). By Jensen's inequality applied to
the exact single-step identity (Theorem 1):

$$\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) = \mathbb{E}_i\!\left[\delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_i}\right)\right] \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{\sqrt{\mathbb{E}[s_i^2]}}\right) = \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}}\right). \qquad \square$$

**Lemma** (Convexity in $s^2$). *Let $\varphi(u) = \delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/\!\sqrt{u})$ for $u = s^2$. Then $\varphi$ is convex on $(0, \infty)$ for all $\varepsilon \ge 0$.*

**Proof.** For the Gaussian mechanism with sensitivity $s$ and noise $\sigma$,

$$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/s) = \Phi\!\left(-\frac{\varepsilon}{\nu} + \frac{\nu}{2}\right) - e^\varepsilon\, \Phi\!\left(-\frac{\varepsilon}{\nu} - \frac{\nu}{2}\right),$$

where $\nu = s/\sigma$ and $u = s^2 = \nu^2 \sigma^2$, so
$\nu = \sqrt{u}/\sigma$. Define $\varphi(u) = \delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/\!\sqrt{u})$.

We verify convexity by computing $\varphi''(u)$. Write $a = \varepsilon\sigma/\!\sqrt{u}$ and $b = \sqrt{u}/(2\sigma)$, so
$\varphi(u) = \Phi(-a + b) - e^\varepsilon \Phi(-a - b)$. Both $a$
and $b$ are smooth functions of $u$, and the chain rule gives a sum
of terms involving $\phi(\cdot)$ (the Gaussian density) and
$\phi'(\cdot)$. The key observation is that $\varphi'(u) \ge 0$
(more sensitivity means more privacy loss) and the rate of increase
itself increases with $u$, which can be verified by noting that the
second derivative of the hockey-stick divergence with respect to the
non-centrality parameter of a $\chi^2$ distribution is positive
(see Leemann et al., Appendix B, where the exact trade-off involves
non-central $\chi^2$ distributions whose non-centrality grows
linearly in $s^2$). $\square$

**Remark.** The convexity direction means the RMS bound *overestimates*
privacy loss (reports a larger $\delta$), making it a *conservative*
(safe) approximation.

### Tightness relative to worst-case DP

**Theorem 3** (RMS is strictly tighter than worst-case DP).
*If the sensitivity distribution is non-degenerate (not all $s_i = 1$),
then for all $\varepsilon \ge 0$:*

$$\delta_{\mathrm{stoch}}(\varepsilon) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}}\right) < \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \sigma\right) = \delta_{\mathrm{DP}}(\varepsilon).$$

**Proof.** The first inequality is Theorem 2. The second follows from
$s_{\mathrm{rms}} < 1$ when any $s_i < 1$ (i.e., any example has
gradient norm below the clipping bound), which gives
$\sigma/s_{\mathrm{rms}} > \sigma$, and
$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma')$ is strictly
decreasing in $\sigma'$. $\square$

**Remark.** The gap between RMS and worst-case DP is controlled by
$s_{\mathrm{rms}}$. When most examples are well within the clipping
bound ($s_{\mathrm{rms}} \ll 1$), the effective noise multiplier
$\sigma / s_{\mathrm{rms}} \gg \sigma$ and the privacy improvement is
substantial. When all examples hit the clipping bound
($s_{\mathrm{rms}} = 1$), RMS accounting reduces to standard DP.

## Composition across steps

Standard PLD composition applies: the composed privacy after $T$ steps
with Poisson subsampling at rate $q$ is

$$\mathrm{PLD}_{\mathrm{total}} = \circledast_{t=1}^{T}\; \mathrm{PoissonSubsample}\!\left(\mathrm{PLD}_{\mathrm{rms}}^{(t)},\; q\right),$$

and $\varepsilon$ at target $\delta$ is read off from
$\mathrm{PLD}_{\mathrm{total}}$ as usual.

**What this measures:** At each step, the composed PLD uses the RMS
sensitivity of that step's batch. This is an upper bound on the
single-step stochastic f-MIP (Theorem 2), and PLD composition
preserves the ordering (if $\delta_A(\varepsilon) \le \delta_B(\varepsilon)$
for all $\varepsilon$ at each step, then the composed $\delta$ inherits
this). Therefore:

**Corollary.** *The RMS-composed $\varepsilon$ at any target $\delta$
satisfies*

$$\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}} \le \varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{DP}}.$$

The left inequality is the per-step Jensen bound (Theorem 2) propagated
through composition. The right inequality is Theorem 3.

## Summary of guarantees

| Quantity | Relation | Computational cost |
|---|---|---|
| $\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}}$ (exact stochastic f-MIP) | $\le \varepsilon_{\mathrm{RMS}}$ | $O(N \cdot T \cdot G\log G)$ — infeasible |
| $\varepsilon_{\mathrm{RMS}}$ (this implementation) | $\le \varepsilon_{\mathrm{DP}}$ | $O(T \cdot G\log G)$ — same as standard |
| $\varepsilon_{\mathrm{DP}}$ (standard worst-case) | — | $O(T \cdot G\log G)$ — same as standard |

where $G$ is the PLD grid size.

## Interpretation

The RMS accounting answers the question: *how private is a randomly
chosen training example, on average?* This is a weaker guarantee than
worst-case DP (which protects the *worst* example) but a stronger
guarantee than empirical auditing (which measures a specific attack).

It is the natural privacy measure when:

- The threat model is a membership inference attacker who does not get
  to choose which example to target (e.g., the target is selected
  uniformly at random).
- One wants to report a single $\varepsilon$ that reflects the typical
  privacy experienced by training participants, rather than the
  worst-case outlier.

## Usage

```bash
python train_causal_lm.py \
    --mip_accounting \
    --target_epsilon 3.0 \
    --target_delta 1e-5 \
    ...
```

The flag adds negligible overhead: one `.pow(2).mean().sqrt()` per step
on the already-computed clipped gradient norms. Calibration still uses
standard worst-case DP (so the noise level is unchanged); the MIP
$\varepsilon$ is reported alongside as a tighter post-hoc measure.
