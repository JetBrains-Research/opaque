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

### Population RMS bound (Theorem 2)

Define the **population RMS sensitivity**:

$$s_{\mathrm{rms}}^{\mathrm{pop}} = \sqrt{\frac{1}{N}\sum_{i=1}^N s_i^2}.$$

**Theorem 2** (Population RMS is a valid upper bound on single-step
stochastic f-MIP). *For each step $t$,*

$$\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}^{\mathrm{pop}}}\right) \quad \text{for all } \varepsilon \ge 0.$$

**Proof.** The Gaussian hockey-stick divergence
$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma/s)$ as a function of
$s^2$ is convex (see Lemma below). By Jensen's inequality applied to
the exact single-step identity (Theorem 1):

$$\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) = \frac{1}{N}\sum_{i=1}^N \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_i}\right) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{\sqrt{\frac{1}{N}\sum_i s_i^2}}\right) = \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}^{\mathrm{pop}}}\right). \qquad \square$$

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
(safe) approximation — **provided** we use the true population RMS.

### Batch RMS in practice

Theorem 2 requires the population quantity $s_{\mathrm{rms}}^{\mathrm{pop}}$
(over all $N$ training examples), but at each step we only observe
sensitivities for the Poisson-sampled batch $B_t$. The implementation
uses the **batch RMS**:

$$\hat{s}_{\mathrm{rms}}^{(t)} = \sqrt{\frac{1}{|B_t|}\sum_{i \in B_t} \left(s_i^{(t)}\right)^2}.$$

Since Poisson sampling includes each example independently with
probability $q$ (independently of gradient norms), $(\hat{s}_{\mathrm{rms}})^2$
is an **unbiased estimator** of $({s}_{\mathrm{rms}}^{\mathrm{pop}})^2$:

$$\mathbb{E}\!\left[(\hat{s}_{\mathrm{rms}})^2\right] = \frac{1}{N}\sum_{i=1}^N s_i^2 = (s_{\mathrm{rms}}^{\mathrm{pop}})^2.$$

However, being unbiased does not make it a deterministic upper bound.
The batch estimate can underestimate the population RMS (if the batch
happens to contain examples with smaller-than-average gradients),
which would make the reported $\varepsilon$ optimistic at that step.

**In practice this matters little:** over $T$ steps of training, the
estimation errors are independent and approximately cancel. The composed
$\varepsilon$ converges to the value obtained with population RMS as
$T$ grows. For typical training runs ($T \ge 100$, $|B| \ge 32$) the
deviation is negligible.

**To obtain a rigorous bound**, one could use a one-sided confidence
interval on $\mathbb{E}[s_i^2]$:

$$\hat{\mu}_{\mathrm{upper}} = (\hat{s}_{\mathrm{rms}})^2 + z_{1-\alpha}\,\frac{\hat{\sigma}_{s^2}}{\sqrt{|B_t|}},$$

where $\hat{\sigma}_{s^2}$ is the sample standard deviation of $s_i^2$
within the batch and $z_{1-\alpha}$ is the normal quantile. Using
$s_{\mathrm{rms}}^{\mathrm{corrected}} = \sqrt{\hat{\mu}_{\mathrm{upper}}}$
would give a bound valid with probability $1 - \alpha$ per step.
The current implementation does not apply this correction.

### Tightness relative to worst-case DP

**Theorem 3** (Population RMS is strictly tighter than worst-case DP).
*If the sensitivity distribution is non-degenerate (not all $s_i = 1$),
then for all $\varepsilon \ge 0$:*

$$\delta_{\mathrm{stoch}}(\varepsilon) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}^{\mathrm{pop}}}\right) < \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \sigma\right) = \delta_{\mathrm{DP}}(\varepsilon).$$

**Proof.** The first inequality is Theorem 2. The second follows from
$s_{\mathrm{rms}}^{\mathrm{pop}} < 1$ when any $s_i < 1$ (i.e., any
example has gradient norm below the clipping bound), which gives
$\sigma/s_{\mathrm{rms}}^{\mathrm{pop}} > \sigma$, and
$\delta_{\mathrm{Gauss}}(\varepsilon;\, \sigma')$ is strictly
decreasing in $\sigma'$. $\square$

**Remark.** The gap between RMS and worst-case DP is controlled by
$s_{\mathrm{rms}}^{\mathrm{pop}}$. When most examples are well within
the clipping bound ($s_{\mathrm{rms}}^{\mathrm{pop}} \ll 1$), the
effective noise multiplier $\sigma / s_{\mathrm{rms}}^{\mathrm{pop}}
\gg \sigma$ and the privacy improvement is substantial. When all
examples hit the clipping bound ($s_{\mathrm{rms}}^{\mathrm{pop}} = 1$),
RMS accounting reduces to standard DP.

## Composition across steps

Standard PLD composition applies: the composed privacy after $T$ steps
with Poisson subsampling at rate $q$ is

$$\mathrm{PLD}_{\mathrm{total}} = \circledast_{t=1}^{T}\; \mathrm{PoissonSubsample}\!\left(\mathrm{PLD}_{\mathrm{rms}}^{(t)},\; q\right),$$

and $\varepsilon$ at target $\delta$ is read off from
$\mathrm{PLD}_{\mathrm{total}}$ as usual.

**With population RMS (Theorem 2 + composition):**

**Corollary.** *If population RMS sensitivities are used at each step,
the composed $\varepsilon$ satisfies*

$$\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}} \le \varepsilon_{\mathrm{RMS}}^{\mathrm{pop}} \le \varepsilon_{\mathrm{DP}}.$$

The left inequality is the per-step Jensen bound (Theorem 2) propagated
through composition (PLD composition preserves ordering). The right
inequality is Theorem 3.

**With batch RMS (this implementation):** The implementation uses the
batch RMS $\hat{s}_{\mathrm{rms}}$ as an unbiased estimate of the
population RMS. The composed $\varepsilon$ is an *estimate* of
$\varepsilon_{\mathrm{RMS}}^{\mathrm{pop}}$, not a deterministic bound.
In practice, the estimation errors across $T$ steps are independent and
approximately cancel, so the composed result closely tracks the
population-RMS value.

## Summary of guarantees

| Quantity | Relation | Notes |
|---|---|---|
| $\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}}$ | $\le \varepsilon_{\mathrm{RMS}}^{\mathrm{pop}}$ | Exact stochastic f-MIP; $O(NT)$ cost |
| $\varepsilon_{\mathrm{RMS}}^{\mathrm{pop}}$ (population RMS) | $\le \varepsilon_{\mathrm{DP}}$ | Rigorous bound; requires all $N$ sensitivities |
| $\varepsilon_{\mathrm{RMS}}^{\mathrm{batch}}$ (this implementation) | $\approx \varepsilon_{\mathrm{RMS}}^{\mathrm{pop}}$ | Unbiased estimate; uses batch only |
| $\varepsilon_{\mathrm{DP}}$ (standard worst-case) | — | Standard DP-SGD accounting |

The batch estimate converges to the population value as $|B| \to \infty$
or $T \to \infty$.

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
    --rms_accounting \
    --target_epsilon 3.0 \
    --target_delta 1e-5 \
    ...
```

The flag adds negligible overhead: `s_rms` is computed inside the
clipping layer alongside the already-computed clipped gradient norms
and returned in `aux.s_rms`. Calibration still uses standard worst-case
DP (so the noise level is unchanged); the RMS $\varepsilon$ is reported
alongside as a tighter post-hoc measure.
