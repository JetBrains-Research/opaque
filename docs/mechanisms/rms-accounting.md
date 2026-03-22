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

### RMS bound (Theorem 2)

Define the **RMS sensitivity**:

$$s_{\mathrm{rms}} = \sqrt{\mathbb{E}_i[s_i^2]} = \sqrt{\frac{1}{N}\sum_{i=1}^N s_i^2}.$$

**Theorem 2** (RMS is a valid upper bound on single-step stochastic
f-MIP). *For each step $t$,*

$$\delta_{\mathrm{stoch}}^{(t)}(\varepsilon) \le \delta_{\mathrm{Gauss}}\!\left(\varepsilon;\, \frac{\sigma}{s_{\mathrm{rms}}}\right) \quad \text{for all } \varepsilon \ge 0.$$

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

### Batch RMS as sample estimate

Theorem 2 bounds $\delta_{\mathrm{stoch}}$ in terms of
$\mathbb{E}_i[s_i^2]$ (the population mean of squared sensitivities
over all $N$ examples). At each step, we only observe sensitivities
for the Poisson-sampled batch $B_t$. The implementation uses the
**batch mean** as a sample estimate:

$$(\hat{s}_{\mathrm{rms}}^{(t)})^2 = \frac{1}{|B_t|}\sum_{i \in B_t} \left(s_i^{(t)}\right)^2.$$

Since Poisson sampling includes each example independently with
probability $q$ (independently of gradient norms), this is an
**unbiased estimator** of $\mathbb{E}_i[s_i^2]$. Every example goes
through the identical Poisson-subsampled Gaussian mechanism — the
only thing that varies across examples is $s_i$. So after Jensen's
collapses the $N$-example average to a single PLD parameterized by
$\mathbb{E}[s_i^2]$, estimating that parameter from a batch sample
is standard Monte Carlo.

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

**Corollary.** *The RMS-composed $\varepsilon$ at any target $\delta$
satisfies*

$$\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}} \le \varepsilon_{\mathrm{RMS}} \le \varepsilon_{\mathrm{DP}}.$$

The left inequality is the per-step Jensen bound (Theorem 2) propagated
through composition (PLD composition preserves ordering). The right
inequality is Theorem 3. In the implementation, $\mathbb{E}[s_i^2]$ is
estimated from the batch at each step (see "Batch RMS as sample
estimate" above).

## Summary of guarantees

| Quantity | Relation | Computational cost |
|---|---|---|
| $\varepsilon_{\mathrm{stoch}}^{\mathrm{exact}}$ (exact multi-step f-MIP) | — | $O(N \cdot T \cdot G\log G)$ — infeasible |
| $\varepsilon_{\mathrm{mixture}}$ (`--accounting mixture`) | $\approx \varepsilon_{\mathrm{stoch}}^{\mathrm{exact}}$ | $O(T \cdot B \cdot G\log G)$ — ~100 bins/step |
| $\varepsilon_{\mathrm{RMS}}$ (`--accounting rms`) | $\le \varepsilon_{\mathrm{DP}}$ | $O(T \cdot G\log G)$ — same as standard |
| $\varepsilon_{\mathrm{DP}}$ (`--accounting standard`) | — | $O(T \cdot G\log G)$ — same as standard |

where $G$ is the PLD grid size and $B$ is the number of sensitivity bins.

**RMS vs mixture:** RMS applies Jensen's inequality per step, losing
tightness when the sensitivity distribution has high variance (mix of
near-zero and near-1 sensitivities). Mixture keeps the full binned
distribution and computes the exact single-step stochastic f-MIP.
Both share the same multi-step approximation: they average per step
then compose, rather than composing per-example then averaging.
When all examples are clipped ($s_i = 1$), the mixture automatically
reduces to a standard Gaussian PLD (no extra cost).

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
# Jensen bound (fast, slightly loose)
python train_causal_lm.py --accounting rms ...

# Binned mixture (exact per-step, tighter)
python train_causal_lm.py --accounting mixture ...
```

Both modes add negligible overhead. Calibration still uses standard
worst-case DP (the noise level is unchanged); the tighter $\varepsilon$
is reported alongside as a post-hoc measure.
