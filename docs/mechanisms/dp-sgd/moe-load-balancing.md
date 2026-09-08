# MoE load balancing

Mixture-of-experts checkpoints such as Mellum 2.0 are trained with a
Switch-Transformer load-balancing loss that keeps the router from
collapsing onto a few experts. That loss is a statistic of the **whole
batch** (the fraction of tokens each expert received), so it has no
per-example gradient to clip and cannot enter a per-example DP objective
as written. The *router-load release* is Opaque's DP realisation of the
same objective: each step releases a differentially private estimate of
the batch router load alongside the clipped gradient, and the
load-balancing gradient is evaluated at that public estimate.

The release rides on the same clipped pytree as the gradient through a
zero-valued *probe parameter* that forms its own per-group clipping
group with a structural (never active) bound. The gradient and the load
statistic are one Gaussian (or matrix) mechanism under Opaque's per-group
noise allocation, so **the privacy accountant call is unchanged**; the
whole price is a $\sqrt{1+\rho}$ inflation of the gradient noise, where
$\rho$ is the share of the clipping budget given to the load group
($\rho = 0.02$ by default, a 1 % inflation).

The feature is reached through `TrainingArguments.router_load_release`
on `DPTrainer` (including the SFT and DPO trainers) and, for hand-written
functional loops, through the helper module
`opaque.api.transformers.moe_load` that the example scripts use. It
works with DP-SGD (Gaussian noise, Poisson sampling) and with every
DP-FTRL matrix mechanism; the [DP-FTRL mechanisms
page](../dp-ftrl/index.md#router-load-release-under-matrix-mechanisms)
covers what changes under correlated noise.

## The per-example loss

For one collated row $x$ with binary token mask $m_{x,t} =
\mathbb{1}\{\texttt{attention\_mask}_{x,t} \neq 0\}$, valid-token count
$T_x = \sum_t m_{x,t}$, routed layers $l = 1..L$, experts $e = 1..E$
and $k$ experts per token, the router of layer $l$ computes logits
$z^l_{x,t}$, probabilities $p^l_{x,t} = \mathrm{softmax}(z^l_{x,t})$ in
fp32 and the executed set $S^l_{x,t} = \mathrm{top}_k(p^l_{x,t})$. The
per-example statistics are

$$
P_e(x) = \frac{1}{L\,T_x} \sum_{l,t} m_{x,t}\, p^l_{x,t,e},
\qquad
h^l_e(x) = \frac{1}{T_x} \sum_t m_{x,t}\, \mathbb{1}\{e \in S^l_{x,t}\},
\qquad
d^l(x) = h^l(x) - \tfrac{k}{E}\mathbf{1},
$$

the mean router probability (differentiable) and the per-layer executed
load fraction (piecewise constant, zero gradient). With the public
constants $\bar T$ (`router_load_mean_tokens`, default $T_{\max}$) and
the token weight $w_x = T_x / \bar T$, the loss of one example at step
$t$ is

$$
\ell_x(\theta; \tilde f_t) = \mathrm{CE}_x(\theta)
+ \alpha \big( S_x - \mathrm{sg}[S_x] \big)
+ \zeta \big( Z_x - \mathrm{sg}[Z_x] \big)
+ \big\langle z,\ \lambda\, w_x\, d^{(L,E)}(x) \big\rangle_{\mathrm{sg}},
$$

$$
S_x = E\, w_x \sum_e \big( \tilde f_{t,e} - \tfrac{k}{E} \big) P_e(x),
$$

where $\mathrm{sg}$ is `detach`, $S_x$ is the load-balancing surrogate
at the public load estimate $\tilde f_t$, $Z_x$ is the optional router
z-loss (coefficient $\zeta$, `router_z_loss_coef`, zero by default) and
$z \in \mathbb{R}^{L \times E}$ is the zero probe parameter. Every added
term is **value-neutral**: it is exactly zero in value and contributes
only its gradient. The loss value the clipper records is therefore
$\mathrm{CE}_x$ bit for bit, the surrogate gradient $\alpha\,\nabla S_x$
reaches the model parameters, and the probe receives the gradient
$\lambda\, w_x\, d^{(L,E)}(x)$, which is the quantity to be released. A
fully masked row ($T_x = 0$) has $h = 0$, $P = 0$ and $w_x = 0$ and
contributes nothing.

### Faithfulness to the Hugging Face objective

Hugging Face pools $f$ and $P$ over all tokens of the batch with one
denominator $L \cdot T_{\mathrm{tot}}$, $T_{\mathrm{tot}} = \sum_{x \in
B} T_x$, so $f(B) = \sum_x (T_x / T_{\mathrm{tot}})\, h(x)$. Because
$w_x$ is linear in $T_x$, the batch mean of the per-example gradient is

$$
\frac{1}{\bar B} \sum_{x \in B_t} \nabla \ell_x
= \frac{1}{\bar B} \sum_x \nabla \mathrm{CE}_x
+ \alpha\, \frac{T_{\mathrm{tot}}}{\bar B\, \bar T}\,
  \nabla L^{\mathrm{HF}}_{\mathrm{aux}}(B_t)\big|_{f = \tilde f_t}
$$

exactly, for ragged and for packed rows alike. The *direction* is the
Hugging Face gradient at the load vector $\tilde f_t$; the scale factor
$T_{\mathrm{tot}} / (\bar B \bar T)$ has mean $\bar T_{\mathrm{true}} /
\bar T$. Set `router_load_mean_tokens` to the mean length measured on a
**public** held-out split to make it one in expectation; with the
default $\bar T = T_{\max}$ the auxiliary term is down-weighted by the
mean length fraction, well inside the usual uncertainty of the
coefficient itself. Two conventions are unchanged by the feature and
worth knowing: Opaque's cross-entropy weights examples equally (Hugging
Face weights tokens), and the estimate $\tilde f_t$ lags the batch it is
applied to (the coefficient multiplies the gradient of the same
functional form at a nearby load vector; the cost table below prices
"nearby").

## The released statistic and its structural bound

For every example and layer, $0 \le h^l_e \le 1$ and $\sum_e h^l_e = k$,
hence $\lVert d^l(x) \rVert^2 \le k (1 - k/E)$ and, over $L$ layers and
with $w_x \le T_{\max} / \bar T$,

$$
\lVert \lambda\, w_x\, d^{(L,E)}(x) \rVert
\;\le\; \lambda\, \Delta_L,
\qquad
\Delta_L = \frac{T_{\max}}{\bar T} \sqrt{k\, L\, (1 - k/E)} .
$$

The bound is **structural**: it depends only on public constants, never
on the data, and it is attained (route every token to the same $k$
experts at $T_{\max}$). The probe scale and the probe group's clipping
bound are

$$
\lambda = \frac{\rho\, C_g}{\Delta_L},
\qquad
C_h = \lambda\, \Delta_L\, (1 + g) = \rho\, C_g\, (1 + g),
\qquad g = 10^{-3},
$$

with $C_g$ the gradient clipping norm (`clipping_norm`; the `fallback`
bound, or the largest gradient group when several are configured). The
guard $g$ covers the small relative shrink the clipper applies to every
ratio for floating-point safety, so the probe group is a bound that
**never clips**: its clip rate is zero by construction and the release
is an unbiased sum. Two checks keep the bound honest: the mask is
binarised as `attention_mask != 0` (an additive or mixed-sign mask
would otherwise score padding or break the bound), and the number of
captured router-logit tensors must equal the number of routed layers (a
duplicated capture would double the load and the bound).

Opaque's default adjacency is add-or-remove. Under replace-one the
gradient bound doubles to $2 C_g$ and the tight per-layer load bound is
$\sqrt{2 \min(k, E-k)\, L}$ per unit of $\lambda\, T_{\max} / \bar T$.

## One joint release, unchanged accountant

Per-group clipping hands the noise mechanism a `PerGroup` bound with the
gradient groups at $C_g$ and the probe at $C_h$; the sums are divided
by the public expected batch size $\bar B$. Opaque's per-group
allocation puts $\sigma_i = \mathrm{nm} \sqrt{C_i \sum_j C_j}$ on group
$i$, so with $S = C_g + C_h$

$$
\sigma_g = \frac{\mathrm{nm}\, C_g \sqrt{1 + \rho}}{\bar B},
\qquad
\sigma_h = \frac{\mathrm{nm}\, C_h \sqrt{1 + 1/\rho}}{\bar B},
\qquad
\frac{(C_g/\bar B)^2}{\sigma_g^2} + \frac{(C_h/\bar B)^2}{\sigma_h^2}
= \frac{1}{\mathrm{nm}^2}
$$

with equality (the *Mahalanobis identity*). The step releases one
Gaussian on the concatenation of the clipped gradient sum and the probe
sum with diagonal covariance. Whitening by $\Sigma^{-1/2}$ is a
bijection, so the mechanism has the privacy of its whitened form, whose
add-or-remove L2 sensitivity is $\sqrt{C_g^2/\sigma_g^2 +
C_h^2/\sigma_h^2} = 1/\mathrm{nm}$: a sensitivity-one Gaussian at
multiplier $\mathrm{nm}$, exactly what `gaussian(nm)` accounts
([Dong, Roth, Su (2019)](https://arxiv.org/abs/1905.02383), Theorem 2.7;
[Zhu, Dong, Wang (2021)](https://arxiv.org/abs/2106.08567), Definition 7
for the dominating pair). The same joint-release argument is the one
[Andrew et al. (2021)](https://arxiv.org/abs/1905.03871) make in Theorem 1
for adaptive clipping's quantile release. Poisson subsampling amplifies
the **one** joint mechanism because both halves share the sampling coin
([Feldman, Shenfeld (2026)](https://arxiv.org/abs/2602.17284), Lemma 3.2 /
Theorem 3.3, the analysis behind `opaque.dpsgd.accounting.poisson`), and
the dependence of $\tilde f_t$ on previous outputs is charged nothing by
adaptive composition of dominating pairs (Zhu, Dong, Wang, Theorem 10),
exactly like the dependence of $\ell_x$ on $\theta_t$.

So the accountant is literally the one you would build without the
feature:

```python
import opaque.dpsgd.accounting as dpsgd_acc

step = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q)
training = step * num_steps  # identical with and without the probe
```

The naive allocation ($\mathrm{nm}\, C_g$ on the gradient and
$\mathrm{nm}\, C_h$ on the load, independently) would be
`gaussian(nm / sqrt(2))`, not `gaussian(nm)`; Opaque never uses it.
`DPTrainer` calibrates $\mathrm{nm}$ from the target $\varepsilon$, so
$\varepsilon$ is held and the entire price shows up as the
$\sqrt{1+\rho}$ factor on the gradient noise.

Under a matrix mechanism the same `PerGroup` bound is latched by
`mf_gaussian_noise` on the first call, the correlated noise $C^{-1} Z$
is applied leaf-wise (probe included) and the realised per-step
standard deviation on every leaf is the allocated base value times
$\lVert \mathrm{row}_t(C^{-1}) \rVert$. The participation pattern is
shared across groups (same example, same step), so the gradient's
`min_sep` / `max_participations` apply to the probe group and the
whole-run accountant `mf_gaussian(nm, strategy)` is unchanged
([Denisov et al. (2022)](https://arxiv.org/abs/2202.08312), Theorem 2.1,
applied to the per-group-whitened stream).

## Post-processing

Everything after the noise is public. Given the noised probe leaf
$\hat y_t \in \mathbb{R}^{L \times E}$ (already divided by $\bar B$):

1. $\hat d^{(L,E)}_t = \hat y_t / \lambda$, an unbiased estimate of
   $\frac{1}{\bar B} \sum_{x \in B_t} w_x\, d^{(L,E)}(x)$ with per-entry
   noise $\sigma_h / \lambda$.
2. Pool over layers, $\hat d_t = \mathrm{mean}_l\, \hat d^{(L,E)}_t$.
   Averaging divides the per-entry noise by exactly $\sqrt L$, so the
   pooled estimate has the noise of a direct pooled release; the
   per-layer entries are a free diagnostic at $\sqrt L$ times the noise.
3. Project onto the sum-zero subspace, $\hat d_t \leftarrow \hat d_t -
   \mathrm{mean}_e\, \hat d_t$ (the signal already sums to zero; the
   projection removes pure noise and scales its variance by $(E-1)/E$).
4. Filter: a bias-corrected EMA $m_{t} = \beta m_{t-1} + (1-\beta)
   \hat d_t$, $\tilde d_t = m_t / (1 - \beta^t)$ with $\beta = 0.99$
   (`router_load_filter_kind="ema"`), or the mean of the last $W$
   releases (`"window"`, divisor $\min(t, W)$). The noise standard
   deviation of $\tilde d_t$ is known exactly: $s_t = \frac{\sigma_h}
   {\lambda \sqrt L} \cdot \phi_t / (1 - \beta^t)$, where $\phi_t$ is
   the row norm of the filter applied to the noise operator. Under
   DP-SGD $\phi_t$ follows the closed recursion $\phi_t^2 = \beta^2
   \phi_{t-1}^2 + (1-\beta)^2 (E-1)/E$; under a matrix mechanism it is
   $\lVert \mathrm{row}_t(F C^{-1}) \rVert$ computed from the
   instantiated strategy's streaming Toeplitz inverse (never a dense
   solve). Both are tabulated once at setup.
5. Dead zone, then shrinkage (`router_load_shrink=True`): $\tilde d^+ =
   0$ when $\lVert \tilde d_t \rVert^2 < c\, E\, s_t^2$ (`router_load_dead_zone`
   $c = 2$), otherwise the positive-part James-Stein factor $\tilde d_t
   (1 - E s_t^2 / \lVert \tilde d_t \rVert^2)$. Below the dead zone the
   surrogate is identically what the true objective does at balance
   (zero); a pure-noise step passes the dead zone with probability
   $P(\chi^2_{E-1} > 2E)$, about $4 \cdot 10^{-6}$ at $E = 64$ (and
   about $5 \cdot 10^{-2}$ at $E = 8$, so small toy models see the
   dead zone open more often).
6. $\tilde f_{t+1} = \mathrm{clamp}(k/E + \tilde d^+, 0, 1)$, the load
   estimate the next step's surrogate consumes. The clamp is inactive
   unless an expert is dead or hot.
7. Monitors: $D_t = \max_e |\tilde d_{t,e}| / (k/E)$ (the largest
   relative deviation of an expert's load from its share) and its
   per-layer analogue $D^l_t$ from a per-layer filter.

**Decision rule.** The trainer evaluates the monitor on the estimate
that enters the surrogate (after the dead zone and the shrinkage) at
the logging cadence; $D > \tau$ (`router_load_trip`, default 0.5: some
expert carries at least 1.5 times or at most half its share) on two
consecutive logged evaluations sets `router_load/tripped`. In
`monitor_then_surrogate` the trip switches the surrogate coefficient
from 0 to the configured value. The switch changes neither the clipping
bound, nor the noise, nor the accountant, nor the matrix mechanism's
latched bound: choosing the next step's loss from previous releases is
post-processing. At $\rho = 0.02$ the smoothed per-entry noise is about
2.4 % of $k/E$ under DP-SGD, so $\tau = 0.5$ sits more than twenty
standard deviations away from noise.

## Cost table

Preset regime: $\bar B = 256$, $k = 8$, $E = 64$, $C_g = 0.9$, $L = 28$,
$q = 256 / 5 \cdot 10^5$, $T = 15\,625$ steps, $\delta = 10^{-6}$,
$\bar T = T_{\max}$. The DP-SGD baseline `poisson(gaussian(0.5622), q) *
T` gives $\varepsilon = 3.0$. Noise is quoted in units of $k/E = 0.125$
per entry of the pooled estimate; the single-release value is $r_1 =
\mathrm{nm}\, \Delta_h \sqrt{1 + 1/\rho} / (\bar B\, k/E)$ with $\Delta_h
= \sqrt{k(1-k/E)}$ (multiply by $T_{\max}/\bar T$ when $\bar T <
T_{\max}$). Filter factors: DP-SGD EMA 0.99 gives 0.0709, window 256
gives 0.0625; band-MF (64 bands, momentum 0.95) gives 1.431 per step,
0.0249 after EMA 0.99 and 0.0198 after window 256.

| $\rho = C_h / C_g$ | gradient-noise inflation $\sqrt{1+\rho}$ ($\varepsilon$ held) | $\varepsilon$ if $\mathrm{nm}$ were held instead | $r_1$, single release | DP-SGD after EMA 0.99 / window 256 | band-MF after EMA 0.99 at $\mathrm{nm}$ 0.5622 / 1.544 | per-layer entries after EMA 0.99, DP-SGD / MF at 1.544 | dead zone engages below RMS imbalance $\delta$, DP-SGD / MF at 0.5622 / MF at 1.544 |
|---|---|---|---|---|---|---|---|
| 0.50 | 1.225 | 5.305 | 8.1 % | 0.57 % / 0.50 % | 0.20 % / 0.55 % | 3.0 % / 2.9 % | 0.008 / 0.003 / 0.008 |
| 0.20 | 1.095 | 4.100 | 11.4 % | 0.81 % / 0.71 % | 0.28 % / 0.77 % | 4.3 % / 4.1 % | 0.011 / 0.004 / 0.011 |
| 0.10 (router or experts trainable) | 1.049 | 3.590 | 15.4 % | 1.09 % / 0.96 % | 0.38 % / 1.04 % | 5.8 % / 5.5 % | 0.015 / 0.005 / 0.015 |
| 0.05 | 1.025 | 3.306 | 21.3 % | 1.51 % / 1.33 % | 0.53 % / 1.46 % | 8.0 % / 7.7 % | 0.021 / 0.007 / 0.020 |
| **0.02 (default)** | **1.010** | 3.126 | 33.2 % | **2.35 %** / 2.07 % | 0.83 % / **2.28 %** | 12.5 % / 12.1 % | **0.033 / 0.012 / 0.032** |
| 0.01 | 1.005 | 3.064 | 46.7 % | 3.31 % / 2.92 % | 1.16 % / 3.19 % | 17.5 % / 16.9 % | 0.047 / 0.016 / 0.045 |

Notes on reading it:

- **The shipped route holds $\varepsilon$** and pays in gradient noise
  (second column). The third column is the same mechanism family
  re-parametrised: keep $\mathrm{nm}$ and let $\varepsilon$ grow. Its
  values are the tight ones for the same-coin joint Gaussian,
  `poisson(gaussian(nm_eff), q) * T` with $\mathrm{nm}_{\mathrm{eff}} =
  \mathrm{nm} \sqrt{(1+\rho)/(1+2\rho)}$; composing the two halves as
  separate mechanisms through the generic subsampled path gives looser
  numbers (3.234 at $\rho = 0.02$, 3.703 at $\rho = 0.1$) and is not
  what Opaque does.
- **The band-MF columns bracket $\mathrm{nm}_{\mathrm{MF}}$.** The
  b-min-separation amplified band-MF multiplier at $\varepsilon = 3$ lies
  between 0.5622 (the DP-SGD Poisson value, a lower bound) and 1.544,
  the deterministic un-amplified band-MF bound
  (`mf_gaussian(nm, band_mf_strategy(bands=64, momentum=0.95),
  n_steps=15625, min_sep=64)`). At the worst admissible multiplier the
  band-MF smoothed error (2.28 %) equals the DP-SGD one (2.35 %): the
  2.8 times better filtering of anti-correlated noise is what pays for
  the un-amplified multiplier.
- **The per-layer column prices the accuracy of the per-layer entries**
  (diagnostics), not the release: the pooled surrogate pays nothing for
  the $(L, E)$ carrier.
- **Error against the objective.** With a per-coordinate RMS imbalance
  of $\delta \cdot k/E$, the relative error of the auxiliary gradient at
  the default row is about $0.024 / \delta$ (DP-SGD) or $0.008 / \delta$
  to $0.023 / \delta$ (band-MF): 8 to 24 % at $\delta = 0.1$, a
  well-balanced checkpoint, and 3 to 8 % at $\delta = 0.3$. Below the
  dead zone the term is exactly zero, where the true gradient is
  negligible too.

### Choosing $\rho$

$\rho = 0.02$ is the default for the preset regime. At another
$(\bar B, \mathrm{nm}, \beta, k/E)$ apply the rule **"the smallest
$\rho$ whose smoothed load noise is below about 30 % of the expected
imbalance"**: compute the stationary $s_\infty / (k/E)$ for the
candidate $\rho$ (the trainer logs it at setup as the stationary release
noise per entry in units of $k/E$, together with the dead-zone
threshold) and compare it with the imbalance you expect to correct.
Larger $\rho$ buys a less noisy estimate at a larger gradient-noise
inflation; $\rho = 0.1$ is the recommended value when the router or the
experts themselves are trainable, because then the load signal matters
more and the extra 5 % gradient noise is cheap by comparison.

## Modes and defaults

| Field | Default | Meaning |
|---|---|---|
| `router_load_release` | `"off"` | `"monitor"` releases the load and logs the monitor without touching the objective; `"surrogate"` also adds the surrogate with coefficient $\alpha$; `"monitor_then_surrogate"` starts as monitor and switches the surrogate on at the trip. |
| `router_load_ratio` | `0.02` | $\rho = C_h / C_g$. |
| `router_aux_loss_coef` | `None` | $\alpha$; `None` reads the model config's `router_aux_loss_coef` in the surrogate modes and means 0 in `"monitor"`. |
| `router_load_mean_tokens` | `None` | $\bar T$; `None` uses $T_{\max}$. |
| `router_load_max_tokens` | `None` | $T_{\max}$; `None` uses the SFT / DPO `max_length` or the model's `max_position_embeddings`. Longer rows raise. |
| `router_load_filter_kind` / `_beta` / `_window` | `"ema"` / `0.99` / `256` | The smoothing filter. |
| `router_load_shrink` / `router_load_dead_zone` | `True` / `2.0` | Dead zone and James-Stein shrinkage. |
| `router_load_trip` | `0.5` | $\tau$ of the decision rule. |
| `router_aux` | `"pooled"` | `"pooled"` is the Hugging Face objective at the public estimate; `"per_sequence"` uses the example's own $h(x)$ (no release needed, a different regulariser); `"per_layer"` is not supported yet. |
| `router_z_loss_coef` | `0.0` | $\zeta$; per-token separable, no privacy cost. |
| `router_fp32` | `False` | fp32-logit router (below). |
| `packed_sequences` | `None` | Public statement that every row is fully valid (below). |

Any mode other than `"off"` requires `clipping_mode="fixed"` with a
finite `clipping_norm`, no private second moments, a model family whose
backbone records router logits (`config.num_experts`,
`config.num_experts_per_tok` and a top-k router module) and the chunked
causal-LM forward, which the trainer enables by passing
`fused_linear_cross_entropy=True` to the patches (an explicit
`performance_kernels_config={"fused_linear_cross_entropy": False}`
raises). The example scripts pin `surrogate` with $\alpha = 10^{-4}$
(Mellum 2.0's own SFT coefficient) and $\rho = 0.02$ for causal-LM
fine-tuning, and `monitor` with $\rho = 0.02$ for DPO.

## Using it

### DPTrainer

```python
from transformers import AutoModelForCausalLM
from opaque.transformers import DPTrainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("JetBrains/Mellum2-12B-A2.5B-Base")
args = TrainingArguments(
    output_dir="run-0",
    per_device_train_batch_size=256,
    privacy_target_epsilon=3.0,
    clipping_norm=0.9,
    router_load_release="surrogate",
    router_aux_loss_coef=1e-4,
    router_load_ratio=0.02,
    router_load_mean_tokens=640,  # mean length of a public held-out split
)
trainer = DPTrainer(model=model, args=args, train_dataset=train_ds)
trainer.train()
```

The trainer attaches the probe before partitioning the trainable
parameters, builds the two-group clipping bound, tabulates the filter
factors from the configured mechanism (the DP-FTRL strategy when one is
selected), requests router logits from the chunked forward inside the
per-example loss, and registers a callback that consumes the noised
probe leaf after clipping, DDP reduction and noise and before the
optimizer update, zeroing the probe's update so the parameter stays a
public constant zero.

The DPO trainer treats the preference pair as the protected unit: the
chosen and rejected policy forwards are pooled with the common
denominator $L\,(T_c + T_r)$ and weight $w = (T_c + T_r) / \bar
T_{\mathrm{pair}}$ (`router_load_mean_tokens` is read as $\bar
T_{\mathrm{pair}}$ and defaults to $2 T_{\max}$); the reference forward
never records router logits. The TRL converters map a user-set positive
`router_aux_loss_coef` to `router_load_release="surrogate"` (SFT) or
`"monitor"` (DPO) with the coefficient forwarded; pass
`router_load_release="off"` to opt out.

### Logged metrics

With the feature on, `grad_norm` and `clipped_grad_norm` are computed
over the non-probe groups, the probe group is skipped in the per-group
metrics, and the public monitors are logged under `router_load/*`:
`D`, `D_layer_max`, `f_min`, `f_max`, `entropy` (of $\tilde f$
normalised to a distribution), `shrink` (the James-Stein factor, 0
inside the dead zone), `dead_zone`, `noise_std` ($s_t$) and `tripped`.
The per-example loss value is the cross-entropy alone, so the logged
`loss` carries no router-derived term. Nothing derived from private
routing other than the noised probe leaf is ever logged.

### Checkpoints and DDP

Checkpoints carry a sidecar `router_load_state.pt` next to the DP
runtime bundle holding the public post-processing state (the filter
accumulators, $\tilde f$, the step and the trip flag). A resume
continues the filter bit for bit and rejects a change of `ratio`, the
filter kind / beta / window, `dead_zone`, $E$, $k$, $L$, the token
constants or the probe scale (which encodes $C_g$) with a
`CheckpointError`, because a continued filter would otherwise mix
releases of two different mechanisms.

Under DDP the probe leaf is all-reduced with every other leaf before the
noise, and the noise key is shared across ranks, so the release, the
filter state and $\tilde f$ are bit-identical on every rank. On a
checkpoint resume each rank restores the saved sampler cursor onto its
own rank-folded stream key, so cross-rank inclusion coins stay
independent through the resume, as the subsampling amplification
requires.

### Manual functional loops

The mechanism is factored into `opaque.api.transformers.moe_load`, the
helper `examples/train_dpftrl.py` and `examples/train_dpo.py` call at
four seams: `attach_probe` before `make_functional`, `probe_bounds` in
place of the clipping-norm resolution, `router_load_terms` inside the
per-example loss, and `initial_state` / `filter_factors` / `update`
between the noise function and the optimizer update.

```python
from opaque.api.transformers import moe_load
from opaque.dpsgd.clipping import clipped_grad
from opaque.functional import make_functional

probe = moe_load.attach_probe(
    model, num_layers=L, num_experts=E
)  # before make_functional
fmodel, params = make_functional(model, partition_trainable=True)
max_norm, lam = moe_load.probe_bounds(
    clipping_norm,
    params,
    ratio=0.02,
    num_layers=L,
    num_experts=E,
    top_k=k,
    mean_tokens=T_bar,
    max_tokens=T_max,
)


def loss_fn(params, input_ids, attention_mask, labels):
    out = fmodel(
        params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        opaque_router_logits=True,
    )
    return moe_load.router_load_terms(
        out.loss,
        out.router_logits,
        attention_mask,
        params,
        f_tilde=f_tilde,
        alpha=alpha,
        lam=lam,
        top_k=k,
        num_layers=L,
        num_experts=E,
        mean_tokens=T_bar,
    )


grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=max_norm, batch_argnums=(1, 2, 3), normalize_by=B_bar
)
phi = moe_load.filter_factors(
    strategy_or_none, n_steps=T, kind="ema", beta=0.99, window=256, num_experts=E
)
state = moe_load.initial_state(
    num_layers=L,
    num_experts=E,
    top_k=k,
    ratio=0.02,
    lam=lam,
    max_norm=max_norm / B_bar,
    noise_multiplier=nm,
    alpha=alpha,
    mean_tokens=T_bar,
    max_tokens=T_max,
    phi=phi,
)

for batch in loader:
    grads, clip_state = grad_fn(params, *batch, state=clip_state)
    noisy, noise_state = noise_fn(grads, noise_state)
    state = moe_load.update(state, noisy.pytree[probe])  # public post-processing
    noisy.pytree[probe].zero_()  # the probe's update is 0
    f_tilde.copy_(state.f_tilde)  # next step's estimate
    params = optimizer_step(params, noisy)
```

`RouterLoadState` round-trips through `opaque.serialization.state_dict`
/ `from_state_dict`, which is how the loops checkpoint it. Under DDP the
loop must all-reduce the probe with the rest of the pytree and use a
shared noise key, as the trainer does.

## Router precision

The stock Hugging Face router computes its logits with `F.linear` in
the hidden-state dtype and only the softmax in fp32. Under bf16 the
logits carry exact ties, and the top-k set the batch-level aux loss sees
(a bf16 softmax) can differ from the executed one. `router_fp32=True`
(`apply_model_patches(model, router_fp32=True)` outside the trainer)
binds an fp32-logit forward on the family's router instances: logits in
fp32, fp32 softmax and top-k, scores cast back to the hidden dtype. This
is the router precision Mellum 2.0 was pretrained with, exact ties
disappear, and the logits the load statistics derive from are the
executed ones. The cost is about 0.3 % of the routed expert compute.

It is **not** a fix for routing drift: on bf16 hidden states about 1 %
of tokens per layer sit inside a rounding tie regardless of the router
precision, and the fp32 router changes the executed routing function on
those tokens rather than the weights. Adapters served through stock
Hugging Face run bf16 routes, so the option is off by default. Routes
are always computed per example inside `vmap` from the current model;
there is no pinning, and Opaque's bf16 `vmap(grad)` matches Hugging
Face's eager forward at equal precision with zero route flips.

## Grouped MoE and packed sequences

`apply_model_patches` selects the sparse grouped-GEMM experts path by
default wherever the host has one (CUDA with Triton, or
`torch._grouped_mm` with at least 16 experts), independently of the
Triton kernel group; the dense every-token-through-every-expert path
costs roughly eight times the routed expert compute and remains the
fallback for small MoEs, fp32 on CUDA and hosts without a grouped route.
The load statistics are identical on both paths. See [Model patches:
MoE](../../user-guide/huggingface/model-patches.md#mixture-of-experts-moe-models).

`packed_sequences` makes the attention-kernel choice a **public**
property. Without it the vmap-safe mask builder probes the physical
microbatch to decide whether the SDPA causal fast path applies, so one
padded example changes the kernel for every microbatch mate. With the
feature on the trainer sets the policy from the flag (`True`: all rows
are fully valid, fast path allowed; `False` or unset: the mask is
always materialised), so every per-example gradient is a function of
the example and the parameters alone, up to the kernel's own
accumulation order.

## Privacy statement

When `router_load_release` is `monitor`, `surrogate` or
`monitor_then_surrogate`, each step releases one Gaussian (or matrix)
mechanism on the concatenation of the clipped per-example gradients and
the per-example token-weighted centred router-load vectors $\lambda
(T_x / \bar T)(h^{(L,E)}(x) - k/E)$, with per-record bounds $C_g$ and
$C_h = \lambda (T_{\max} / \bar T) \sqrt{k L (1 - k/E)} (1 + g)$; the
two are one mechanism under Opaque's per-group allocation and the
accountant is unchanged (`gaussian(nm)` per step under the stated
sampler, or `mf_gaussian(nm, strategy)` for the horizon). The gradient
noise is inflated by $\sqrt{1 + \rho}$. The load estimate $\tilde f_t$
consumed by the loss and the monitors $D_t$, $D^l_t$ are post-processing
of previous releases. No other quantity derived from private routing is
released: the per-example probe-group norms are excluded from
telemetry, the logged gradient norms are computed over the non-probe
groups, and the per-example loss value carries no router-derived term.
The guarantee holds as run single-process and under DDP, including
checkpoint resume (the release is reduced before the shared-key noise,
and each rank resumes its own sampler stream). Residual
non-per-example effects are floating-point only: attention-kernel
selection is derived from the public `packed_sequences` flag, not from
the batch, and per-example gradients differ from real arithmetic by
accumulation order. The pre-existing un-noised logging of the
batch-mean loss, gradient norms, clip rate and realised batch size is
unchanged by this feature and outside its accounting. If the data are
packed into fixed-length rows, the protected unit is one packed row,
which may hold pieces of several documents and split a long document
across rows; state it in the privacy claim.

Two further hygiene points belong to the run, not to the mechanism: the
calibration pass that picks $C_g$ and measures $\bar T$ from unclipped
per-example norms is itself a private query, so run it on a held-out
split disjoint from the protected training set (or account it); and
`clipping_mode="auto"` / `"adaptive"` and private second moments are
rejected with the feature on because they would rescale or duplicate
the probe's release.

## References

- **Fedus, Zoph, Shazeer (2022)**, [Switch Transformers](https://arxiv.org/abs/2101.03961),
  section 2.2, equations (4) to (6): the load-balancing loss and the
  pooled $f$ / $P$ statistics.
- **Zoph et al. (2022)**, [ST-MoE](https://arxiv.org/abs/2202.08906),
  section 3.1, equation (5): the router z-loss.
- **Kojic et al. (2026)**, [Mellum 2 Technical Report](https://arxiv.org/abs/2605.31268):
  the running-average load (section 3.6), the SFT coefficient
  $10^{-4}$ (section 5.1.2) and the fp32 router (appendix).
- **Andrew, Thakkar, McMahan, Ramaswamy (2021)**, [Differentially Private Learning with Adaptive Clipping](https://arxiv.org/abs/1905.03871),
  Theorem 1: the joint gradient-plus-statistic Gaussian release.
- **Dong, Roth, Su (2019)**, [Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383),
  Theorem 2.7: the Gaussian mechanism's privacy under whitening.
- **Zhu, Dong, Wang (2021)**, [Optimal Accounting of Differential Privacy via Characteristic Function](https://arxiv.org/abs/2106.08567),
  Definition 7 and Theorem 10: dominating pairs and their adaptive
  composition, which charges nothing for the lagged estimate.
- **Feldman, Shenfeld (2026)**, [arXiv:2602.17284](https://arxiv.org/abs/2602.17284),
  Lemma 3.2 and Theorem 3.3: Poisson amplification of the joint
  mechanism, as implemented by `opaque.dpsgd.accounting.poisson`.
- **Denisov, McMahan, Rush, Smith, Thakurta (2022)**, [Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams](https://arxiv.org/abs/2202.08312),
  Theorem 2.1: the matrix mechanism on adaptive streams, applied to the
  per-group-whitened stream.
- **DeepSeek-AI (2024)**, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437),
  section 2.1.2: the per-sequence balance loss that `router_aux="per_sequence"`
  corresponds to.
