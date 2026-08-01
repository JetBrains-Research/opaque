# Privacy amplification by random allocation — design note

**Status:** proposal / research note. Nothing here is implemented yet.

**Scope.** How to extend Opaque's balls-in-bins support in two directions the
Feldman–Shenfeld line of work now makes possible:

1. **PLD accounting** for random allocation, not just Rényi/(ε, δ) bounds.
2. **Correlated-noise DP-FTRL** (matrix mechanisms), not just the identity /
   DP-SGD strategy.

Plus several adjacent wins the same machinery unlocks (§7).

---

## 1. TL;DR

- The paper usually cited here — Feldman & Shenfeld, *Privacy amplification by
  random allocation* ([arXiv:2502.08202]) — is deliberately **analytic**: its
  own discussion section names "does not lend itself to tight accounting of
  composition" as the open problem. Both halves of the question therefore have
  dedicated follow-ups, and neither needs new theory from us:
  - **PLD** → Feldman & Shenfeld, *Efficient privacy loss accounting for
    subsampling and random allocation* ([arXiv:2602.17284]). Gives an exact PLD
    transform for random allocation plus a validity- and tightness-preserving
    numerical algorithm.
  - **Matrix mechanisms** → Schuchardt & Kalinin, *Sampling-Free Privacy
    Accounting for Matrix Mechanisms under Random Allocation*
    ([arXiv:2601.21636]). Gives a deterministic Rényi accountant (exact for
    banded strategies) and a conditional-composition accountant that emits
    **per-step dominating pairs** — which plug straight into Opaque's existing
    PLD composition.
- There is a **scheme distinction** that has to be made explicit before any of
  this lands (§3). Opaque today implements *fixed-assignment balls-in-bins*;
  the Feldman–Shenfeld results are about *k-out-of-t random allocation*. They
  are different mechanisms with different privacy. Measured on the identity
  strategy, re-randomising the assignment each epoch is worth **32–62% lower ε**
  (§3.3) — but it is not free for correlated noise (§3.4).
- Prototypes of the two core algorithms were written and validated against the
  authors' reference implementation and against closed forms (§4.4, §5.4). The
  PLD route came out **8–33% tighter than the best bound in [arXiv:2502.08202]**
  and tighter than Poisson at the matched rate, with a self-certifying
  upper/lower sandwich of 0.4–1%.
- **Start with the identity path (§4.5).** Opaque's *existing* fixed-assignment
  BnB with `IdentityStrategy` turns out to be *exactly* 1-out-of-`b` random
  allocation of a Gaussian at `σ_eff = σ/√E`. So the PLD accountant is a
  deterministic, composable, drop-in replacement for `bnb_mc_pld_identity`
  **with no sampler change and no scheme split** — and it measured 8–27%
  tighter than the Rényi route on the same mechanism. This is the shortest path
  to value and it should lead the work.

---

## 2. Where Opaque stands today

| Piece | Location | Method |
|---|---|---|
| BnB sampler | `opaque/api/dpftrl/sampling/_balls_in_bins.py` | fixed assignment, reused every epoch |
| BnB accountant (correlated) | `amplification/balls_in_bins/monte_carlo.rs` | Monte Carlo over the Choquette-Choo Lemma 3.2 dominating pair, banded Cholesky |
| BnB accountant (identity) | `amplification/balls_in_bins/identity.rs` | specialised importance-sampled MC |
| Python surface | `accounting/dpftrl/amplification/_balls_in_bins.py` | `balls_in_bins(inner, num_bins, n_steps)` |

Three limitations follow from the accountant being Monte Carlo:

1. **The guarantee is probabilistic, not deterministic.** An MC accountant
   certifies "(ε, δ)-DP with high confidence", which is a strictly weaker
   statement than a bound on δ. Recovering a deterministic guarantee requires
   modifying the mechanism with random abstentions.
2. **Cost scales with 1/δ.** Showing (ε, 10⁻⁸)-DP is ~10⁵× more expensive than
   (ε, 10⁻³)-DP. This is the binding constraint at production δ.
3. **It does not compose.** `BallsInBins` is documented as *total* cost —
   `pld()` returns the whole-run PLD and callers are told not to compose it.
   That is a direct consequence of the MC route, not a modelling choice.
4. **Results are not reproducible across machines.** Every MC driver shards work
   by `rayon::current_num_threads()` and seeds per-thread streams as `seed + tid`
   (`monte_carlo.rs:277-290`, `identity.rs:261-274`), so a different core count
   yields a different sample partition and a different ε from the *same* seed.
   For a number that goes into a privacy claim this is an unwelcome property.
5. **The importance-sampling weights are never self-normalised.**
   `weighted_samples_to_pmf` (`identity.rs:179-195`) deposits `w/n` per bucket
   and `Pmf::new` does no renormalisation, so total mass equals 1 only in
   expectation. The hardcoded `_IDENTITY_IS_TILT = 1.0` sits in a safe spot, but
   the margin is thin — the tilt is a load-bearing constant, not a free knob.

Two coverage gaps also exist:

- `BandMfStrategy.gram_matrix()` **raises** — BandMF cannot use balls-in-bins at
  all today, and falls back to Poisson / b-min-sep. BandMF is exactly the
  *p*-banded case where the Schuchardt–Kalinin dynamic program is **exact and
  fastest**, so this is the cheapest gap to close.
- `IdentityStrategy.gram_matrix()` also raises; the identity path is served by a
  bespoke MC primitive rather than the general machinery.

---

## 3. Two schemes, not one

This distinction is the single most important thing in this note, because
mixing the two up silently produces an **unsound** accountant.

### 3.1 Scheme A — fixed-assignment balls-in-bins (what Opaque implements)

Each record draws a bin `i ~ Uniform([b])` **once**, and participates in steps
`i, b+i, 2b+i, …, (E−1)b+i`. `BallsInBinsSampler` says so explicitly, and it
must: the Choquette-Choo Lemma 3.2 dominating pair requires it. Participation
is *perfectly correlated across epochs* — learning the bin in epoch 1 reveals it
for every later epoch. Min separation is exactly `b`, which is what the MF
strategies are tuned for.

This is the scheme in [arXiv:2601.21636] (their "balls-in-bins", `k` epochs,
`b` batches per epoch).

### 3.2 Scheme B — k-out-of-t random allocation (what Feldman–Shenfeld analyse)

Each record picks `k` of the `t` steps **uniformly at random**, independently of
other records. For `k = 1` this is one bin per record with no epoch structure;
for `k > 1` [arXiv:2602.17284] reduces it to a composition of `m_f` copies of
1-out-of-⌊t/k⌋ and `m_c` copies of 1-out-of-⌈t/k⌉ allocation.

Operationally, "re-shuffle the bin assignment at the start of every epoch" is
the `k = E`, `t = N` case, and its accounting is the `E`-fold composition of
1-out-of-`b` allocation.

**These are different mechanisms.** Applying an allocation bound to Scheme A, or
a Lemma 3.2 bound to Scheme B, is not conservative in either direction — it is
simply wrong. Any implementation must therefore pair each sampler with its own
accountant and refuse the cross product.

### 3.3 Scheme B is materially better for uncorrelated noise

Measured through identical Rényi machinery and an identical RDP→(ε, δ)
conversion, so the comparison is apples-to-apples (δ = 10⁻⁸, identity strategy):

| b | epochs | σ | Scheme A (fixed) | Scheme B (re-shuffled) | ε reduction |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 1.0 | 6.270 | 4.205 | 32.9% |
| 16 | 4 | 1.0 | 10.714 | 5.792 | 45.9% |
| 32 | 2 | 1.0 | 5.586 | 3.439 | 38.4% |
| 32 | 4 | 1.0 | 10.021 | 4.658 | 53.5% |
| 32 | 2 | 2.0 | 1.418 | 0.973 | 31.4% |
| 64 | 4 | 2.0 | 2.487 | 1.180 | 52.6% |
| 64 | 8 | 2.0 | 5.025 | 1.936 | 61.5% |

The gap widens with epoch count, which is what the correlation argument
predicts: fixed assignment adds no fresh sampling randomness after epoch 1.

For DP-SGD this is close to free — re-shuffling every epoch is what ordinary
training loops already do, and it is *easier* to implement than pinning an
assignment.

### 3.4 …but not obviously better for correlated noise

Scheme B destroys the min-separation guarantee: a record can land in bin `b` of
one epoch and bin `1` of the next, giving separation 1. Opaque's MF strategies
(`min_sep`, `max_participations`) are tuned against `min_sep = num_bins`, and
their sensitivity is computed over the worst-case participation pattern. Under
Scheme B that worst case collapses to adjacent participation, and the
sensitivity — hence σ — rises.

So the recommendation splits:

- **Identity / DP-SGD:** offer Scheme B and make it the default. Clear win.
- **Correlated MF:** keep Scheme A as the default. Scheme B is only worth it if
  the strategy is re-tuned at `min_sep = 1` and the sensitivity increase is
  smaller than the amplification gain — an empirical question per strategy,
  not something to assume.

---

## 4. Question 1 — PLD accounting for random allocation

### 4.1 The result

[arXiv:2602.17284] introduces a **PLD realization**: a random variable `L` over
`[−∞, ∞]` with `E[e^{−L}] ≤ 1` and no mass at `−∞`. Its **dual** `L̃` is defined
by `f_{L̃}(l) = f_L(−l)·e^{l}`, with an atom at `+∞` of mass `1 − E[e^{−L}]`. If
`L` is the privacy loss of `(P, Q)` then `L̃` is the privacy loss of `(Q, P)`.

Random allocation is then *exactly* a transform of the base PLD:

```
remove:   ψ⃗_t(L)  =  ln( (1/t) · ( e^{L₀} + Σ_{i=1}^{t−1} e^{−L̃ᵢ} ) )
add:      ψ⃖_t(L)  =  −ln( (1/t) · Σ_{i=1}^{t} e^{−Lᵢ} )
```

with `L₀, L₁, …` independent copies. So the PLD of random allocation is a
**convolution of exponentiated PLDs** — the log of a sum of `t` independent
`exp`-PLDs. For the Gaussian the `exp`-PLDs are lognormal, and the allocation
PLD is the log of a sum of `t` lognormals.

Two facts make this practical:

- **Errors do not accumulate.** Unlike composition (where discretisation error
  adds across convolutions), if `Lᵢ ⪯_{(α,βᵢ)} Uᵢ` then
  `ln(e^{L₁}+e^{L₂}) ⪯_{(α, β₁+β₂)} ln(e^{U₁}+e^{U₂})`. The `α` term does not
  grow. This is what makes `t = 10⁶` tractable.
- **`log t` convolutions suffice** via exponentiation by squaring.

Total cost: `O((IQR_{β/t}/α)² · log³ t)`, deterministic, no sampling.

### 4.2 Why this does not drop into Opaque's `Pmf` as-is

Opaque's `Pmf` is an **evenly spaced additive** grid:
`loss = (lower_loss_index + i) · discretization`, with `infinity_mass` and
`negative_infinity_mass` atoms. That representation is right for PLDs and is
what FFT composition needs.

The allocation transform convolves in the **exponentiated** domain. Evenly
spaced bins in the loss become **geometrically** spaced bins in `e^{loss}`, and
the paper is explicit that FFT on a uniform grid is the wrong tool here: for
σ = 1, β = 10⁻¹⁰ the PLD spans ≈ 12.5 but its exponent spans ≈ 950, so a uniform
grid over the exponent either explodes or loses the left tail.

> **A shortcut that does not work.** It is tempting to observe that Opaque's
> arithmetic loss grid *is* a geometric grid in `e^l` with ratio `r = e^Δ`, and
> conclude that exp-PLD convolution is just `numerics::fft::convolve` in index
> space. It is not. Additive convolution in index space gives the law of
> `L₁ + L₂` — that is, of the **product** `e^{L₁}·e^{L₂}` — whereas the
> allocation transform needs the law of the **sum** `e^{L₁} + e^{L₂}`. A sum of
> two geometric-grid values is not on the grid, which is exactly why the paper's
> `conv` is a direct `O(n²)` pass with a `range-renorm` step rather than an FFT.

So this needs a **new representation** next to `Pmf`, not a change to it:

```rust
/// PMF on a geometrically spaced grid: values are
/// [0, v_min·r^0, …, v_min·r^{n-1}, +∞].  Used only for exp-PLDs.
pub struct GeomPmf {
    pub v_min: f64,
    pub ratio: f64,        // r = e^{alpha'}
    pub probs: Vec<f64>,
    pub zero_mass: f64,    // image of the -inf atom under exp
    pub infinity_mass: f64,
}
```

### 4.2b Preconditions Opaque's PLDs do not currently meet

Three of them, all load-bearing:

1. **Stochastic, not hockey-stick, domination.** Every theorem in
   [arXiv:2602.17284] (`def:stochDom`, `lem:mult_tight`, `thm:num_acc_RA`) is
   stated for *first-order stochastic* domination. Opaque's PLDs are built by
   `create_pmf_connect_the_dots_uniform` (`connect_the_dots.rs:82`), which
   inverts a δ(ε) curve — that is tight in the **hockey-stick** sense, which is
   strictly weaker. The paper explicitly cites a pair that hockey-stick-dominates
   without stochastically dominating. **`gaussian_pld` output is therefore not a
   valid input to the allocation transform.** The base PLD must be built by
   `disc_dist` applied to the analytic loss CDF with round-up rounding, which
   *is* stochastically dominating by construction. (The prototype does this; it
   is why its sandwich holds.)
2. **No mass at −∞.** A PLD realization requires `f_L(−∞) = 0`, because the dual
   reweights by `e^{−l}` and `e^{+∞}` is not a number. But `gaussian_pld`,
   `mf_gaussian_pld` and `poisson_gaussian_pld` all end with
   `.with_tail_budgets(tail/2, tail/2)` (`gaussian.rs:36`), and `self_compose`
   moves left-tail mass into `negative_infinity_mass`. `pld_dual` must **assert**
   `negative_infinity_mass == 0` and reject composed or left-truncated inputs.
   (The dual *produces* `+∞` mass; it is undefined *on* `−∞` mass. An earlier
   draft of this note had that backwards.)
3. **`E[e^{−L}] ≤ 1`.** Connect-the-Dots clamps negatives and renormalises
   (`connect_the_dots.rs:145-160`), so whether `Σ pᵢe^{−lᵢ} ≤ 1` survives
   discretisation is an open numerical question. Assert it at the entry point
   and add a test sweeping σ and Δ.

On the dual specifically: mathematically `D(L(P,Q)) = L(Q,P)`, so it is tempting
to read the stored `pmf_add` as the dual of `pmf_remove`. **Do not implement it
that way.** The two are independently round-up-discretised objects, and
[arXiv:2602.17284] shows stochastic domination is *not* preserved by the dual
transform — which is exactly why Alg. `rand-alloc-rem` takes the dual from the
**raw, undiscretised** input, and why the lower-bound variant is undefined
otherwise. The API must therefore take an analytic loss/CDF handle for the base
mechanism, not just a discretised `Pmf`. (For the Gaussian the dual is available
in closed form — `L̃ ~ N(1/(2σ²), 1/σ²)`, the same law as `L` — which makes it the
natural golden test case.)

### 4.3 Proposed Rust surface

```rust
// pld/realization.rs
pub fn pld_dual(pmf: &Pmf) -> Pmf;
pub fn disc_dist(pmf: &Pmf, alpha: f64, beta: f64, dir: Rounding) -> Pmf;

// pld/geom.rs
impl GeomPmf {
    pub fn from_pmf_exp(pmf: &Pmf) -> Self;
    pub fn conv(&self, other: &GeomPmf, dir: Rounding) -> Self;   // O(n²), rayon
    pub fn self_conv(&self, t: usize, dir: Rounding) -> Self;     // exp-by-squaring
    pub fn into_pmf_log(self, scale: f64) -> Pmf;
}

// amplification/random_allocation.rs
pub fn random_allocation_pld(
    base: &PrivacyLossDistribution,
    t: usize, k: usize,
    alpha: f64, beta: f64,
    dir: Rounding,                 // Upper => valid bound; Lower => audit sandwich
    config: &DiscretizationConfig,
) -> Result<PrivacyLossDistribution>;

// generic Poisson subsampling on an arbitrary PLD (see §7.1)
pub fn subsample_pld(base: &PrivacyLossDistribution, lambda: f64)
    -> Result<PrivacyLossDistribution>;
```

The `O(n²)` direct convolution is precisely why this belongs in the Rust core:
at production settings `n ≈ 2·10⁴`, so one convolution is ~4·10⁸ fused
multiply-adds — trivial for `rayon`, hopeless in Python.

Three `Pmf` details will bite an implementer:

- `Pmf::new` (`dense.rs:113`) hardcodes `negative_infinity_mass = 0.0` and there
  is **no constructor that sets it**. The dual transform produces −∞ mass, so it
  must build the result with a struct literal (all fields are `pub`) or a new
  builder must be added.
- `Pmf::new` performs **zero validation** — normalisation, ordering and
  contiguity are all caller-enforced. New transforms must assert their own
  invariants.
- `pmf_beta_asymmetric` (`metrics.rs:289-298`) deliberately *drops*
  `negative_infinity_mass` while the symmetric path keeps it (`metrics.rs:208`).
  A PLD carrying −∞ mass will therefore read differently through the two paths.
  This needs reconciling before the dual transform is exposed.

One numerical trap: the dual reweights each bucket by `e^{−l}`, which
**amplifies the small negative probabilities FFT composition leaves behind**.
Clamp to zero before exponentiating.

Because the output is a genuine `PrivacyLossDistribution`, the `k > 1` reduction
and multi-epoch composition reuse Opaque's **existing** composition path. That
removes the "do NOT compose externally" restriction that the MC accountant
forces, so `RandomAllocation` can implement `_pld_at_horizon` honestly and
participate in calibration.

### 4.4 Validation

A Python prototype of Algorithms `disc-dist`, `PLD-dual`, `range-renorm`,
`conv`, `self-conv`, `rand-alloc-rem` and `rand-alloc-add` was written and
checked (δ = 10⁻⁸, k = 1, Gaussian):

| σ | t | RA-PLD remove (upper) | RA-PLD remove (lower) | Poisson 1/t | FS25 `direct` | FS25 `decomposition` |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 8 | **3.726** | 3.711 | 5.102 | 4.036 | 5.635 |
| 1.0 | 64 | **1.806** | 1.792 | 1.938 | 2.298 | 2.409 |
| 2.0 | 64 | **0.345** | 0.315 | 0.389 | 0.512 | 0.570 |
| 1.0 | 128 | **1.270** | 1.253 | 1.330 | 1.893 | 1.748 |

`direct` / `decomposition` are the authors' own reference implementation
(`github.com/moshenfeld/random_allocation`, v1.0.5, MIT), which implements the
[arXiv:2502.08202] bounds. Reading:

- The PLD bound is **below every reference upper bound**, by 8–33%. Expected:
  PLD accounting is tighter than the analytic and RDP routes.
- The PLD bound is **below Poisson at the matched rate** in all four regimes,
  matching the paper's claim that allocation can beat Poisson.
- The **upper/lower sandwich is 0.4–1%** (9% in the σ = 2 case, which just needs
  a finer α). This is self-certifying: the algorithm bounds its own error
  without any reference to compare against — the property the MC accountant
  cannot offer.
- The add direction is uniformly looser than remove, consistent with both
  papers.

The reference implementation contains **no PLD method** — it predates
[arXiv:2602.17284]. So this route has no public implementation to copy, and the
sandwich is the correctness argument.

### 4.5 The identity path is already a random allocation — start here

§3 says Scheme A and Scheme B are different mechanisms, and they are. But for
`IdentityStrategy` specifically, Scheme A **collapses onto a k = 1 allocation**.

With `C = I`, the mixture mean `mᵢ = Σ_{j=0}^{E−1} |C|[:, b·j+i]` is the
indicator of `{i, b+i, …, (E−1)b+i}`, so `‖mᵢ‖² = E` and `⟨mᵢ, mⱼ⟩ = 0` for
`i ≠ j` — the Gram is exactly `E·I_b`. Projecting onto `span{mᵢ}` and rescaling
by `1/√E`:

```
P ≅ (1/b) Σᵢ N(eᵢ, (σ²/E)·I_b),   Q ≅ N(0, (σ²/E)·I_b)
```

which is *precisely* the k = 1 random-allocation dominating pair `(P̄_b, Q^b)`
for a Gaussian randomizer with `σ_eff = σ/√E`. Opaque's own
`amplification/balls_in_bins/identity.rs` header states the `G = E·I_b` /
`σ_eff` structure; the reduction to random allocation is the step not taken.

Verified numerically — both routes on the same mechanism, δ = 10⁻⁸:

| b | E | σ | σ_eff | RA-PLD at (t = b, σ_eff) | Rényi at G = E·I_b |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 1.0 | 0.7071 | **5.769** | 6.270 |
| 16 | 4 | 1.0 | 0.5000 | **9.984** | 10.714 |
| 32 | 2 | 1.0 | 0.7071 | **5.079** | 5.587 |
| 32 | 4 | 2.0 | 1.0000 | **2.410** | 2.829 |
| 64 | 4 | 2.0 | 1.0000 | **1.806** | 2.487 |

Three consequences, all good:

1. `random_allocation_pld(gaussian_pld(σ/√E), t = num_bins, k = 1)` is a
   **drop-in replacement for `bnb_mc_pld_identity`** — deterministic,
   composable, no 1/δ cost, and 8–27% tighter than the Rényi route. It needs
   **no sampler change and no Scheme A/B split**, so it can ship ahead of
   everything in §3.
2. §6 claimed no external oracle exists. That is wrong for this path: **every
   k = 1 bound in [arXiv:2502.08202], and the whole MIT reference
   implementation, applies to Opaque's existing identity BnB** at
   `(t = b, σ → σ/√E)`. That is a real, independent cross-check.
3. It does not extend to correlated `C`, where the `mᵢ` are neither orthogonal
   nor equal-norm. Those still need §5.

---

## 5. Question 2 — correlated-noise DP-FTRL

### 5.1 The result

For the Lemma 3.2 dominating pair `P = (1/b)Σᵢ N(mᵢ, σ²I)`, `Q = N(0, σ²I)`
with Gram `G_{ij} = ⟨mᵢ, mⱼ⟩`, the remove-direction Rényi divergence is

```
R_α(P‖Q) = [ log Σ_{r ∈ [b]^α} exp( Σ_{j₁≠j₂} G_{r_{j₁}, r_{j₂}} / (2σ²) )
             − α·log b ] / (α − 1)
```

Grouping by the count vector `c` (`cᵢ = #{j : r_j = i}`) gives the form that
actually computes:

```
Σ_r (…) = Σ_{c : Σcᵢ = α} multinomial(α; c) · exp( (cᵀGc − Σᵢ cᵢG_{ii}) / (2σ²) )
```

Evaluating this is **#P-complete** for general `G`. But when `C` is *p*-banded,
`G` becomes **cyclically** *p*-banded (banded with the corner blocks filled),
`cᵀGc` only couples `cᵢ` with `c_{i±(p−1)}`, and a dynamic program over bins —
carrying the last `p−1` counts and the running total — evaluates it in
`O(b·p·α^{2p})`. For `p = 1` (DP-SGD) that is `O(bα²)`, versus the `O(2^α)` of
the [arXiv:2502.08202] integer-partition formula.

For near-banded `G`, the elementwise bound `G ≤ G^{(p)} + τE` with
`τ = max_{min(|i−j|, b−|i−j|) ≥ p} G_{ij}` gives a valid upper bound by adding
`ατ/(2σ²)`.

The add direction has a closed form:
`R_α(Q‖P) ≤ (1/(2bσ²))Σⱼ G_{jj} + ((α−1)/(2b²σ²))ΣᵢΣⱼ G_{ij}`.

> **Note on the paper's statement.** As printed, the trailing term reads
> `− α·log b / (1 − α)`, i.e. `+α·log b/(α − 1)`. Deriving
> `E_Q[(P/Q)^α] = b^{−α} Σ_r exp(·)` gives the opposite sign, and the `b = 1`
> case settles it: only the negative sign reproduces the Gaussian closed form
> `R_α = α‖m‖²/(2σ²)`, which the prototype confirms to 10 decimal places
> (§5.4). We should implement the derived sign.

### 5.2 How Opaque's strategies map onto it

| Strategy | Structure of `C` | Gram over bins | Route |
|---|---|---|---|
| `IdentityStrategy` | `C = I` (p = 1) | `G = E·I_b`, exactly diagonal | **exact DP**, `O(bα²)` |
| `BandMfStrategy` | *p*-banded Toeplitz | cyclically *p*-banded | **exact DP** — closes a gap, BnB is unsupported today |
| `BsrStrategy` | banded square root, explicit `bandwidth` | cyclically banded | **exact DP** |
| `BisrStrategy` | banded *inverse* square root | near-banded, fast decay | truncation + `ατ/(2σ²)` |
| `LambdaCgdStrategy` | AR(1)-like, `G_{ij} ∼ E·λ^{\|i−j\|}` | near-banded, decay set by λ | truncation — **but see §5.4** |
| `BltStrategy` | Toeplitz, buffer decay | near-banded | truncation |

The banded cases are strictly better served than by MC. The near-banded cases
need the caveat in §5.4.

### 5.3 Conditional composition — the part that unifies both questions

The Rényi route is lossy for small ε (the RDP→(ε, δ) conversion captures large
deviations). The second accountant in [arXiv:2601.21636] fixes that, and it is
the piece that matters most architecturally for Opaque.

The obstruction to PLD accounting for matrix mechanisms is that the Lemma 3.2
dominating pair **does not factorise** across steps — noise correlation and a
fixed participation count create shared randomness. Conditional composition
(Choquette-Choo et al. 2023 Thm 3.1; "posterior sampling" in Feldman–Shenfeld)
replaces it with a *factorising* pair at a bounded cost:

```
H_γ(P‖Q) ≤ H_γ( ⊗ₙ P⁽ⁿ⁾ ‖ ⊗ₙ Q⁽ⁿ⁾ ) + δ_E
```

where `δ_E` is the probability that the per-step pair fails to dominate. Their
Algorithm returns exactly those `(P⁽ⁿ⁾, Q⁽ⁿ⁾)` given a significance `β = δ_E/N`,
picking mixture weights via a reverse-hazard dominance criterion and analytic
AM-GM tail bounds.

**This is the bridge.** `⊗ₙ P⁽ⁿ⁾` is a product of per-step Gaussian mixtures —
which is exactly the input Opaque's existing PLD composition already consumes.
So the correlated-MF path becomes:

```
strategy → mixture means m_i → per-step dominating pairs (Alg. cond-comp)
        → per-step PLDs → existing FFT composition → PLD → ε(δ)
```

with `δ_E` added to the final δ. No new composition machinery, and it produces a
**composable, horizon-truncatable** PLD — which is what `_pld_at_horizon` wants
and what MC cannot give.

Their amortisation result matters for Opaque specifically: most of the cost of
finding the thresholds `τᵢ` is linear in σ, so re-evaluating at a new σ drops
from `O(N²b²)` to `O(Nb²)`. Opaque's `calibration.py` searches over σ; this
makes that search roughly `N`× cheaper.

**Two caveats — this is the least-derisked part of the note.**

- **Unlike §4 and §5.1, none of this was prototyped.** §5.4's validation covers
  the *Rényi* accountant only. Treat §5.3 as a promising direction, not a
  verified one.
- **Opaque has no mixture-Gaussian PLD primitive.** The per-step pairs are
  `(Σᵢ pᵢ N(μ_{i,n}, σ²), N(0, σ²))` with `b` arbitrary weights and `b`
  arbitrary scalar means. Nothing in the codebase computes that:
  `mf_gaussian_pld` is a single Gaussian, and `parallel_poisson.rs`'s mixture
  has *collinear* means with binomial weights. [arXiv:2601.21636] leans on
  Google `dp_accounting`'s `MixtureGaussianPrivacyLoss` here. So phase 5
  implicitly includes a new
  `mixture_gaussian_get_delta(ε, adjacency, &weights, &means, σ)` fed through
  `discretize_asymmetric_mechanism`. That is a substantial work item in its own
  right and should be scoped separately.

Their Algorithm 2 as printed also needs care before implementation — the loop
computes upper bounds `λ̄ᵢ` but the return line reads `λᵢ`, and the mixtures are
written with `(n−1)`-dimensional prefix means where the per-step definition
calls for the scalar step-`n` means. Reconcile against the proofs rather than
transcribing.

### 5.4 Validation

A prototype of the Rényi accountant (brute force, cyclic-banded DP, add-direction
closed form, RDP→(ε, δ)) was checked:

- **`b = 1` reduces to the Gaussian closed form** — `R_α = α‖m‖²/(2σ²)` to 10
  decimal places across `(σ, ‖m‖², α)` combinations. This is what settles the
  sign question in §5.1.
- **The DP reproduces brute force exactly** on cyclically *p*-banded `G` for
  `(b, p, α)` ∈ {(5,1,3), (5,1,4), (6,2,3), (6,2,4), (7,2,3), (8,3,3)},
  agreeing to < 10⁻⁸.
- **Truncation is sound** — for λ-CGD the truncated bound stayed above the exact
  value in every configuration tested.

**The Gram really is cyclically banded.** Re-deriving Opaque's λ-CGD Gram from
`gram_matrix.rs` gives
`G_{ij} = Σ_{p,q} λ^{|b(p−q)+(i−j)|} ≈ E·λ^{d} + (E−1)·λ^{b−d}` for `d = |i−j|` —
a *cyclic* structure, not a plain AR(1) band. At λ = 0.9, b = 100, E = 4 that
predicts `G[0][0] = 4.0002`, `G[0][1] = 3.6002`, `G[0][50] = 0.0361`,
`G[0][99] = 2.7002`, which reproduces the values measured from the Rust builder
exactly. The corner is **67% of the diagonal** while the mid-row entry is under
1%. This is precisely the structure [arXiv:2601.21636] assumes, and it confirms
the strategy table above — but see §5.5, because the same fact has consequences
for the *existing* MC accountant.

But one **negative result worth acting on**: with that corrected Gram, the
truncation bound still *degrades* as the retained bandwidth grows once λ is
large (b = 8, E = 2, σ = 1, α = 3):

| λ | exact | p = 1 | p = 2 | p = 3 |
|---:|---:|---:|---:|---:|
| 0.3 | 1.239 | 2.020 | 1.486 | 1.317 |
| 0.5 | 1.447 | 2.643 | 2.124 | 1.834 |
| 0.7 | 1.982 | 3.538 | 3.377 | 3.161 |
| 0.9 | 3.740 | 6.247 | 6.769 | 6.914 |

At λ ≤ 0.7 the bound tightens with `p`, as intended. At λ = 0.9 it gets *worse*,
because `τ` decays too slowly for the `ατ/(2σ²)` term to pay for the larger
retained band. Opaque's own MC docstring cites λ = 0.9, b = 1953 as a realistic
configuration — squarely in the bad regime. At b = 100, E = 4, λ = 0.9 you would
need `p ≈ 20–40` to get `τ` down to 0.49–0.065, and the DP cost is `α^{2p}`.

**Two conclusions:**

1. The Rényi route should **not** be advertised as a general replacement for MC
   on slowly-decaying Grams. For those, the conditional-composition route (§5.3)
   or the existing MC path remains necessary.
2. There is a hard **tractability ceiling on `p`**. With `O(b·p·α^{2p})` and a
   typical α in the low tens, only very small bandwidths are affordable — `p = 1`
   (identity/DP-SGD) is an outright win, and `p = 2–4` is plausible, but
   BSR/BandMF at the `p = 64` used in the paper's own experiments is not
   obviously reachable by the DP as stated. This must be resolved against the
   paper's experimental section before committing to phase 4 (§9, open question 5).

### 5.5 A pre-existing fragility this analysis surfaced

The cyclic structure of the Gram (§5.4) interacts badly with two independent
heuristics already in the codebase:

- `BandedCholesky::compute` (`monte_carlo.rs:61-127`) estimates a bandwidth by
  walking `d = 1, 2, …` and **breaking at the first `d` whose entries all fall
  below `1e-6 · max_diag`**. It only ever looks at `|i − j|`, never at the cyclic
  distance, so a Gram that decays into the middle of the row and then *rises
  again at the corner* stops the scan early.
- `lambda_cgd_gram_matrix` (`gram_matrix.rs:235-237`) drops cross-epoch terms —
  and therefore the wrap — when `λ^b < 1e-15`.

Most of the time these are accidentally complementary: when the wrap matters,
the scan runs to the end and returns full bandwidth; when the scan stops early,
`skip_cross_epoch` has already removed the wrap. Every configuration in the
docstring's own regime (λ = 0.9, b = 1953) is safe.

But the two thresholds — `1e-15` on `λ^b` and `1e-6` on the Gram entries — are
unrelated, and there is a window between them. Re-implementing the detection
loop and sweeping λ ∈ [0.7, 0.85], b ∈ [100, 240], E ∈ {2, 4, 8} finds **51
configurations** where the wrap is retained but the detected bandwidth truncates
it:

| λ | b | E | λ^b | `skip_cross_epoch` | est_bw | bw | corner / diagonal |
|---:|---:|---:|---:|:--:|---:|---:|---:|
| 0.72 | 100 | 4 | 5.4e−15 | false | 42 | 94 | **0.540** |
| 0.72 | 105 | 8 | 1.0e−15 | false | 42 | 94 | **0.630** |
| 0.75 | 110 | 4 | 1.8e−14 | false | 48 | 106 | **0.563** |

In these the banded Cholesky zeroes entries carrying more than a third of the
diagonal magnitude, so the sampler draws `u ~ N(m_i, σ²·LLᵀ)` with `LLᵀ ≠ G`.
That is not conservative in either direction — it is simply the wrong
distribution, and the resulting ε could be too small.

Two caveats on this finding: it comes from re-implementing the detection logic
in Python, not from running the Rust; and the `est_bw·2 + 10` safety margin
plus the `b − 1` cap rescue most nearby settings. **It should be confirmed
against the real builder before being called a bug.** Independently of the
random-allocation work, two cheap defensive fixes are worth making:

1. Measure bandwidth by **cyclic** distance `min(|i−j|, b−|i−j|)`, and stop
   subsampling rows (`step_by((b/20).max(1))`, `monte_carlo.rs:71`, checks ~20
   entries per diagonal and can under-detect on its own).
2. Assert the dropped mass is negligible — e.g. `max|G − LLᵀ| ≤ tol · max_diag`
   — rather than trusting the heuristic.

This is also an argument for the deterministic accountants: a dynamic program
over an explicitly *cyclically* banded Gram cannot make this class of mistake,
because the cyclic structure is in the data model rather than in a heuristic.

---

## 6. Validating an implementation

For the **identity path** an external oracle does exist (§4.5): the MIT reference
implementation's `allocation_epsilon_*` functions apply directly at
`(t = num_bins, σ → σ/√E, k = 1)`. Use it.

Everywhere else there is no reference implementation, so the test strategy has
to be self-supporting:

1. **Sandwich.** Both algorithms have upper- and lower-bound variants
   (`Rounding::Upper` / `Lower`). Every test asserts `lower ≤ upper` and that the
   gap shrinks as α shrinks. This certifies without an oracle.
2. **Degenerate cases.** `t = 1` must return the base PLD; `b = 1` must return
   the Gaussian closed form; `G` diagonal must agree with the
   integer-partition formula of [arXiv:2502.08202] Thm 4.8.
3. **Cross-check against the reference bounds.** The authors' package (MIT) can
   be a dev dependency; every new bound must sit at or below
   `allocation_epsilon_direct` and `allocation_epsilon_decomposition`.
4. **Cross-check against the existing MC accountant.** For configurations where
   MC is affordable (δ ≈ 10⁻³), the deterministic bound must lie above the MC
   estimate but within its confidence band's slack.
5. **Monotonicity**, matching the invariants already asserted in
   `packages/opaque-dpftrl/tests/accounting/`: ε decreasing in σ, increasing in
   epochs, increasing in k.

---

## 7. Other things the same machinery unlocks

### 7.1 Generic subsampling on an arbitrary PLD (the sleeper result)

[arXiv:2602.17284] Thm `PLD_subsam` gives Poisson subsampling as a transform of
the PLD *realization*:

```
remove:  f_{φ⃗_λ(L)}(l) = λ·f_L(φ_λ(l)) + (1−λ)·f_{−L̃}(φ_λ(l))
add:     f_{φ⃖_λ(L)}(l) = f_L(−φ_λ(−l)),        φ_λ(l) = ln(1 + (e^l − 1)/λ)
```

Every mainstream accounting library — Google's `dp_accounting`, Opacus, Meta's —
implements subsampled-Gaussian and subsampled-Laplace via *closed-form,
mechanism-specific* PLDs. This theorem removes that restriction: subsampling
becomes a `O(support)` transform of any PLD, so **any** mechanism with a
dominating PLD gets amplification for free.

For Opaque concretely: `AdaClip` (`transformations/adaclip.rs`) and any future
non-Gaussian noise get Poisson amplification without bespoke analysis. This is
arguably the highest value-per-line item in this note and is independent of
random allocation entirely.

### 7.2 Deterministic replacement for b-min-sep

`amplification/b_min_sep/mc.rs` is Monte Carlo for the same reasons BnB is. The
conditional-composition construction applies to the same class of
non-factorising dominating pairs, so the same per-step-pair trick should
deterministically bound b-min-sep too. Worth scoping after §5.3 lands.

### 7.3 Auditing gets a two-sided bracket

`opaque-auditing` currently validates ε empirically. The lower-bound variant of
the allocation PLD gives a *numerical* lower bound on the same quantity, so
audits can be checked against a bracket `[ε_lower, ε_upper]` rather than a single
number. An empirical estimate outside that bracket is an unambiguous bug signal.

### 7.4 Calibration speedup

§5.3's σ-amortisation makes the conditional-composition accountant roughly `N`×
cheaper across a calibration sweep. Combined with §4's removal of the
"non-composable" restriction, `calibration.py` can calibrate BnB configurations
that are impractical today.

### 7.5 Close the BandMF gap

`BandMfStrategy.gram_matrix()` raising is the cheapest concrete win here:
BandMF is genuinely *p*-banded, so the Rényi DP is exact and fast, and BnB
support for it needs the Gram plus a dispatch entry.

---

## 8. Suggested sequencing

| Phase | Content | Depends on |
|---|---|---|
| 1 | `GeomPmf` + `pld_dual` + `disc_dist` + conv/self-conv in Rust, with the sandwich tests and the §4.2b precondition assertions | — |
| 2 | **Identity path (§4.5):** `random_allocation_pld` at `(t = b, σ/√E, k = 1)` as a drop-in for `bnb_mc_pld_identity`, cross-checked against the MIT reference impl. **No sampler change.** | 1 |
| 3 | `subsample_pld` — generic subsampling transform (§7.1) | 1 |
| 4 | Rényi DP accountant for banded Grams (`p = 1` first); add Gram support to BandMF | — |
| 5 | Split the sampler API so Scheme A and Scheme B are distinct types, each accountant refusing the other's sampler (§3); general Scheme B accounting with the `k > 1` reduction | 1, 2 |
| 6 | Mixture-Gaussian PLD primitive (§5.3) | — |
| 7 | Conditional-composition accountant → per-step pairs → existing composition | 4, 6 |
| 8 | Retire MC as the default where 2/4/7 dominate; keep it for slowly-decaying Grams (§5.4) | 2, 4, 7 |

Phase 2 is the one to do first: it is the shortest path to a deterministic,
composable accountant, it replaces an existing MC primitive rather than adding
a new mechanism, and it comes with an external oracle. Phases 1–3, 4 and 6 are
otherwise independent.

Independently of all of this, the two defensive fixes in §5.5 are worth making
on their own schedule.

---

## 9. Open questions

1. **Is Scheme B actually worse for correlated MF?** §3.4 argues it is, but the
   trade-off (amplification gain vs sensitivity loss at `min_sep = 1`) has not
   been measured per strategy. This should be settled empirically before
   choosing defaults.
2. **`δ_E` budgeting.** Conditional composition spends δ on the non-dominance
   event. How that is exposed — an implicit split, or a user-visible knob — is
   an API decision with correctness consequences.
3. **Choosing (α, β).** These are accuracy knobs, not privacy parameters. They
   should follow the precedent set for `_IDENTITY_IS_TILT` and be fixed
   internally with a documented default rather than exposed.
4. **BLT's retuning behaviour.** `_pld_at_horizon` already special-cases BLT
   because re-running L-BFGS at a shorter horizon changes the mechanism. Any new
   accountant must preserve that pinning argument.
5. **The `α^{2p}` ceiling** (§5.4). Resolve what `p` means in the stated
   complexity — the bandwidth of `C` in training steps, or the cyclic bandwidth
   of `G` in bins — and how [arXiv:2601.21636] runs its own `p = 64` experiments,
   before committing to the Rényi DP for anything beyond `p = 1`.

---

## References

- [arXiv:2502.08202] Feldman & Shenfeld, *Privacy amplification by random allocation*.
- [arXiv:2602.17284] Feldman & Shenfeld, *Efficient privacy loss accounting for subsampling and random allocation*.
- [arXiv:2601.21636] Schuchardt & Kalinin, *Sampling-Free Privacy Accounting for Matrix Mechanisms under Random Allocation*.
- Choquette-Choo, Ganesh, Haque, Steinke & Thakurta, *Near-exact privacy amplification for matrix mechanisms*, arXiv:2410.06266.
- Choquette-Choo, Ganesh, Steinke & Thakurta, *Privacy amplification for matrix mechanisms*, ICLR 2024.
- Chua et al., *Balls-and-bins sampling for DP-SGD*, AISTATS 2025.
- Zhu, Dong & Wang, *Optimal accounting of differential privacy via characteristic function*, AISTATS 2022.
- Reference implementation: `github.com/moshenfeld/random_allocation` (MIT).

[arXiv:2502.08202]: https://arxiv.org/abs/2502.08202
[arXiv:2602.17284]: https://arxiv.org/abs/2602.17284
[arXiv:2601.21636]: https://arxiv.org/abs/2601.21636
