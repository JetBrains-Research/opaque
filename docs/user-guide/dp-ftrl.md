# Correlated noise (DP-FTRL)

Opaque's DP-FTRL mechanisms add **correlated** Gaussian noise across
training steps via matrix factorization. Compared to independent noise at
each step (standard DP-SGD), correlated noise reduces variance on the
**cumulative** updates that the optimizer effectively applies, for the same
calibrated privacy guarantee.

The mechanisms live in `opaque.dpftrl`. Mechanism-specific math and API
details live in [Mechanisms](../mechanisms/index.md).

## What Opaque implements

1. A **strategy** object (e.g. `band_mf_strategy`, `bsr_strategy`) holds:
   - coefficients that define a lower-triangular linear map used for noise,
   - **sensitivity** (and sometimes a **Gram matrix**) for privacy accounting,
   - a streaming representation for efficient noise generation.

2. `mf_noise(grad_template, strategy, noise_multiplier=..., key=...)` returns `(noise_fn, state)` that injects noise each step. The mechanism reads the per-step contribution bound from each `ClippedPytree` input (typically produced by `clipped_grad`) and uses `noise_multiplier × bound` as the realized standard deviation; the latched first-call bound must remain constant across steps.

3. **Privacy accounting** uses the same sensitivity (and Gram matrix when needed) as the strategy passed to `mf_noise`. Always build the accounting mechanism from **the same** strategy object you use for noise.

## Two notions of “correct”

Understanding MF in Opaque is easier if you separate:

### 1. Mechanism / DP correctness

The **differential privacy guarantee** applies to the **randomized algorithm you actually run**: the linear map implied by the strategy, the noise scale, and the data collection / subsampling process.

If accounting uses the same sensitivity (and compatible amplification) as the noise function, and the sampler matches what the analysis assumes, the DP statement is about **that** mechanism—not about whether the strategy was numerically optimal for an idealized workload.

### 2. Workload fidelity / utility

Many strategies are **designed or optimized** under a **model** of the optimizer (e.g. Polyak momentum, constant learning rate, workload decay such as BSR’s paper \(\alpha\)). If the **real** training loop differs (different optimizer, schedule, accumulation pattern), **utility** may be worse than the paper’s ideal, even when DP is still valid for the implemented \(C\).

Opaque’s BandMF and BLT pass **workload coefficients** into a Toeplitz optimization problem. For **non-constant** learning rate schedules, the encoded workload is a **Toeplitz surrogate**: it does not exactly match every entry of the time-varying triangular map \(W_{t,s} = \eta_t \beta^{t-s}\) unless \(\eta_t\) is constant. Privacy remains correct for the constructed strategy; the gap is in how tightly the optimization target matches your true discrete-time operator.

See [BandMF — Assumptions and limitations](../mechanisms/band-mf.md#assumptions-and-limitations) for a concise statement.

## Choosing a mechanism (summary)

| Mechanism | Noise pattern | Typical amplification | Extra memory | Notes |
|-----------|---------------|----------------------|--------------|-------|
| [BandMF](../mechanisms/band-mf.md) | Optimized banded Toeplitz | Cyclic Poisson or b-min-sep | \(O(\text{bands})\) | General default; L-BFGS at init |
| [BLT](../mechanisms/blt.md) | Buffered linear Toeplitz | BnB (+ sequential data order) | \(O(\text{buffers})\) | Long runs / multi-epoch |
| [DP-λCGD](../mechanisms/lambda-cgd.md) | PRNG replay, bandwidth 2 | BnB | \(O(1)\) | Minimal memory |
| [BISR](../mechanisms/bisr.md) | Banded inverse square root | BnB | \(O(p)\) | Analytic inverse coefficients |
| [BSR](../mechanisms/bsr.md) | Banded square root (closed form) | BnB | \(O(p)\) | Paper `alpha`, `beta` (kw-only); bind `beta` from SGD momentum or Adam \(\beta_1\) |
| Identity | Independent (DP-SGD style) | Poisson / standard | \(O(1)\) | Baseline via MF API |

## Private Second Moments And MF

Private second-moment estimation uses **two** correlated noise streams
(gradients and squared gradients) with a **joint sensitivity**. It is
**not** the same workload model as single-stream SGD+momentum mechanisms.

When using `mf_noise(..., second_moment=True)`, pass
`second_moment_strategy` explicitly. This keeps first-moment and
second-moment workload choices visible, especially for λCGD where there
is no single universally correct mapping from optimizer β₂ to strategy λ.

## BSR scope

[BSR](../mechanisms/bsr.md) ships **closed-form** coefficients for the Kalinin–Lampert workload in \((\alpha,\beta)\). It does **not** accept arbitrary `lr_schedule` inside the closed-form path. For general schedules or optimizers, use **BandMF** (numerical Toeplitz optimization) or **BLT**.

With private second moments + BSR, the second stream is a second
`bsr_strategy(..., alpha=..., beta=β₂)`; require \(\alpha > \beta\) for
each stream’s \(\beta\).

## LR schedule and workload modeling

**No MF mechanism in Opaque currently implements schedule-aware factorizations** from Kalinin & Andersson (arXiv:2511.17994). That paper proposes constructions provably better under non-constant LR for the schedule workload \(A_\chi = A_1 D\), but the closed-form path only covers **exponential decay** and is not generic enough for general schedules (including warmup).

BandMF and BLT accept `lr_schedule` as a **Toeplitz-surrogate** workload optimization input (utility heuristic, not exact schedule-aware construction). BISR, BSR, and λCGD do **not** accept `lr_schedule`.

## Further reading

- [Noise addition](noise.md) — Gaussian vs MF entry points
- [Mechanisms index](../mechanisms/index.md) — per-mechanism docs
- [Optimizers](optimizers.md) — SGD vs private second-moment AdamW
