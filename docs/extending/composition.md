# Composition

This page describes the small set of types that flow between
Opaque's clipping, noise, optimisation, and accounting stages today,
and what shape a new extension needs to emit if it wants to plug into
that flow without code on either side of the seam knowing the
extension exists.

Like the rest of `docs/extending/`, this is descriptive of today's
design rather than a versioned contract. The patterns documented
here are what the current code does and what extensions are expected
to follow; the patterns themselves may evolve.

## The types your code emits

Three types in `opaque/api/engine/types.py` form the seam between
the privacy-bookkeeping side and the rest of the pipeline:

- **`ClippedPytree(pytree, max_norm)`** — what a clipping rule or
  sensitivity oracle returns. `pytree` is the (already-summed,
  already-clipped) per-record contribution. `max_norm` is the public
  bound on one record's L2 contribution; it's the only privacy
  parameter that flows downstream.
- **`NoisedPytree(pytree, max_norm, noise_stddev)`** — what a noise
  mechanism returns. Extends `ClippedPytree` with the realised
  noise stddev (kept for telemetry / debugging; not used by the
  privacy accounting).
- **`PerGroup(groups, values)`** — a dict-like that flows through
  the pipeline as a per-parameter-group scalar. Used as the
  `max_norm` field when clipping is per-group, and reused for
  per-group noise stddevs downstream.

What flows downstream of a clipping rule today is *just* a
`ClippedPytree`. The route that produced its `max_norm` — fixed
threshold, AUTO-S smooth-min, adaptive moving threshold, an
architecturally-derived Lipschitz constant — doesn't enter into
what the noise mechanism, the optimiser, or the accountant does.
That's what makes adding a new sensitivity source today a local
change.

## `max_norm` as the privacy parameter

`max_norm` is the L2 sensitivity of one record's contribution to the
sum. Today the rest of Opaque trusts this field — the noise
mechanism reads it to set σ, the accountant uses it implicitly via
the noise multiplier, and the optimiser doesn't touch it but
preserves it on each output. Things to know:

- Scalar `max_norm` produces uniform-stddev Gaussian noise:
  `σ = noise_multiplier · max_norm`.
- `PerGroup` `max_norm` lets per-group noise be allocated
  non-uniformly. See "`PerGroup` and MSE-optimal allocation" below.
- A `ClippedPytree.sensitivity` property collapses either case to a
  scalar effective L2 sensitivity (`max_norm.effective` for
  `PerGroup`, or `float(max_norm)` for scalar). This is what the
  accountant ends up consuming.
- Arithmetic on `ClippedPytree` is intentionally narrow: scalar
  multiplication / division / negation preserve the clipped-query
  interpretation; addition, subtraction, power, and reverse
  division raise. The intent is that a downstream component can't
  silently corrupt `max_norm` semantics — anything fancier than
  scaling should operate on `.pytree` and reconstruct an explicit
  `max_norm`.

## `PerGroup` and MSE-optimal allocation

`PerGroup` carries pre-resolved (parameter-key → group-name) and
(group-name → value) mappings, with the arithmetic to flow
naturally through the pipeline. When a per-group `max_norm` reaches
a Gaussian noise mechanism, the mechanism today applies the
MSE-optimal Mahalanobis allocation
`σᵢ = noise_multiplier · √(Cᵢ · ΣⱼCⱼ)` directly (see
`per_group_noise_stddev` in `opaque/api/engine/noise_allocation.py`).
The same σ values are available as a preview via
`ClippedPytree.noise_stddev_for(noise_multiplier=…, allocation="optimal")`
on the input — useful when you want to see what the mechanism will
apply without actually running it. The privacy accounting stays
`gaussian(noise_multiplier)` either way: there is no composition
penalty for the per-group allocation because the Mahalanobis
constraint is satisfied with equality.

For the user-facing treatment of the same idea, see the per-group
discussion in [Per-Example Gradient Clipping](../user-guide/clipping.md);
that page has the worked tuning intuitions. From a contributor
perspective the thing to know is that emitting
`ClippedPytree(max_norm=PerGroup(...))` is enough — the downstream
allocation is automatic.

## DP-FTRL and constant `max_norm`

Matrix-factorisation mechanisms (DP-FTRL) require the per-record
sensitivity bound to be constant across the run, since the
correlation matrix is fixed at calibration time. Today the MF
dispatcher enforces this by latching the `max_norm` from the first
step and raising if it differs on a later step.

The check is in
`opaque/api/dpftrl/noise/_dispatcher.py:251`
(`_validate_constant_max_norm`). Practical implications for a new
extension:

- Fixed clipping and AUTO-S already pass: the bound is `R` (or
  `R, R²` for paired streams) and constant by construction.
- Adaptive clipping, by construction, does *not* pass — its
  threshold moves per step. Adaptive-clipping users stay on DP-SGD
  for this reason.
- A sensitivity-oracle extension whose `max_norm` is derived from
  the architecture (rather than from a norm computation) is
  typically constant across steps, and so composes with DP-FTRL
  without further work. As an illustration, a Lipschitz-layer
  wheel that bakes the per-layer Lipschitz constant into the
  architecture would emit a fixed `max_norm` and pass this latch.

## Things to know

A handful of patterns to avoid when building a clipping-side or
sensitivity-oracle extension today:

- **Returning a bare pytree.** Downstream components key on the
  `ClippedPytree` / `NoisedPytree` types — a raw tensor pytree out
  of a custom clipper won't carry `max_norm` and the noise
  mechanism won't know how to set σ. Always wrap.
- **Mutating `max_norm` mid-pipeline.** `ClippedPytree` is a frozen
  dataclass; mutation isn't possible without going around the type.
  Use `dataclasses.replace` or the manual `clipped()` factory if
  you need to reconstruct.
- **Varying per-step `max_norm` with MF noise downstream.** The
  latch above will raise; don't pair an adaptive-style clipping
  rule with MF noise.
- **Per-group bookkeeping outside the `PerGroup` container.**
  Downstream code expects per-group sensitivities via `PerGroup`
  specifically, not ad-hoc dicts; the arithmetic and accountant
  hookups are on `PerGroup`.

## Illustration: a sensitivity-oracle extension

Suppose a hypothetical `opaque-lipschitz` wheel ships per-layer
Lipschitz-bounded modules and computes a global per-record bound
`R` from the architecture. Today, its public entry point would
look like:

```python
from opaque.api.engine.types import clipped, ClippedPytree


def lipschitz_clipped(loss_pytree, *, R: float) -> ClippedPytree:
    """Wrap a pre-summed gradient pytree with an architecturally-derived bound."""
    return clipped(loss_pytree, max_norm=R)
```

That's the whole seam from the extension's side. The rest of the
pipeline doesn't need to know that `R` came from a Lipschitz
analysis rather than from a per-example norm computation —
`opaque.dpsgd.noise.gaussian(noise_multiplier).apply(out)` works on
the returned `ClippedPytree` the same as it does for fixed clipping
or AUTO-S, and `opaque.dpsgd.accounting.gaussian(noise_multiplier)`
gives the same privacy story.

The `max_norm=R` value is constant by construction (architecture
doesn't change between steps), so MF noise composes too. No new
`DpProcess` is needed — see "When you don't need a new accounting
primitive" in [Adding a new mechanism family](new-mechanism.md).

## See also

- [Adding a new mechanism family](new-mechanism.md) — the broader
  recipe.
- [`opaque.types` (Pytree Wrappers)](../reference/clipping.md#pytree-wrappers) — full reference for
  the types described here.
- [Per-Example Gradient Clipping](../user-guide/clipping.md) — the
  user-facing treatment of `PerGroup`.
