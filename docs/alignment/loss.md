# Alignment losses

Per-example preference/SFT losses. Each is a pure, `vmap`-safe function on
per-example tensors; the caller wraps it with `vmap(grad(...))` → clip → noise.
DPO variants carry a `DPSpec` declaring their DP tier.

## DP-purity tiers

- **Tier 1** — loss for example `i` depends only on example `i`; a single-record
  swap moves only that row's gradient. Verified by a NaN-injection contract
  test. Every live loss (SFT `nll`/`dft`, all DPO variants) is Tier 1.
- **Tier 3** — rejected: the per-example contribution depends on a batch order
  statistic (sort/rank/quantile, e.g. DPO's `aot*`), so one swap can change a
  clipped gradient by `O(1)`. Not exposed; `resolve_dpo_loss("aot")` raises
  `NotImplementedError`.

## DPO family (`opaque.alignment.loss.dpo`)

`DPO_LOSSES` maps a variant name to a per-pair callable
`dpo_<v>(chosen_logratio, rejected_logratio, *, beta, **kw)`. All 14 variants
are Tier 1: `sigmoid`, `hinge`, `robust`, `ipo`, `sigmoid_norm`, `discopop`,
`sft`, `squarechipo`, `apo_zero`, `apo_down`, `exo_pair`, `nca_pair`,
`bco_pair`, `sppo_hard`. Helpers: `f_divergence_logits`/`f_divergence_remap`
(reverse_kl / forward_kl / js / alpha), `mpo_combine`, `wpo_weights`,
`ld_dpo_split`. `resolve_dpo_loss(name)` dispatches and raises
`NotImplementedError` for the rejected `aot*` family.

::: opaque.alignment.loss.dpo

## SFT family (`opaque.alignment.sft.loss`)

`nll_loss` (causal-LM CE, per-example mean over non-ignored tokens) and
`dft_loss` (DFT: detached-softmax-weighted, with a **DP-corrected per-example
divisor** — `mask.sum()`, not TRL's batch `num_items_in_batch`). Both are
direct functions (no string registry); both are strict per-example (Tier 1).

::: opaque.alignment.sft.loss

## DP-purity records

::: opaque.alignment.loss.types
