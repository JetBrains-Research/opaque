# Alignment losses

Per-example preference/SFT losses. Each is a pure, `vmap`-safe function on
per-example tensors; the caller wraps it with `vmap(grad(...))` → clip → noise.
Every loss carries a `DPSpec` declaring its DP tier.

## DP-purity tiers

- **Tier 1** — loss for example `i` depends only on example `i`. Verified by a
  NaN-injection contract test.
- **Tier 2** — example `i` + a *detached* batch aggregate with `O(1/n)`
  leverage (KTO's batch-mean KL). The caller computes the aggregate **outside**
  vmap, `.detach()`-es it, and broadcasts it in. Verified by an autograd-graph
  detach audit + a leverage test.
- **Tier 3** — rejected (sort/rank across batch). Not exposed.

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

## KTO family (`opaque.alignment.loss.kto`)

`kto_loss` is **Tier 2**: it takes a scalar detached `kl` (the batch-mean KL
the caller computes outside vmap). `apo_zero_unpaired` is Tier 1. `KTO_SPEC`
records the tier + `kl_mean` aggregate; `KTO_AGGREGATES` declares the
`LossAggregateSpec` (per-rank in v1; cross-rank all-reduce is a v2 item).

::: opaque.alignment.loss.kto

## SFT family (`opaque.alignment.sft.loss`)

`nll_loss` (causal-LM CE, per-example mean over non-ignored tokens) and
`dft_loss` (DFT: detached-softmax-weighted, with a **DP-corrected per-example
divisor** — `mask.sum()`, not TRL's batch `num_items_in_batch`). Both are
direct functions (no string registry); both are strict per-example (Tier 1).

::: opaque.alignment.sft.loss

## DP-purity records

::: opaque.alignment.loss.types
