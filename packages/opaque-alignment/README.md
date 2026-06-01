# opaque-alignment

Functional, mechanism-agnostic primitives for **DP-safe preference learning**:
per-example loss functions (DPO / KTO / SFT families), logprob helpers,
preference collators, dataset transforms, reference-model helpers, alignment
metrics, and an alignment-specific fused-linear preference kernel.

Built on `opaque-engine` (clipping, functional, distributed) and `opaque-base`
(serialization). Consumed by both functional training scripts
(`examples/train_dpo.py`-style) and the TRL-style class trainers in
`opaque.transformers.trl`.

## Design

- **Functional, no hidden state.** Every public symbol is a pure function, a
  factory returning a callable, or an inert dataclass — no `nn.Module`
  subclasses, no user-instantiated classes. `vmap(grad(...))` composes cleanly
  only over pure functions.
- **Mechanism-agnostic.** Depends only on the substrate packages
  (`opaque-engine`, `opaque-base`); never on `opaque-dpsgd`, `opaque-dpftrl`,
  or `opaque-optimizers`. The DP mechanism and optimizer are chosen at the call
  site. This is enforced by `tests/contracts/test_dependency_direction.py`.
- **Two-tier DP-purity invariant.** Every public per-example loss is labelled
  Tier 1 (strict per-example, NaN-injection verified) or Tier 2 (per-example +
  detached batch aggregate with bounded `O(1/n)` leverage, aggregate-detach
  audit verified). Tier 3 (rank/sort/quantile across batch) is rejected at the
  registry level.

## Import layout

```
opaque.alignment                         <- public façade (re-exports)
opaque.api.alignment                     <- implementation namespace
opaque.alignment.loss.{dpo,kto,sft}      <- per-example loss families + registries
opaque.alignment.logprob                 <- selective_log_softmax, sequence_logp
opaque.alignment.collator                <- factory fns returning collator callables
opaque.alignment.data                    <- prompt extraction, packing, chat templates
opaque.alignment.reference               <- ref-logp precompute, null_ref_context, EMA
opaque.alignment.metric                  <- rewards, KL, token accuracy
opaque.alignment.kernel                  <- fused-linear preference kernel (paired + unpaired)
```

## Status

Planning / early implementation. See `docs/development/opaque-alignment-plan.md`
for the full package plan and phase breakdown.
