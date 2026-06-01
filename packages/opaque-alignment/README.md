# opaque-alignment

Functional, mechanism-agnostic primitives for **DP-safe preference learning**:
per-example loss functions (DPO / SFT families), logprob helpers, preference
collators, dataset transforms, reference-model helpers, alignment metrics, and
an alignment-specific fused-linear preference kernel.

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
- **DP-purity invariant.** Every public per-example loss is Tier 1 — strict
  per-example, NaN-injection verified (a single-record swap moves only that
  row's gradient). Tier 3 (rank/sort/quantile across the batch, e.g. DPO's
  `aot` family) is rejected — `resolve_dpo_loss("aot")` raises
  `NotImplementedError`.

## Import layout

```
opaque.alignment                         <- public façade (re-exports)
opaque.api.alignment                     <- implementation namespace
opaque.alignment.sft.{loss,collator}     <- SFT method (nll, dft; LM collator)
opaque.alignment.loss.dpo                <- DPO per-pair loss family + registry
opaque.alignment.logprob                 <- selective_log_softmax, sequence_logp
opaque.alignment.collator                <- preference (DPO) collator factory
opaque.alignment.data                    <- prompt extraction, chat templates
opaque.alignment.reference               <- ref-logp precompute, null_ref_context, EMA
opaque.alignment.metric                  <- rewards, KL, token accuracy
opaque.alignment.kernel                  <- fused-linear DPO preference kernel
```

## Status

Planning / early implementation. See `docs/development/opaque-alignment-plan.md`
for the full package plan and phase breakdown.
