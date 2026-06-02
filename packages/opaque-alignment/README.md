# opaque-alignment

Functional, mechanism-agnostic primitives for **DP-safe preference learning**:
per-example loss functions (DPO / SFT families), logprob helpers, preference
collators, dataset transforms, reference-model helpers, alignment metrics, and
memory-efficient fused-linear twins (`fused_nll_loss` / `fused_dft_loss` /
`fused_sequence_logp`) that project hidden states through the `lm_head` via the
optional opaque-patches linear-CE kernel.

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
  row's gradient). Tier 3 variants (rank/sort/quantile across the batch, e.g.
  DPO's `aot` family) are simply not shipped.
- **Direct functions, no registry.** Each method exposes its loss functions
  directly (`dpo_sigmoid`, `nll_loss`, …) — there is no string registry,
  resolver, or variant enum. A config-string consumer (trainer / CLI) builds
  its own name→function mapping at the call site (see `examples/train_dpo.py`).

## Import layout

Method-first, mirroring `opaque.dpsgd` / `opaque.dpftrl`: each method owns its
primitives under its own namespace.

```
opaque.alignment                         <- top-level façade (exposes dpo, sft)
opaque.api.alignment                     <- implementation namespace
opaque.alignment.dpo                     <- DPO method façade (aggregates the below)
opaque.alignment.dpo.loss                <- logp + 14 per-pair heads + log-ratio combinators
opaque.alignment.dpo.collator            <- preference (DPO) collator factory
opaque.alignment.dpo.reference           <- ref-logp precompute, null_ref_context, EMA
opaque.alignment.dpo.metric              <- preference reward telemetry
opaque.alignment.dpo.data                <- preference prompt extraction
opaque.alignment.sft.loss                <- nll_loss, dft_loss + fused_nll_loss, fused_dft_loss
opaque.alignment.sft.collator            <- language-modeling (SFT) collator
```

`opaque.alignment.dpo.loss` is the DPO loss-construction toolkit: the per-sequence
log-prob primitives `sequence_logp` / `fused_sequence_logp`, the 14 per-pair
heads, and the log-ratio combinators all live there (a `per_example_loss` is
`head(sequence_logp(...) - ref_logp, …)`). The lower-level `selective_log_softmax`
building block, `entropy_from_logits` / `mean_token_accuracy` (token metrics), and
chat-template helpers stay internal impl under `opaque.api.alignment.*`, surfaced
through the method that consumes them, following the shared-impl re-import pattern of
`opaque.dpsgd.clipping`.

## Status

Planning / early implementation. See `docs/development/opaque-alignment-plan.md`
for the full package plan and phase breakdown.
