# Alignment (`opaque.alignment`)

`opaque-alignment` ships **functional, mechanism-agnostic primitives for
DP-safe preference learning and supervised fine-tuning**: per-example loss
functions (SFT and DPO), per-sequence log-probability helpers, batch
collators, dataset transforms, reference-model helpers, and reward metrics.
Every public symbol is a pure function or a factory returning a callable, so
each composes cleanly under `vmap(grad(...))` — the per-example differentiation
that DP-SGD and DP-FTRL clipping is built on.

The package builds only on the engine (clipping, functional conversion,
distributed) and base (serialization) layers. It does **not** depend on a DP
mechanism or an optimizer: the noise mechanism and optimizer are chosen at the
call site, so a researcher can assemble a DP-SGD or DP-FTRL alignment run from
`opaque.alignment` primitives plus a mechanism plus an optimizer — no trainer
subclass required. The same loss closure runs under either mechanism; only the
two noise/sampling imports change.

## Design

- **Functional, no hidden state.** Losses are pure functions of per-example
  tensors; collators and reference helpers are factories returning callables.
  The DP-SGD/DP-FTRL pipeline supplies its own clip and noise state.
- **Per-example by construction.** Each loss output for example `i` depends
  only on example `i`'s data, so per-example sensitivity stays `O(C)` after
  clipping. SFT divisors are per-example token counts (not a batch aggregate),
  and the DPO collator keeps chosen/rejected as separate `(B, ...)` tensors so
  one preference pair maps to one clipped gradient.
- **Mechanism-agnostic.** No mechanism or optimizer is imported in the
  package source; the mechanism is a call-site choice.
- **Direct functions, no registries.** Losses are exposed by name as plain
  functions. Mapping a config string to one of them is the caller's concern —
  the example scripts keep that tiny `name -> fn` map at the CLI boundary.

## Module map

| Module | Contents |
|---|---|
| `opaque.alignment.sft.loss` | `nll_loss`, `dft_loss`, and the fused twins `fused_nll_loss`, `fused_dft_loss` |
| `opaque.alignment.sft.collator` | `language_modeling_collator` (output schema `LMBatch` in `…collator.types`) |
| `opaque.alignment.dpo.loss` | per-sequence logp (`sequence_logp`, `fused_sequence_logp`), 14 per-pair heads, and the log-ratio combinators |
| `opaque.alignment.dpo.collator` | `preference_collator` |
| `opaque.alignment.dpo.reference` | `compute_ref_logprobs_for_dataset`, `null_ref_context`, `with_disabled_adapter`, `ema_update_reference` |
| `opaque.alignment.dpo.metric` | `reward_metrics` |
| `opaque.alignment.dpo.data` | `extract_prompt` |

## See also

- [SFT end-to-end](sft.md) — build a DP-SGD supervised fine-tuning run from
  the language-modeling collator and the per-example NLL/DFT losses.
- [DPO end-to-end](dpo.md) — reference-logp precompute, the preference
  collator, choosing a per-pair head, and reward-metric eval.
- [Alignment API reference](../reference/alignment.md) — every public
  function with its import path and a one-line description.
- [DP-SGD end-to-end](../user-guide/dp-sgd.md) — the clipping/noise/optimizer
  pipeline these losses plug into.
