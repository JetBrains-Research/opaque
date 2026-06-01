# Alignment (`opaque.alignment`)

`opaque-alignment` ships **functional, mechanism-agnostic primitives for
DP-safe preference learning**: per-example loss functions (DPO / KTO / SFT),
logprob helpers, preference collators, dataset transforms, reference-model
helpers, alignment metrics, and a chunked fused-linear preference kernel.

It builds only on `opaque-engine` (clipping, functional, distributed) and
`opaque-base` (serialization). It does **not** depend on `opaque-dpsgd`,
`opaque-dpftrl`, or `opaque-optimizers` — the DP mechanism and optimizer are
chosen at the call site. A researcher can build a DP-DPO training script using
only `opaque-alignment` + a mechanism + an optimizer; no `DPTrainer` subclass
is required.

## Design principles

- **Functional, no hidden state.** Every public symbol is a pure function, a
  factory returning a callable, or an inert dataclass. `vmap(grad(...))`
  composes cleanly only over pure functions.
- **Mechanism-agnostic.** Enforced as a CI gate
  (`tests/contracts/test_dependency_direction.py` forbids mechanism/optimizer
  imports in this wheel's source).
- **Two-tier DP-purity invariant.** Every public per-example loss is labelled
  **Tier 1** (strict per-example; NaN-injection verified) or **Tier 2**
  (per-example + a *detached* batch aggregate with bounded `O(1/n)` leverage;
  aggregate-detach + leverage audited). **Tier 3** (rank/sort/quantile across
  the batch, e.g. the `aot` family) is rejected at the registry level —
  `resolve_dpo_loss("aot")` raises `NotImplementedError`.

## Module map

| Module | Contents |
|---|---|
| `opaque.alignment.logprob` | `selective_log_softmax`, `sequence_logp`, `get_batch_logps` |
| `opaque.alignment.loss.dpo` | 14 DPO variants + `DPO_LOSSES`/`DPO_SPEC` + f-divergence/MPO/WPO/LD helpers |
| `opaque.alignment.loss.kto` | `kto_loss` (Tier 2), `apo_zero_unpaired` + `KTO_LOSSES`/`KTO_SPEC` |
| `opaque.alignment.sft.loss` | `nll_loss`, `dft_loss` (direct functions) |
| `opaque.alignment.sft.collator` | `language_modeling_collator` |
| `opaque.alignment.collator` | `preference_collator`, `unpaired_preference_collator` |
| `opaque.alignment.data` | `extract_prompt`, `rotate_kto_completions`, chat-template helpers |
| `opaque.alignment.reference` | `compute_ref_logprobs_for_dataset`, `null_ref_context`, `ema_update_reference` |
| `opaque.alignment.metric` | `reward_metrics`, `kl_estimator`, `entropy_from_logits`, `mean_token_accuracy` |
| `opaque.alignment.kernel` | `opaque_fused_linear_dpo_loss`, `opaque_fused_linear_kto_loss` |

## Functional examples

- `examples/train_sft.py` — DP-SGD SFT with `language_modeling_collator` + `nll_loss`/`dft_loss`.
- `examples/train_dpo.py` — DP-SGD DPO with precomputed reference logps.
- `examples/train_kto.py` — DP-SGD KTO demonstrating the Tier-2 caller pattern.

See [Losses](loss.md), [Collators](collator.md), and
[Reference handling](reference.md) for details.
