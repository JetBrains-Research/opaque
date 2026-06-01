# Reference-model handling

DPO needs reference log-probabilities. These helpers run **outside vmap**
(a forward pass over a dataset, or PEFT adapter toggles).

## Precompute + cache

`compute_ref_logprobs_for_dataset(dataset, ref, collator, output_columns, *,
batch_size, cache_key, cache_dir)` runs a one-shot pass under `torch.no_grad()`,
gathers across ranks (via the `opaque.distributed` metric helpers), and caches
to a content-addressed `.npz` keyed on a SHA-256 fingerprint of the dataset
fingerprint + `cache_key` + `output_columns`. A cache hit skips the forward
entirely. `ref` is a `Callable[[batch], dict[str, Tensor]]` returning per-example
logps — wrap a model into such a callable, keeping precompute mechanism-agnostic.

## The four reference configurations

`null_ref_context(model, ref_model=None)` dispatches per `RefSpec`:

| Config | Detected by | Behavior |
|---|---|---|
| **Separate model** | `ref_model` is not PEFT-derived | no-op (use `ref_model` directly) |
| **LoRA with `ref` adapter** | `is_peft_model(model)` and `"ref"` in `peft_config` | `set_adapter("ref")` on enter, restore on exit |
| **LoRA without separate ref** | `is_peft_model(model)`, `ref_model is None` | `disable_adapter()` — base model is the reference |
| **Explicit callable** | user passes a `ref_fn` | no-op |

## EMA reference sync (TR-DPO)

`ema_update_reference(ref_params, policy_params, alpha)` returns a new pytree
`(1 - alpha) * ref + alpha * policy` leafwise.

::: opaque.alignment.reference

::: opaque.alignment.reference.types
