# DPO end-to-end

This guide walks through a full DP-SGD Direct Preference Optimization
(DPO) run built from `opaque.alignment` primitives: precompute the
reference policy's log-probabilities, collate preference pairs, compute a
per-example DPO loss, differentiate it under `vmap(grad(...))`, clip,
noise, and step the optimizer, then evaluate with reward metrics. The
alignment imports come from the `opaque.alignment.dpo.*` façades; the DP
plumbing is the same [DP-SGD pipeline](../user-guide/dp-sgd.md) used
everywhere else.

## Why DPO is different

DPO scores each response against a frozen reference policy `pi_ref`, so
two things change relative to [SFT](sft.md):

1. **A reference is needed.** Every preference pair contributes
   `log pi(y) - log pi_ref(y)` log-ratios. The reference forward must run
   *outside* the per-example `vmap(grad(...))` region — it is a separate
   model (or an adapter toggle), not part of the differentiated closure.
2. **The unit of privacy is the pair.** The chosen and rejected forwards
   for one example share one clipped gradient, so the collator keeps
   chosen and rejected as separate `(B, ...)` tensors rather than TRL's
   concatenated `(2B, L)` layout. One preference pair maps to one
   per-example gradient, and each loss output depends only on that pair's
   data, keeping per-record sensitivity `O(C)` after clipping.

The mechanism is still the caller's choice: swap the `opaque.dpsgd` noise
and sampling imports for `opaque.dpftrl` to run
[DP-FTRL](../user-guide/dp-ftrl.md); the loss closure is unchanged.

## 1. Reference log-probabilities

Run the reference once over the dataset, *before* training, and cache the
per-example chosen/rejected logps.
`opaque.alignment.dpo.reference.compute_ref_logprobs_for_dataset` takes a
`ref(batch) -> {col: (B,) tensor}` callable (wrap your model into one),
runs a single `torch.no_grad()` pass, gathers across ranks, and caches to
a content-addressed `.npz` keyed on the dataset fingerprint plus your
`cache_key`. A cache hit skips the forward entirely.

```python
from opaque.alignment.dpo.reference import compute_ref_logprobs_for_dataset

dataset = compute_ref_logprobs_for_dataset(
    dataset,
    ref,                                  # batch -> ref_*_logps dict
    collator=collate,                     # the preference collator (below)
    output_columns=("ref_chosen_logps", "ref_rejected_logps"),
    batch_size=8,
    cache_key=("dpo", model_name),
    cache_dir=cache_dir,
)
```

When the policy is a PEFT/LoRA adapter, the *base* model is the reference:
enter `null_ref_context(model)` (or `with_disabled_adapter(model)`) around
the reference forward to disable the adapter, so no second model is
needed. `null_ref_context` dispatches over the reference configurations —
separate `ref_model`, a LoRA `"ref"` adapter clone, a disabled adapter, or
a no-op for an explicit callable. Both helpers run **outside vmap** (they
mutate `nn.Module` adapter state). For TR-DPO, periodically move the
reference toward the policy with `ema_update_reference(ref_params,
policy_params, alpha)` between steps.

## 2. Preference collator

`opaque.alignment.dpo.collator.preference_collator` is a factory
returning `collate(examples)`. Its output carries the chosen and rejected
trios — `chosen_input_ids`/`chosen_attention_mask`/`chosen_completion_mask`
and the `rejected_*` trio (each `(B, L)`) — plus the precomputed
`ref_chosen_logps` / `ref_rejected_logps` `(B,)` columns once the
reference pass has attached them. The completion mask is `0` over prompt
tokens and `1` over the response span, so only completion tokens score.
Use `opaque.alignment.dpo.data.extract_prompt` to split an implicit prompt
out of a chat example before tokenizing.

## 3. Per-example loss

Inside `vmap(grad(...))`, run the two forwards (chosen, rejected), turn
each into a completion logp with `sequence_logp`, subtract the precomputed
reference logps to form per-example log-ratios, and feed those to a
per-pair head:

```python
from opaque.functional import make_functional
from opaque.alignment.dpo.loss import sequence_logp, sigmoid_loss

fmodel, trainable, frozen = make_functional(
    model, disable_autograd_tracking=True, partition_trainable=True
)

def per_example_loss(
    trainable_params,
    chosen_ids, chosen_mask, chosen_cmask,
    rejected_ids, rejected_mask, rejected_cmask,
    ref_chosen_logps, ref_rejected_logps,
):
    merged = {**frozen, **trainable_params}
    chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = fmodel(merged, input_ids=rejected_ids, attention_mask=rejected_mask)
    chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)
    rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask)
    return sigmoid_loss(
        chosen_logp - ref_chosen_logps,
        rejected_logp - ref_rejected_logps,
        beta=beta,
    )
```

A DPO `per_example_loss` is always `head(sequence_logp(...) - ref_logp,
...)`. `sequence_logp(logits, input_ids, completion_mask)` applies the
causal-LM shift, gathers per-token log-probs, masks to the completion
span, and sums. On large vocabularies the `(T, V)` logits dominate memory:
`fused_sequence_logp(hidden_states, lm_head_weight, input_ids,
completion_mask)` is the drop-in that projects hidden states through the
`lm_head` without materializing logits (fused linear-CE kernel on CUDA +
half precision with `opaque-alignment[patches]`, eager fallback
otherwise). Like the fused SFT losses it is strictly per-example — pass
one sequence and let the outer `vmap` batch it.

### Choosing a per-pair head

All 14 heads take `(chosen_logratio, rejected_logratio, *, beta, ...)` and
return a per-example scalar. `sigmoid_loss` is the standard DPO objective
and the right default. The others trade off robustness, length
normalization, and reference handling — `hinge_loss`, `robust_loss`
(label-smoothed), `ipo_loss`, `sigmoid_norm_loss` (length-normalized),
`discopop_loss`, `chosen_nll_loss` (chosen-completion NLL regularizer for
MPO/RPO blends), `squarechipo_loss`, `apo_zero_loss`/`apo_down_loss`,
`exo_loss`, `nca_loss`, `bco_loss`, `sppo_loss`. As with SFT, map a config
string to a head at the CLI boundary (`{"sigmoid": sigmoid_loss, ...}`) —
the library exposes direct functions, not a registry. For composite
objectives, the log-ratio combinators (`f_divergence_remap` /
`f_divergence_logits`, `mpo_combine`, `wpo_weights`, `ld_dpo_split`)
remap or blend log-ratios before the head; see the
[reference](../reference/alignment.md#losses).

## 4. DP-SGD loop

The clip/noise/optimizer/sampler glue is identical to
[SFT](sft.md#3-vmapgrad-clip-noise-optimizer); only `batch_argnums` grows
to cover all eight per-example arguments:

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise

grad_fn, clip_state = clipped_grad(
    per_example_loss,
    argnums=0,
    batch_argnums=(1, 2, 3, 4, 5, 6, 7, 8),  # the 8 per-example tensors
    clipping_norm=1.0,
    normalize_by=batch_size,
    return_aux=True,
)
noise_fn, noise_state = gaussian_noise(noise_multiplier=nm, key=key(seed))

for indices in sampler:
    batch = collate_to_device([rows[i] for i in indices])  # 8-tuple
    (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = opt.update(noisy_grads, opt_state, params=trainable)
    trainable = torchopt.apply_updates(trainable, updates)
```

## 5. Reward-metric eval

DPO has no token-level CE eval objective, so evaluate with reward metrics
on held-out pairs. `opaque.alignment.dpo.metric.reward_metrics` takes the
same per-example log-ratios and returns detached `rewards/chosen`,
`rewards/rejected`, `rewards/accuracies`, and `rewards/margins` scalars:

```python
from opaque.alignment.dpo.metric import reward_metrics

metrics = reward_metrics(
    chosen_logp - ref_chosen_logps,
    rejected_logp - ref_rejected_logps,
    beta=beta,
)  # {"rewards/chosen": ..., "rewards/accuracies": ..., ...}
```

Run eval forwards under `torch.no_grad()` outside the clipped path — they
are not part of the private gradient.

## Runnable references

- [`examples/train_dpo.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpo.py)
  — full DP-SGD LoRA DPO script: reference precompute with
  `null_ref_context`, the preference collator, per-pair head selection,
  the per-example `vmap(grad)` loop, calibration/auditing, and reward-metric
  eval (plus a `--smoke` CPU run).

## See also

- [Alignment overview](index.md) — the package, its design, and the
  module map.
- [SFT end-to-end](sft.md) — the supervised fine-tuning companion guide.
- [Alignment API reference](../reference/alignment.md) — every public
  function with its import path.
- [DP-SGD end-to-end](../user-guide/dp-sgd.md) — calibration, clipping,
  noise, sampling, and optimizer details.
- [DP-FTRL end-to-end](../user-guide/dp-ftrl.md) — the correlated-noise
  mechanism swap.
