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

Implemented preference heads in this guide follow the primary papers for
[DPO](https://arxiv.org/abs/2305.18290), [IPO](https://arxiv.org/abs/2310.12036),
[DiscoPOP](https://arxiv.org/abs/2406.08414), [SimPO](https://arxiv.org/abs/2405.14734),
[ORPO](https://arxiv.org/abs/2403.07691), [WPO](https://arxiv.org/abs/2406.11827),
[LD-DPO](https://arxiv.org/abs/2409.06411), [APO](https://arxiv.org/abs/2408.06266),
[SquareChiPO](https://arxiv.org/abs/2505.21395), [NCA](https://arxiv.org/abs/2402.05369),
[BCO](https://arxiv.org/abs/2404.04656), and
[SPPO](https://arxiv.org/abs/2405.00675). TR-DPO reference sync uses the
EMA-updated reference policy variant exposed on the trainer surface.

## 1. Reference log-probabilities

Run the frozen reference once over the dataset, *before* training, and cache
the per-example chosen/rejected logps when reuse is needed.
`opaque.alignment.dpo.reference.compute_ref_logprobs_for_dataset` takes a
`ref(batch) -> {col: (B,) tensor}` callable (wrap your model into one),
runs a single `torch.no_grad()` pass, gathers across ranks, and caches to
a content-addressed `.safetensors` file whose name is the SHA-256 digest of
`(dataset._fingerprint, cache_identity, output_columns)`.
By default that file lives under `<tempdir>/opaque_ref_cache/`; pass
`cache_dir=` to pin a different location. A cache hit skips the forward
entirely. Cache directories are owner-only and archive files are owner-readable
and writable only because they contain private per-example values. Remove a
selected cache directory when its contents are no longer needed.
This protects the artifacts from other local accounts; it does not isolate
processes running as the same account, so choose and clean cache locations
according to that threat model.

Pass `use_cache=False` for one-shot results that cannot be reused. TR-DPO does
this automatically for its initial seed columns: it recomputes reference logps
from the evolving EMA reference before every training and evaluation step, so
those seed values are never persisted.

On disk, the cache stores the requested tensors under their native torch dtypes
via `opaque.serialization.state_dict(...)` and `safetensors`; for example, a
bf16 reference forward stays bf16 in the archive. The only dtype demotion
happens later, when the cached tensors are attached back to a Hugging Face
dataset column: those values are converted to Python `float` because PyArrow
has no bf16 column type.

```python
import hashlib

from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    with_disabled_adapter,
)


def ref_state_digest(model) -> str:
    """Hash the effective adapter-disabled reference weights."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".lora_" in name or ".modules_to_save." in name:
            continue
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        start = value.storage_offset() * value.element_size()
        end = start + value.numel() * value.element_size()
        hasher.update(bytes(value.untyped_storage()[start:end]))
    return hasher.hexdigest()


with with_disabled_adapter(model):
    dataset = compute_ref_logprobs_for_dataset(
        dataset,
        ref,                                  # batch -> ref_*_logps dict
        collator=collate,                     # the preference collator (below)
        output_columns=("ref_chosen_logps", "ref_rejected_logps"),
        batch_size=8,
        cache_identity={
            "kind": "dpo-reference-logprobs",
            "reference": {
                "adapter_mode": "disabled",
                "state_sha256": ref_state_digest(model),
            },
        },
        cache_dir=cache_dir,
    )
```

`cache_identity` must be JSON-like — scalars, string-keyed mappings, and
sequences. Mapping order does not affect the digest, and unsupported or
non-deterministic values raise `TypeError` rather than falling back to `repr`.
A dataset without a deterministic `_fingerprint` is rejected for the same
reason.

A hit reuses the stored logprobs without re-examining the model, so the
identity must encode every input that changes reference behavior: an immutable
revision or weight digest, plus the adapter mode. Keying on a mutable name
alone — a model name, a checkpoint directory — silently returns logprobs from
whatever last occupied that name. `DPOTrainer` derives this identity from the
effective reference state on its own; only manual callers build it. Hashing the
weights as above is the conservative choice — an immutable Hub revision pin is
cheaper and equally safe when the reference is loaded straight from the Hub and
never mutated in-process. `examples/train_dpo.py` builds the same identity for
its manual precompute path.

For a PEFT/LoRA policy where the *base* model is the reference, use
`with_disabled_adapter(model)` around the reference forward as above, so no
second model is needed. `null_ref_context` instead dispatches over the
reference configurations — separate `ref_model`, a LoRA `"ref"` adapter clone,
a disabled adapter, or a no-op for an explicit callable. If it selects a
separate model or `"ref"` adapter clone, derive the cache identity from that
selected reference: its adapter mode must be accurate and its adapter weights
must be included in the digest. Both helpers run **outside vmap** (they mutate
`nn.Module` adapter state). For TR-DPO, periodically move the reference toward
the policy with `ema_update_reference(ref_params, policy_params, alpha)`
between steps.

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

The per-pair heads take `(chosen_logratio, rejected_logratio, *, beta, ...)`
and return a per-example scalar. `sigmoid_loss` is the standard DPO objective
and the right default. The others trade off robustness and reference
handling — `hinge_loss`, `robust_loss` (label-smoothed cDPO), `ipo_loss`,
`discopop_loss`, `chosen_nll_loss` (chosen-completion NLL regularizer for
MPO/RPO/CPO blends), `apo_zero_loss`/`apo_down_loss`,
`exo_loss`, `nca_loss`, `bco_loss`, `sppo_loss`. As with SFT, map a config
string to a head at the CLI boundary (`{"sigmoid": sigmoid_loss, ...}`) —
the library exposes direct functions, not a registry. For composite
objectives, the log-ratio combinators (`f_divergence_remap` /
`f_divergence_logits`, `mpo_combine`, `wpo_weights`, `ld_dpo_split`)
remap or blend log-ratios before the head; see the
[reference](../reference/alignment.md#losses).

### Reference-free methods (SimPO, CPO, ORPO)

These need no reference model — they score the policy log-prob directly, so you
skip the reference precompute (section 1) entirely. SimPO and ORPO use the
**length-normalized** per-token reward `r = log π(y) / |y|`, via
`sequence_logp(..., length_normalized=True)`:

```python
from opaque.alignment.dpo.loss import (
    sequence_logp, simpo_loss, odds_ratio_loss,
    chosen_nll_loss, sigmoid_loss, mpo_combine,
)

# length_normalized=True → the per-token mean reward log π(y)/|y|
c = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask, length_normalized=True)
r = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask, length_normalized=True)
chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)  # CPO uses the raw sum
rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask)

# SimPO — length-normalized sigmoid with a target margin γ:
loss = simpo_loss(c, r, beta=beta, gamma=gamma)

# ORPO — odds-ratio term + λ·NLL on the chosen completion:
loss = mpo_combine(
    {"or": odds_ratio_loss(c, r), "nll": chosen_nll_loss(c)},
    {"or": 1.0, "nll": orpo_lambda},
)

# CPO — reference-free sigmoid (raw logp as the log-ratio) + λ·NLL:
loss = mpo_combine(
    {"pref": sigmoid_loss(chosen_logp, rejected_logp, beta=beta),
     "nll": chosen_nll_loss(chosen_logp)},
    {"pref": 1.0, "nll": cpo_alpha},
)
```

`odds_ratio_loss` is the only one with genuinely new math (it works on the
log-probs, not log-ratios); SimPO is the length-normalized sigmoid plus a margin,
and CPO is pure composition of existing heads.

## 4. DP-SGD loop

The clip/noise/optimizer/sampler glue is identical to
[SFT](sft.md#3-vmapgrad-clipping-noise-and-optimization); only `batch_argnums` grows
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

## 5. Reward-metric evaluation

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
