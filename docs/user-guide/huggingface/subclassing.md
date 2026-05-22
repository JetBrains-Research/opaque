# Subclassing DPTrainer

DPTrainer exposes one DP-correct extension point: the
`compute_per_example_loss` method.  Subclasses override that single
method to compute one example's loss; the trainer composes `vmap` →
`grad` → clip → noise → step around it.  DP-correctness is preserved by
construction because the override sits *inside* the vmap'd loss closure,
not around it.

If you've been writing custom HF Trainer subclasses by overriding
`compute_loss(...)`, the shape is the same — `compute_per_example_loss`
is the per-example, vmap-shaped equivalent.

## The hook

```python
def compute_per_example_loss(
    self,
    fmodel: Callable[..., Any],
    params: dict[str, Tensor],
    inputs: dict[str, Tensor],
    *,
    return_logits: bool = False,
) -> Tensor | tuple[Tensor, Any]:
    ...
```

Arguments:

- `fmodel` — functional model from `opaque.functional.make_functional`.
  Call as `fmodel(params, **inputs)` to get the model's
  `ModelOutput`.
- `params` — all parameters merged (`frozen | trainable`).  Under vmap,
  `trainable` is per-example replicated; `frozen` is broadcast.  You
  don't have to know which is which — pass the merged dict to `fmodel`.
- `inputs` — **one example's** input dict.  The trainer's vmap layer
  has already stripped the leading batch dim, so every tensor in
  `inputs` has shape `(seq, ...)`, not `(1, seq, ...)`.
- `return_logits` — when `True`, also return the model's `logits`
  tensor.  Used by the per-example eval path
  (`include_for_metrics=["loss"]`) so one forward yields both
  per-example losses and predictions.

Return:

- `Tensor` scalar loss when `return_logits=False`.
- `(loss, logits)` tuple when `return_logits=True`.

## Default implementation

The base method forwards through `fmodel(params, **inputs)`, reads
`output["loss"]` (HF's per-model `LOSS_MAPPING` dispatch handles
causal-LM, classification, seq2seq, etc.), and applies
`args.label_smoothing_factor` via a `cross_entropy(...,
label_smoothing=...)` rebuild on the exposed logits.

The base raises `RuntimeError` if the model output has no `"loss"`
field and no `compute_loss_func` was supplied.

## Vmap-safety constraints

`compute_per_example_loss` runs under `vmap(grad(...))`.  Operations
that work in a normal forward pass break under vmap; the constraints
are:

- **No mutations of `self` from inside the override.**  `vmap` traces
  the function once per example; side effects on instance attributes
  fire `batch_size` times in nondeterministic order.  Record diagnostic
  state outside the vmap path (e.g. from `training_step` or a callback).
- **No data-dependent Python control flow on input tensors.**
  `if labels.sum() > 0:` runs once per example but vmap can't
  partition the trace by example.  Use `torch.where` /
  `torch.masked_select` instead.
- **No `torch.nonzero`, no `.item()`, no `.tolist()`** — they all
  break the vmap batching rules.
- **Shapes must be data-independent.**  Mask construction and reshapes
  should depend only on input shapes, not values.
- **Don't modify `params` in place.**  Functional pattern — read,
  combine, never write.

The trainer's vmap layer also wraps the closure with autocast and the
fp16 loss scaler when those are configured; subclass code doesn't need
to handle them.

## SFT pattern — response-only loss

A classic supervised fine-tuning pattern: compute cross-entropy only on
the response tokens, ignoring the prompt.  The dataset emits
`labels = -100` for prompt positions; the override masks the loss to
positions where the label is not the ignore index.

```python
import torch
from opaque.transformers import DPTrainer


class SFTTrainer(DPTrainer):
    def compute_per_example_loss(
        self, fmodel, params, inputs, *, return_logits=False
    ):
        labels = inputs["labels"]  # (seq,) — leading batch dim already stripped
        outputs = fmodel(params, **{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs["logits"]  # (seq, vocab)

        # Standard causal-LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # `cross_entropy` with `ignore_index=-100` does the response-only
        # masking — prompt positions in the dataset are set to -100.
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=float(self.args.label_smoothing_factor),
        )

        if return_logits:
            return loss, logits
        return loss
```

The trainer does the vmap, grad, clip, and noise; the subclass owns the
math.  The override is identical to a normal `Trainer.compute_loss`
implementation except shapes are `(seq, ...)` instead of
`(batch, seq, ...)`.

## DPO pattern — double-forward margin loss

DPO computes a margin between log-probabilities under the trained
policy vs. a frozen reference model on `(chosen, rejected)` response
pairs.  The override does both forwards inside the same per-example
closure so vmap sees one self-contained example.

```python
class DPOTrainer(DPTrainer):
    def __init__(self, *args, ref_model, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._ref_model = ref_model.eval()
        self._beta = beta
        # ``make_functional`` once outside the vmap path so we capture
        # the reference model's parameters as a constant.
        from opaque.functional import make_functional
        self._ref_fmodel, self._ref_params = make_functional(self._ref_model)

    def compute_per_example_loss(
        self, fmodel, params, inputs, *, return_logits=False
    ):
        chosen_in = {k.removeprefix("chosen_"): v for k, v in inputs.items() if k.startswith("chosen_")}
        rejected_in = {k.removeprefix("rejected_"): v for k, v in inputs.items() if k.startswith("rejected_")}

        # Policy forwards
        chosen_logp = _seq_logprob(fmodel(params, **chosen_in), chosen_in["labels"])
        rejected_logp = _seq_logprob(fmodel(params, **rejected_in), rejected_in["labels"])

        # Reference forwards (frozen parameters captured at __init__)
        ref_chosen_logp = _seq_logprob(self._ref_fmodel(self._ref_params, **chosen_in), chosen_in["labels"])
        ref_rejected_logp = _seq_logprob(self._ref_fmodel(self._ref_params, **rejected_in), rejected_in["labels"])

        logits = self._beta * (
            (chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp)
        )
        loss = -torch.nn.functional.logsigmoid(logits)

        if return_logits:
            # Bundle the policy chosen logits so downstream metrics can
            # read them; pick whichever shape your `compute_metrics`
            # wants.
            return loss, None
        return loss
```

`_seq_logprob` is a per-example helper that sums log-probabilities over
non-ignored tokens.  The reference model's parameters are captured once
at `__init__` and treated as constants under vmap — they're broadcast,
not replicated per example.

The collator is responsible for emitting `chosen_input_ids`,
`chosen_labels`, `rejected_input_ids`, `rejected_labels` as separate
batch keys; the trainer's per-example dispatch picks them up via
`_discover_batch_keys`.

## KTO pattern — single-sample preference loss

KTO operates on individual `(input, label, is_desirable)` samples.  The
override branches on the `is_desirable` flag inside the loss math —
using `torch.where`, never a Python `if`, so vmap can trace through
both branches:

```python
class KTOTrainer(DPTrainer):
    def __init__(self, *args, ref_model, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # ... (same ref-model setup as DPO)
        self._beta = beta
        self._lambda_d = desirable_weight
        self._lambda_u = undesirable_weight

    def compute_per_example_loss(
        self, fmodel, params, inputs, *, return_logits=False
    ):
        is_desirable = inputs["is_desirable"]  # scalar (0 or 1)

        policy_logp = _seq_logprob(fmodel(params, **inputs), inputs["labels"])
        ref_logp = _seq_logprob(self._ref_fmodel(self._ref_params, **inputs), inputs["labels"])
        kl = (policy_logp - ref_logp).clamp(min=0)

        v_d = self._lambda_d * torch.sigmoid(self._beta * (policy_logp - ref_logp - kl))
        v_u = self._lambda_u * torch.sigmoid(self._beta * (kl - (policy_logp - ref_logp)))

        # ``torch.where`` keeps the trace single-pathed; both branches
        # are evaluated, the predicate selects.
        loss = torch.where(is_desirable.bool(), 1.0 - v_d, 1.0 - v_u)

        if return_logits:
            return loss, None
        return loss
```

`is_desirable` is per-example: under vmap it arrives as a scalar tensor,
the `torch.where` selects the right branch for the example being
processed.

## Callable bypass — `compute_loss_func`

For one-off losses that don't justify a subclass, pass a callable to
the constructor:

```python
def my_loss(outputs, labels):
    # outputs: one example's ModelOutput; labels: one example's labels
    return torch.nn.functional.cross_entropy(
        outputs["logits"].view(-1, outputs["logits"].size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

trainer = DPTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    compute_loss_func=my_loss,
)
```

The signature is `(outputs, labels) -> Tensor` — called *per example
under vmap*.  This is NOT HF's `(outputs, labels, num_items_in_batch) ->
scalar` shape: there's no `num_items_in_batch` argument because each
call already sees only one example, and the trainer's
`normalize_by` arithmetic at the clip / noise layer takes care of the
batch-mean math.

When `compute_loss_func` is set, the base
`compute_per_example_loss` forwards, picks the first available label
column from `args.label_names`, and calls `compute_loss_func(output,
labels)`.  Label-smoothing rebuild is **skipped** on this path — the
user-supplied loss owns smoothing semantics.

## Eval semantics

`compute_per_example_loss` covers training *and* eval — the same
override produces both.  When `args.include_for_metrics=["loss"]`,
`prediction_step` calls the vmap'd closure with `return_logits=True` so
one forward pass produces both real per-example eval losses (instead of
the batch-mean repeated) and the prediction tensors.

The default eval path (no `include_for_metrics=["loss"]`) goes through
`prediction_step` directly, which calls `self._model(**inputs)` —
*not* `compute_per_example_loss`.  If your override semantics need to
hold at eval too (e.g. masked SFT loss), set
`include_for_metrics=["loss"]` to route eval through the same closure.

## Logging side effects

The trainer measures and logs per-step diagnostics — `clip_rate`,
`grad_norm`, `noise_stddev` — outside the vmap'd path, by reading the
returned clip / noise state.  Subclasses that want their own
per-example diagnostics should compute them in a `TrainerCallback`'s
`on_step_end`, not from inside `compute_per_example_loss`.

## Composition with other features

| Feature | Override interaction |
|---|---|
| `args.label_smoothing_factor` | Honored by the base; subclasses overriding the method own smoothing themselves. |
| `args.fp16` / `args.bf16` | Autocast wraps the override; loss is returned in the autocast dtype, the loss scaler handles fp16. |
| `args.torch_compile` | The full per-example closure (override + autocast + scaler) is compiled. |
| `args.gradient_checkpointing` | Set up on the model before the override runs; the override sees a checkpointed forward path transparently. |
| PEFT / LoRA | `params` carries the LoRA merged set; the trainer partitions trainable vs. frozen automatically. |
| `clipping_norm` per-group | Applied outside `compute_per_example_loss` — the override returns one scalar, per-group clipping happens at the grad level. |

## See also

- [DPTrainer API](dptrainer.md) — the public `train()` / `evaluate()` /
  `predict()` surface.
- [PEFT and LoRA](peft.md) — `make_functional` and the trainable /
  frozen partition.
- [Per-example gradient clipping](../clipping.md) — what `clipped_grad`
  does to the gradients of the scalar your override returns.
