# Mixed Precision

Opaque supports `bfloat16`, `float16`, and `float32` training. Three knobs
control what runs at what precision; all three are independent, and the safe
defaults already cover the common cases. This page documents the contract so
you can reason about edge cases — especially the **DP-critical** ones — without
reverse-engineering individual modules.

| Knob | Where | What it controls |
|------|-------|------------------|
| `torch.autocast(...)` | Around the forward / loss closure | Op-level dtype dispatch (matmul, conv, …). Standard PyTorch. |
| `loss_scaler` | `opaque.precision.loss_scaler` | Dynamic loss scaling for `fp16` — multiplies loss before backward, unscales grads before clipping. |
| `compute_dtype` | Kwarg on clipping + noise factories | Precision at which sensitivity-bound and noise sampling are computed. **DP-critical**: must be high enough for the privacy accountant to be calibrated to a real C, not a rounded-to-zero one. |

## The DP-critical invariant

The privacy accountant is calibrated to `noise_multiplier · C`, where C is the
per-example clip-norm threshold passed to `clipped_grad`. If anything makes the
gradient norm seen by the clip mechanism differ from the gradient norm that the
optimizer step actually consumes, the recorded privacy guarantee is wrong.

There are two places this shows up:

1. **Loss scaling.** When you scale the loss by `S` before backward (the
   underflow workaround for `fp16`), per-example gradients come back at
   magnitude `S · g`. Clipping that to C means the *effective* per-example
   bound is `C/S`, not C. The accountant doesn't know about S; it still records
   "noise calibrated to C". To preserve the invariant, the unscale must run
   **before** the clip-norm. `clipped_grad` exposes a `pre_clipping_transform`
   slot for exactly this purpose — `loss_scaler.unscale_grads` is shaped to
   plug in there.

2. **Compute precision.** When per-example L2-norm reductions run in `bfloat16`
   (which has 8-bit mantissa), small gradients silently round to zero. The
   sensitivity bound C the mechanism observes can be much smaller than the
   geometric truth, and the realized noise stddev is too small for the actual
   sensitivity. Clipping and noise both default `compute_dtype=torch.float32`
   for this reason — even when training is otherwise in `bfloat16` /
   `float16`.

## Recipe: `bfloat16` training

Easiest path. `bfloat16` has the same 8-bit exponent as `float32`, so loss
scaling isn't needed — gradients don't underflow. Just enter an autocast
context around the loss closure.

```python
def per_example_loss(params, x, y):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return loss(model_fn(params, x), y)

grad_fn, clip_state = clipped_grad(
    per_example_loss, argnums=0, batch_argnums=(1, 2), clipping_norm=C,
)
```

`compute_dtype` defaults to `torch.float32` on both `clipped_grad` and
`gaussian_noise`, so the sensitivity-bound and noise sampling run at safe
precision even though the forward is in `bfloat16`.

## Recipe: `float16` training (with loss scaling)

`float16` has a 5-bit exponent, so gradients can underflow. Use
`opaque.precision.loss_scaler` to multiply the loss before backward, then wire
the unscale as `pre_clipping_transform` so clipping sees the true magnitudes.

```python
from opaque.precision import loss_scaler, all_finite

scaler, scaler_state = loss_scaler()  # defaults match torch.amp.GradScaler

def per_example_loss(params, x, y):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model_fn(params, x), y)
    return scaler.scale_loss(loss, scaler_state)

grad_fn, clip_state = clipped_grad(
    per_example_loss,
    argnums=0,
    batch_argnums=(1, 2),
    clipping_norm=C,
    pre_clipping_transform=lambda g: scaler.unscale_grads(g, scaler_state),
)

# in the step:
grads, clip_state = grad_fn(params, x, y, state=clip_state)
grads_finite = all_finite(grads)
if grads_finite:
    noisy, noise_state = noise_fn(grads, noise_state)
    opt_state, params = optimizer.update(noisy, opt_state, params=params)
    acc_state = accountant.advance(acc_state, ...)
scaler_state = scaler.update(scaler_state, grads_finite)
```

On a non-finite step the wrapper *must* skip the noise mechanism, the optimizer
update, **and** the accountant advance — together. Skipped steps consume zero
privacy budget. The scaler's `update` advances the grow/backoff schedule
regardless of the skip decision.

## Recipe: `float32` training (everything disabled)

Pass `enabled=False` to keep the call sites uniform; the scaler degrades to
identity functions, the autocast block is just omitted.

```python
scaler, scaler_state = loss_scaler(enabled=False)
# the same wiring as the fp16 recipe; scale_loss / unscale_grads / update
# are no-ops.
```

## The `compute_dtype` contract

`compute_dtype` is **not** an autocast surface. Autocast intercepts op dispatch
inside the forward pass; clipping / noise reductions run *outside* the autocast
region (inside `vmap`, where autocast does not propagate) and produce the
sensitivity bound the privacy accountant relies on. Three rules:

- **Safe defaults on every public knob**, but the precise default differs
  by primitive. Clipping defaults to `compute_dtype=None`, which auto-promotes
  `bfloat16` / `float16` inputs to `float32` for the L2 reduction and leaves
  `float32` / `float64` inputs untouched (so an fp64 forward stays fp64
  through clipping). Both `gaussian_noise(...)` and `mf_gaussian_noise(...)`
  default to the literal `compute_dtype=torch.float32` and sample at that
  precision regardless of the input pytree's dtype. If you want `compute_dtype
  =torch.float64` end-to-end, you must pass it explicitly to clipping *and*
  noise — the clipping side does not promote upward on its own.
- **Don't lower it without recalibrating.** Setting
  `compute_dtype=torch.bfloat16` is a numerical regression of the privacy
  guarantee — the accountant still records `noise_multiplier · C`, but the
  realized noise stddev no longer matches.
- **You can raise it.** `compute_dtype=torch.float64` is safe (and
  occasionally useful for very small clip thresholds or aggressive
  noise multipliers).

## What's compatible with `torch.amp` and what isn't

| `torch.amp` primitive | Opaque counterpart |
|-----------------------|---------------------|
| `torch.amp.autocast(device_type, dtype=...)` | Used directly; nothing to wrap. The trainer enters it around the loss closure. |
| `torch.amp.GradScaler` | `opaque.precision.loss_scaler` — functional analog. State is a frozen dataclass; defaults match `GradScaler` (`init_scale=2**16`, `growth_factor=2.0`, `backoff_factor=0.5`, `growth_interval=2000`). |
| `torch.amp.custom_fwd` / `custom_bwd` | Not used — the functional DP step goes through `vmap(grad(...))`, which does not interact with custom autograd. Triton kernels in `opaque-patches` consult `torch.is_autocast_enabled()` directly at the wrapper boundary. |
| `GradScaler.step(optimizer)` (fuses inf-check + optimizer.step + skip) | Caller-owned. `loss_scaler` returns the schedule and the unscale; the surrounding loop owns the skip decision because `optimizer.update(...)` is composed by the user, not by the scaler. |

The structural gap — the scaler doesn't own the optimizer call — is forced by
the functional path: per-example gradients are returned as a pytree by
`vmap(grad(...))`, never attached to `Parameter.grad`. There's no
`scaler.unscale_(optimizer)` analogue because there's no optimizer to walk
parameters of. The trade is that the unscale runs *inside* `vmap`, per-example,
before any reduction — which is exactly where DP needs it.
