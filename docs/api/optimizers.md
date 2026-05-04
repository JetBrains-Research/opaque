# Optimizers

Opaque ships its own functional optimizer library at
[`opaque.optimizers`](#opaque.optimizers): Opaque-built factories with
DP-aware paths, plus a curated set of `torchopt` re-exports for the
stateless primitives where vanilla behaviour is acceptable under DP
noise.  All factories return
[TorchOpt](https://torchopt.readthedocs.io/) `GradientTransformation`s
and accept optional DP-aware kwargs (`noise_stddev`,
`noisy_squared_grads`) at update time.

---

## API shape

Functional: no hidden mutable state, explicit
`(updates, new_state) = optimizer.update(grads, state)` interface.

```python
import torchopt
from opaque.optimizers import adamw

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Explicit state in, state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

---

## What's in `opaque.optimizers`

### Opaque-built factories (DP-aware)

All accept `noise_stddev` (constructor default + per-step
`update()` override) to activate the optimizer's noise-aware path.
Where applicable they also accept `noisy_squared_grads` for JME
paired-stream substitution.

| Factory | DP-aware mode | When to use |
|---|---|---|
| **`adamw`** | φ-EMA on v̂ (DP-AdamW-BC) + JME | Default for DP training |
| **`ademamix`** | φ-EMA on v̂ + JME | Long-horizon training (slow EMA captures long-range signal) |
| **`adafactor`** | (deferred) | Memory-constrained large-LM fine-tuning; ship vanilla + WD only for now |
| **`lion`** | (planned: sign gating) | Smaller state than Adam; vanilla works under noise but the gated mode is the real DP variant |
| **`rmsprop`** | φ-EMA on v + JME | Adaptive without first moment; cheaper than Adam |
| **`adagrad`** | cumulative `Φ_acc` subtraction | Sparse-gradient settings; **the correction is mandatory** — vanilla Adagrad's denominator runs away under DP noise |
| **`schedule_free`** | post-processing (transparent forward) | Wrapper around any base optimizer; replaces external LR schedules |

Constructor knobs vary per optimizer (e.g. `decoupled_weight_decay`,
`update_rms_clip` on `adamw` and `ademamix`); see the docstrings.

### Re-exported from `torchopt`

For users who want a vanilla baseline or who're explicitly OK with the
non-corrected behaviour under DP.  Same names as torchopt's, no DP-aware
modes — slow under noise but functional.

| Re-export | DP behaviour |
|---|---|
| `sgd` | Update is unbiased (`E[g + ξ] = g`); momentum's variance is bounded but doesn't bias direction.  Canonical DP baseline. |
| `adam` | Has bias from the squared-noise term; bounded by EMA decay.  Slow without correction; prefer `adamw(decoupled_weight_decay=False, noise_stddev=σ)`. |
| `adadelta` | Two EMAs whose ratio partially self-corrects under noise.  Functional, not optimal. |
| `radam` | Same DP-BC story as Adam, just no Opaque-built variant yet. |

### Not re-exported

- `torchopt.adamax` — the max-norm tracker `v_t = max(β v_{t-1}, |g_t + ξ|)`
  permanently absorbs the half-normal noise mean (`σ·√(2/π)`); per-coordinate
  effective LR is floored by accumulated noise magnitude regardless of the
  true gradient.  No clean DP-aware variant; for non-DP use,
  `from torchopt import adamax` directly.

---

## DP-aware kwargs

Two optional kwargs on `update()` activate the optimizer's noise-aware
behaviour.  Both default to the constructor-time value (or absent).

### `noise_stddev`

Tells the optimizer the per-step noise σ.  Each optimizer activates
whatever noise-aware machinery it has:

| Optimizer | What `noise_stddev` does |
|---|---|
| `adamw`, `ademamix` | β₂-EMA of σ², subtracted from v̂ before sqrt (Chooi et al., [arXiv:2511.07843](https://arxiv.org/abs/2511.07843)) |
| `rmsprop` | α-EMA of σ², subtracted from v before sqrt (no `(1−α^t)` divide; v and φ accumulate at the same rate) |
| `adagrad` | Cumulative Σ σ², subtracted from v_acc before sqrt (no decay in either) |
| `lion` | (planned) sign gating when per-coordinate SNR is below threshold |
| `schedule_free` | Forwarded transparently to the wrapped base |

Constructor takes a default (scalar `float` or
[`PerGroup`](clipping.md#per-group-allocation)); per-step override at
`update()` time:

```python
optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_stddev=initial_sigma)

# Adaptive clipping changes the per-step σ → override per call:
updates, state = optimizer.update(
    noisy_grads, state, params=p,
    noise_stddev=noise_multiplier * clip_state.sensitivity,
)
```

`noise_stddev = 0` (the default) disables the noise-aware path; the
optimizer reduces to its standard math.

### `noisy_squared_grads`

Substitutes a JME paired-stream privately-estimated `g²` in place of
squaring the (already noised) gradient.  Required by `jme_noise()`'s
output shape; opt-in everywhere it applies (Adam-family + RMSprop):

```python
from opaque.dpftrl.noise import jme_noise, blt_strategy
from opaque.optimizers import adamw

strategy = blt_strategy(n_steps=1000, ...)
noise_fn, noise_state = jme_noise(grad_template, strategy, ...)

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Per-step:
(noisy_grads, noisy_sq), noise_state = noise_fn(grads, noise_state)
updates, opt_state = optimizer.update(
    noisy_grads, opt_state, params=p, noisy_squared_grads=noisy_sq,
)
```

Reference: Kalinin et al., [arXiv:2502.06597](https://arxiv.org/abs/2502.06597).

`noise_stddev` and `noisy_squared_grads` are **mutually exclusive** at
any single `update()` call; passing both raises `ValueError`.

---

## Complete pattern (DP-SGD)

```python
import torchopt
from opaque.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import adamw
from opaque.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
sigma = noise_multiplier * clip_state.sensitivity
noise_fn, noise_state = gaussian_noise(stddev=sigma, key=key(42))

# Optimizer with DP-AdamW-BC active
optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_stddev=sigma)
opt_state = optimizer.init(params)

for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(
        noisy_grads, opt_state, params=params,
        noise_stddev=noise_multiplier * clip_state.sensitivity,  # adaptive σ
    )
    params = torchopt.apply_updates(params, updates)
```

---

## Schedule-free wrapper

`schedule_free` wraps any base `GradientTransformation` (Opaque-built
or torchopt) with Defazio's schedule-free averaging
([arXiv:2405.15682](https://arxiv.org/abs/2405.15682)).  Three weight
sequences internally: `z` (raw iterate), `x` (Polyak-Ruppert average,
the published params), `y = (1-β)z + βx` (forward-pass weights).

```python
from opaque.optimizers import adamw, schedule_free

optimizer = schedule_free(adamw(lr=1e-3), beta=0.9, warmup_steps=100)
opt_state = optimizer.init(params)

for step in range(num_steps):
    # Trainer treats params as y_t; wrapper internally maintains z, x.
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)

# At save / eval time, read the published x_t:
from opaque.optimizers.schedule_free import get_eval_params
eval_params = get_eval_params(opt_state)
```

**DP-utility note**: under DP, `x_t = (1/n) Σ z_s` is a Polyak-Ruppert
average of noised iterates.  When per-step iterate noise is approximately
independent, `Var[x_n] ≈ Var[z]/n` — the published checkpoint has
significantly lower noise than the final iterate at the same privacy
budget.  This is a real DP-utility win specific to averaging-based
methods.

---

## Serialisation

`state_dict` / `load_state_dict` live in
`opaque.optimizers.serialization` (a less-common building block, not
in the package's top-level `__all__`).  Walks the chain state,
encoding every tensor leaf and Python primitive into a flat
`{path: value}` dict ready for `torch.save`.

```python
from opaque.optimizers import adamw
from opaque.optimizers.serialization import state_dict, load_state_dict

opt = adamw(lr=1e-3, weight_decay=0.01)
state = opt.init(params)
# ... train ...

# Save
torch.save(state_dict(state), "opt.pt")

# Load — template must have the same shape (init from same params).
template = opt.init(params)
state = load_state_dict(template, torch.load("opt.pt"))
```

Forward-compatible: paths missing from the saved dict keep the
template's value, so optimizers that gain new state fields between
releases load cleanly from older checkpoints.

---

## DDP compatibility

When using `torch.nn.parallel.DistributedDataParallel`, Opaque's
functional gradient pipeline runs *inside* each rank.  DDP handles the
all-reduce of noisy gradients across ranks; the optimizer state stays
synchronised because `optimizer.update` is a pure function and all
ranks receive identical noisy gradients after `sum_gradients` + noise
addition with the same key on all ranks.

Use `local_shard()` to partition the dataset across ranks and pass a
per-rank key via `fold_in(key, rank)` to each `PoissonSampler`.

---

## See also

- [Optimizers User Guide](../user-guide/optimizers.md) — concept-level
  explanation of the second-moment problem, when to use which DP mode.
- [Gradient Clipping API](clipping.md) — `clipped_grad`,
  `adaptive_clipped_grad`, `auto_clipped_grad`, `PerGroup`.
- [Schedules API](schedules.md) — LR schedules that pair with
  optimizer factories.
- [TorchOpt Documentation](https://torchopt.readthedocs.io/) — for
  re-exports and lower-level transforms.
