# DP-FTRL end-to-end

This guide walks through the full DP-FTRL pipeline: pick a
matrix-factorization strategy, calibrate the noise multiplier for the
*whole training run*, clip gradients, add correlated MF noise, run an
explicit-state optimizer step, and checkpoint state. DP-FTRL-specific imports
on this page come from the `opaque.dpftrl.*` public façade; shared state,
optimizer, RNG, and accounting APIs use their own public namespaces.

For DP-FTRL theory and a side-by-side comparison of mechanisms, see
[DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md). For the DP-SGD
counterpart, see [DP-SGD end-to-end](dp-sgd.md).

## Providers and execution model

The functional DP-FTRL path runs eagerly with Torch, JAX, and MLX. Install
`opaque-dpftrl` with the provider wheel used by the application, or use the
root extras `opaque[dpftrl]`, `opaque[dpftrl,jax]`, or
`opaque[dpftrl,mlx]`. The first provider-native gradient template passed to
`mf_gaussian_noise` activates its provider; explicit activation through
`torch_backend()`, `jax_backend()`, or `mlx_backend()` is also available.

Gradient pytrees, noised outputs, streaming correlation buffers, and paired
private second-moment streams remain provider-native. Opaque preserves each
leaf's dtype and device at the public boundary. Unless overridden,
`compute_dtype=None` uses the active provider's `float32` internally for
normal sampling and linear-combination arithmetic.

Opaque's immutable keys give deterministic replay for the same configuration,
inputs, and state within one provider. Torch, JAX, and MLX use different native
random implementations, so samples are not required to be bit-identical across
providers. DP-FTRL is documented as an eager path; do not infer full-loop JIT
safety from a provider's optional compilation capability.

Hugging Face `DPTrainer`, model patches, and Triton kernels remain Torch-only.
JAX and MLX users compose the provider-neutral functional primitives directly.

## Why DP-FTRL

DP-FTRL adds **correlated** Gaussian noise across training steps via
matrix factorization. Compared to independent noise at each step
(DP-SGD), correlated noise reduces variance on the **cumulative**
updates that the optimizer actually applies, for the same calibrated
privacy guarantee.

The trade-off: DP-FTRL accountants describe **whole training runs**.
The amplification factory takes `n_steps` at calibration time, the
strategy commits to a sensitivity / Gram matrix at construction time,
and the noise mechanism latches the per-step contribution bound on
the first call. Changing the training length, the per-step bound, or
the strategy mid-run breaks the privacy claim.

## Two notions of "correct"

DP-FTRL has two distinct notions of "correctness" worth keeping
separate:

1. **DP correctness** — the privacy guarantee applies to the
   randomized algorithm you actually run. As long as the accounting
   uses the same sensitivity (and Gram matrix when needed) as the
   strategy passed to `mf_gaussian_noise`, and the sampler matches the
   amplification analysis, the DP statement is valid.
2. **Workload fidelity / utility** — strategies are designed for a
   workload model (Polyak momentum, constant LR, exponential decay).
   If the real loop differs (different optimizer, different schedule,
   accumulation pattern), utility may be worse than the paper's
   ideal even when the DP statement is unchanged.

## 1. Strategy choice

Pick a matrix-factorization strategy by mechanism. The strategy
object holds: coefficients defining the lower-triangular linear map
used for noise, the sensitivity (and sometimes a Gram matrix) used by
the accountant, and a streaming representation for efficient noise
generation.

```python
from opaque.dpftrl.noise import (
    band_mf_strategy,    # numerical Toeplitz optimization
    blt_strategy,        # buffered linear Toeplitz, multi-epoch
    bisr_strategy,       # banded inverse square root
    bsr_strategy,        # banded square root, closed-form
    lambda_cgd_strategy, # PRNG replay, O(1) memory
    identity_strategy,   # no correlation; baseline
)

strategy = band_mf_strategy(bands=10)
```

Strategies are provider-independent host recipes. Their
`coefficients(n_steps=..., min_sep=..., max_participations=...)` queries return
NumPy arrays for inspection and accounting, while `mf_gaussian_noise` projects
the recipe into an immutable numeric execution plan and applies that plan to
native Torch, JAX, or MLX arrays.

See [DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md) for the
choice criteria.

## 2. Calibration

DP-FTRL accountants describe a whole training run. Build the strategy
first, then build the matching accounting mechanism using its
sensitivity / Gram matrix:

```python
import opaque.accounting as acc                  # cross-cutting
import opaque.dpftrl.accounting as dpftrl_acc    # DP-FTRL factories

# Same strategy that will go into mf_gaussian_noise below.
strategy = band_mf_strategy(bands=10)

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpftrl_acc.poisson(
        dpftrl_acc.mf_gaussian(nm, strategy),
        sample_rate=0.01,
        n_steps=1000,
    ),
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param
```

Three amplification factories under
`opaque.dpftrl.accounting` — pick the one that matches your sampler:

- `dpftrl_acc.poisson(...)` — Poisson subsampling (cyclic-Poisson
  under banded MF).
- `dpftrl_acc.b_min_sep(...)` — b-min-separation participation
  pattern.
- `dpftrl_acc.balls_in_bins(...)` — fixed-partition participation.

Each amplification factory wraps a mechanism into a single
`DpProcess` describing the full training run. **Always pass the same
strategy object** to `mf_gaussian_noise` and the accounting factory — that is
how DP correctness is preserved.

### Step-by-step ε reporting (`per_step`)

DP-FTRL processes are *whole-process*: feeding the bare process to
`Accountant`'s `acct |= step` would over-count. Wrap with
`acc.per_step(...)` to get a step-shaped adapter whose
`per_step(proc) * K` materialises the strategy-aware K-prefix PLD —
identical in shape to the DP-SGD `acct |= step` idiom but with the
correct K-step bound under MF correlations:

```python
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
step = acc.per_step(proc)
acct = Accountant(budget=acc.epsilon_budget(3.0, delta=1e-5))

for batch in dataloader:
    # ... train ...
    acct |= step
    eps_so_far = acct.epsilon_at(delta=1e-5)
```

For analytic PLDs, K-step ε is monotone and bounded by full-horizon ε.
Monte Carlo PLDs are point estimates without that guarantee.
`K > proc.n_steps` raises.

## 3. Clipping

Same engine clipping primitives as DP-SGD, just imported from
`opaque.dpftrl.clipping`. Adaptive clipping is **not** available
under DP-FTRL — its threshold drifts across steps, violating the
constant per-step sensitivity assumption MF privacy proofs require.

```python
from opaque.dpftrl.clipping import clipped_grad

def loss_fn(params, batch):
    return loss

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    argnums=0,
    batch_argnums=1,
    normalize_by=batch_size,
)
```

`auto_clipped_grad` (AUTO-S) is also available and compatible with
DP-FTRL — its per-record sensitivity bound
`sup ‖R · g / (‖g‖ + γ)‖ ≤ R` is constant in the input, satisfying
the same invariant MF accounting requires of fixed clipping.

**Choosing among the three.** Scalar `clipped_grad` is the right
default. Reach for `per_group` clipping when one or more parameter
groups have substantially different gradient magnitudes than the
rest — a freshly initialized head on top of frozen pretrained
layers, or a LoRA target whose gradients sit far below its
siblings: the non-uniform `σᵢ ∝ √Cᵢ` noise allocation concentrates
less noise on small-gradient groups without sacrificing much
sensitivity on the dominant ones, and on heterogeneous workloads
this recovers a small evaluation-loss improvement over scalar clipping
at the same joint budget. Setting `Cᵢ` substantially tighter than
the per-group typical magnitudes (e.g. half the per-group median)
regresses below scalar — clipping bias dominates the
noise-redistribution benefit, so target `Cᵢ` near per-group typical
gradient magnitudes. Reach for `auto_clipped_grad` when you don't
want to tune `R` per workload — on the workloads we've measured
AUTO-S matches or slightly beats fixed clipping at the same `R`,
with no per-workload tuning needed.

## 4. Noise

`opaque.dpftrl.noise.mf_gaussian_noise` injects correlated noise:

```python
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.random import key

# grad_template is the structure of clipped_grad's output —
# typically a ClippedPytree from a single warm-up call.
warmup_grads, _ = grad_fn(params, warmup_batch, state=clip_state)

noise_fn, noise_state = mf_gaussian_noise(
    warmup_grads,
    strategy,                       # same object you used in accounting
    n_steps=1000,                   # total training steps
    noise_multiplier=noise_multiplier,
    key=key(0),
    compute_dtype=None,              # active provider's float32
)
```

The `grad_template` and each later clipped pytree must contain native arrays
from the same provider. Internal noise and correlation state use
`compute_dtype`; `None` deliberately resolves to provider `float32`, while the
returned leaves are cast back to their input dtypes and remain on their input
devices.

`mf_gaussian_noise` reads the per-step contribution bound from the
`ClippedPytree` input on the **first call** and latches it for the
rest of the run. The bound is `noise_multiplier × max_norm`, so each
step must produce gradients with the same `max_norm` for the privacy
claim to hold.

For private second-moment estimation (Adam-style optimizers), pass
`second_moment_strategy=...` — see [Optimizers](optimizers.md). The
joint Mahalanobis allocation between the two streams keeps privacy
accounting at the same `gaussian(nm)` shape as the first-moment-only
release: with the same strategy used for both streams, the
first-moment σ picks up exactly a `√(1+C)` factor over the no-paired
baseline (where `C` is the per-record clipping bound — scalar for
`clipped_grad`, `PerGroup` for per-group clipping; AUTO-S uses `R`
for the same role). With distinct strategies for the two streams the
inflation depends on the ratio of their column norms — see
[Noise](noise.md). Either way, pick `C` as small as the optimizer
tolerates when the paired release is on (cross-reference: the same
guidance as in the [Clipping](clipping.md) empirical evidence).

## 5. Sampling

DP-FTRL has its own sampler family under `opaque.dpftrl.sampling`:

```python
from opaque.dpftrl.sampling import (
    CyclicPoissonSampler,    # banded MF: cyclic Poisson subsampling
    BMinSepSampler,          # b-min-separation
    BallsInBinsSampler,      # fixed-partition
    SequentialBatchSampler,  # deterministic order, used by BLT
)

sampler = CyclicPoissonSampler(
    dataset, sample_rate=0.01, bands=10, n_steps=1000, key=key(42),
)
```

The sampler must match the amplification factory you used in
calibration.

## 6. Optimizer

Same surface as DP-SGD:

```python
from opaque.optimizers import adamw

optimizer_step, opt_state = adamw(params, lr=1e-3, noise_bias_correction=True)
```

Private second-moment AdamW pairs with `mf_gaussian_noise(...,
second_moment_strategy=...)` — the noise mechanism produces a
`SecondMomentNoiseOutput` and the optimizer's DP-aware path consumes
it. See [Optimizers](optimizers.md) for the full second-moment story.

**Stability under the paired release.** The math holds under MF
(the predicted σ_first inflation matches the formula referenced in
[Noise](noise.md)), but Adam-family optimizer stability is
workload-dependent. The destabilization risk comes from the
v update: when per-coordinate gradient signal is small relative to
the second-stream σ — common when some parameter groups have very
small gradients — Adam's per-coordinate scaling accumulates bias
and the average gradient norm grows across training. Watch for a
rising clipping rate and a growing per-step gradient norm in the
early steps. Mitigations: lower the learning rate (a ~3× drop is
sometimes enough), or use `per_group` clipping to scope `C` per
group so the per-group σ on small-gradient groups doesn't dominate
the v signal there.

## 7. End-to-end loop

```python
from opaque.optimizers import apply_updates
from opaque.serialization import state_dict

for step, batch in enumerate(sampler):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer_step(noised, opt_state, params=params)
    params = apply_updates(params, updates)

# Hand this flat mapping to the application's checkpoint writer.
checkpoint = state_dict({
    "params": params,
    "opt_state": opt_state,
    "clip_state": clip_state,
    "noise_state": noise_state,
})
```

Torch users can obtain functional `params` with
`opaque.torch.functional.make_functional`. JAX and MLX users pass their native
functional parameter pytrees directly; the loop and Opaque state threading are
otherwise the same.

To resume, reconstruct the clipping, optimizer, and MF noise mechanisms with
the same configuration, assemble a matching state template, and call
`from_state_dict(template, checkpoint)`. Native-array handlers preserve Torch,
JAX, or MLX state leaves when restored against a matching provider template.
The restored `MFNoiseState` or `SecondMomentMFNoiseState` carries the saved key,
step counter, and provider-native correlation buffers, so the next noise call
matches uninterrupted execution within that provider.

## Runnable references

- [`examples/train_dpftrl.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpftrl.py)
  — full DP-FTRL training script.
- `tests/integration/test_dpftrl_pipeline.py` — minimal smoke test
  exercising the same flow on a tiny LlamaConfig + LoRA model (and
  the Qwen2 variant).

## See also

- [Clipping](clipping.md) — fixed and AUTO-S variants
  (adaptive is DP-SGD-only).
- [Noise](noise.md) — `mf_gaussian_noise` shape, strategy types,
  per-step bound latching.
- [Sampling](sampling.md) — DP-FTRL sampler family.
- [Accounting](accounting.md) — `DpProcess`, the
  whole-process model, MF-specific composition.
- [Optimizers](optimizers.md) — second-moment integration.
- [DP-FTRL mechanisms](../mechanisms/dp-ftrl/index.md) — per-mechanism
  reference pages.
- [DP-SGD end-to-end](dp-sgd.md) — the per-step companion pipeline.
