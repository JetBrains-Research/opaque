# DP-SGD end-to-end

This guide covers the DP-SGD pipeline: calibrate a noise multiplier,
clip per-example gradients, add Gaussian noise, update parameters, and
checkpoint state. DP-SGD mechanism imports use the `opaque.dpsgd.*` public
façade — no engine paths or internal `opaque.api.*` paths.

For the conceptual deep-dives on each component, see the topic pages
under [User Guide](index.md): [clipping](clipping.md),
[noise](noise.md), [sampling](sampling.md), [accounting](accounting.md),
[optimizers](optimizers.md). For the full DP-FTRL counterpart, see
[DP-FTRL end-to-end](dp-ftrl.md).

## Backend selection

Install `opaque-dpsgd` together with the provider for your array runtime
(`opaque-torch` or `opaque-mlx`). Passing native
parameter and batch arrays to the clipping or noise function selects that
provider automatically. To choose a provider before the first array-bearing
call, use `opaque.backend.set_backend()` with the provider factory, such as
`opaque.torch.torch_backend()`, or its name: `set_backend("torch")`.

On Apple Silicon, install `opaque-mlx`, `opaque-dpsgd`, and
`opaque-optimizers`, then select MLX with `set_backend("mlx")`. MLX has no
native `float64`; mechanisms retain device-native `float32` accumulation and
an explicit request for `opaque.ops.float64()` fails instead of using a host
fallback.

### Native MLX modules

`opaque.mlx.functional.make_functional` adapts a conventional `mlx.nn.Module`
to the explicit-parameter callable used by the training loop. It preserves the
caller-owned module while the returned callable receives the current parameter
pytree on each invocation:

```python
import mlx.core as mx
import mlx.nn as nn

from opaque.backend import set_backend
from opaque.mlx.functional import make_functional

set_backend("mlx")
model = nn.Linear(2, 1)
model_fn, params = make_functional(model)

def loss_fn(explicit_params, features, targets):
    return mx.mean(mx.square(model_fn(explicit_params, features) - targets))
```

The usual fixed, AUTO-S, and adaptive clipping factories accept this loss
function unchanged. Keyed Gaussian and bounded Gaussian noise, per-group
allocation, functional optimizers, loss scaling, schedules, and checkpoint
restore all operate on the returned MLX parameter pytree. See
[`examples/train_mlx_dpsgd.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_mlx_dpsgd.py)
for a complete small loop.

MLX compilation accepts array pytrees and scalar constants, not Opaque state
dataclasses. To compile a clipping calculation, capture its explicit clip
state in a closure, return the gradient array pytree, and keep threading the
returned state in the surrounding Python loop. Do not pass `ClipState`,
`NoiseState`, or optimizer-state objects directly to a compiled MLX function.

### Native MLX-LM causal-LM example

`examples/train_dpsgd_mlx.py` is a self-contained MLX/MLX-LM causal-LM example.
It accepts a Hugging Face repository ID or local MLX model directory, loads the
model and tokenizer through `mlx_lm.load`, installs native MLX-LM LoRA layers,
and trains the explicit adapter tree with Opaque. Install the root examples
dependency group first:

```bash
uv sync --group examples --all-packages --extra all
uv run python examples/train_dpsgd_mlx.py --help
```

This short Apple Silicon smoke performs one private update, evaluates, writes
a resumable checkpoint, exports an MLX-LM adapter, reloads it through
`mlx_lm.load`, and executes a forward check:

```bash
uv run python examples/train_dpsgd_mlx.py \
  --model-name HuggingFaceTB/SmolLM2-135M \
  --dataset JetBrains/KExercises --dataset-text-field solution \
  --num-train-samples 3 --num-eval-samples 1 --max-seq-len 16 \
  --batch-size 2 --num-epochs 1 --stop-at-step 1 \
  --clipping-mode fixed --noise-multiplier 0.2 --optimizer sgd \
  --target-delta 1e-5 --no-wandb \
  --checkpoint-path runs/mlx-smoke/checkpoint \
  --adapter-path runs/mlx-smoke/adapter
```

Continue the same deterministic sampler/noise stream for the second step:

```bash
uv run python examples/train_dpsgd_mlx.py \
  --model-name HuggingFaceTB/SmolLM2-135M \
  --dataset JetBrains/KExercises --dataset-text-field solution \
  --num-train-samples 3 --num-eval-samples 1 --max-seq-len 16 \
  --batch-size 2 --num-epochs 1 --stop-at-step 2 \
  --clipping-mode fixed --noise-multiplier 0.2 --optimizer sgd \
  --target-delta 1e-5 --no-wandb \
  --resume-from runs/mlx-smoke/checkpoint \
  --checkpoint-path runs/mlx-smoke/checkpoint \
  --adapter-path runs/mlx-smoke/adapter
```

The checkpoint contains explicit parameters, optimizer state, clipping state,
noise RNG state, accountant state, and progress. It is private training state;
do not publish it. The adapter directory is the inference artifact:

```python
import mlx_lm

model, tokenizer = mlx_lm.load(
    "HuggingFaceTB/SmolLM2-135M",
    adapter_path="runs/mlx-smoke/adapter",
)
```

One complete dataset row is one sampled privacy unit. Fixed token padding,
prompt masking through `--dataset-prompt-field`, and causal shifting alter only
which tokens contribute to that row's scalar loss; they never turn tokens into
separately sampled or accounted units.

## Why DP-SGD

DP-SGD adds independent Gaussian noise to a clipped gradient at every
training step. The privacy accountant composes per-step costs into a
total budget: each call to the noise mechanism is a separate
`DpProcess`, multiplied by the number of training steps.

Compared with DP-FTRL, DP-SGD needs no fixed training horizon or
matrix strategy; its independent noise can be less useful on cumulative
updates.

## 1. Calibration

Calibrate the noise multiplier to your privacy budget before training:

```python
import opaque.accounting as acc            # cross-cutting (compose, calibrate)
import opaque.dpsgd.accounting as dpsgd_acc  # DP-SGD factories

dataset_size = 50_000
batch_size = 256
sample_rate = batch_size / dataset_size
num_steps = 1000

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=5.0,
)
noise_multiplier = result.param
```

`dpsgd_acc.gaussian(nm)` is the per-step Gaussian mechanism;
`dpsgd_acc.poisson(..., sample_rate)` wraps it with Poisson
amplification; `* num_steps` composes across the run. The resulting
`DpProcess` is what `acc.calibrate` solves for the noise multiplier.

## 2. Clipping

`opaque.dpsgd.clipping.clipped_grad` wraps a per-example loss
function in `vmap(grad(...))` semantics, clips each per-example
gradient to a fixed norm, and sums:

```python
from opaque.dpsgd.clipping import clipped_grad

def loss_fn(params, batch):
    # ... per-example loss (NO mean over batch) ...
    return loss

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    argnums=0,
    batch_argnums=1,
    normalize_by=batch_size,
)
```

For automatic threshold tuning across steps:
`opaque.dpsgd.clipping.adaptive_clipped_grad`
([Andrew et al., 2021](https://arxiv.org/abs/1905.03871)).
For AUTO-S smooth scaling: `opaque.dpsgd.clipping.auto_clipped_grad`
([Bu et al., 2023](https://arxiv.org/abs/2206.07136)).

## 3. Noise

`opaque.dpsgd.noise.gaussian_noise` adds calibrated Gaussian noise to
the clipped gradient sum:

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key, split

key_sampling, key_noise = split(key(42), num=2)

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key_noise,
)
```

For bounded noise support, pass `bound=...` to
`opaque.dpsgd.noise.gaussian_noise` — same accounting, inverse-CDF
sampling, and accepts a positive scalar (symmetric `[-B, B]`) or a
`(low, high)` tuple.

## 4. Sampling

DP-SGD pairs with Poisson subsampling:

```python
from opaque.dpsgd.sampling import PoissonSampler

sampler = PoissonSampler(
    dataset, sample_rate=sample_rate, key=key_sampling,
)
```

`PoissonSampler` accepts `truncated_batch_size=` for the
truncated-Poisson variant; calibration must use
`dpsgd_acc.poisson(..., truncated_batch_size=..., dataset_size=...)`
to match.

## 5. Optimizer

```python
from opaque.optimizers import adamw

optimizer_step, opt_state = adamw(
    params,
    lr=1e-3,
    weight_decay=0.0,
    noise_bias_correction=True,  # DP-aware second-moment correction
)
```

`opaque.optimizers` ships `adamw`, `adam`, `sgd`, `radam`,
`adafactor`, `lion`, `ademamix`, and `schedule_free`. The
`noise_bias_correction=True` flag corrects the
biased second moment that arises when the optimizer sees noised
gradients.

## 6. End-to-end loop

```python
from opaque.serialization import state_dict
from opaque.optimizers import apply_updates

# ``params`` and ``batches`` contain native arrays from the selected provider.
for batch in batches:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer_step(noised, opt_state, params=params)
    params = apply_updates(params, updates)

# Checkpoint at the end (or any step):
ckpt = {
    "params": params,
    "opt_state": opt_state,
    "clip_state": clip_state,
    "noise_state": noise_state,
}
checkpoint = state_dict(ckpt)
```

Persist the resulting flat state dict with the selected provider's checkpoint
facility (or your application's storage layer), then restore it with
`opaque.serialization.from_state_dict`.

## Runnable references

- [`examples/train_dpsgd.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd.py)
  — full Torch causal-LM training script.
- [`examples/train_dpsgd_mlx.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_dpsgd_mlx.py)
  — full native MLX-LM causal-LM training, resume, and adapter export script.
- [`examples/train_mlx_dpsgd.py`](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_mlx_dpsgd.py)
  — small native MLX linear-model primitive loop.
- `tests/integration/test_dpsgd_pipeline.py` — minimal smoke test
  exercising the same flow on a tiny LlamaConfig + LoRA model
  (and the Qwen2 variant for the real-HF case).

## See also

- [Clipping](clipping.md) — fixed, AUTO-S, adaptive variants and
  per-group norms.
- [Noise](noise.md) — Gaussian (optionally bounded), when to choose
  which.
- [Sampling](sampling.md) — Poisson sampler details and the
  truncated-Poisson trade-off.
- [Accounting](accounting.md) — `DpProcess`, calibration, budgets.
- [Optimizers](optimizers.md) — DP bias correction and the
  second-moment story.
- [DP-SGD mechanisms](../mechanisms/dp-sgd/index.md) — Gaussian
  reference page.
- [DP-FTRL end-to-end](dp-ftrl.md) — the correlated-noise companion
  pipeline.
