# opaque-mlx

MLX provider for Opaque's backend-neutral engine primitives.

## Install

On Apple Silicon, install the standalone wheel:

```bash
pip install opaque-mlx
```

## Usage

MLX arrays select this provider automatically. Explicit selection is available
through `opaque.mlx.mlx_backend()`.

Provider activation registers the portable core, eager process-level
distributed and baseline observability profiles, the optional
`ExecutionProfile` transforms (`compile`, `checkpoint`,
`optimize_saved_activations`), allocator cache/peak controls, and native
`mlx.core.array` serialization. MLX trace annotations and device-capacity
observations are intentionally unavailable.

The MLX execution provider implements:

- `compile(fn)` → `mlx.core.compile(fn)`
- `checkpoint(fn)` → `mlx.core.checkpoint(fn)`
- `optimize_saved_activations(fn)` → identity transform that emits a one-time
  warning: MLX uses unified memory, so there is no separate host/device
  transfer and total activation storage is not reduced.

## Engine support

`opaque-mlx` implements Opaque's backend contract for native MLX arrays on
Apple Silicon. Engine fixed and AUTO-S clipping conformance runs eagerly
through the MLX autodiff and vectorization primitives.

DP-SGD mechanisms dispatch through this provider, so adaptive clipping and
Gaussian (including bounded Gaussian) noise operate on native MLX arrays.
Accounting remains array-framework-independent, and distributed behavior uses
the provider's distributed runtime profile.

| Capability | Support |
|---|---|
| Portable array, autodiff, pytree, and keyed-random primitives | Yes |
| Engine fixed and AUTO-S clipping conformance | Yes, eager |
| `loss_scaler` and `all_finite` precision helpers | Yes, eager |
| `ExecutionProfile` transforms (`compile`, `checkpoint`, `optimize_saved_activations`) | Yes [^1] |
| Native `mlx.core.array` serialization | Yes |
| `opaque.dpsgd.clipping.adaptive_clipped_grad` and Gaussian mechanisms | Yes, eager |
| `opaque.optimizers` | Yes, eager on native MLX arrays |
| Accounting-integrated DP-SGD | Yes |
| Distributed MLX training | Yes, through the distributed profile |

[^1]: `optimize_saved_activations` is an identity transform on MLX and emits
a one-time warning because unified memory removes the separate host/device
placement problem.