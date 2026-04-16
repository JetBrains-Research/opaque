# Gradient denoising (DiSK)

Optional **post-processing** on **already noisy** gradients: a separate step
after the DP noise mechanism, using the same functional pattern as noise
addition: `(denoise_fn, state)` with immutable state.

Opaque currently ships **DiSK** (Zhang et al., ICLR 2025): a simple Kalman
filter per tensor element that exploits temporal structure in the gradient
sequence. It is complementary to DP noise (Gaussian, MF, etc.): you still run
your usual `noise_fn`; denoising runs on the **released** noisy gradients.

## API pattern

```python
from opaque.denoising import disk_denoiser
from opaque.distributed import sync

denoise, denoiser_state = disk_denoiser(
    grad_template,
    noise_stddev=noise_stddev,   # same σ as gaussian_noise / noise_fn
    process_stddev=process_std,  # random-walk scale (see below)
)

noisy_grads, noise_state = noise_fn(grads, noise_state, stddev=noise_stddev)
if denoise is not None:
    denoised_grads, denoiser_state = denoise(
        noisy_grads,
        denoiser_state,
        noise_stddev=noise_stddev,
    )
# Under DDP (example pattern from train_causal_lm.py):
# if denoise is not None:
#     noise_state, denoiser_state = sync(noise_state, denoiser_state)
# else:
#     noise_state = sync(noise_state)
```

- **`grad_template`**: a PyTree with the same structure as gradients (e.g.
  trainable parameters); leaves must be tensors so shapes and devices match.
- **`noise_stddev`**: same units as **`gaussian_noise(..., stddev=...)`** —
  scalar or **`PerGroup`** when you use per-group noise allocation.
- **`denoise(..., noise_stddev=None)`**: omit to reuse the factory default; pass
  each step when σ changes (adaptive clipping).

Internally, the filter uses measurement variance **R = noise_stddev²** and
process variance **Q = process_stddev²** for the random-walk prior. The public
API only exposes **standard deviations** (same convention as `gaussian_noise`).

## Privacy

If denoising is applied **only** to the signal that is already a valid DP
release (the same tensor you would have fed to the optimizer without
denoising), it is **post-processing** and does **not** consume additional
privacy budget under standard DP composition.

Do **not** feed denoised gradients back into mechanisms that assume a
different sensitivity (for example, re-clipping or changing what enters the
noise function) without a fresh privacy analysis.

## Choosing `process_stddev`

- **`noise_stddev`** (hence \(R\)) is set by the **mechanism** (clip norm,
  noise multiplier, accounting).
- **`process_stddev`** encodes how much you allow the **underlying** gradient
  estimate to move between steps **before** seeing the next noisy observation
  (random-walk prior). Smaller values → more smoothing across time; larger
  values → tracking fresh noisy gradients more closely.

There is no closed-form “best” value from the noise multiplier alone: the
multiplier fixes measurement noise; process scale is a **separate** modeling
choice, often tuned by validation or by scanning a ratio to `noise_stddev`.

## Example script

The causal LM example (`examples/train_causal_lm.py` in the repository) supports:

- `--denoiser none|disk`
- `--denoiser-process-std` — process noise scale (default matches the library
  default in `disk_denoiser`; same units as noise stddev).

## State types

- **`DenoiserState`** — abstract base (step counter), same role as
  `NoiseState` / `ClipState`.
- **`DiskDenoiserState`** — concrete immutable state for DiSK.

## See also

- [Noise Addition](noise.md) — mechanism and `stddev` calibration
- [DiSK API](../api/denoising.md) — full signatures

## Reference

- Zhang, Bu, Balle, Hong, Razaviyayn, Mirrokni — *DiSK: Kalman Filter-Based
  Gradient Denoising for Private Learning* (ICLR 2025),
  [arXiv:2410.03883](https://arxiv.org/abs/2410.03883).
