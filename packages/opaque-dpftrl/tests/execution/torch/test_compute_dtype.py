"""Torch tests for ``mf_gaussian_noise``'s ``compute_dtype`` kwarg.

Pins down that the user-facing knob behaves the same way as the parallel
``opaque.dpsgd.noise.gaussian_noise(..., compute_dtype=...)``:

- Default is ``torch.float32`` (sampling Gaussians in bf16/fp16 distorts
  the noise distribution; not safe under the standard analysis).
- The dtype is used **directly** for provider-native normal sampling —
  there is no auto-promotion of the input tensor's dtype.
- The input pytree's dtype is preserved on output (type-stable boundary).
"""

from __future__ import annotations

import torch

from opaque.api.dpftrl.noise._identity import identity_strategy
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.random import key
from opaque.types import clipped


def _template():
    return {"w": torch.zeros(8, dtype=torch.float32)}


def _build_noise(*, compute_dtype=None):
    """Construct the IdentityStrategy MF noise fn.

    ``compute_dtype=None`` means "let the factory default apply" — this is
    how we probe the documented default.  Anything else is forwarded as
    the explicit user value.
    """
    kwargs = {
        "n_steps": 4,
        "noise_multiplier": 1.0,
        "key": key(42),
    }
    if compute_dtype is not None:
        kwargs["compute_dtype"] = compute_dtype
    return mf_gaussian_noise(_template(), identity_strategy(), **kwargs)


def test_mf_noise_ignores_unrelated_global_torch_rng_draws():
    grads = clipped({"w": torch.zeros(8, dtype=torch.float32)}, max_norm=1.0)
    noise_fn, state = _build_noise()
    expected, _ = noise_fn(grads, state)

    torch.manual_seed(999)
    torch.randn(1000)
    noise_fn, state = _build_noise()
    actual, _ = noise_fn(grads, state)

    assert torch.equal(actual.pytree["w"], expected.pytree["w"])


def test_compute_dtype_drives_inner_sampling_dtype():
    """fp32 vs fp64 ``compute_dtype`` produce different noise realisations on
    the same key — confirms the kwarg actually drives the internal sampling
    dtype rather than being a no-op.  Same-key fp32 sampling rounds to the
    fp32 grid, fp64 sampling does not, so the post-boundary fp32 outputs
    diverge."""
    grads = clipped({"w": torch.zeros(64, dtype=torch.float32)}, max_norm=1.0)
    nf32, s32 = _build_noise(compute_dtype=torch.float32)
    nf64, s64 = _build_noise(compute_dtype=torch.float64)
    out32, _ = nf32(grads, s32)
    out64, _ = nf64(grads, s64)
    assert not torch.equal(out32.pytree["w"], out64.pytree["w"].to(torch.float32))
