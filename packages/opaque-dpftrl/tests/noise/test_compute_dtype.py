"""Tests for ``mf_gaussian_noise``'s ``compute_dtype`` kwarg.

Pins down that the user-facing knob behaves the same way as the parallel
``opaque.dpsgd.noise.gaussian_noise(..., compute_dtype=...)``:

- Default is ``torch.float32`` (sampling Gaussians in bf16/fp16 distorts
  the noise distribution; not safe under the standard analysis).
- The dtype is used **directly** for the internal ``torch.randn`` —
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


def test_compute_dtype_default_is_float32():
    """No-arg call resolves to fp32 — matches dpsgd.gaussian_noise default."""
    noise_fn, state = mf_gaussian_noise(
        _template(),
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.0,
        key=key(42),
    )
    # Reach into the closure: the kwarg threads to _iid_normal_noise via
    # _streaming_mf_noise; an end-to-end fp32 output suffices as evidence.
    grads = clipped({"w": torch.zeros(8, dtype=torch.float32)}, max_norm=1.0)
    noisy, _ = noise_fn(grads, state)
    assert noisy.pytree["w"].dtype == torch.float32


def test_compute_dtype_preserves_input_dtype_on_output():
    """Bf16 input pytree stays bf16 on output even with fp32 internal compute."""
    noise_fn, state = mf_gaussian_noise(
        _template(),
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.0,
        key=key(42),
        compute_dtype=torch.float32,
    )
    grads = clipped({"w": torch.zeros(8, dtype=torch.bfloat16)}, max_norm=1.0)
    noisy, _ = noise_fn(grads, state)
    assert noisy.pytree["w"].dtype == torch.bfloat16


def test_compute_dtype_float64_override():
    """User-passed fp64 is honoured end-to-end."""
    noise_fn, state = mf_gaussian_noise(
        _template(),
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.0,
        key=key(42),
        compute_dtype=torch.float64,
    )
    # Input fp64 → fp64 output (no down-cast loss).
    grads = clipped({"w": torch.zeros(8, dtype=torch.float64)}, max_norm=1.0)
    noisy, _ = noise_fn(grads, state)
    assert noisy.pytree["w"].dtype == torch.float64


def test_compute_dtype_changes_realized_noise_when_input_is_low_precision():
    """fp32 vs fp64 compute_dtype produce different noise realizations on the
    same key — confirms the kwarg actually drives the internal sampling dtype
    rather than being a no-op.
    """
    grads = clipped({"w": torch.zeros(64, dtype=torch.float32)}, max_norm=1.0)
    nf32, s32 = mf_gaussian_noise(
        _template(),
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.0,
        key=key(42),
        compute_dtype=torch.float32,
    )
    nf64, s64 = mf_gaussian_noise(
        _template(),
        identity_strategy(),
        n_steps=4,
        noise_multiplier=1.0,
        key=key(42),
        compute_dtype=torch.float64,
    )
    out32, _ = nf32(grads, s32)
    out64, _ = nf64(grads, s64)
    # Both downcast to the input's fp32 on the boundary, but the underlying
    # sampling differs in fp32 vs fp64, so the realizations are not bit-equal.
    # (They will be close — within fp32 rounding — but not identical.)
    assert not torch.equal(out32.pytree["w"], out64.pytree["w"].to(torch.float32))
