# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the chunked fused-linear SFT loss (opt-in memory path).

Mirrors the DPO kernel tests (``tests/kernel/test_fused_linear_dpo.py``):

- **Eager-vs-fused parity**: ``fused_linear_sft_loss`` matches the all-at-once
  eager reference ``loss_fn(hidden @ Wᵀ, labels)`` for both ``nll_loss`` and
  ``dft_loss`` and every ``chunk_size`` — chunking is a pure batch partition.
- **chunk_size invariance**: identical result across ``chunk_size`` ∈ {1..8}.
- **grad composability**: ``torch.func.grad`` w.r.t. both ``hidden_states`` and
  ``lm_head_weight`` is finite and matches the eager-reference gradient (the
  gradient flows through the chunk boundaries identically).
- **autocast sanity** on CPU and the ``chunk_size < 1`` error path.

float64 is used so parity/gradient tolerances are not dominated by float32 noise
(the fused path is the same ops on a batch slice, so parity is effectively exact).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opaque.api.alignment.sft.loss._dft import dft_loss
from opaque.api.alignment.sft.loss._fused import fused_linear_sft_loss
from opaque.api.alignment.sft.loss._nll import nll_loss

_B, _T, _H, _V = 4, 6, 8, 16
_ATOL = 1e-6


def _make_inputs(*, seed: int = 0, dtype: torch.dtype = torch.float64):
    """Deterministic (hidden, lm_head, labels) bundle with an ignore span."""
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_B, _T, _H, generator=gen, dtype=dtype)
    lm_head = torch.randn(_V, _H, generator=gen, dtype=dtype)
    labels = torch.randint(0, _V, (_B, _T), generator=gen)
    # Mask the first two positions (prompt boundary): after the causal shift this
    # leaves at least one ``-100`` in the shifted labels, exercising the mask.
    labels[:, :2] = -100
    return hidden, lm_head, labels


def _eager(hidden, lm_head, labels, loss_fn):
    """All-at-once reference: materialise the full (B, T, V) logits."""
    logits = hidden @ lm_head.transpose(-2, -1)
    return loss_fn(logits, labels)


@pytest.mark.parametrize("loss_fn", [nll_loss, dft_loss], ids=["nll", "dft"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_eager_vs_fused_parity(loss_fn, chunk_size):
    hidden, lm_head, labels = _make_inputs(seed=1)
    expected = _eager(hidden, lm_head, labels, loss_fn)
    actual = fused_linear_sft_loss(
        hidden, lm_head, labels, loss_fn=loss_fn, chunk_size=chunk_size
    )
    assert actual.shape == (_B,)
    assert torch.allclose(actual, expected, atol=_ATOL, rtol=0.0)


@pytest.mark.parametrize("loss_fn", [nll_loss, dft_loss], ids=["nll", "dft"])
def test_chunk_size_invariance(loss_fn):
    hidden, lm_head, labels = _make_inputs(seed=2)
    base = fused_linear_sft_loss(hidden, lm_head, labels, loss_fn=loss_fn, chunk_size=1)
    for chunk_size in (2, 3, 4, 8):  # 3 -> uneven final chunk; 8 -> single chunk
        other = fused_linear_sft_loss(
            hidden, lm_head, labels, loss_fn=loss_fn, chunk_size=chunk_size
        )
        assert torch.allclose(base, other, atol=_ATOL, rtol=0.0), (
            f"chunk_size={chunk_size} diverged from chunk_size=1"
        )


@pytest.mark.parametrize("loss_fn", [nll_loss, dft_loss], ids=["nll", "dft"])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_grad_composability(loss_fn, chunk_size):
    hidden, lm_head, labels = _make_inputs(seed=3)

    def fused_h(h):
        return fused_linear_sft_loss(
            h, lm_head, labels, loss_fn=loss_fn, chunk_size=chunk_size
        ).sum()

    def eager_h(h):
        return _eager(h, lm_head, labels, loss_fn).sum()

    fused_g = grad(fused_h)(hidden)
    assert fused_g.shape == hidden.shape
    assert torch.isfinite(fused_g).all()
    assert torch.allclose(fused_g, grad(eager_h)(hidden), atol=_ATOL, rtol=0.0)

    # Gradient w.r.t. the lm_head weight must match too (full-FT case); the
    # frozen case is simply *not* differentiating this argument.
    def fused_w(w):
        return fused_linear_sft_loss(
            hidden, w, labels, loss_fn=loss_fn, chunk_size=chunk_size
        ).sum()

    def eager_w(w):
        return _eager(hidden, w, labels, loss_fn).sum()

    assert torch.allclose(grad(fused_w)(lm_head), grad(eager_w)(lm_head), atol=_ATOL)


def test_invalid_chunk_size_raises():
    hidden, lm_head, labels = _make_inputs(seed=4)
    with pytest.raises(ValueError, match="chunk_size"):
        fused_linear_sft_loss(hidden, lm_head, labels, chunk_size=0)


def test_autocast_runs_finite_on_cpu():
    """Inside a bf16 CPU autocast region the kernel stays finite and tracks fp32."""
    hidden, lm_head, labels = _make_inputs(seed=5, dtype=torch.float32)
    plain = fused_linear_sft_loss(hidden, lm_head, labels)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        wrapped = fused_linear_sft_loss(hidden, lm_head, labels)
    assert torch.isfinite(wrapped).all()
    assert torch.allclose(wrapped.float(), plain.float(), atol=1e-1, rtol=0.0)
