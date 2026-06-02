# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused-linear SFT losses ``fused_nll_loss`` / ``fused_dft_loss``.

Each is a **per-example** drop-in for its eager twin (``nll_loss`` / ``dft_loss``)
that takes hidden states + the ``lm_head`` weight instead of logits, and is
driven by ``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path).

- **CPU** (float64): the fused function takes its eager fallback
  (``loss(hidden @ W.T, labels)``), so these assert the per-example contract,
  shapes, and ``vmap(grad)`` composability against the eager reference.
- **GPU** (bf16, ``[patches]``): the same parity, now exercising the fused
  opaque-patches linear-CE kernel path (NLL plain; DFT via ``use_token_scaling``).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.alignment.sft.loss import (
    dft_loss,
    fused_dft_loss,
    fused_nll_loss,
    nll_loss,
)

_B, _T, _H, _V = 4, 9, 6, 17

# (fused, eager) twins, keyed for parametrization.
_PAIRS = {"nll": (fused_nll_loss, nll_loss), "dft": (fused_dft_loss, dft_loss)}


def _make_inputs(seed: int, *, dtype=torch.float64, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(_B, _T, _H, generator=gen, dtype=dtype)
    weight = torch.randn(_V, _H, generator=gen, dtype=dtype)
    labels = torch.randint(0, _V, (_B, _T), generator=gen)
    labels[:, :2] = -100  # a prompt span ignored per example
    return hidden.to(device), weight.to(device), labels.to(device)


@pytest.mark.parametrize("name", sorted(_PAIRS))
def test_fused_matches_eager_forward_cpu(name: str) -> None:
    """Per-example forward (under vmap) matches eager ``loss(hidden @ W.T, …)``."""
    fused, eager = _PAIRS[name]
    hidden, weight, labels = _make_inputs(seed=1)

    got = vmap(lambda h, lab: fused(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: eager(h @ weight.T, lab))(hidden, labels)

    assert got.shape == (_B,)
    assert torch.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("name", sorted(_PAIRS))
def test_fused_vmap_grad_matches_eager_cpu(name: str) -> None:
    """``vmap(grad(...))`` w.r.t. hidden and weight matches the eager reference."""
    fused, eager = _PAIRS[name]
    hidden, weight, labels = _make_inputs(seed=2)

    g_h_fused = vmap(grad(lambda h, lab: fused(h, weight, lab)))(hidden, labels)
    g_h_eager = vmap(grad(lambda h, lab: eager(h @ weight.T, lab)))(hidden, labels)
    assert g_h_fused.shape == hidden.shape
    assert torch.allclose(g_h_fused, g_h_eager, atol=1e-10)

    # Grad w.r.t. the shared weight, summed over the per-example batch.
    g_w_fused = grad(
        lambda w: vmap(lambda h, lab: fused(h, w, lab))(hidden, labels).sum()
    )(weight)
    g_w_eager = grad(
        lambda w: vmap(lambda h, lab: eager(h @ w.T, lab))(hidden, labels).sum()
    )(weight)
    assert torch.allclose(g_w_fused, g_w_eager, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("name", sorted(_PAIRS))
def test_fused_lce_path_matches_eager_gpu(name: str) -> None:
    """The fused kernel path (CUDA + bf16) matches the eager reference.

    Exercises the opaque-patches linear-CE kernel (NLL plain; DFT via
    ``use_token_scaling``); bf16 matmul is coarse, so tolerances are loose.
    """
    fused, eager = _PAIRS[name]
    hidden, weight, labels = _make_inputs(seed=3, dtype=torch.bfloat16, device="cuda")

    got = vmap(lambda h, lab: fused(h, weight, lab))(hidden, labels)
    want = vmap(lambda h, lab: eager(h @ weight.T, lab))(hidden, labels)
    assert torch.allclose(got.float(), want.float(), atol=1e-2, rtol=0.0)

    g_fused = vmap(grad(lambda h, lab: fused(h, weight, lab)))(hidden, labels)
    g_eager = vmap(grad(lambda h, lab: eager(h @ weight.T, lab)))(hidden, labels)
    assert torch.isfinite(g_fused).all()
    assert torch.allclose(g_fused.float(), g_eager.float(), atol=1e-2, rtol=0.0)
