# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the DPO-specific helpers (plan §7.1, §3.4).

Covers the three combinator/weighting helpers that live inside ``loss/dpo/``:

- :func:`mpo_combine` — the TRL ``loss_type=list`` weighted blend (MPO). A
  weighted-sum reference case, the subset-of-keys behaviour, and the
  missing-key ``KeyError`` guard.
- :func:`wpo_weights` — the WPO per-example reweighting (arXiv:2406.11827). A
  hand-computed avg-logp→exp case, the **detach** invariant (a non-detached
  weight would couple the gradient and break DP Tier 1, §3.3), and the
  all-zero-mask ``clamp(min=1)`` div-by-zero guard.
- :func:`ld_dpo_split` — the LD-DPO length-desensitised logp split
  (arXiv:2409.10524). Full-prefix coverage reducing to the plain masked-sum,
  the ``alpha=0`` prefix-only case, and a ``torch.func.vmap(torch.func.grad)``
  finite-gradient contract test (§3.4) with a tensor ``shared_prefix_len``.

Imports target the concrete impl paths because the public façade is wired by a
later unit (γ.W).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.alignment.dpo.loss._ld_dpo import ld_dpo_split
from opaque.api.alignment.dpo.loss._mpo import mpo_combine
from opaque.api.alignment.dpo.loss._wpo import wpo_weights

# ---------------------------------------------------------------------------
# mpo_combine — weighted blend (TRL loss_type=list)
# ---------------------------------------------------------------------------


def test_mpo_combine_weighted_sum() -> None:
    """{sigmoid, sft} with weights {1.0, 0.5} → t1 + 0.5 * t2."""
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([10.0, 20.0, 30.0])
    out = mpo_combine({"sigmoid": t1, "sft": t2}, {"sigmoid": 1.0, "sft": 0.5})
    assert torch.allclose(out, t1 + 0.5 * t2)


def test_mpo_combine_subset_weights_ok() -> None:
    """weights may select a strict subset of the available losses."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([4.0, 8.0])
    out = mpo_combine({"sigmoid": t1, "sft": t2}, {"sigmoid": 2.0})
    assert torch.allclose(out, 2.0 * t1)


def test_mpo_combine_single_term() -> None:
    """A single-term blend returns the scaled term."""
    t1 = torch.tensor([1.0, 2.0, 3.0])
    out = mpo_combine({"sigmoid": t1}, {"sigmoid": 0.25})
    assert torch.allclose(out, 0.25 * t1)


def test_mpo_combine_broadcasts() -> None:
    """Loss tensors that broadcast together are combined per broadcast rules."""
    scalar = torch.tensor(1.0)
    vec = torch.tensor([1.0, 2.0, 3.0])
    out = mpo_combine({"a": scalar, "b": vec}, {"a": 2.0, "b": 1.0})
    assert torch.allclose(out, 2.0 * scalar + vec)
    assert out.shape == (3,)


def test_mpo_combine_missing_key_raises() -> None:
    """A weight key absent from losses raises KeyError."""
    t1 = torch.tensor([1.0, 2.0])
    with pytest.raises(KeyError):
        mpo_combine({"sigmoid": t1}, {"sigmoid": 1.0, "sft": 0.5})


def test_mpo_combine_vmap_grad_finite() -> None:
    """vmap(grad(mpo_combine-sum)) over (B,) inputs yields finite grads (§3.4)."""
    a = torch.tensor([0.5, -1.0, 2.0, 0.0])
    b = torch.tensor([1.0, 0.0, -2.0, 3.0])

    def per_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mpo_combine({"a": x, "b": y}, {"a": 1.0, "b": 0.5}).sum()

    g = vmap(grad(per_example))(a, b)
    assert g.shape == (4,)
    assert torch.isfinite(g).all()
    # grad w.r.t. the "a" term is its weight (1.0).
    assert torch.allclose(g, torch.ones(4))


# ---------------------------------------------------------------------------
# wpo_weights — per-example reweighting (arXiv:2406.11827)
# ---------------------------------------------------------------------------


def test_wpo_weights_hand_computed() -> None:
    """avg_logp = masked-mean per row, weight = exp(avg_logp)."""
    # Row 0: logps [-1, -2, 0], mask [1, 1, 0] → avg = -1.5 → exp(-1.5).
    # Row 1: logps [-0.5, -0.5, -0.5], mask [1, 1, 1] → avg = -0.5 → exp(-0.5).
    logps = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    expected = torch.tensor([torch.tensor(-1.5).exp(), torch.tensor(-0.5).exp()])
    assert out.shape == (2,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_wpo_weights_is_detached() -> None:
    """The weight is detached even with grad-tracking inputs (DP Tier 1, §3.3)."""
    logps = torch.tensor([[-1.0, -2.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    assert out.requires_grad is False
    assert out.grad_fn is None


def test_wpo_weights_all_zero_mask_no_div0() -> None:
    """An all-zero mask row uses clamp(min=1): avg_logp = 0 → weight = 1."""
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]])
    mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = wpo_weights(logps, mask)
    assert torch.isfinite(out).all()
    # Row 0: numerator is 0 (all masked), denom clamped to 1 → exp(0) = 1.
    assert torch.allclose(out[0], torch.tensor(1.0), atol=1e-6)


def test_wpo_weights_vmap_safe() -> None:
    """wpo_weights runs under vmap over a batch axis and stays detached."""
    logps = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    out = vmap(wpo_weights)(logps, mask)
    assert out.shape == (4,)
    assert out.requires_grad is False


# ---------------------------------------------------------------------------
# ld_dpo_split — length-desensitised logp split (arXiv:2409.10524)
# ---------------------------------------------------------------------------


def test_ld_dpo_full_prefix_equals_masked_sum() -> None:
    """shared_prefix_len covering the whole sequence → plain masked-sum."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=4, alpha=0.3)
    plain = (logps * mask).sum(dim=-1)
    assert torch.allclose(out, plain, atol=1e-6)


def test_ld_dpo_alpha_one_equals_masked_sum() -> None:
    """alpha=1 recovers the plain masked-sum regardless of prefix length."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=1, alpha=1.0)
    plain = (logps * mask).sum(dim=-1)
    assert torch.allclose(out, plain, atol=1e-6)


def test_ld_dpo_alpha_zero_prefix_only() -> None:
    """alpha=0 → only the prefix tokens contribute."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=2, alpha=0.0)
    # Only positions 0 and 1 count: -1 + -2 = -3.
    assert torch.allclose(out, torch.tensor([-3.0]), atol=1e-6)


def test_ld_dpo_tail_weighting() -> None:
    """Tail tokens (pos >= prefix) are weighted by alpha."""
    logps = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = ld_dpo_split(logps, mask, shared_prefix_len=2, alpha=0.5)
    # prefix: -1 + -2 = -3 ; tail: 0.5 * (-3 + -4) = -3.5 ; total -6.5.
    assert torch.allclose(out, torch.tensor([-6.5]), atol=1e-6)


def test_ld_dpo_per_example_tensor_prefix() -> None:
    """A per-example tensor shared_prefix_len broadcasts against the pos axis."""
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
    mask = torch.ones(2, 3)
    prefix = torch.tensor([[1], [3]])  # (B, 1) broadcasts against (T,)
    out = ld_dpo_split(logps, mask, shared_prefix_len=prefix, alpha=0.0)
    # Row 0: prefix len 1 → only pos 0 → -1.
    # Row 1: prefix len 3 → whole seq → -6.
    assert torch.allclose(out, torch.tensor([-1.0, -6.0]), atol=1e-6)


def test_ld_dpo_vmap_grad_finite_tensor_prefix() -> None:
    """vmap(grad(ld_dpo_split-sum)) over (4, T) → finite (§3.4 vmap-safety)."""
    seq_len = 5
    lp = torch.randn(4, seq_len)
    mask = torch.ones(4, seq_len)

    def per_example(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return ld_dpo_split(x, m, shared_prefix_len=2, alpha=0.5).sum()

    g = vmap(grad(per_example))(lp, mask)
    assert g.shape == (4, seq_len)
    assert torch.isfinite(g).all()
    # Per-token grad equals the position weight: 1.0 for pos<2, else 0.5.
    expected_row = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5])
    assert torch.allclose(g, expected_row.expand(4, seq_len), atol=1e-6)


def test_ld_dpo_vmap_grad_per_example_tensor_prefix() -> None:
    """vmap(grad) with a vmapped per-example tensor prefix stays finite (§3.4).

    Exercises the broadcast of a per-example ``shared_prefix_len`` tensor
    against the position axis *inside* the vmap region (adversarial check that
    the comparison/where path is genuinely vmap-safe with a tensor prefix).
    """
    seq_len = 4
    lp = torch.randn(4, seq_len)
    mask = torch.ones(4, seq_len)
    prefix = torch.tensor([[1], [2], [3], [4]])  # (B, 1)

    def per_example(x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return ld_dpo_split(x, m, shared_prefix_len=p, alpha=0.5).sum()

    g = vmap(grad(per_example))(lp, mask, prefix)
    assert g.shape == (4, seq_len)
    assert torch.isfinite(g).all()
