# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DPO sigmoid, hinge, and robust loss variants (plan §7.1, §11.3).

Covers:

- :func:`dpo_sigmoid` (Rafailov 2023) — §7.1
- :func:`dpo_hinge` (Liu 2023) — §7.1
- :func:`dpo_robust` (label-smoothed Rafailov) — §7.1

Each variant has ≥ 3 hand-computed reference cases, label-smoothing effect
tests (for sigmoid + robust), vmap-safety contract tests (§3.4), and the
NaN-injection Tier-1 DP-purity contract (§11.3): setting one example's
chosen_logratio to NaN must only corrupt that example's output, not its
neighbours.

Imports target the concrete implementation paths because the public façade
wiring is handled by unit γ.W.
"""

from __future__ import annotations

import math

import torch
from torch.func import grad, vmap

from opaque.api.alignment.dpo.loss._hinge import dpo_hinge
from opaque.api.alignment.dpo.loss._robust import dpo_robust
from opaque.api.alignment.dpo.loss._sigmoid import dpo_sigmoid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _t(x: float) -> torch.Tensor:
    """Create a 0-dim float32 tensor."""
    return torch.tensor(x, dtype=torch.float32)


def _batch(vals: list[float]) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float32)


# ---------------------------------------------------------------------------
# dpo_sigmoid — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_sigmoid_delta0_beta1_ls0() -> None:
    """delta=0, beta=1, ls=0 → -log(sigma(0)) = log(2) ≈ 0.6931."""
    loss = dpo_sigmoid(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.isfinite(loss)
    assert torch.allclose(loss, _t(math.log(2.0)), atol=_ATOL)


def test_sigmoid_delta1_beta1_ls0() -> None:
    """delta=1, beta=1, ls=0 → -log(sigma(1)) ≈ 0.3133."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = dpo_sigmoid(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_delta_neg1_beta1_ls0() -> None:
    """delta=-1, beta=1, ls=0 → -log(sigma(-1)) ≈ 1.3133 (penalises reversal)."""
    expected = -math.log(1.0 / (1.0 + math.exp(1.0)))
    loss = dpo_sigmoid(_t(-1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_delta2_beta05_ls0() -> None:
    """delta=2, beta=0.5 → same as delta=1, beta=1 (scaling equivalence)."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = dpo_sigmoid(_t(2.0), _t(0.0), beta=0.5, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# dpo_sigmoid — label_smoothing effect


def test_sigmoid_ls_increases_loss_for_positive_delta() -> None:
    """Label smoothing blends in the reversed term → higher loss for delta > 0."""
    loss_no_ls = dpo_sigmoid(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    loss_ls = dpo_sigmoid(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert loss_ls > loss_no_ls


def test_sigmoid_ls_reference_value() -> None:
    """delta=2, beta=1, ls=0.1 → hand-computed blend ≈ 0.3269."""
    sig2 = 1.0 / (1.0 + math.exp(-2.0))
    sig_neg2 = 1.0 / (1.0 + math.exp(2.0))
    expected = -math.log(sig2) * 0.9 - math.log(sig_neg2) * 0.1
    loss = dpo_sigmoid(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_sigmoid_ls0_matches_ls_formula_explicitly() -> None:
    """At ls=0, the formula collapses to -logsigmoid(beta*delta)."""
    c, r = _t(1.5), _t(0.5)
    delta = 1.0
    expected = -math.log(1.0 / (1.0 + math.exp(-delta)))
    loss = dpo_sigmoid(c, r, beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# ---------------------------------------------------------------------------
# dpo_hinge — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_hinge_delta0_beta1() -> None:
    """delta=0, beta=1 → relu(1 - 0) = 1.0."""
    loss = dpo_hinge(_t(0.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(1.0), atol=_ATOL)


def test_hinge_delta2_beta1_zero() -> None:
    """delta=2, beta=1 → relu(1 - 2) = 0.0 (margin satisfied)."""
    loss = dpo_hinge(_t(2.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(0.0), atol=_ATOL)


def test_hinge_delta05_beta1() -> None:
    """delta=0.5, beta=1 → relu(1 - 0.5) = 0.5."""
    loss = dpo_hinge(_t(0.5), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(0.5), atol=_ATOL)


def test_hinge_delta2_beta03() -> None:
    """delta=2, beta=0.3 → relu(1 - 0.6) = 0.4."""
    loss = dpo_hinge(_t(2.0), _t(0.0), beta=0.3)
    assert torch.allclose(loss, _t(0.4), atol=_ATOL)


def test_hinge_delta_neg1_beta1() -> None:
    """delta=-1, beta=1 → relu(1 - (-1)) = 2.0 (chosen < rejected)."""
    loss = dpo_hinge(_t(-1.0), _t(0.0), beta=1.0)
    assert torch.allclose(loss, _t(2.0), atol=_ATOL)


def test_hinge_uses_relu_not_python_branch() -> None:
    """Differentiable at 0: the subgradient is 0 exactly at the hinge boundary."""
    # At the hinge boundary (beta*delta == 1), the relu subgradient selects 0.
    # torch.relu is differentiable a.e.; autograd picks subgrad=0 at 0.
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.zeros(1)
    loss = dpo_hinge(x, y, beta=1.0)
    loss.backward()
    # At delta=1 the relu input is 0; its subgradient = 0 → grad on x is 0
    assert x.grad is not None
    assert x.grad.item() == 0.0


# ---------------------------------------------------------------------------
# dpo_robust — hand-computed reference cases
# ---------------------------------------------------------------------------


def test_robust_delta0_beta1_ls0() -> None:
    """delta=0, ls=0 → same as sigmoid: log(2) ≈ 0.6931."""
    loss = dpo_robust(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(math.log(2.0)), atol=_ATOL)


def test_robust_delta1_beta1_ls0() -> None:
    """delta=1, ls=0 → reduces to -log(sigma(1)) ≈ 0.3133 (same as sigmoid)."""
    expected = -math.log(1.0 / (1.0 + math.exp(-1.0)))
    loss = dpo_robust(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_delta2_beta1_ls0() -> None:
    """delta=2, ls=0 → -log(sigma(2)) ≈ 0.1269."""
    sig2 = 1.0 / (1.0 + math.exp(-2.0))
    expected = -math.log(sig2)
    loss = dpo_robust(_t(2.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


# dpo_robust — label_smoothing effect


def test_robust_ls_reference_value_delta0() -> None:
    """delta=0, ls=0.1 → normalised blend; denominator (1-0.2)=0.8."""
    # logsigmoid(0) == logsigmoid(-0) == log(0.5)
    # numerator: -log(0.5)*0.9 + log(0.5)*0.1 = log(2)*(0.9 - 0.1) = log(2)*0.8
    # divided by 0.8 → log(2), same as ls=0 for this symmetric point.
    expected = math.log(2.0)
    loss = dpo_robust(_t(0.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_ls_reference_value_delta1() -> None:
    """delta=1, ls=0.1 → hand-computed: (-log(sig(1))*0.9 + log(sig(-1))*0.1) / 0.8."""
    sig1 = 1.0 / (1.0 + math.exp(-1.0))
    sig_neg1 = 1.0 / (1.0 + math.exp(1.0))
    expected = (-math.log(sig1) * 0.9 + math.log(sig_neg1) * 0.1) / 0.8
    loss = dpo_robust(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.1)
    assert torch.allclose(loss, _t(expected), atol=_ATOL)


def test_robust_ls_changes_loss_vs_sigmoid() -> None:
    """At ls>0 robust and sigmoid differ: robust normalises by (1-2*ls)."""
    delta_c, delta_r = _t(1.5), _t(0.5)
    ls = 0.15
    loss_sig = dpo_sigmoid(delta_c, delta_r, beta=1.0, label_smoothing=ls)
    loss_rob = dpo_robust(delta_c, delta_r, beta=1.0, label_smoothing=ls)
    # They should differ (robust divides by 0.7, sigmoid does not normalise)
    assert not torch.allclose(loss_sig, loss_rob, atol=_ATOL)


def test_robust_ls0_equals_sigmoid_ls0() -> None:
    """At ls=0 the robust loss must match the sigmoid loss exactly."""
    c, r = _t(1.3), _t(0.2)
    loss_sig = dpo_sigmoid(c, r, beta=0.1, label_smoothing=0.0)
    loss_rob = dpo_robust(c, r, beta=0.1, label_smoothing=0.0)
    assert torch.allclose(loss_sig, loss_rob, atol=_ATOL)


def test_robust_ls_near_05_is_large() -> None:
    """As ls → 0.5 the denominator → 0; loss magnitude grows without bound."""
    # At ls=0.49 the denominator is 0.02; loss should be large.
    loss = dpo_robust(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.49)
    assert torch.isfinite(loss)  # still finite
    loss_ls0 = dpo_robust(_t(1.0), _t(0.0), beta=1.0, label_smoothing=0.0)
    assert loss.abs() > loss_ls0.abs()


# ---------------------------------------------------------------------------
# vmap-safety: sigmoid + hinge
# ---------------------------------------------------------------------------


def test_sigmoid_vmap_finite_grads() -> None:
    """vmap(grad(dpo_sigmoid)) over (4,) batch produces finite per-example grads."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return dpo_sigmoid(c, r, beta=0.1).sum()

    # grad(argnums=(0,1)) returns a 2-tuple; vmap maps over the batch axis
    graded = grad(loss_fn, argnums=(0, 1))
    vmapped = vmap(graded)

    chosen = torch.linspace(-1.0, 1.0, 4)
    rejected = torch.zeros(4)
    gc, _gr = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    assert torch.all(torch.isfinite(gc))


def test_hinge_vmap_finite_grads() -> None:
    """vmap(grad(dpo_hinge)) over (4,) batch produces finite per-example grads."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return dpo_hinge(c, r, beta=0.1).sum()

    # grad(argnums=(0,1)) returns a 2-tuple; vmap maps over the batch axis
    graded = grad(loss_fn, argnums=(0, 1))
    vmapped = vmap(graded)

    chosen = torch.linspace(-0.5, 1.5, 4)
    rejected = torch.zeros(4)
    gc, _gr = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    assert torch.all(torch.isfinite(gc))


def test_sigmoid_vmap_correct_gradient_direction() -> None:
    """Grad of sigmoid loss w.r.t. chosen_logratio is negative (loss decreases as chosen increases)."""

    def loss_fn(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return dpo_sigmoid(c, r, beta=1.0).sum()

    # Default argnums=0: returns single tensor (grad w.r.t. chosen_logratio)
    graded = grad(loss_fn)
    vmapped = vmap(graded)

    chosen = torch.tensor([0.0, 1.0, -0.5, 2.0])
    rejected = torch.zeros(4)
    gc = vmapped(chosen, rejected)

    assert gc.shape == (4,)
    # Increasing chosen_logratio decreases the loss → gradient is negative
    assert torch.all(gc < 0)


# ---------------------------------------------------------------------------
# NaN-injection Tier-1 DP-purity contract (§11.3)
# ---------------------------------------------------------------------------


def test_sigmoid_nan_injection_local() -> None:
    """NaN at index 2 of chosen_logratio must corrupt only output[2]."""
    chosen = torch.tensor([0.5, 1.0, float("nan"), -0.5])
    rejected = torch.zeros(4)
    out = dpo_sigmoid(chosen, rejected, beta=1.0)

    assert torch.isfinite(out[0])
    assert torch.isfinite(out[1])
    assert torch.isnan(out[2])
    assert torch.isfinite(out[3])


def test_hinge_nan_injection_local() -> None:
    """NaN at index 2 of chosen_logratio must corrupt only output[2]."""
    chosen = torch.tensor([0.5, 1.0, float("nan"), -0.5])
    rejected = torch.zeros(4)
    out = dpo_hinge(chosen, rejected, beta=1.0)

    assert torch.isfinite(out[0])
    assert torch.isfinite(out[1])
    assert torch.isnan(out[2])
    assert torch.isfinite(out[3])


def test_robust_nan_injection_local() -> None:
    """NaN at index 1 of rejected_logratio must corrupt only output[1]."""
    chosen = torch.zeros(4)
    rejected = torch.tensor([0.5, float("nan"), 1.0, -0.5])
    out = dpo_robust(chosen, rejected, beta=1.0, label_smoothing=0.1)

    assert torch.isfinite(out[0])
    assert torch.isnan(out[1])
    assert torch.isfinite(out[2])
    assert torch.isfinite(out[3])


def test_sigmoid_nan_injection_rejected() -> None:
    """NaN injected into rejected_logratio at index 0 corrupts only output[0]."""
    chosen = torch.ones(4)
    rejected = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    out = dpo_sigmoid(chosen, rejected, beta=1.0)

    assert torch.isnan(out[0])
    assert torch.isfinite(out[1])
    assert torch.isfinite(out[2])
    assert torch.isfinite(out[3])


# ---------------------------------------------------------------------------
# 0-dim scalar compatibility (vmap maps over 0-dim within batch)
# ---------------------------------------------------------------------------


def test_sigmoid_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(1.0)
    r = torch.tensor(0.0)
    out = dpo_sigmoid(c, r, beta=1.0)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_hinge_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(0.5)
    r = torch.tensor(0.0)
    out = dpo_hinge(c, r, beta=1.0)
    assert out.ndim == 0
    assert torch.allclose(out, torch.tensor(0.5), atol=_ATOL)


def test_robust_zero_dim_input() -> None:
    """0-dim tensor inputs produce a 0-dim tensor output."""
    c = torch.tensor(1.0)
    r = torch.tensor(0.0)
    out = dpo_robust(c, r, beta=1.0, label_smoothing=0.0)
    assert out.ndim == 0
    assert torch.isfinite(out)
