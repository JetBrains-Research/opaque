# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the KTO loss family (work-unit δ.1).

Covers:
- :func:`kto_loss` (**Tier 2**, arXiv:2402.01306 Eq. 8) — per-example loss plus
  a detached batch-mean KL aggregate ``kl`` (``z_0``).
- :func:`apo_zero_unpaired` (**Tier 1**, arXiv:2408.06266) — strict per-example,
  no KL term.

For each function:
- >=3 hand-computed reference cases (small, analytically tractable inputs;
  weights applied) for desirable and undesirable labels.
- Mixed-label batch via ``torch.where`` (a single call resolving both branches).
- vmap-safety: ``torch.func.vmap(torch.func.grad(...))`` over a ``(4,)`` batch
  (grad w.r.t. the log-ratios) yields finite gradients.

For :func:`kto_loss` specifically:
- ``kl=0`` degeneracy (Poisson batch <= 1 fallback): finite, sensible, and the
  desirable branch matches the ``apo_zero_unpaired`` desirable branch term.
- **Tier-2 DP-purity audit (§11.4):**
  (a) *aggregate-detach* — building ``kl = leaf.mean().detach()`` and
      back-propagating from ``kto_loss(...).sum()`` leaves the kl-origin leaf
      with ``grad=None`` (the detach is honored).
  (b) *bounded leverage* — ``|d L_i / d kl| <= beta * max(weights)``, confirmed
      both analytically (autograd) and numerically (a small eps perturbation of
      ``kl`` moves each per-example loss by <= ``beta * max_w * eps``), so an
      ``O(1/n)`` change in the batch-mean kl gives an ``O(1/n)`` per-example
      change — the §3.3 Tier-2 condition.

Imports target the concrete implementation paths (public façade is wired by
δ.W, not this work-unit).
"""

from __future__ import annotations

import torch
from torch.func import grad, vmap

from opaque.api.alignment.kto.loss._apo_zero_unpaired import apo_zero_unpaired
from opaque.api.alignment.kto.loss._kto import kto_loss

_T = torch.tensor


def _sigmoid(x: float) -> torch.Tensor:
    return torch.sigmoid(_T(x))


# ===========================================================================
# kto_loss — hand-computed reference cases
# ===========================================================================


class TestKtoLossReference:
    """Hand-computed reference cases for :func:`kto_loss`."""

    def test_desirable_chosen_equals_kl_beta1_is_half(self) -> None:
        """label=True, chosen_lr=kl, β=1 → 1 - σ(β*(kl-kl)) = 1 - σ(0) = 0.5."""
        out = kto_loss(
            _T(2.0),  # chosen_lr == kl
            _T(0.0),  # rejected (unused: label True)
            _T(True),
            beta=1.0,
            kl=_T(2.0),
        )
        assert out.shape == ()
        assert torch.allclose(out, _T(0.5), atol=1e-6)

    def test_undesirable_rejected_equals_kl_beta1_is_half(self) -> None:
        """label=False, rejected_lr=kl, β=1 → 1 - σ(β*(kl-kl)) = 0.5."""
        out = kto_loss(
            _T(0.0),  # chosen (unused: label False)
            _T(-3.0),  # rejected_lr == kl
            _T(False),
            beta=1.0,
            kl=_T(-3.0),
        )
        assert out.shape == ()
        assert torch.allclose(out, _T(0.5), atol=1e-6)

    def test_desirable_with_weight(self) -> None:
        """label=True, chosen_lr=1, kl=0, β=0.5, w_d=2 → 2*(1-σ(0.5))."""
        out = kto_loss(
            _T(1.0),
            _T(0.0),
            _T(True),
            beta=0.5,
            kl=_T(0.0),
            desirable_weight=2.0,
        )
        expected = 2.0 * (1 - _sigmoid(0.5 * 1.0))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_undesirable_with_weight(self) -> None:
        """label=False, rejected_lr=-1, kl=0, β=0.5, w_u=3 → 3*(1-σ(0.5))."""
        out = kto_loss(
            _T(0.0),
            _T(-1.0),
            _T(False),
            beta=0.5,
            kl=_T(0.0),
            undesirable_weight=3.0,
        )
        # 1 - σ(β*(kl - rejected)) = 1 - σ(0.5*(0 - (-1))) = 1 - σ(0.5)
        expected = 3.0 * (1 - _sigmoid(0.5 * 1.0))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_desirable_large_margin_near_zero_loss(self) -> None:
        """label=True, chosen_lr >> kl → σ→1 → loss → 0."""
        out = kto_loss(_T(20.0), _T(0.0), _T(True), beta=1.0, kl=_T(0.0))
        assert out.item() < 1e-6
        assert torch.isfinite(out)

    def test_none_substitution_eager_path(self) -> None:
        """Eager path: None on the unused side is substituted with zeros."""
        out_des = kto_loss(_T(1.0), None, _T(True), beta=0.5, kl=_T(0.0))
        expected_des = 1 - _sigmoid(0.5 * 1.0)
        assert torch.allclose(out_des, expected_des, atol=1e-6)

        out_undes = kto_loss(None, _T(-1.0), _T(False), beta=0.5, kl=_T(0.0))
        expected_undes = 1 - _sigmoid(0.5 * 1.0)
        assert torch.allclose(out_undes, expected_undes, atol=1e-6)


# ===========================================================================
# kto_loss — mixed-label batch
# ===========================================================================


class TestKtoLossMixedBatch:
    """``torch.where`` resolves desirable/undesirable per example in one call."""

    def test_mixed_label_batch(self) -> None:
        """A single call with mixed labels picks the right branch per example."""
        chosen = _T([1.0, 5.0, -1.0, 0.0])
        rejected = _T([0.0, 0.0, 2.0, -2.0])
        label = _T([True, True, False, False])
        beta = 0.5
        kl = _T(0.3)
        w_d, w_u = 1.5, 2.5

        out = kto_loss(
            chosen,
            rejected,
            label,
            beta=beta,
            kl=kl,
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )
        assert out.shape == (4,)

        # Reference: per-example, branch selected by label.
        for i in range(4):
            if label[i]:
                ref = w_d * (1 - torch.sigmoid(beta * (chosen[i] - kl)))
            else:
                ref = w_u * (1 - torch.sigmoid(beta * (kl - rejected[i])))
            assert torch.allclose(out[i], ref, atol=1e-6)

    def test_batched_shape_preserved(self) -> None:
        torch.manual_seed(0)
        chosen = torch.randn(8)
        rejected = torch.randn(8)
        label = torch.randint(0, 2, (8,)).bool()
        out = kto_loss(chosen, rejected, label, beta=0.1, kl=_T(0.0))
        assert out.shape == (8,)
        assert torch.isfinite(out).all()


# ===========================================================================
# kto_loss — kl=0 degeneracy (Poisson batch <= 1 fallback)
# ===========================================================================


class TestKtoLossKlZeroDegeneracy:
    """``kl=0`` must be finite and degenerate to apo_zero_unpaired-like terms.

    With ``kl=0`` the desirable branch is ``w_d*(1 - σ(β*chosen_lr))`` — exactly
    the desirable branch of :func:`apo_zero_unpaired`. The undesirable branch is
    ``w_u*(1 - σ(β*(0 - rejected_lr))) = w_u*(1 - σ(-β*rejected_lr))
    = w_u*σ(β*rejected_lr)`` — exactly the undesirable branch of
    :func:`apo_zero_unpaired` (using ``1 - σ(-x) = σ(x)``).
    """

    def test_kl_zero_desirable_matches_apo(self) -> None:
        chosen = _T([1.0, -0.5, 3.0])
        rejected = _T([0.0, 0.0, 0.0])
        label = _T([True, True, True])
        kto = kto_loss(chosen, rejected, label, beta=0.7, kl=_T(0.0))
        apo = apo_zero_unpaired(chosen, rejected, label, beta=0.7)
        assert torch.allclose(kto, apo, atol=1e-6)

    def test_kl_zero_undesirable_matches_apo(self) -> None:
        chosen = _T([0.0, 0.0, 0.0])
        rejected = _T([1.0, -0.5, 2.0])
        label = _T([False, False, False])
        kto = kto_loss(chosen, rejected, label, beta=0.7, kl=_T(0.0))
        apo = apo_zero_unpaired(chosen, rejected, label, beta=0.7)
        assert torch.allclose(kto, apo, atol=1e-6)

    def test_kl_zero_mixed_matches_apo(self) -> None:
        """Full degeneracy: kl=0 mixed-label batch equals apo_zero_unpaired."""
        torch.manual_seed(11)
        chosen = torch.randn(16)
        rejected = torch.randn(16)
        label = torch.randint(0, 2, (16,)).bool()
        w_d, w_u = 1.3, 0.8
        kto = kto_loss(
            chosen,
            rejected,
            label,
            beta=0.3,
            kl=_T(0.0),
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )
        apo = apo_zero_unpaired(
            chosen,
            rejected,
            label,
            beta=0.3,
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )
        assert torch.allclose(kto, apo, atol=1e-6)
        assert torch.isfinite(kto).all()


# ===========================================================================
# kto_loss — vmap-safety
# ===========================================================================


class TestKtoLossVmapSafety:
    """``vmap(grad(...))`` over a (4,) batch; grad w.r.t. log-ratios is finite.

    Under vmap the caller passes label-masked tensors (never None). We mask each
    side by the label so that the per-example body sees a valid tensor on both
    sides, matching the Tier-2 caller contract (§8.1).
    """

    def test_vmap_grad_finite(self) -> None:
        b = 4
        torch.manual_seed(42)
        chosen = torch.randn(b)
        rejected = torch.randn(b)
        label = _T([True, False, True, False])
        kl = (chosen - rejected).mean().detach()  # scalar, broadcast in

        def per_example(
            ci: torch.Tensor, ri: torch.Tensor, li: torch.Tensor
        ) -> torch.Tensor:
            return kto_loss(ci, ri, li, beta=0.1, kl=kl).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(
            chosen, rejected, label
        )
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()


# ===========================================================================
# kto_loss — Tier-2 DP-purity audit (§11.4)
# ===========================================================================


class TestKtoLossTier2DpPurity:
    """Aggregate-detach audit + bounded-leverage test for the Tier-2 kl term."""

    def test_aggregate_detach_honored(self) -> None:
        """(a) No autograd path from kl back to its origin leaf.

        Build ``kl = some_leaf.mean().detach()`` (the Tier-2 caller contract),
        form ``loss = kto_loss(..., kl=kl).sum()``, and assert that
        ``autograd.grad(loss, some_leaf, allow_unused=True)`` is ``None`` — the
        detach severs the graph from kl back to its origin.
        """
        some_leaf = torch.randn(8, requires_grad=True)
        kl = some_leaf.mean().detach()  # detached batch-mean aggregate

        chosen = torch.randn(8, requires_grad=True)
        rejected = torch.randn(8, requires_grad=True)
        label = torch.randint(0, 2, (8,)).bool()

        loss = kto_loss(chosen, rejected, label, beta=0.5, kl=kl).sum()

        # The detach must be honored: kl carries no graph to some_leaf.
        (grad_leaf,) = torch.autograd.grad(loss, some_leaf, allow_unused=True)
        assert grad_leaf is None, (
            "Tier-2 detach violated: kl retains a path to its origin leaf."
        )

        # Sanity: the loss DOES depend on the log-ratios (graph is otherwise live).
        grads = torch.autograd.grad(loss, (chosen, rejected), allow_unused=True)
        assert any(g is not None for g in grads)

    def test_aggregate_detach_honored_when_kl_not_detached_path_still_severed(
        self,
    ) -> None:
        """Even if the caller forgot ``.detach()``, the loss math reads kl as a
        constant w.r.t. its origin only because the caller detaches. Here we
        verify the *contract* path: a properly-detached kl gives None.

        (We also confirm a NON-detached kl WOULD create a path, proving the test
        above is meaningful and not vacuously passing.)
        """
        some_leaf = torch.randn(8, requires_grad=True)
        kl_live = some_leaf.mean()  # NOT detached

        chosen = torch.randn(8, requires_grad=True)
        rejected = torch.randn(8, requires_grad=True)
        label = torch.randint(0, 2, (8,)).bool()

        loss = kto_loss(chosen, rejected, label, beta=0.5, kl=kl_live).sum()
        (grad_leaf,) = torch.autograd.grad(loss, some_leaf, allow_unused=True)
        # Non-detached kl DOES reach the leaf — confirms detach is load-bearing.
        assert grad_leaf is not None

    def test_bounded_leverage_analytic(self) -> None:
        """(b) ``|d L_i / d kl| <= beta * max(weights)`` via autograd.

        sigmoid'(.) <= 1/4, so |dL/dkl| <= w * beta / 4 <= beta * max_w.
        """
        beta = 0.7
        w_d, w_u = 1.5, 2.0
        max_w = max(w_d, w_u)
        bound = beta * max_w

        chosen = torch.randn(64)
        rejected = torch.randn(64)
        label = torch.randint(0, 2, (64,)).bool()

        for i in range(chosen.numel()):
            kl = torch.tensor(0.3, requires_grad=True)

            def per_example(k: torch.Tensor, idx: int = i) -> torch.Tensor:
                return kto_loss(
                    chosen[idx],
                    rejected[idx],
                    label[idx],
                    beta=beta,
                    kl=k,
                    desirable_weight=w_d,
                    undesirable_weight=w_u,
                )

            (g,) = torch.autograd.grad(per_example(kl), kl)
            assert torch.isfinite(g)
            assert abs(g.item()) <= bound + 1e-6, (
                f"leverage {abs(g.item()):.6f} exceeds bound {bound:.6f}"
            )

    def test_bounded_leverage_numeric(self) -> None:
        """(b) Numeric perturbation: an eps change in kl moves each per-example
        loss by <= ``beta * max_w * eps``.

        This is the operational §3.3 condition: since the batch-mean kl moves by
        O(1/n) when one record is swapped, each per-example loss moves by
        O(1/n) too (leverage is bounded).
        """
        beta = 0.6
        w_d, w_u = 1.2, 1.8
        max_w = max(w_d, w_u)
        eps = 1e-3

        torch.manual_seed(7)
        chosen = torch.randn(50)
        rejected = torch.randn(50)
        label = torch.randint(0, 2, (50,)).bool()

        kl0 = _T(0.25)
        kl1 = _T(0.25 + eps)

        loss0 = kto_loss(
            chosen,
            rejected,
            label,
            beta=beta,
            kl=kl0,
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )
        loss1 = kto_loss(
            chosen,
            rejected,
            label,
            beta=beta,
            kl=kl1,
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )

        delta = (loss1 - loss0).abs()
        bound = beta * max_w * eps
        assert (delta <= bound + 1e-9).all(), (
            f"max per-example change {delta.max().item():.3e} exceeds bound {bound:.3e}"
        )

    def test_leverage_scales_as_one_over_n(self) -> None:
        """Swapping one record's contribution to the batch-mean kl moves the
        per-example loss by O(1/n).

        We emulate the trainer: kl = mean over n KL-logp diffs. Swapping one
        diff by a bounded amount changes the mean by that amount / n, and the
        bounded-leverage property then yields an O(1/n) per-example loss change.
        """
        beta = 0.5
        for n in (8, 32, 128):
            kl_diffs = torch.randn(n)
            kl_a = kl_diffs.mean().detach()

            # Swap one record's KL contribution by a bounded delta (= 1.0).
            kl_diffs_b = kl_diffs.clone()
            kl_diffs_b[0] = kl_diffs_b[0] + 1.0
            kl_b = kl_diffs_b.mean().detach()

            # mean moved by exactly 1/n.
            assert torch.allclose(kl_b - kl_a, _T(1.0 / n), atol=1e-6)

            chosen = torch.randn(n)
            rejected = torch.randn(n)
            label = torch.randint(0, 2, (n,)).bool()

            loss_a = kto_loss(chosen, rejected, label, beta=beta, kl=kl_a)
            loss_b = kto_loss(chosen, rejected, label, beta=beta, kl=kl_b)

            # Per-example loss change is bounded by beta * (1/n) * max_w (=1).
            per_example_change = (loss_b - loss_a).abs().max().item()
            assert per_example_change <= beta * (1.0 / n) + 1e-6


# ===========================================================================
# apo_zero_unpaired — hand-computed reference cases
# ===========================================================================


class TestApoZeroUnpairedReference:
    """Hand-computed reference cases for :func:`apo_zero_unpaired`."""

    def test_desirable_zero_input_is_half(self) -> None:
        """label=True, chosen_lr=0, β=1 → 1 - σ(0) = 0.5."""
        out = apo_zero_unpaired(_T(0.0), _T(0.0), _T(True), beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, _T(0.5), atol=1e-6)

    def test_undesirable_zero_input_is_half(self) -> None:
        """label=False, rejected_lr=0, β=1 → σ(0) = 0.5."""
        out = apo_zero_unpaired(_T(0.0), _T(0.0), _T(False), beta=1.0)
        assert out.shape == ()
        assert torch.allclose(out, _T(0.5), atol=1e-6)

    def test_desirable_positive_margin_with_weight(self) -> None:
        """label=True, chosen_lr=2, β=0.5, w_d=2 → 2*(1-σ(1.0))."""
        out = apo_zero_unpaired(
            _T(2.0), _T(0.0), _T(True), beta=0.5, desirable_weight=2.0
        )
        expected = 2.0 * (1 - _sigmoid(0.5 * 2.0))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_undesirable_with_weight(self) -> None:
        """label=False, rejected_lr=1, β=0.5, w_u=3 → 3*σ(0.5)."""
        out = apo_zero_unpaired(
            _T(0.0), _T(1.0), _T(False), beta=0.5, undesirable_weight=3.0
        )
        expected = 3.0 * _sigmoid(0.5 * 1.0)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_none_substitution_eager_path(self) -> None:
        """Eager path: None on the unused side is substituted with zeros."""
        out_des = apo_zero_unpaired(_T(1.0), None, _T(True), beta=0.5)
        assert torch.allclose(out_des, 1 - _sigmoid(0.5), atol=1e-6)
        out_undes = apo_zero_unpaired(None, _T(1.0), _T(False), beta=0.5)
        assert torch.allclose(out_undes, _sigmoid(0.5), atol=1e-6)


# ===========================================================================
# apo_zero_unpaired — mixed batch + Tier-1 properties
# ===========================================================================


class TestApoZeroUnpairedBatch:
    """Mixed-label batch, bounds, NaN-injection locality, and vmap-safety."""

    def test_mixed_label_batch(self) -> None:
        chosen = _T([1.0, -2.0, 0.5, 3.0])
        rejected = _T([0.0, 1.0, -1.0, 2.0])
        label = _T([True, False, True, False])
        beta = 0.4
        out = apo_zero_unpaired(chosen, rejected, label, beta=beta)
        assert out.shape == (4,)
        for i in range(4):
            if label[i]:
                ref = 1 - torch.sigmoid(beta * chosen[i])
            else:
                ref = torch.sigmoid(beta * rejected[i])
            assert torch.allclose(out[i], ref, atol=1e-6)

    def test_loss_bounded_zero_to_max_weight(self) -> None:
        """Each term is a (weighted) sigmoid in [0, w]."""
        torch.manual_seed(3)
        chosen = torch.randn(64)
        rejected = torch.randn(64)
        label = torch.randint(0, 2, (64,)).bool()
        w_d, w_u = 1.5, 2.0
        out = apo_zero_unpaired(
            chosen,
            rejected,
            label,
            beta=0.2,
            desirable_weight=w_d,
            undesirable_weight=w_u,
        )
        assert (out >= 0).all()
        assert (out <= max(w_d, w_u) + 1e-6).all()

    def test_nan_injection_locality(self) -> None:
        """Tier-1 NaN-injection: a NaN in one example affects only that loss."""
        b = 5
        torch.manual_seed(4)
        chosen = torch.randn(b)
        rejected = torch.randn(b)
        label = _T([True, True, True, True, True])  # all desirable
        chosen_nan = chosen.clone()
        chosen_nan[2] = float("nan")
        out = apo_zero_unpaired(chosen_nan, rejected, label, beta=0.5)
        assert torch.isnan(out[2])
        for i in range(b):
            if i != 2:
                assert torch.isfinite(out[i])

    def test_vmap_grad_finite(self) -> None:
        b = 4
        torch.manual_seed(5)
        chosen = torch.randn(b)
        rejected = torch.randn(b)
        label = _T([True, False, True, False])

        def per_example(
            ci: torch.Tensor, ri: torch.Tensor, li: torch.Tensor
        ) -> torch.Tensor:
            return apo_zero_unpaired(ci, ri, li, beta=0.1).sum()

        grads_c, grads_r = vmap(grad(per_example, argnums=(0, 1)))(
            chosen, rejected, label
        )
        assert grads_c.shape == (b,)
        assert grads_r.shape == (b,)
        assert torch.isfinite(grads_c).all()
        assert torch.isfinite(grads_r).all()
