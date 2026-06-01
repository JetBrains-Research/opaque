# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + contract tests for SFT loss primitives (plan §7.3, §11.2, §11.3).

Covers work-unit ε.1 of the opaque-alignment plan.

Verified properties
-------------------
* :func:`nll_loss` — ≥3 hand-computed reference cases (uniform logits → log(V),
  ignore-index exclusion, all-ignored div-0 guard).
* :func:`dft_loss` — hand-computed case with detached-prob weighting; gradient
  detach invariant (§11.3 corollary); DP-corrected divisor assertion.
* **DP-corrected divisor** (§3.3, §8.2): 2-example batch with different
  completion lengths → per-example divisor differs from a batch-level divisor;
  both functions are verified against the per-example result.
* **NaN-injection** (Tier 1, §11.3): NaN in one example's logits produces NaN
  only for that example; neighbours are finite.
* **Vmap-safety** (§3.4, §11.2): ``vmap(grad(nll_loss.sum))`` over a
  ``(4, T, V)`` batch with all-valid masks yields finite gradients.

Imports target concrete implementation paths because the public façade
``__init__.py`` is wired in the separate ε.W unit.
"""

from __future__ import annotations

import math

import torch
from torch.func import grad, vmap

from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.loss.sft._dft import dft_loss
from opaque.api.alignment.loss.sft._nll import nll_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _logits_uniform(t: int, v: int) -> torch.Tensor:
    """Return all-zero (uniform distribution) logits of shape ``(t, v)``."""
    return torch.zeros(t, v)


# ---------------------------------------------------------------------------
# nll_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


class TestNllLoss:
    """Verified reference cases for the standard causal-LM NLL loss."""

    def test_uniform_logits_all_valid(self) -> None:
        """Uniform logits → per-token NLL = log(V); mean = log(V).

        With all-zero logits over V=4 tokens the log-softmax at every position
        is log(1/4) = -log(4).  Three shifted tokens, all valid → mean NLL = log(4).
        """
        V = 4
        T = 4  # seq len; shifted to T-1=3 predictions
        logits = _logits_uniform(T, V)
        labels = torch.tensor([0, 1, 2, 3])  # shifted labels: [1, 2, 3], all valid
        out = nll_loss(logits, labels)
        expected = math.log(V)
        assert out.shape == (), "per-example call must return a scalar"
        assert torch.allclose(out, torch.tensor(expected), atol=_ATOL)

    def test_ignore_index_excluded(self) -> None:
        """Tokens with label -100 are excluded from the mean.

        With V=3 uniform logits and two of three shifted positions valid,
        the mean NLL is log(3), regardless of the ignored position's logit row.
        """
        V = 3
        T = 4
        logits = _logits_uniform(T, V)
        labels = torch.tensor([0, 1, -100, 2])  # shifted: [1, -100, 2]; 2 valid
        out = nll_loss(logits, labels)
        expected = math.log(V)  # log(3) — only the 2 valid tokens count
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=_ATOL)

    def test_all_ignored_returns_zero(self) -> None:
        """All-ignored row: divisor clamped to 1 → loss = 0 (no div-by-zero).

        When every shifted label is -100 the mask sum is 0.  After
        ``clamp(min=1)`` the divisor is 1 and the numerator is 0 → 0.0.
        This guards the edge case that would otherwise produce NaN.
        """
        T = 3
        V = 4
        logits = _logits_uniform(T, V)
        labels = torch.full((T,), -100, dtype=torch.long)
        out = nll_loss(logits, labels)
        assert out.shape == ()
        assert torch.isfinite(out), "all-ignored row must not produce NaN/inf"
        assert torch.allclose(out, torch.zeros(()), atol=_ATOL)

    def test_batched_shape(self) -> None:
        """Batched (B, T, V) input returns shape (B,)."""
        B, T, V = 3, 5, 8
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = nll_loss(logits, labels)
        assert out.shape == (B,)

    def test_per_example_matches_batched(self) -> None:
        """Stacking per-example calls must equal the batched result."""
        B, T, V = 3, 6, 5
        torch.manual_seed(1)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[0, 3] = -100  # exercise ignore-index path

        batched = nll_loss(logits, labels)
        per_example = torch.stack([nll_loss(logits[i], labels[i]) for i in range(B)])
        assert torch.allclose(batched, per_example, atol=_ATOL)

    # ------------------------------------------------------------------
    # DP-corrected divisor: per-example token count, not batch token count
    # ------------------------------------------------------------------

    def test_dp_corrected_divisor(self) -> None:
        """Per-example divisor differs from a batch-level total-token divisor.

        Constructs a 2-example batch where example 0 has 3 valid shifted tokens
        and example 1 has 1 valid shifted token.  The per-example divisors (3
        and 1) differ from the batch-level divisor (4), so the two approaches
        produce different per-example losses.  This test asserts the per-example
        result — verifying that the implementation uses ``mask.sum(-1)`` per
        example, not a cross-example aggregate (plan §3.3, §8.2).
        """
        torch.manual_seed(2)
        V = 4
        T = 6  # seq len
        logits = torch.randn(2, T, V)

        # Example 0: shifted labels [1, 2, 3, -100, -100] → 3 valid tokens
        # Example 1: shifted labels [1, -100, -100, -100, -100] → 1 valid token
        labels_0 = torch.tensor([0, 1, 2, 3, -100, -100])
        labels_1 = torch.tensor([0, 1, -100, -100, -100, -100])
        labels = torch.stack([labels_0, labels_1])  # (2, 6)

        out = nll_loss(logits, labels)
        assert out.shape == (2,)

        # Reference: hand-compute per-example losses
        sl0 = labels_0[1:]  # shifted labels for example 0
        sl1 = labels_1[1:]  # shifted labels for example 1
        sl0_logits = logits[0, :-1, :]  # shifted logits (T-1, V)
        sl1_logits = logits[1, :-1, :]

        mask0 = (sl0 != -100).float()
        mask1 = (sl1 != -100).float()
        c0 = sl0.clamp(min=0)
        c1 = sl1.clamp(min=0)
        logp0 = selective_log_softmax(sl0_logits, c0)
        logp1 = selective_log_softmax(sl1_logits, c1)
        expected_0 = (-logp0 * mask0).sum() / mask0.sum().clamp(min=1)
        expected_1 = (-logp1 * mask1).sum() / mask1.sum().clamp(min=1)

        assert torch.allclose(out[0], expected_0, atol=_ATOL)
        assert torch.allclose(out[1], expected_1, atol=_ATOL)

        # Confirm that a batch-level divisor would give a DIFFERENT answer,
        # proving the test distinguishes the two approaches.
        total_valid = mask0.sum() + mask1.sum()  # 4.0
        batch_0 = (-logp0 * mask0).sum() / total_valid
        batch_1 = (-logp1 * mask1).sum() / total_valid
        assert not torch.allclose(out[0], batch_0, atol=1e-7), (
            "per-example and batch divisors must differ for this construction"
        )
        assert not torch.allclose(out[1], batch_1, atol=1e-7), (
            "per-example and batch divisors must differ for this construction"
        )

    # ------------------------------------------------------------------
    # NaN-injection (Tier 1, §11.3)
    # ------------------------------------------------------------------

    def test_nan_injection_only_corrupt_one_example(self) -> None:
        """NaN logits in one example → only that example's loss is NaN.

        This is the Tier-1 NaN-injection contract (plan §11.3): replacing one
        example's input with NaN must not corrupt the gradients/outputs of any
        other example in the batch.
        """
        B, T, V = 4, 5, 6
        torch.manual_seed(3)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        # Inject NaN into example 1's logits (all positions)
        logits[1] = float("nan")

        out = nll_loss(logits, labels)
        assert out.shape == (B,)
        assert torch.isfinite(out[0]), "example 0 must be finite"
        assert torch.isnan(out[1]), "example 1 (NaN logits) must be NaN"
        assert torch.isfinite(out[2]), "example 2 must be finite"
        assert torch.isfinite(out[3]), "example 3 must be finite"

    # ------------------------------------------------------------------
    # Vmap-safety (§3.4, §11.2)
    # ------------------------------------------------------------------

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(nll_loss.sum)) over (4, T, V) with all-valid labels is finite.

        Uses ``torch.func.vmap(torch.func.grad(...))`` to verify that the
        function is free of Python control flow on tensor values, ``.item()``
        calls, and module-state side-effects — all of which would break vmap.
        """
        B, T, V = 4, 6, 8
        torch.manual_seed(4)
        logits = torch.randn(B, T, V)
        # All-valid labels: integers in [0, V), no -100 ignore tokens.
        labels = torch.randint(0, V, (B, T))

        def per_example_fn(lg: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
            return nll_loss(lg, lab).sum()

        grads = vmap(grad(per_example_fn))(logits, labels)
        assert grads.shape == (B, T, V)
        assert torch.isfinite(grads).all(), (
            "vmap(grad(nll_loss)) must yield finite grads"
        )


# ---------------------------------------------------------------------------
# dft_loss — hand-computed reference cases
# ---------------------------------------------------------------------------


class TestDftLoss:
    """Verified reference cases for the DFT (Dynamic Fine-Tuning) loss.

    DFT arXiv:2508.05629 — per-token loss = -detach(softmax_prob) * logp.
    """

    def test_hand_computed_detached_weighting(self) -> None:
        """Small case: -p.detach() * logp per-token, averaged over 2 valid tokens.

        logits = [[2.0, 0.0], [0.0, 1.0]], labels = [0, 1, 1]
        shifted logits: [[2.0, 0.0], [0.0, 1.0]]
        shifted labels: [1, 1] (both valid)

        Row 0: lse0 = log(e^2 + 1); logp0 = -lse0; p0 = 1/exp(lse0)
        Row 1: lse1 = log(1 + e^1); logp1 = 1 - lse1; p1 = e^1/exp(lse1)
        DFT mean = (-p0*logp0 + -p1*logp1) / 2
        """
        lse0 = math.log(math.exp(2.0) + math.exp(0.0))
        lse1 = math.log(math.exp(0.0) + math.exp(1.0))
        logp0 = 0.0 - lse0  # log-prob of token 1 from row [2, 0]
        logp1 = 1.0 - lse1  # log-prob of token 1 from row [0, 1]
        p0 = math.exp(0.0) / math.exp(lse0)  # softmax prob of token 1 from row [2, 0]
        p1 = math.exp(1.0) / math.exp(lse1)  # softmax prob of token 1 from row [0, 1]
        expected = (-p0 * logp0 + -p1 * logp1) / 2.0

        # Build (1, T=3, V=2) shaped input; last logit row is irrelevant (shifted away)
        logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        labels = torch.tensor([[0, 1, 1]])

        out = dft_loss(logits, labels)
        assert out.shape == (1,)
        assert torch.allclose(out, torch.tensor([expected]), atol=_ATOL)

    def test_all_ignored_returns_zero(self) -> None:
        """All-ignored row: div-0 guard → 0.0 (same invariant as nll_loss)."""
        T = 3
        V = 4
        logits = _logits_uniform(T, V)
        labels = torch.full((T,), -100, dtype=torch.long)
        out = dft_loss(logits, labels)
        assert out.shape == ()
        assert torch.isfinite(out)
        assert torch.allclose(out, torch.zeros(()), atol=_ATOL)

    def test_batched_shape(self) -> None:
        """Batched (B, T, V) input returns shape (B,)."""
        B, T, V = 3, 5, 8
        torch.manual_seed(5)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = dft_loss(logits, labels)
        assert out.shape == (B,)

    def test_per_example_matches_batched(self) -> None:
        """Stacking per-example calls must equal the batched result."""
        B, T, V = 3, 6, 5
        torch.manual_seed(6)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[1, 2] = -100  # exercise ignore-index path

        batched = dft_loss(logits, labels)
        per_example = torch.stack([dft_loss(logits[i], labels[i]) for i in range(B)])
        assert torch.allclose(batched, per_example, atol=_ATOL)

    # ------------------------------------------------------------------
    # Gradient detach invariant — the weighting probability p must be detached
    # ------------------------------------------------------------------

    def test_gradient_flows_only_through_logp_not_p(self) -> None:
        """Gradient of dft_loss equals gradient of -p.detach() * logp.

        If ``p`` were not detached the gradient w.r.t. the logits would include
        a second term from ∂p/∂logits, changing the gradient entirely.  This
        test confirms that:
          1. grad(dft_loss) == grad(-p.detach()*logp)   [detached reference]
          2. grad(dft_loss) != grad(-p_no_detach*logp)  [non-detached differs]
        """
        torch.manual_seed(7)
        logits_base = torch.randn(1, 3, 4)
        labels = torch.randint(0, 4, (1, 3))

        # --- (1) grad of dft_loss ---
        lg1 = logits_base.clone().requires_grad_(True)
        dft_loss(lg1, labels).sum().backward()
        grad_dft = lg1.grad.clone()

        # --- (2) reference: -p.detach() * logp ---
        lg2 = logits_base.clone().requires_grad_(True)
        sl2 = lg2[:, :-1, :]  # shifted logits
        shifted_labels = labels[:, 1:]  # shifted labels
        clamped = shifted_labels.clamp(min=0)
        logp_ref = selective_log_softmax(sl2, clamped)
        p_det = (
            torch.softmax(sl2, dim=-1)
            .gather(-1, clamped.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        mask_ref = (shifted_labels != -100).float()
        divisor_ref = mask_ref.sum(-1).clamp(min=1)
        ((-p_det * logp_ref * mask_ref).sum(-1) / divisor_ref).sum().backward()
        grad_ref = lg2.grad.clone()

        # --- (3) non-detached p (must differ from dft_loss gradient) ---
        lg3 = logits_base.clone().requires_grad_(True)
        sl3 = lg3[:, :-1, :]
        logp_nodetch = selective_log_softmax(sl3, clamped)
        p_nodetch = (
            torch.softmax(sl3, dim=-1).gather(-1, clamped.unsqueeze(-1)).squeeze(-1)
        )
        mask_nd = (labels[:, 1:] != -100).float()
        divisor_nd = mask_nd.sum(-1).clamp(min=1)
        ((-p_nodetch * logp_nodetch * mask_nd).sum(-1) / divisor_nd).sum().backward()
        grad_nodetch = lg3.grad.clone()

        # Assertions
        assert torch.allclose(grad_dft, grad_ref, atol=1e-6), (
            "dft_loss gradient must equal -p.detach()*logp gradient"
        )
        assert not torch.allclose(grad_dft, grad_nodetch, atol=1e-7), (
            "a non-detached p changes the gradient — dft_loss must detach p"
        )

    # ------------------------------------------------------------------
    # DP-corrected divisor
    # ------------------------------------------------------------------

    def test_dp_corrected_divisor(self) -> None:
        """Per-example divisor (own token count) differs from batch-level total.

        2-example batch: example 0 has 3 valid tokens, example 1 has 1.
        The per-example implementation divides by 3 and 1 respectively.
        A batch-level divisor of 4 would produce different values.
        """
        torch.manual_seed(8)
        V = 4
        T = 6
        logits = torch.randn(2, T, V)

        labels_0 = torch.tensor([0, 1, 2, 3, -100, -100])  # 3 valid shifted
        labels_1 = torch.tensor([0, 1, -100, -100, -100, -100])  # 1 valid shifted
        labels = torch.stack([labels_0, labels_1])

        out = dft_loss(logits, labels)
        assert out.shape == (2,)

        # Reference: hand-compute per-example DFT losses
        sl0 = labels_0[1:]
        sl1 = labels_1[1:]
        sl0_logits = logits[0, :-1, :]
        sl1_logits = logits[1, :-1, :]

        mask0 = (sl0 != -100).float()
        mask1 = (sl1 != -100).float()
        c0 = sl0.clamp(min=0)
        c1 = sl1.clamp(min=0)
        logp0 = selective_log_softmax(sl0_logits, c0)
        logp1 = selective_log_softmax(sl1_logits, c1)
        p0 = (
            torch.softmax(sl0_logits, dim=-1)
            .gather(-1, c0.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        p1 = (
            torch.softmax(sl1_logits, dim=-1)
            .gather(-1, c1.unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )
        expected_0 = (-p0 * logp0 * mask0).sum() / mask0.sum().clamp(min=1)
        expected_1 = (-p1 * logp1 * mask1).sum() / mask1.sum().clamp(min=1)

        assert torch.allclose(out[0], expected_0, atol=_ATOL)
        assert torch.allclose(out[1], expected_1, atol=_ATOL)

        # Confirm that a batch-level divisor of 4 would yield DIFFERENT values
        total_valid = mask0.sum() + mask1.sum()  # 4.0
        batch_0 = (-p0 * logp0 * mask0).sum() / total_valid
        batch_1 = (-p1 * logp1 * mask1).sum() / total_valid
        assert not torch.allclose(out[0], batch_0, atol=1e-7), (
            "per-example and batch divisors must differ for example 0"
        )
        assert not torch.allclose(out[1], batch_1, atol=1e-7), (
            "per-example and batch divisors must differ for example 1"
        )

    # ------------------------------------------------------------------
    # NaN-injection (Tier 1, §11.3)
    # ------------------------------------------------------------------

    def test_nan_injection_only_corrupt_one_example(self) -> None:
        """NaN logits in one example → only that example's loss is NaN.

        Tier-1 NaN-injection contract (plan §11.3).  Example 2 of a 4-example
        batch has all NaN logits; examples 0, 1, 3 must remain finite.
        """
        B, T, V = 4, 5, 6
        torch.manual_seed(9)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        logits[2] = float("nan")

        out = dft_loss(logits, labels)
        assert out.shape == (B,)
        assert torch.isfinite(out[0]), "example 0 must be finite"
        assert torch.isfinite(out[1]), "example 1 must be finite"
        assert torch.isnan(out[2]), "example 2 (NaN logits) must be NaN"
        assert torch.isfinite(out[3]), "example 3 must be finite"

    # ------------------------------------------------------------------
    # Vmap-safety (§3.4, §11.2)
    # ------------------------------------------------------------------

    def test_vmap_grad_finite(self) -> None:
        """vmap(grad(dft_loss.sum)) over (4, T, V) with all-valid labels is finite."""
        B, T, V = 4, 6, 8
        torch.manual_seed(10)
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))

        def per_example_fn(lg: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
            return dft_loss(lg, lab).sum()

        grads = vmap(grad(per_example_fn))(logits, labels)
        assert grads.shape == (B, T, V)
        assert torch.isfinite(grads).all(), (
            "vmap(grad(dft_loss)) must yield finite grads"
        )
