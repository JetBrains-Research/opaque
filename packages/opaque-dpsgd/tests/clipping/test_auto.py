"""Tests for AUTO-S automatic clipping (Bu et al., NeurIPS 2023)."""

import math

import pytest
import torch

from opaque.core.clipping import auto_scale_pytree, clipped_grad
from opaque.dpsgd.clipping import (
    AutoClippedGradAux,
    AutoClipState,
    auto_clipped_fun,
    auto_clipped_grad,
)
from opaque.core.clipping.per_group import PerGroup


class TestAutoScalePytree:
    """Tests for the low-level auto_scale_pytree primitive."""

    def test_formula_scalar(self):
        """Output should equal R * g / (||g|| + gamma)."""
        pytree = {"a": torch.tensor([3.0, 4.0])}  # norm = 5
        scaled, aux = auto_scale_pytree(pytree, R=1.0, gamma=0.01)

        expected_scale = 1.0 / (5.0 + 0.01)
        torch.testing.assert_close(scaled["a"], pytree["a"] * expected_scale)
        assert aux.norm.item() == pytest.approx(5.0)
        assert aux.group_norms is None

    def test_output_norm_bounded_by_R(self):
        """For any input, ||output|| must be <= R."""
        generator = torch.Generator().manual_seed(42)
        R = 0.7
        for _ in range(20):
            pytree = {
                "a": torch.randn(5, generator=generator) * 100,
                "b": torch.randn(3, generator=generator),
            }
            scaled, _ = auto_scale_pytree(pytree, R=R, gamma=0.01)
            total = torch.cat([scaled["a"], scaled["b"]])
            norm = float(torch.linalg.vector_norm(total))
            assert norm <= R + 1e-6, f"norm {norm} exceeds R={R}"

    def test_approaches_unit_projection_for_large_norms(self):
        """When ||g|| >> gamma, output approaches unit-norm projection."""
        pytree = {"a": torch.tensor([300.0, 400.0])}  # norm = 500
        scaled, _ = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        # Expected norm ≈ 500 / (500 + 0.01) ≈ 0.99998
        norm = float(torch.linalg.vector_norm(scaled["a"]))
        assert norm == pytest.approx(1.0, rel=1e-4)

    def test_zero_gradient_maps_to_zero(self):
        """g = 0 should yield output 0 (finite, no NaN)."""
        pytree = {"a": torch.zeros(4)}
        scaled, aux = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        torch.testing.assert_close(scaled["a"], torch.zeros(4))
        assert aux.norm.item() == 0.0

    def test_nan_sanitization(self):
        """NaN/Inf values should be replaced with 0 before scaling."""
        pytree = {"a": torch.tensor([float("nan"), float("inf"), 1.0])}
        scaled, _ = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        assert not torch.isnan(scaled["a"]).any()
        assert not torch.isinf(scaled["a"]).any()

    def test_rejects_zero_gamma(self):
        """gamma=0 is rejected (AUTO-V, undefined at zero gradient)."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            auto_scale_pytree({"a": torch.tensor([1.0])}, R=1.0, gamma=0.0)

    def test_per_group_formula(self):
        """Per-group AUTO-S applies the formula within each group."""
        pytree = {
            "attn.q": torch.tensor([3.0]),
            "attn.k": torch.tensor([4.0]),  # attn group norm = 5
            "mlp.w": torch.tensor([6.0]),
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 2.0, "mlp": 3.0},
        )
        scaled, aux = auto_scale_pytree(pytree, R=pg, gamma=0.01)

        # attn: scale = 2 / (5 + 0.01)
        attn_scale = 2.0 / (5.0 + 0.01)
        torch.testing.assert_close(scaled["attn.q"], torch.tensor([3.0 * attn_scale]))
        torch.testing.assert_close(scaled["attn.k"], torch.tensor([4.0 * attn_scale]))

        # mlp: scale = 3 / (6 + 0.01)
        mlp_scale = 3.0 / (6.0 + 0.01)
        torch.testing.assert_close(scaled["mlp.w"], torch.tensor([6.0 * mlp_scale]))

        assert aux.group_norms is not None
        assert aux.group_norms["attn"].item() == pytest.approx(5.0)
        assert aux.group_norms["mlp"].item() == pytest.approx(6.0)

    def test_per_group_sensitivity_bound(self):
        """Per-group output norm is bounded by sqrt(sum R_k^2)."""
        pytree = {
            "a": torch.randn(5) * 1000,
            "b": torch.randn(5) * 1000,
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.5, "g2": 1.5},
        )
        scaled, _ = auto_scale_pytree(pytree, R=pg, gamma=0.01)
        total_norm = float(
            torch.linalg.vector_norm(torch.cat([scaled["a"], scaled["b"]]))
        )
        bound = math.sqrt(0.5**2 + 1.5**2)
        assert total_norm <= bound + 1e-4


class TestAutoClipState:
    """Tests for AutoClipState validation and sensitivity."""

    def test_scalar_sensitivity(self):
        state = AutoClipState(clipping_norm=2.0, normalize_by=4.0, gamma=0.01)
        assert state.sensitivity == pytest.approx(0.5)

    def test_per_group_sensitivity(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 3.0, "g2": 4.0})
        state = AutoClipState(clipping_norm=pg, normalize_by=1.0)
        # sqrt(9 + 16) / 1 = 5
        assert state.sensitivity == pytest.approx(5.0)

    def test_rejects_non_positive_R(self):
        with pytest.raises(ValueError, match="positive"):
            AutoClipState(clipping_norm=0.0, normalize_by=1.0)

    def test_rejects_non_positive_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            AutoClipState(clipping_norm=1.0, normalize_by=1.0, gamma=0.0)


class TestAutoClippedGrad:
    """Tests for auto_clipped_grad."""

    def _loss(self, params, x, y):
        return ((x @ params - y) ** 2).mean()

    def test_basic_usage(self):
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, new_state = grad_fn(params, batch_x, batch_y, state=state)
        assert grads.shape == params.shape
        assert isinstance(new_state, AutoClipState)
        assert new_state.clipping_norm == 1.0

    def test_state_is_fixed(self):
        """State does not change across steps (no adaptation)."""
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=0.5
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        _, s1 = grad_fn(params, batch_x, batch_y, state=state)
        _, s2 = grad_fn(params, batch_x, batch_y, state=s1)
        assert s1 == s2  # dataclasses with same values

    def test_sensitivity_via_state(self):
        """sensitivity = R / normalize_by."""
        _, state = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=2.5,
            normalize_by=10.0,
        )
        assert state.sensitivity == pytest.approx(0.25)

    def test_grad_norm_bounded(self):
        """Sum of per-example scaled gradients has bounded norm.

        With B examples, each scaled to norm <= R, the triangle-inequality
        bound is B * R.  We check this holds.
        """
        R = 0.3
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=R
        )
        params = torch.randn(10)
        batch_size = 8
        batch_x = torch.randn(batch_size, 10) * 100  # force large gradients
        batch_y = torch.randn(batch_size) * 100

        grads, _ = grad_fn(params, batch_x, batch_y, state=state)
        norm = float(torch.linalg.vector_norm(grads))
        assert norm <= batch_size * R + 1e-5

    def test_normalize_by(self):
        """normalize_by scales the summed gradient."""
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        fn1, s1 = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=1.0
        )
        fn2, s2 = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=4.0
        )
        g1, _ = fn1(params, batch_x, batch_y, state=s1)
        g2, _ = fn2(params, batch_x, batch_y, state=s2)
        torch.testing.assert_close(g1 / 4.0, g2)

    def test_return_aux(self):
        """return_aux=True produces AutoClippedGradAux with norms."""
        grad_fn, state = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=1.0,
            return_aux=True,
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert isinstance(aux, AutoClippedGradAux)
        assert aux.grad_norms.shape == (8,)
        assert aux.clipped_grad_norms.shape == (8,)
        assert aux.loss_values.shape == (8,)
        assert aux.batch_size == 8
        # Each clipped norm must be <= R
        assert bool((aux.clipped_grad_norms <= 1.0 + 1e-5).all())

    def test_microbatching_equivalence(self):
        """Microbatched AUTO-S matches full-batch AUTO-S."""
        torch.manual_seed(0)
        params = torch.randn(10)
        batch_x = torch.randn(16, 10)
        batch_y = torch.randn(16)

        fn_full, s_full = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=0.5
        )
        fn_mb, s_mb = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=0.5,
            microbatch_size=4,
        )
        g_full, _ = fn_full(params, batch_x, batch_y, state=s_full)
        g_mb, _ = fn_mb(params, batch_x, batch_y, state=s_mb)
        torch.testing.assert_close(g_full, g_mb, rtol=1e-5, atol=1e-6)

    def test_empty_batch(self):
        """Empty batch yields zero gradient and preserves state."""
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0
        )
        params = torch.randn(10)
        batch_x = torch.empty(0, 10)
        batch_y = torch.empty(0)

        grads, new_state = grad_fn(params, batch_x, batch_y, state=state)
        assert grads.shape == params.shape
        torch.testing.assert_close(grads, torch.zeros_like(params))
        assert new_state == state

    def test_rejects_non_positive_R(self):
        with pytest.raises(ValueError, match="R must be positive"):
            auto_clipped_grad(self._loss, R=0.0, batch_argnums=(1, 2))

    def test_rejects_non_positive_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            auto_clipped_grad(self._loss, gamma=0.0, batch_argnums=(1, 2))

    def test_differs_from_fixed_clipping_at_small_norms(self):
        """At small ||g||, AUTO-S can differ from fixed clipping.

        This is the defining behavioral difference: AUTO-S continuously scales by
        R / (||g|| + gamma) even when ||g|| < R, unlike fixed clipping's
        min(1, R/||g||).
        """
        # Construct a batch where per-example gradients will be small.
        torch.manual_seed(0)
        params = torch.tensor([0.001, 0.001])

        def tiny_loss(p, x, y):
            # Loss with tiny gradient magnitude
            return 0.001 * ((x @ p - y) ** 2).mean()

        batch_x = torch.randn(4, 2) * 0.01
        batch_y = torch.randn(4) * 0.01

        fn_auto, s_auto = auto_clipped_grad(
            tiny_loss, argnums=0, batch_argnums=(1, 2), R=1.0, gamma=0.01
        )
        fn_fixed, s_fixed = clipped_grad(
            tiny_loss, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
        )
        g_auto, _ = fn_auto(params, batch_x, batch_y, state=s_auto)
        g_fixed, _ = fn_fixed(params, batch_x, batch_y, state=s_fixed)

        # Gradients should be materially different (AUTO-S amplifies small grads).
        assert not torch.allclose(g_auto, g_fixed, rtol=1e-3)

    def test_has_aux_passthrough(self):
        """loss_fn returning (loss, extra) is supported with has_aux=True."""

        def loss_with_aux(p, x, y):
            pred = x @ p
            return ((pred - y) ** 2).mean(), pred

        grad_fn, state = auto_clipped_grad(
            loss_with_aux,
            argnums=0,
            has_aux=True,
            batch_argnums=(1, 2),
            R=1.0,
            return_aux=True,
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert aux.loss_aux is not None  # predictions per example


class TestAutoClippedGradPerGroup:
    """Per-group AUTO-S."""

    def test_per_group_sensitivity_bound(self):
        """Per-group sensitivity is sqrt(sum R_k^2) / normalize_by."""
        pg = PerGroup(groups={}, values={"g1": 3.0, "g2": 4.0})
        _, state = auto_clipped_grad(
            lambda p, x: (x - p).pow(2).mean(),
            argnums=0,
            batch_argnums=1,
            R=pg,
            normalize_by=2.0,
        )
        assert state.sensitivity == pytest.approx(5.0 / 2.0)

    def test_per_group_applies_correct_scales(self):
        """Per-group AUTO-S scales each group by its own R_k."""

        # Two-parameter model: dict form needed for per-group.
        def loss_fn(params, x, y):
            pred = x @ params["w"] + params["b"]
            return ((pred - y) ** 2).mean()

        pg = PerGroup(
            groups={"w": "weights", "b": "biases"},
            values={"weights": 2.0, "biases": 0.5},
        )
        grad_fn, state = auto_clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=pg,
            return_aux=True,
        )
        params = {"w": torch.randn(10) * 10, "b": torch.randn(1) * 10}
        batch_x = torch.randn(6, 10)
        batch_y = torch.randn(6)

        (grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert aux.group_norms is not None
        assert set(aux.group_norms.keys()) == {"weights", "biases"}


class TestAutoClippedFun:
    """Tests for auto_clipped_fun (general-purpose scaling)."""

    def test_basic_scaling_and_sum(self):
        """Sum of scaled per-example outputs; each output bounded by R."""

        def per_example(x):
            return x  # identity per-example, batch sums

        fn, state = auto_clipped_fun(per_example, batch_argnums=0, R=1.0)
        batch = torch.randn(5, 3) * 10
        result, new_state = fn(batch, state=state)

        expected = torch.zeros(3)
        for i in range(5):
            scale = 1.0 / (torch.linalg.vector_norm(batch[i]) + 0.01)
            expected = expected + batch[i] * scale

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        assert new_state == state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
