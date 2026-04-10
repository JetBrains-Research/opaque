"""Tests for per-group clipping through clip_pytree, clipped_fun, and clipped_grad."""

import pytest
import torch

from opaque.clipping import clip_pytree, clipped_grad
from opaque.clipping.types import FixedClipState
from opaque.utils.per_group import PerGroup, per_group


class TestClipPytreePerGroup:
    """Tests for clip_pytree with PerGroup clipping_norm."""

    def _make_pg(self, keys, group_map, values):
        """Helper to build a PerGroup from key lists."""
        groups = {k: group_map[k] for k in keys}
        return PerGroup(groups=groups, values=values)

    def test_no_clipping_when_within_bounds(self):
        """Gradients within group norms should pass through unchanged."""
        pytree = {
            "attn.q": torch.tensor([1.0, 0.0]),  # norm = 1
            "attn.k": torch.tensor([0.0, 1.0]),  # norm = 1
            "mlp.w": torch.tensor([0.5, 0.0]),  # norm = 0.5
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 10.0, "mlp": 10.0},  # Large bounds
        )
        clipped, aux = clip_pytree(pytree, pg)
        for key in pytree:
            torch.testing.assert_close(clipped[key], pytree[key])

    def test_clips_groups_independently(self):
        """Each group should be clipped to its own norm bound."""
        # attn group: norm = sqrt(9 + 16) = 5, bound = 1 → scale = 0.2
        # mlp group: norm = 3, bound = 6 → no clipping (scale = 1)
        pytree = {
            "attn.q": torch.tensor([3.0]),
            "attn.k": torch.tensor([4.0]),
            "mlp.w": torch.tensor([3.0]),
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 1.0, "mlp": 6.0},
        )
        clipped, aux = clip_pytree(pytree, pg)

        # attn group clipped: scale = 1/5
        torch.testing.assert_close(clipped["attn.q"], torch.tensor([0.6]))
        torch.testing.assert_close(clipped["attn.k"], torch.tensor([0.8]))

        # mlp group not clipped
        torch.testing.assert_close(clipped["mlp.w"], torch.tensor([3.0]))

    def test_aux_returns_global_norm(self):
        """aux.norm should be the global L2 norm across all groups."""
        pytree = {
            "a": torch.tensor([3.0]),
            "b": torch.tensor([4.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        _, aux = clip_pytree(pytree, pg)
        assert aux.norm.item() == pytest.approx(5.0)  # sqrt(9+16)

    def test_nan_sanitization(self):
        """NaN values should be replaced with 0 before clipping."""
        pytree = {
            "a": torch.tensor([float("nan"), 1.0]),
            "b": torch.tensor([2.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg)
        assert not torch.isnan(clipped["a"]).any()

    def test_return_zero(self):
        """return_zero should zero out the output regardless of inputs."""
        pytree = {"a": torch.tensor([5.0]), "b": torch.tensor([3.0])}
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg, return_zero=True)
        torch.testing.assert_close(clipped["a"], torch.tensor([0.0]))
        torch.testing.assert_close(clipped["b"], torch.tensor([0.0]))

    def test_zero_norm_group(self):
        """Group with all-zero tensors should produce zero output without NaN."""
        pytree = {
            "a": torch.tensor([0.0, 0.0]),
            "b": torch.tensor([1.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg)
        assert not torch.isnan(clipped["a"]).any()
        torch.testing.assert_close(clipped["a"], torch.tensor([0.0, 0.0]))


class TestFixedClipStatePerGroup:
    """Tests for FixedClipState with PerGroup clipping_norm."""

    def test_sensitivity_is_scalar_for_per_group(self):
        """Even with PerGroup clipping_norm, sensitivity is scalar ||C||_2 / n."""
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 2.0, "g2": 4.0},
        )
        state = FixedClipState(clipping_norm=pg, normalize_by=2.0)
        sens = state.sensitivity
        import math

        assert isinstance(sens, float)
        assert sens == pytest.approx(math.sqrt(2.0**2 + 4.0**2) / 2.0)

    def test_sensitivity_scalar_single_group(self):
        pg = PerGroup(groups={"x": "attn"}, values={"attn": 3.0})
        state = FixedClipState(clipping_norm=pg, normalize_by=1.0)
        assert isinstance(state.sensitivity, float)
        assert state.sensitivity == pytest.approx(3.0)

    def test_validation_rejects_non_positive_group(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": -1.0})
        with pytest.raises(ValueError, match="positive"):
            FixedClipState(clipping_norm=pg)

    def test_float_sensitivity_unchanged(self):
        """Float clipping_norm should still return float sensitivity."""
        state = FixedClipState(clipping_norm=2.0, normalize_by=4.0)
        assert state.sensitivity == pytest.approx(0.5)
        assert isinstance(state.sensitivity, float)


class TestClippedGradPerGroup:
    """Tests for clipped_grad with PerGroup clipping_norm."""

    def test_basic_per_group_clipping(self):
        """clipped_grad should clip per group when given PerGroup norm."""

        def loss(params, data):
            pred = params["w1"] * data + params["w2"] * data
            return (pred**2).mean()

        params = {"w1": torch.tensor(3.0), "w2": torch.tensor(4.0)}
        pg = per_group(params, w1=1.0, w2=1.0)

        grad_fn, clip_state = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=pg
        )

        data = torch.tensor([1.0, 2.0, 3.0])
        grads, _ = grad_fn(params, data, state=clip_state)

        assert isinstance(grads, dict)
        assert "w1" in grads and "w2" in grads

    def test_sensitivity_is_scalar(self):
        """clip_state.sensitivity should be scalar (L2 norm) even with PerGroup clipping."""

        def loss(params, data):
            return (params["a"] * data).mean()

        params = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
        pg = per_group(params, a=2.0, b=4.0)

        _, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            normalize_by=10.0,
        )

        sens = clip_state.sensitivity
        assert isinstance(sens, float)
        # sensitivity = sqrt(2^2 + 4^2) / 10 = sqrt(20) / 10
        import math

        assert sens == pytest.approx(math.sqrt(20) / 10)

    def test_per_group_with_microbatch(self):
        """Per-group clipping should work with microbatching."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        pg = per_group(params, w=1.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            microbatch_size=2,
        )

        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        grads, _ = grad_fn(params, data, state=clip_state)
        assert isinstance(grads, dict)

    def test_per_group_with_return_aux(self):
        """return_aux should work with per-group clipping."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        pg = per_group(params, w=1.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            return_aux=True,
        )

        data = torch.tensor([1.0, 2.0, 3.0])
        (grads, aux), _ = grad_fn(params, data, state=clip_state)
        assert isinstance(grads, dict)
        assert aux.grad_norms is not None

    def test_global_vs_per_group_single_group(self):
        """Single-group PerGroup should behave like global clipping."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        norm_val = 1.5

        # Global clipping
        grad_fn_g, cs_g = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=norm_val
        )
        # Per-group with single group
        pg = per_group(params, w=norm_val)
        grad_fn_pg, cs_pg = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=pg
        )

        data = torch.tensor([5.0, 10.0, -3.0])
        grads_g, _ = grad_fn_g(params, data, state=cs_g)
        grads_pg, _ = grad_fn_pg(params, data, state=cs_pg)

        torch.testing.assert_close(grads_g["w"], grads_pg["w"])

    def test_noise_multiplier_arithmetic(self):
        """noise_multiplier * clip_state.sensitivity should return scalar."""

        def loss(params, data):
            return (params["a"] * data).mean()

        params = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
        pg = per_group(params, a=1.0, b=2.0)

        _, clip_state = clipped_grad(loss, argnums=0, batch_argnums=1, clipping_norm=pg)

        noise_multiplier = 1.1
        stddev = noise_multiplier * clip_state.sensitivity
        assert isinstance(stddev, float)
        # sensitivity = sqrt(1^2 + 2^2) / 1 = sqrt(5)
        import math

        assert stddev == pytest.approx(1.1 * math.sqrt(5))
