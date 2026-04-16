"""Tests for jme_adamw optimizer."""

import pytest
import torch


@pytest.fixture
def params():
    return {"w": torch.randn(4, 3), "b": torch.randn(4)}


@pytest.fixture
def grads():
    return {"w": torch.randn(4, 3), "b": torch.randn(4)}


@pytest.fixture
def sq_grads(grads):
    return {k: v**2 for k, v in grads.items()}


def _requires_torchopt():
    try:
        import torchopt  # noqa: F401

        return False
    except ImportError:
        return True


pytestmark = pytest.mark.skipif(_requires_torchopt(), reason="torchopt not installed")


class TestJmeAdamW:
    def test_import(self):
        from opaque.optimizers import jme_adamw, JmeAdamWState  # noqa: F401

    def test_is_gradient_transformation(self):
        from opaque.optimizers import jme_adamw
        from torchopt.base import GradientTransformation

        opt = jme_adamw(lr=1e-3)
        assert isinstance(opt, GradientTransformation)

    def test_init_state(self, params):
        from opaque.optimizers import jme_adamw, JmeAdamWState

        opt = jme_adamw(lr=1e-3)
        state = opt.init(params)
        assert isinstance(state, tuple)
        s_adam = state[0]
        assert isinstance(s_adam, JmeAdamWState)
        assert s_adam.step == 0
        assert s_adam.mu["w"].shape == (4, 3)
        assert s_adam.nu["b"].shape == (4,)

    def test_update_produces_correct_shapes(self, params, grads, sq_grads):
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=1e-3)
        state = opt.init(params)
        updates, new_state = opt.update(
            grads,
            state,
            noisy_squared_grads=sq_grads,
        )
        assert updates["w"].shape == (4, 3)
        assert updates["b"].shape == (4,)

    def test_step_increments(self, params, grads, sq_grads):
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, noisy_squared_grads=sq_grads)
        assert state[0].step == 1
        _, state = opt.update(grads, state, noisy_squared_grads=sq_grads)
        assert state[0].step == 2

    def test_updates_descend(self, params, grads, sq_grads):
        """Updates should point opposite to gradient direction."""
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=0.1)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, noisy_squared_grads=sq_grads)
        for k in grads:
            dot = (updates[k] * grads[k]).sum()
            assert dot < 0, f"Update should descend for param '{k}'"

    def test_callable_lr(self, params, grads, sq_grads):
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=lambda step: 1e-3 * (0.1 if step > 5 else 1.0))
        state = opt.init(params)
        updates, _ = opt.update(grads, state, noisy_squared_grads=sq_grads)
        assert updates["w"].shape == (4, 3)

    def test_weight_decay(self, params, grads, sq_grads):
        """Weight decay should affect updates when params are provided."""
        from opaque.optimizers import jme_adamw

        opt_no_wd = jme_adamw(lr=1e-3, weight_decay=0.0)
        opt_wd = jme_adamw(lr=1e-3, weight_decay=0.1)

        s1 = opt_no_wd.init(params)
        s2 = opt_wd.init(params)

        u1, _ = opt_no_wd.update(grads, s1, noisy_squared_grads=sq_grads)
        u2, _ = opt_wd.update(
            grads,
            s2,
            params=params,
            noisy_squared_grads=sq_grads,
        )
        assert not torch.allclose(u1["w"], u2["w"])

    def test_missing_noisy_squared_grads_raises(self, params, grads):
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(ValueError, match="noisy_squared_grads"):
            opt.update(grads, state)

    def test_apply_updates(self, params, grads, sq_grads):
        """Full cycle: init → update → apply."""
        import torchopt
        from opaque.optimizers import jme_adamw

        opt = jme_adamw(lr=1e-3)
        old_w = params["w"].clone()
        state = opt.init(params)
        updates, state = opt.update(
            grads,
            state,
            noisy_squared_grads=sq_grads,
        )
        new_params = torchopt.apply_updates(params, updates)
        assert new_params["w"].shape == old_w.shape
        assert not torch.allclose(new_params["w"], old_w)
