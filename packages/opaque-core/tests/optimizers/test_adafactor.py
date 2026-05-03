"""Tests for opaque.optimizers.adafactor."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import AdafactorState, adafactor  # noqa: E402


@pytest.fixture
def matrix_params():
    torch.manual_seed(0)
    return {
        "fc1.weight": torch.randn(8, 4),  # factored
        "fc2.weight": torch.randn(4, 2),  # factored
        "bias": torch.randn(2),  # scalar v
    }


@pytest.fixture
def matrix_grads(matrix_params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in matrix_params.items()}


def _af_state(chain_state) -> AdafactorState:
    return chain_state[0]


class TestVanilla:
    def test_state_factored_for_matrices_scalar_for_vectors(self, matrix_params):
        opt = adafactor(lr=1e-3)
        st = _af_state(opt.init(matrix_params))
        assert isinstance(st, AdafactorState)
        # Three flat leaves; per-leaf v_state tuple of length 2 (factored)
        # for matrices, length 1 (scalar) for the bias vector.
        assert len(st.v_flat) == 3
        v_lengths = [len(v) for v in st.v_flat]
        assert sorted(v_lengths) == [1, 2, 2]

    def test_factored_v_shapes_match_axes(self):
        """For a (rows, cols) matrix, v_row has shape (rows,) and v_col (cols,)."""
        params = {"w": torch.randn(8, 4)}
        opt = adafactor(lr=1e-3)
        st = _af_state(opt.init(params))
        v_row, v_col = st.v_flat[0]
        assert v_row.shape == (8,)
        assert v_col.shape == (4,)

    def test_step_increments(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        _, state = opt.update(matrix_grads, state, params=matrix_params)
        assert _af_state(state).step == 1
        _, state = opt.update(matrix_grads, state, params=matrix_params)
        assert _af_state(state).step == 2

    def test_apply_updates_changes_params(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        orig = {k: v.clone() for k, v in matrix_params.items()}
        state = opt.init(matrix_params)
        updates, _ = opt.update(matrix_grads, state, params=matrix_params)
        new = torchopt.apply_updates(matrix_params, updates)
        assert any(not torch.equal(new[k], orig[k]) for k in matrix_params)

    def test_first_moment_optional(self, matrix_params, matrix_grads):
        opt0 = adafactor(lr=1e-3, beta1=0.0)
        opt1 = adafactor(lr=1e-3, beta1=0.9)
        st0 = _af_state(opt0.init(matrix_params))
        st1 = _af_state(opt1.init(matrix_params))
        assert st0.m is None
        assert st1.m is not None

    def test_decoupled_wd_zero_grad(self):
        params = {"w": torch.ones(3) * 2.0}
        grads = {"w": torch.zeros(3)}
        opt = adafactor(lr=0.1, weight_decay=0.5, decoupled_weight_decay=True)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # update = -lr * (0 + wd * params)
        expected = -0.1 * 0.5 * params["w"]
        torch.testing.assert_close(updates["w"], expected, atol=1e-6, rtol=0.0)

    def test_finite_updates(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        updates, _ = opt.update(matrix_grads, state, params=matrix_params)
        for k in matrix_params:
            assert torch.isfinite(updates[k]).all()


class TestDeferredDPModes:
    """Phase A: factored-v DP corrections are deferred; both raise."""

    def test_noise_stddev_raises(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(NotImplementedError, match="DP-Adafactor"):
            opt.update(matrix_grads, state, params=matrix_params, noise_stddev=0.5)

    def test_noisy_squared_grads_raises(self, matrix_params, matrix_grads):
        sq = {k: v.pow(2) for k, v in matrix_grads.items()}
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(NotImplementedError, match="DP-Adafactor"):
            opt.update(matrix_grads, state, params=matrix_params, noisy_squared_grads=sq)


class TestValidation:
    def test_negative_decay_rate_required(self):
        with pytest.raises(ValueError, match="decay_rate"):
            adafactor(decay_rate=0.5)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            adafactor(eps_grad=0.0)
