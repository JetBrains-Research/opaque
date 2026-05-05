"""Tests for opaque.optimizers.adafactor."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

# Pre-load clipping.types so opaque.core.noise's import of
# SecondMomentClippingOutput doesn't observe a partial module mid-cycle.
from opaque.clipping.types import ClippedPytree  # noqa: E402, F401
from opaque.core.noise import noised  # noqa: E402
from opaque.clipping.per_group import PerGroup  # noqa: E402
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


class TestExplicitKwargsRejected:
    """``optimizer.update()`` does not take per-step metadata kwargs;
    metadata travels via ``NoisedPytree``.  Stray kwargs surface as a
    Python ``TypeError`` from the unknown-keyword check."""

    def test_noise_stddev_rejected(self, matrix_params, matrix_grads):
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(TypeError, match="noise_stddev"):
            opt.update(matrix_grads, state, params=matrix_params, noise_stddev=0.5)

    def test_noisy_squared_grads_rejected(self, matrix_params, matrix_grads):
        sq = {k: v.pow(2) for k, v in matrix_grads.items()}
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            opt.update(
                matrix_grads, state, params=matrix_params, noisy_squared_grads=sq
            )


class TestBCMode:
    """DP noise-variance bias correction on the row/col factors."""

    def test_default_keeps_phi_zero(self, matrix_params, matrix_grads):
        """Default ``noise_bias_correction=False``: φ stays at 0 even
        under NoisedPytree updates."""
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        for _ in range(5):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=0.5),
                state,
                params=matrix_params,
            )
        assert all(v == 0.0 for v in _af_state(state).phi_flat)

    def test_phi_advances_under_noisy_metadata(self, matrix_params, matrix_grads):
        """With BC on, φ tracks the β₂_t-EMA of σ² per leaf."""
        sigma = 0.5
        opt = adafactor(lr=1e-3, noise_bias_correction=True)
        state = opt.init(matrix_params)
        # Drive 8 steps of constant σ; phi should approach σ² steady state.
        for _ in range(8):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=matrix_params,
            )
        # All leaves have the same scalar σ → all phi entries equal.
        phi = _af_state(state).phi_flat
        assert all(p == pytest.approx(phi[0]) for p in phi)
        assert phi[0] > 0.0
        # Steady-state target: σ² (mostly converged after 8 steps).
        assert phi[0] == pytest.approx(sigma**2, rel=0.2)

    def test_per_group_routes_to_per_leaf_phi(self, matrix_params, matrix_grads):
        """PerGroup noise_stddev with σ varying per group → phi varies per leaf
        according to the group its dotted-path key resolves to."""
        pg = PerGroup(
            groups={
                "fc1.weight": "attn",
                "fc2.weight": "mlp",
                "bias": "mlp",
            },
            values={"attn": 0.2, "mlp": 0.8},
        )
        opt = adafactor(lr=1e-3, noise_bias_correction=True)
        state = opt.init(matrix_params)
        _, state = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=matrix_params,
        )
        af = _af_state(state)
        path_to_phi = dict(zip(af.paths, af.phi_flat))
        # Group "attn" → σ=0.2 → variance 0.04 (× one-step EMA factor)
        # Group "mlp"  → σ=0.8 → variance 0.64
        # The two should differ proportionally to (0.04, 0.64).
        attn_phi = path_to_phi["fc1.weight"]
        mlp_phi = path_to_phi["fc2.weight"]
        bias_phi = path_to_phi["bias"]
        assert mlp_phi > attn_phi
        # Bias is in the same group as fc2.weight → same phi.
        assert mlp_phi == pytest.approx(bias_phi)
        # Variance ratio is (0.8/0.2)² = 16.
        assert mlp_phi / attn_phi == pytest.approx(16.0, rel=1e-4)

    def test_bc_changes_updates(self, matrix_params, matrix_grads):
        """With non-zero σ, BC actually changes the update vs vanilla."""
        sigma = 0.5
        opt_bc = adafactor(lr=1e-3, noise_bias_correction=True)
        opt_no = adafactor(lr=1e-3, noise_bias_correction=False)
        s_bc = opt_bc.init(matrix_params)
        s_no = opt_no.init(matrix_params)
        # Run a few warmup steps so phi has built up.
        for _ in range(3):
            _, s_bc = opt_bc.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                s_bc,
                params=matrix_params,
            )
            _, s_no = opt_no.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
                s_no,
                params=matrix_params,
            )
        u_bc, _ = opt_bc.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
            s_bc,
            params=matrix_params,
        )
        u_no, _ = opt_no.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=sigma),
            s_no,
            params=matrix_params,
        )
        # On at least one leaf the BC and no-BC updates differ.
        any_diff = any(not torch.allclose(u_bc[k], u_no[k]) for k in matrix_params)
        assert any_diff


class TestValidation:
    def test_negative_decay_rate_required(self):
        with pytest.raises(ValueError, match="decay_rate"):
            adafactor(decay_rate=0.5)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            adafactor(eps_grad=0.0)
