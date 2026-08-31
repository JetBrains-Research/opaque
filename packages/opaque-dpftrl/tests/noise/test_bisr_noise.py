"""Tests for BisrStrategy factory, runtime dispatch, and accounting equivalence."""

import dataclasses

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.dpftrl.amplification import _balls_in_bins
from opaque.api.dpftrl.noise import _bisr as bisr_module
from opaque.api.dpftrl.noise._bisr import BisrStrategy, _native, bisr_strategy
from opaque.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    materialize_lower_triangular,
)
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


class TestBisrStrategy:
    def test_returns_correct_type(self):
        assert isinstance(bisr_strategy(bandwidth=4), BisrStrategy)

    def test_sensitivity_positive(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = bisr_strategy(bandwidth=4).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_lr_schedule_is_rejected_with_recalibration_guidance(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            bisr_strategy(bandwidth=3, lr_schedule=lambda _step: 1.0)

    def test_legacy_none_schedule_state_loads(self):
        strategy = from_state_dict(
            bisr_strategy(bandwidth=3),
            {
                "type": "BisrStrategy",
                "bandwidth": 3,
                "normalized": False,
                "momentum": 0.3,
                "lr_schedule": None,
                "inv_coefficients": None,
            },
        )

        assert strategy == bisr_strategy(bandwidth=3, normalized=False, momentum=0.3)
        assert state_dict(strategy)["lr_schedule"] is None

    def test_legacy_non_none_schedule_state_is_rejected(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            from_state_dict(
                bisr_strategy(bandwidth=3),
                {
                    "type": "BisrStrategy",
                    "bandwidth": 3,
                    "normalized": False,
                    "momentum": 0.3,
                    "lr_schedule": {
                        "__opaque_recipe__": "ConstantSchedule",
                        "value": 0.1,
                    },
                    "inv_coefficients": None,
                },
            )

    def test_gram_uses_only_unweighted_native_path(self, monkeypatch):
        expected = (2.0, 0.5, 0.5, 1.0)

        class CapturingNative:
            def bisr_gram_matrix(self, *_args):
                return expected

            def bisr_gram_matrix_lr(self, *_args):
                raise AssertionError("weighted Gram path must not be called")

        monkeypatch.setattr(bisr_module, "_native", CapturingNative)
        bisr_module._bisr_gram_matrix_cached.cache_clear()
        strategy = bisr_strategy(bandwidth=3)

        assert strategy.gram_matrix(
            n_steps=4, min_sep=2, max_participations=2
        ) == pytest.approx(expected)
        bisr_module._bisr_gram_matrix_cached.cache_clear()

    def test_streaming_matrix_present(self):
        assert bisr_strategy(bandwidth=4).streaming_matrix(**_PART) is not None

    def test_exposes_runtime_noise_factory(self):
        strategy = bisr_strategy(bandwidth=4)
        template = {"w": torch.zeros(6)}
        raw = strategy.raw_noise_factory(
            template,
            n_steps=12,
            min_sep=3,
            max_participations=2,
            key=key(0),
            compute_dtype=torch.float32,
        )
        noise_fn, state, row_l2_at = raw
        assert callable(noise_fn)
        assert state._step_counter == 0
        assert row_l2_at(0) > 0

    def test_mf_gaussian_noise_uses_runtime_operator(self, monkeypatch):
        strategy = bisr_strategy(bandwidth=4)
        template = {"w": torch.zeros(6)}
        calls = {"count": 0}
        original = BisrStrategy.raw_noise_factory

        def fake_raw_noise_factory(self, *args, **kwargs):
            calls["count"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(BisrStrategy, "raw_noise_factory", fake_raw_noise_factory)
        noise_fn, state = mf_gaussian_noise(
            template,
            strategy,
            n_steps=12,
            min_sep=3,
            max_participations=2,
            noise_multiplier=1.0,
            key=key(1),
        )
        out, _ = noise_fn(clipped({"w": torch.zeros(6)}, max_norm=1.0), state)
        assert calls["count"] == 1
        assert float(out.noise_stddev) > 0

    @pytest.mark.parametrize("bandwidth", [2, 4])
    @pytest.mark.parametrize("n_steps", [2, 6, 12])
    def test_runtime_operator_uses_full_horizon_strategy(self, bandwidth, n_steps):
        strategy = bisr_strategy(bandwidth=bandwidth, normalized=False, momentum=0.3)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
        runtime_noise_fn, _, runtime_row_l2_at = strategy.raw_noise_factory(
            {"w": torch.zeros(1)},
            n_steps=n_steps,
            min_sep=1,
            max_participations=1,
            key=key(0),
            compute_dtype=torch.float32,
        )
        del runtime_noise_fn

        expected_dense = streaming.materialize(n_steps)
        runtime_row_l2 = torch.tensor(
            [runtime_row_l2_at(step) for step in range(n_steps)], dtype=torch.float64
        )
        expected_row_l2 = expected_dense.pow(2).sum(dim=1).sqrt()

        torch.testing.assert_close(runtime_row_l2, expected_row_l2)

        full_horizon_strategy_coefs = _native().bisr_strategy_coefficients(
            list(strategy._inv_coefs()), n_steps
        )
        assert len(full_horizon_strategy_coefs) == n_steps
        manual_streaming = inverse_as_streaming_matrix(
            torch.tensor(full_horizon_strategy_coefs, dtype=torch.float64)
        )
        torch.testing.assert_close(
            expected_dense, manual_streaming.materialize(n_steps)
        )

    @pytest.mark.parametrize("normalized", [False, True])
    def test_row_norms_closed_form_matches_probing(self, normalized):
        n_steps = 12
        strategy = bisr_strategy(bandwidth=3, normalized=normalized)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
        closed_form = streaming.row_norms_squared(n_steps)
        probing = dataclasses.replace(
            streaming, row_norms_squared_fn=None
        ).row_norms_squared(n_steps)
        torch.testing.assert_close(closed_form, probing, atol=1e-10, rtol=1e-10)

    def test_matches_old_sensitivity(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be >= 2"):
            bisr_strategy(bandwidth=1)

    @pytest.mark.parametrize("momentum", [-0.1, 1.0, float("nan"), float("inf")])
    def test_rejects_invalid_momentum(self, momentum):
        with pytest.raises(
            ValueError, match=r"momentum must be finite and in \[0, 1\)"
        ):
            bisr_strategy(bandwidth=2, momentum=momentum)

    @pytest.mark.parametrize(
        "coefficients",
        [(0.0, 1.0), (1e-31, 1.0), (1.0, float("nan")), (1.0, float("inf"))],
    )
    def test_rejects_invalid_custom_inverse_coefficients(self, coefficients):
        with pytest.raises(ValueError, match="inv_coefficients"):
            bisr_strategy(bandwidth=2, inv_coefficients=coefficients)

    def test_coefficients_reject_nonpositive_horizon(self):
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            bisr_strategy(bandwidth=2).coefficients(n_steps=0)


class TestBisrPld:
    delta = 1e-5

    def test_bisr_pld(self):
        eps = ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4), **_PART).epsilon_at(
            self.delta
        )
        assert eps > 0

    def test_bisr_bnb(self):
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4)),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0

    def test_bnb_uses_absolute_normalized_forward_encoder(self, monkeypatch):
        n_steps, num_bins = 4, 2
        strategy = bisr_strategy(
            bandwidth=2,
            normalized=True,
            inv_coefficients=(1.0, 1.0),
        )
        process = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, strategy),
            num_bins=num_bins,
            n_steps=n_steps,
        )

        actual_native = _balls_in_bins._native
        captured: dict[str, tuple[float, ...]] = {}
        sentinel = object()

        class CapturingNative:
            def __getattr__(self, name):
                return getattr(actual_native, name)

            def bnb_mc_pld(self, gram, *_):
                captured["gram"] = tuple(gram)
                return sentinel

        monkeypatch.setattr(_balls_in_bins, "_native", CapturingNative())
        assert process.pld() is sentinel

        encoder = materialize_lower_triangular(
            strategy.coefficients(n_steps=n_steps), n_steps
        )
        normalized_encoder = encoder / encoder.square().sum(dim=0).sqrt()
        grouped = torch.stack(
            [
                normalized_encoder[:, bin_index::num_bins].abs().sum(dim=1)
                for bin_index in range(num_bins)
            ],
            dim=1,
        )
        expected = grouped.T @ grouped

        actual = torch.tensor(captured["gram"], dtype=torch.float64).reshape(
            num_bins, num_bins
        )
        torch.testing.assert_close(actual, expected)
        assert actual[0, 1] > 0
