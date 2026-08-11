"""Tests for BisrStrategy factory, runtime dispatch, and accounting equivalence."""

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise._bisr import BisrStrategy, _native, bisr_strategy
from opaque.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    materialize_lower_triangular,
)
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.random import key
from opaque.serialization import state_dict
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

    def test_schedule_weighted_gram_matches_dense_step_weighted_operator(self):
        n_steps, min_sep, max_participations = 6, 2, 3
        learning_rates = torch.tensor(
            [1.0, 0.5, 2.0, 1.5, 0.25, 3.0], dtype=torch.float64
        )
        strategy = bisr_strategy(
            bandwidth=3,
            normalized=False,
            momentum=0.3,
            lr_schedule=lambda step: float(learning_rates[step]),
        )

        encoder = materialize_lower_triangular(
            strategy.coefficients(n_steps=n_steps), n_steps
        )
        grouped_columns = torch.stack(
            [encoder[:, bin_index::min_sep].sum(dim=1) for bin_index in range(min_sep)],
            dim=1,
        )
        expected = grouped_columns.T @ torch.diag(learning_rates) ** 2 @ grouped_columns

        gram = torch.tensor(
            strategy.gram_matrix(
                n_steps=n_steps,
                min_sep=min_sep,
                max_participations=max_participations,
            ),
            dtype=torch.float64,
        ).reshape(min_sep, min_sep)
        torch.testing.assert_close(gram, expected)

    def test_uniform_schedule_matches_unweighted_gram(self):
        kwargs = {"n_steps": 12, "min_sep": 3, "max_participations": 4}
        unweighted = bisr_strategy(bandwidth=3, normalized=False, momentum=0.3)
        weighted = bisr_strategy(
            bandwidth=3,
            normalized=False,
            momentum=0.3,
            lr_schedule=lambda _step: 1.0,
        )

        assert weighted.gram_matrix(**kwargs) == pytest.approx(
            unweighted.gram_matrix(**kwargs)
        )

    def test_callable_schedule_is_not_serializable(self):
        strategy = bisr_strategy(bandwidth=3, lr_schedule=lambda _step: 1.0)
        with pytest.raises(TypeError, match="callable strategy field"):
            state_dict(strategy)

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
    @pytest.mark.parametrize("n_steps", [6, 12])
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

    def test_matches_old_sensitivity(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be >= 2"):
            bisr_strategy(bandwidth=1)


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
        ).epsilon_at(self.delta)
        assert eps > 0
