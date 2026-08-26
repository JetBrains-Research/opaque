"""BnB prefix Grams are projections of the matrix deployed by runtime."""

from __future__ import annotations

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core import _native
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy, _column_norm
from opaque.dpftrl.noise import bisr_strategy, lambda_cgd_strategy

_N_STEPS = 12
_NUM_BINS = 3
_MC_KW = {
    "seed": 17,
    "mc_resolution": 0.05,
    "mc_failure_probability": 0.1,
}


def _lambda_noising_matrix(strategy) -> torch.Tensor:
    matrix = torch.zeros((_N_STEPS, _N_STEPS), dtype=torch.float64)
    for step in range(_N_STEPS):
        norm = (
            _column_norm(strategy.lambda_, _N_STEPS, step)
            if strategy.normalized
            else 1.0
        )
        matrix[step, step] = norm
        if step > 0:
            matrix[step, step - 1] = -strategy.lambda_ * norm
    return matrix


def _runtime_encoder(strategy) -> torch.Tensor:
    if isinstance(strategy, LambdaCgdStrategy):
        noising = _lambda_noising_matrix(strategy)
    else:
        noising = strategy.streaming_matrix(n_steps=_N_STEPS).materialize(_N_STEPS)
    return torch.linalg.inv(noising)


def _prefix_gram(encoder: torch.Tensor, prefix_steps: int) -> torch.Tensor:
    # The BnB privacy pair groups columns of |C|. For a K-step release, use the
    # leading principal projection of the actual N-step runtime encoder C_N;
    # do not rebuild or re-normalize a K-step strategy.
    prefix = encoder[:prefix_steps, :prefix_steps].abs()
    modes = torch.stack(
        [prefix[:, bin_index::_NUM_BINS].sum(dim=1) for bin_index in range(_NUM_BINS)],
        dim=1,
    )
    return modes.T @ modes


@pytest.mark.parametrize("normalized", [False, True], ids=["raw", "normalized"])
@pytest.mark.parametrize("kind", ["lambda_cgd", "bisr"])
def test_prefix_gram_matches_deployed_runtime(kind, normalized):
    if kind == "lambda_cgd":
        strategy = lambda_cgd_strategy(
            lambda_=0.9,
            normalized=normalized,
        )
    else:
        strategy = bisr_strategy(
            bandwidth=4,
            momentum=0.8,
            normalized=normalized,
        )

    process = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, strategy),
        num_bins=_NUM_BINS,
        n_steps=_N_STEPS,
    )
    encoder = _runtime_encoder(strategy)
    if normalized:
        torch.testing.assert_close(
            encoder.square().sum(dim=0),
            torch.ones(_N_STEPS, dtype=torch.float64),
            rtol=1e-9,
            atol=1e-10,
        )

    for prefix_steps in range(_NUM_BINS, _N_STEPS + 1, _NUM_BINS):
        expected = _prefix_gram(encoder, prefix_steps)
        actual = torch.tensor(
            process._correlated_gram_at(prefix_steps), dtype=torch.float64
        ).reshape(_NUM_BINS, _NUM_BINS)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("kind", ["lambda_cgd", "bisr"])
def test_public_prefix_pld_uses_deployed_runtime_gram(kind):
    if kind == "lambda_cgd":
        strategy = lambda_cgd_strategy(lambda_=0.9)
    else:
        strategy = bisr_strategy(bandwidth=4, momentum=0.8)

    process = ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(1.0, strategy),
        num_bins=_NUM_BINS,
        n_steps=_N_STEPS,
    )
    prefix_steps = 2 * _NUM_BINS
    config = get_discretization(**_MC_KW)
    expected = _native.bnb_mc_pld(
        list(process._correlated_gram_at(prefix_steps)),
        _NUM_BINS,
        1.0,
        config.to_native(),
    )
    actual = process.pld_at(prefix_steps, **_MC_KW)

    assert actual.epsilon_at(0.1) == pytest.approx(expected.epsilon_at(0.1))
