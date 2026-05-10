"""DP-FTRL ``identity_mf`` truncated Poisson must equal DP-SGD truncated Poisson
self-composed ``n_steps`` times.

Regression guard: the ``ftrl_acc.poisson(identity_mf(...), ...,
truncated_batch_size=, dataset_size=)`` path goes through
``PoissonMf.pld``'s truncated branch, while
``dpsgd_acc.poisson(gaussian(...), ..., truncated_batch_size=,
dataset_size=) * n_steps`` goes through ``Poisson.pld``'s truncated
branch followed by ``self_compose``.  Both reduce to the same native
``truncated_poisson_gaussian_pld`` composed ``n_steps`` times, so the
``epsilon_at(delta)`` of one must equal the other.
"""

from __future__ import annotations

import math

import pytest

import opaque.dpftrl.accounting as ftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc


_DELTA = 1e-5


@pytest.mark.parametrize(
    "nm, sample_rate, n_steps, truncated_batch_size, dataset_size",
    [
        (1.1, 0.01, 200, 64, 50_000),
        (0.8, 0.005, 500, 32, 100_000),
        (1.5, 0.02, 100, 256, 50_000),
    ],
)
def test_ftrl_identity_truncated_matches_dpsgd_truncated_composed(
    nm: float,
    sample_rate: float,
    n_steps: int,
    truncated_batch_size: int,
    dataset_size: int,
):
    ftrl_proc = ftrl_acc.poisson(
        ftrl_acc.identity_mf(nm),
        sample_rate=sample_rate,
        n_steps=n_steps,
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
    dpsgd_proc = (
        dpsgd_acc.poisson(
            dpsgd_acc.gaussian(nm),
            sample_rate,
            truncated_batch_size=truncated_batch_size,
            dataset_size=dataset_size,
        )
        * n_steps
    )

    eps_ftrl = ftrl_proc.epsilon_at(_DELTA)
    eps_dpsgd = dpsgd_proc.epsilon_at(_DELTA)
    assert math.isclose(eps_ftrl, eps_dpsgd, rel_tol=1e-9)
