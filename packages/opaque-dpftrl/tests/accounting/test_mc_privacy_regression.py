"""Public worst-case upper-confidence vectors for full-horizon DP-FTRL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
import torch

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    lambda_cgd_strategy,
)

if TYPE_CHECKING:
    from opaque.dpftrl.noise.types import MfStrategy

pytestmark = [pytest.mark.mc_regression, pytest.mark.slow]

_NOISE_MULTIPLIER = 1.3
_TAIL_DELTA = 2e-5
_BODY_DELTA = 1e-2
_MC_SUITE_FAILURE_BUDGET = 1e-6
# Analytic-tail and composition settings are inactive on these direct MC paths.
_MC_CONFIG = {
    "discretization": 1e-4,
    "max_grid_size": 10_000_000,
    "seed": 42,
    "mc_resolution": 1e-5,
    "mc_failure_probability": 1e-7,
}
_EXPECTED_SAMPLES_PER_DIRECTION = 3_178_294
_EXPECTED_ACHIEVED_RESOLUTION = 9.999999700749385e-6
_EPSILON_REL_TOLERANCE = 1e-9
_EPSILON_ABS_TOLERANCE = 3e-9


@dataclass(frozen=True, slots=True)
class _RegressionCase:
    name: str
    amplifier: Literal["balls_in_bins", "b_min_sep"]
    strategy: MfStrategy
    tail_epsilon: float
    body_epsilon: float


# Treat baseline drift as a privacy review event; never refresh values mechanically.
_CASES = (
    _RegressionCase(
        name="bnb-blt",
        amplifier="balls_in_bins",
        strategy=blt_strategy(max_buffers=3, momentum=0.95),
        tail_epsilon=8.603578123103548,
        body_epsilon=4.587585679253265,
    ),
    _RegressionCase(
        name="bnb-bsr",
        amplifier="balls_in_bins",
        strategy=bsr_strategy(bandwidth=4, alpha=0.9, beta=0.7),
        tail_epsilon=10.648482125516102,
        body_epsilon=5.722947783683076,
    ),
    _RegressionCase(
        name="bnb-bisr",
        amplifier="balls_in_bins",
        strategy=bisr_strategy(bandwidth=4, normalized=True, momentum=0.0),
        tail_epsilon=6.4914807166747845,
        body_epsilon=3.4192837778179794,
    ),
    _RegressionCase(
        name="bnb-lambda-cgd",
        amplifier="balls_in_bins",
        strategy=lambda_cgd_strategy(lambda_=0.8, normalized=True),
        tail_epsilon=7.488544627936765,
        body_epsilon=4.122664877905618,
    ),
    _RegressionCase(
        name="b-min-sep-band-mf",
        amplifier="b_min_sep",
        strategy=band_mf_strategy(bands=8, momentum=0.95),
        tail_epsilon=1.5584442850210172,
        body_epsilon=0.5212456489432249,
    ),
)


def _build_process(case: _RegressionCase):
    inner = ftrl_acc.mf_gaussian(
        _NOISE_MULTIPLIER,
        case.strategy,
        n_steps=1,
    )
    if case.amplifier == "balls_in_bins":
        return ftrl_acc.balls_in_bins(inner, num_bins=8, n_steps=32)
    return ftrl_acc.b_min_sep(inner, n_steps=64, p0=0.02)


def _diagnostic(case: _RegressionCase, process, coefficients: torch.Tensor) -> str:
    if case.amplifier == "balls_in_bins":
        amplifier_parameters = {
            "num_bins": process.num_bins,
            "n_steps": process.n_steps,
            "num_epochs": process.num_epochs,
        }
    else:
        amplifier_parameters = {
            "bands": process.min_sep,
            "n_steps": process.n_steps,
            "p0": process.p0,
            "p": process.sampling_prob,
        }
    has_coefficients = coefficients.numel() > 0
    resolved_config = acc.get_discretization(**_MC_CONFIG)
    details = {
        "case": case,
        "noise_multiplier": _NOISE_MULTIPLIER,
        "amplifier_parameters": amplifier_parameters,
        "mc_config": _MC_CONFIG,
        "samples_per_direction": resolved_config.resolved_num_mc_samples,
        "suite_failure_probability": (
            len(_CASES) * _MC_CONFIG["mc_failure_probability"]
        ),
        "coefficients": {
            "min": float(coefficients.min()) if has_coefficients else None,
            "max": float(coefficients.max()) if has_coefficients else None,
            "norm": float(coefficients.norm()) if has_coefficients else None,
            "values": coefficients.tolist(),
        },
    }
    return repr(details)


def _assert_epsilon(
    *,
    actual: float,
    expected: float,
    delta: float,
    diagnostic: str,
) -> None:
    assert actual == pytest.approx(
        expected,
        rel=_EPSILON_REL_TOLERANCE,
        abs=_EPSILON_ABS_TOLERANCE,
    ), (
        f"epsilon drift at delta={delta!r}: expected={expected!r}, "
        f"observed={actual!r}; {diagnostic}"
    )


def test_mc_suite_confidence_budget():
    config = acc.get_discretization(**_MC_CONFIG)
    suite_failure_probability = len(_CASES) * config.mc_failure_probability

    assert config.resolved_num_mc_samples == _EXPECTED_SAMPLES_PER_DIRECTION, (
        f"expected {_EXPECTED_SAMPLES_PER_DIRECTION} samples per direction, "
        f"observed {config.resolved_num_mc_samples}; config={_MC_CONFIG!r}"
    )
    assert suite_failure_probability <= _MC_SUITE_FAILURE_BUDGET, (
        f"suite failure probability {suite_failure_probability!r} exceeds "
        f"{_MC_SUITE_FAILURE_BUDGET!r}; cases={len(_CASES)}; config={_MC_CONFIG!r}"
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_mc_privacy_regression(case: _RegressionCase):
    process = _build_process(case)
    coefficients = case.strategy.coefficients(
        n_steps=process.n_steps,
        min_sep=process.min_sep,
        max_participations=process.max_participations,
    )
    assert isinstance(coefficients, torch.Tensor), (
        f"case={case!r}; coefficients_type={type(coefficients).__name__}"
    )
    diagnostic = _diagnostic(case, process, coefficients)
    expected_coefficient_count = (
        process.n_steps if case.amplifier == "balls_in_bins" else process.min_sep
    )

    # These constraints keep every vector inside its amplifier's theorem domain.
    assert coefficients.ndim == 1, diagnostic
    assert coefficients.numel() == expected_coefficient_count, diagnostic
    assert torch.isfinite(coefficients).all().item(), diagnostic
    assert (coefficients >= 0).all().item(), diagnostic
    if case.amplifier == "b_min_sep":
        assert float(coefficients.norm()) == pytest.approx(1.0, abs=1e-12), diagnostic

    tail_epsilon = process.epsilon_at(_TAIL_DELTA, **_MC_CONFIG)
    pld = process.pld(**_MC_CONFIG)
    body_epsilon = pld.epsilon_at(_BODY_DELTA)
    diagnostic = (
        f"{diagnostic}; mc_metadata={{'failure_probability': "
        f"{pld.mc_failure_probability!r}, 'resolution': {pld.mc_resolution!r}, "
        f"'infinity_mass': {pld.infinity_mass!r}}}"
    )

    assert pld.mc_failure_probability == pytest.approx(
        _MC_CONFIG["mc_failure_probability"]
    ), diagnostic
    assert pld.mc_confidence == pytest.approx(
        1.0 - _MC_CONFIG["mc_failure_probability"]
    ), diagnostic
    assert pld.mc_resolution == pytest.approx(
        _EXPECTED_ACHIEVED_RESOLUTION,
        rel=1e-12,
        abs=1e-15,
    ), diagnostic
    assert pld.infinity_mass == pytest.approx(
        _EXPECTED_ACHIEVED_RESOLUTION,
        rel=1e-12,
        abs=1e-15,
    ), diagnostic
    assert math.isinf(pld.epsilon_at(pld.infinity_mass / 2.0)), diagnostic
    assert tail_epsilon == pld.epsilon_at(_TAIL_DELTA), diagnostic

    _assert_epsilon(
        actual=tail_epsilon,
        expected=case.tail_epsilon,
        delta=_TAIL_DELTA,
        diagnostic=diagnostic,
    )
    _assert_epsilon(
        actual=body_epsilon,
        expected=case.body_epsilon,
        delta=_BODY_DELTA,
        diagnostic=diagnostic,
    )
