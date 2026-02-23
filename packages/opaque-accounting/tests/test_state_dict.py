"""Tests for DpProcess state_dict serialization."""

import math
from typing import cast

import opaque_accounting as acc
from opaque_accounting.accountant import Accountant
from opaque_accounting.base import DpProcess
from opaque_accounting.mechanisms.band_mf_amplified import (
    BandMfAmplified,
    band_mf_amplified,
)


def test_gaussian_state_dict_structure():
    proc = acc.gaussian(1.1)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "Gaussian"
    assert state["noise_multiplier"] == 1.1


def test_eps_delta_state_dict_structure():
    proc = acc.eps_delta(1.0, 1e-5)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "EpsDelta"
    assert state["epsilon"] == 1.0
    assert state["delta"] == 1e-5


def test_identity_state_dict_structure():
    proc = acc.identity()
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "Identity"


def test_poisson_state_dict_structure():
    proc = acc.poisson(acc.gaussian(0.8), 0.01)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "Poisson"
    assert state["sample_rate"] == 0.01
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Gaussian"


def test_truncated_poisson_state_dict_structure():
    proc = acc.truncated_poisson(acc.gaussian(0.8), 0.01, 128, 10_000)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "TruncatedPoisson"
    assert state["batch_size_cap"] == 128
    assert state["dataset_size"] == 10_000
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Gaussian"


def test_parallel_poisson_state_dict_structure():
    proc = acc.parallel_poisson(acc.gaussian(0.8), sample_rate=0.01, num_workers=4)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "ParallelPoisson"
    assert state["num_workers"] == 4
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Poisson"


def test_adaclip_state_dict_structure():
    proc = acc.adaclip(acc.gaussian(0.8), batch_size=1000)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "AdaClip"
    assert state["quantile_noise_multiplier"] == 0.05
    assert state["batch_size"] == 1000
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Gaussian"


def test_composed_state_dict_structure():
    left = acc.gaussian(0.8)
    right = acc.eps_delta(1.0, 1e-5)
    proc = left | right
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "Composed"
    left = cast(dict[str, object], state["left"])
    right = cast(dict[str, object], state["right"])
    assert left["type"] == "Gaussian"
    assert right["type"] == "EpsDelta"


def test_repeated_state_dict_structure():
    proc = acc.gaussian(0.8) * 3
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "Repeated"
    assert state["count"] == 3
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Gaussian"


def test_cached_state_dict_structure():
    proc = acc.cached(acc.gaussian(0.8))
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "CachedProcess"
    inner = cast(dict[str, object], state["inner"])
    assert inner["type"] == "Gaussian"


def test_band_mf_amplified_state_dict_structure():
    proc = band_mf_amplified(1.0, 2.5, 0.01, 200)
    state = cast(dict[str, object], proc.state_dict())
    assert state["type"] == "BandMfAmplified"
    assert state["noise_multiplier"] == 1.0
    assert state["sensitivity"] == 2.5
    assert state["sample_rate"] == 0.01
    assert state["num_groups"] == 200


def test_band_mf_amplified_round_trip():
    proc = band_mf_amplified(1.0, 2.5, 0.01, 200)
    state = proc.state_dict()
    restored = DpProcess.from_state_dict(state)
    assert isinstance(restored, BandMfAmplified)
    assert restored == proc


def test_accountant_state_dict_roundtrip():
    acct = Accountant()
    step = acc.poisson(acc.adaclip(acc.gaussian(0.8), batch_size=1000), 0.01)
    acct = acct | step
    state = acct.state_dict()
    restored = Accountant.from_state_dict(state)
    eps = restored.epsilon_at(1e-5)
    assert math.isfinite(eps) and eps > 0
