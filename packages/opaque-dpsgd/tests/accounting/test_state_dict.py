"""Tests for DpProcess flat state_dict serialization (opaque.serialization)."""

import math
from typing import cast

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import Accountant
from opaque.dpftrl.accounting.amplification._b_min_sep import BMinSep
from opaque.dpftrl.accounting.amplification._cyclic_poisson import CyclicPoisson
from opaque.dpftrl.accounting.mechanisms._band_mf import BandMf
from opaque.serialization import from_state_dict, state_dict

# Template type selects the registered handler; PLD decode uses the flat dict's
# root ``type`` field, so any concrete :class:`~opaque.accounting._base.DpProcess`
# instance is sufficient (``identity()`` is a stable choice).
_PROCESS_TEMPLATE = acc.identity()


def test_gaussian_state_dict_structure():
    proc = dpsgd_acc.gaussian(1.1)
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "Gaussian"
    assert state["noise_multiplier"] == 1.1


def test_eps_delta_state_dict_structure():
    proc = acc.eps_delta(1.0, 1e-5)
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "EpsDelta"
    assert state["epsilon"] == 1.0
    assert state["delta"] == 1e-5


def test_identity_state_dict_structure():
    proc = acc.identity()
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "Identity"


def test_poisson_state_dict_structure():
    proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "Poisson"
    assert state["sample_rate"] == 0.01
    assert state["inner.type"] == "Gaussian"


def test_truncated_poisson_state_dict_structure():
    proc = dpsgd_acc.truncated_poisson(dpsgd_acc.gaussian(0.8), 0.01, 128, 10_000)
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "TruncatedPoisson"
    assert state["batch_size_cap"] == 128
    assert state["dataset_size"] == 10_000
    assert state["inner.type"] == "Gaussian"


def test_parallel_poisson_state_dict_structure():
    proc = dpsgd_acc.parallel_poisson(
        dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4
    )
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "ParallelPoisson"
    assert state["num_workers"] == 4
    assert state["inner.type"] == "Poisson"
    assert state["inner.inner.type"] == "Gaussian"


def test_adaclip_state_dict_structure():
    proc = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "AdaClip"
    assert state["fraction_noise_std"] == 0.05
    assert state["expected_batch_size"] == 1000
    assert state["inner.type"] == "Gaussian"


def test_composed_state_dict_structure():
    left = dpsgd_acc.gaussian(0.8)
    right = acc.eps_delta(1.0, 1e-5)
    proc = left | right
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "Composed"
    assert state["left.type"] == "Gaussian"
    assert state["right.type"] == "EpsDelta"


def test_repeated_state_dict_structure():
    proc = dpsgd_acc.gaussian(0.8) * 3
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "Repeated"
    assert state["count"] == 3
    assert state["inner.type"] == "Gaussian"


def test_cached_state_dict_structure():
    proc = acc.cached(dpsgd_acc.gaussian(0.8))
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "CachedProcess"
    assert state["inner.type"] == "Gaussian"


def test_cyclic_poisson_state_dict_structure():
    proc = ftrl_acc.cyclic_poisson(
        ftrl_acc.band_mf(1.0, sensitivity=2.5, num_groups=200), sample_rate=0.01
    )
    state = cast(dict[str, object], state_dict(proc))
    assert state["type"] == "CyclicPoisson"
    assert state["sample_rate"] == 0.01
    assert state["inner.type"] == "BandMf"
    assert state["inner.noise_multiplier"] == 1.0
    assert state["inner.sensitivity"] == 2.5
    assert state["inner.num_groups"] == 200


def test_b_min_sep_round_trip():
    proc = ftrl_acc.b_min_sep(
        ftrl_acc.band_mf(1.0, sensitivity=1.2, num_groups=50),
        strategy_coefficients=(0.9, 0.1),
        n_steps=100,
        p0=0.02,
    )
    state = state_dict(proc)
    restored = from_state_dict(_PROCESS_TEMPLATE, state)
    assert isinstance(restored, BMinSep)
    assert restored == proc


def test_cyclic_poisson_round_trip():
    proc = ftrl_acc.cyclic_poisson(
        ftrl_acc.band_mf(1.0, sensitivity=2.5, num_groups=200), sample_rate=0.01
    )
    state = state_dict(proc)
    restored = from_state_dict(_PROCESS_TEMPLATE, state)
    assert isinstance(restored, CyclicPoisson)
    assert isinstance(restored.inner, BandMf)
    assert restored == proc


def test_accountant_state_dict_roundtrip():
    acct = Accountant()
    step = dpsgd_acc.poisson(
        dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000), 0.01
    )
    acct = acct | step
    state = state_dict(acct)
    restored = from_state_dict(Accountant(), state)
    eps = restored.epsilon_at(1e-5)
    assert math.isfinite(eps) and eps > 0
