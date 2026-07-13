"""make_clipping_aggregate: clip math + executor-safe pickling."""

import builtins
import pickle

import pytest
import torch

ifed_client = pytest.importorskip("ifed_client")
cloudpickle = pytest.importorskip("cloudpickle")

from ifed.metrics import MetricsBundle  # noqa: E402
from ifed_client import AgentState  # noqa: E402
from ifed_client._bundle import pickle_defaults_by_value  # noqa: E402

from opaque.federated import make_clipping_aggregate  # noqa: E402


def _agent_state(params):
    return AgentState(
        params=params,
        metrics=MetricsBundle(scalars={"_": 0.0}, histograms={"_": [0.0]}),
    )


def _global_norm(params):
    return float(torch.cat([g.flatten() for g in params.values()]).norm())


def test_clips_each_client_and_sums():
    aggregate = make_clipping_aggregate(1.0)
    big = {"w": torch.full((2,), 30.0), "b": torch.full((1,), 40.0)}  # norm >> 1
    small = {"w": torch.full((2,), 0.1), "b": torch.full((1,), 0.1)}  # norm < 1
    out = aggregate([_agent_state(big), _agent_state(small)])

    assert out.count == 2
    # big was scaled down to norm 1, small passed through unscaled
    scale = 1.0 / _global_norm(big)
    assert torch.allclose(out.grads["w"], big["w"] * scale + small["w"], atol=1e-5)
    assert torch.allclose(out.grads["b"], big["b"] * scale + small["b"], atol=1e-5)


def test_sum_sensitivity_bounded_by_clipping_norm():
    """Any one client moves the sum by at most C in L2."""
    clipping_norm = 2.0
    aggregate = make_clipping_aggregate(clipping_norm)
    base = [_agent_state({"w": torch.randn(4)}) for _ in range(3)]
    adversary = _agent_state({"w": torch.full((4,), 1e6)})

    with_adv = aggregate(base + [adversary]).grads
    without = aggregate(base).grads
    delta = {k: with_adv[k] - without[k] for k in with_adv}
    assert _global_norm(delta) <= clipping_norm + 1e-5


def test_empty_cohort_raises():
    aggregate = make_clipping_aggregate(1.0)
    with pytest.raises(ValueError, match="no agent states"):
        aggregate([])


def test_validates_clipping_norm():
    with pytest.raises(ValueError, match="clipping_norm"):
        make_clipping_aggregate(0.0)


def test_pickles_without_opaque(monkeypatch):
    """The executor venv has no opaque: unpickling must not import opaque.*."""
    aggregate = make_clipping_aggregate(1.0)
    with pickle_defaults_by_value():
        payload = cloudpickle.dumps(aggregate)

    real_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] == "opaque":
            raise ImportError(f"executor venv has no {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    restored = pickle.loads(payload)
    out = restored([_agent_state({"w": torch.full((2,), 30.0)})])
    assert out.count == 1
    assert _global_norm(out.grads) <= 1.0 + 1e-5
