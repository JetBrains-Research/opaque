"""clipped_grad: eager compile, lazy run open, per-round protocol, contracts."""

from typing import Dict

import pytest
import torch

ifed = pytest.importorskip("ifed")
ifed_client = pytest.importorskip("ifed_client")

from ifed_client import RoundInput, RoundResult  # noqa: E402

from opaque.api.engine.clipping.types import FixedClipState  # noqa: E402
from opaque.api.engine.types import ClippedPytree  # noqa: E402
from opaque.federated import DataLoader, MinSepSampler, clipped_grad  # noqa: E402


class Iris(ifed.Dataset):
    sepal_length = ifed.Float()
    sepal_width = ifed.Float()


def loss_fn(
    params: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor]
) -> torch.Tensor:
    pred = data["sepal_length"].unsqueeze(1) @ params["w"] + params["b"]
    return ((pred.squeeze(-1) - data["sepal_width"]) ** 2).mean()


PARAMS = {"w": torch.zeros(1, 1), "b": torch.zeros(1)}


class FakeRun:
    def __init__(self, count_fn):
        self.next_calls = []
        self._count_fn = count_fn

    def next(self, state):
        self.next_calls.append(state)
        return RoundResult(
            grads={"w": torch.full((1, 1), 6.0), "b": torch.full((1,), 3.0)},
            count=self._count_fn(),
        )


class FakePlan:
    def __init__(self, count_fn):
        self.open_calls = []
        self.run = FakeRun(count_fn)

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        return self.run


class FakeClient:
    """Duck-typed ifed_client.Client: records compile, hands out a FakePlan."""

    def __init__(self, count_fn=lambda: 2):
        self.compile_calls = []
        self.plan = FakePlan(count_fn)

    def compile(self, model, **kwargs):
        self.compile_calls.append((model, kwargs))
        return self.plan


@pytest.fixture
def population():
    return ifed.Population("/hive", datasets=[Iris])


def _loader(population, rounds=4, batch_size=2, bands=2):
    sampler = MinSepSampler(population, batch_size=batch_size, bands=bands)
    return DataLoader(population, batch_sampler=sampler, rounds=rounds)


def test_eager_compile_at_factory(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    assert isinstance(clip_state, FixedClipState)
    assert len(client.compile_calls) == 1  # EAGER: before any grad_fn call
    model, kwargs = client.compile_calls[0]
    assert isinstance(model, ifed.FunctionalModel)
    assert kwargs["population"] == "/hive"
    assert callable(kwargs["aggregate"])  # the clipping aggregate
    assert client.plan.open_calls == []  # run NOT opened yet


def test_lazy_open_from_cohort_config(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        client,
        clipping_norm=1.0,
        params=PARAMS,
        data=Iris,
        round_input_timeout=120,
        round_timeout=60.0,
    )
    loader = _loader(population, rounds=4, batch_size=2, bands=3)
    cohorts = iter(loader)

    grads, clip_state = grad_fn(PARAMS, next(cohorts), state=clip_state)
    assert client.plan.open_calls == [
        {
            "rounds": 4,
            "cardinality": 2,
            "separation": 2,  # bands - 1
            "population": "/hive",
            "round_input_timeout": 120,
            "iteration_timeout": 60.0,
        }
    ]
    assert grad_fn.run is client.plan.run

    grad_fn(PARAMS, next(cohorts), state=clip_state)
    assert len(client.plan.open_calls) == 1  # opened exactly once


def test_round_input_and_clipped_output(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    cohort = next(iter(_loader(population, batch_size=2)))
    live_params = {"w": torch.ones(1, 1), "b": torch.ones(1)}
    grads, clip_state2 = grad_fn(live_params, cohort, state=clip_state)

    (sent,) = client.plan.run.next_calls
    assert isinstance(sent, RoundInput)
    assert sent.params is live_params

    assert isinstance(grads, ClippedPytree)
    assert grads.max_norm == pytest.approx(1.0 / 2)  # C / k
    assert torch.equal(grads.pytree["w"], torch.full((1, 1), 3.0))  # 6 / k
    assert torch.equal(grads.pytree["b"], torch.full((1,), 1.5))  # 3 / k
    assert clip_state2 is clip_state  # state threads through


def test_normalize_by_override(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris, normalize_by=4.0
    )
    cohort = next(iter(_loader(population, batch_size=2)))
    grads, _ = grad_fn(PARAMS, cohort, state=clip_state)
    assert grads.max_norm == pytest.approx(0.25)
    assert torch.equal(grads.pytree["w"], torch.full((1, 1), 1.5))


def test_rejects_raw_cohort(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    with pytest.raises(ValueError, match="DataLoader"):
        grad_fn(PARAMS, ifed.Cohort(round=0, size=2), state=clip_state)


def test_rejects_cohort_from_other_loader(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)
    with pytest.raises(ValueError, match="different DataLoader"):
        grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)


def test_rejects_out_of_order_round(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    cohorts = iter(_loader(population, rounds=4))
    grad_fn(PARAMS, next(cohorts), state=clip_state)
    next(cohorts)  # skip round 1
    with pytest.raises(ValueError, match="out-of-order"):
        grad_fn(PARAMS, next(cohorts), state=clip_state)


def test_rejects_population_mismatch(population):
    client = FakeClient()
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        client,
        clipping_norm=1.0,
        params=PARAMS,
        data=Iris,
        population="/prod",
    )
    with pytest.raises(ValueError, match="compiled for"):
        grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)


def test_rejects_wrong_contribution_count(population):
    client = FakeClient(count_fn=lambda: 1)  # one short
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        grad_fn(PARAMS, next(iter(_loader(population, batch_size=2))), state=clip_state)


def test_requires_dataset(population):
    with pytest.raises(TypeError, match="pass data="):
        clipped_grad(loss_fn, FakeClient(), clipping_norm=1.0, params=PARAMS)
