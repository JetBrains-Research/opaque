"""clipped_grad: eager compile, lazy task registration, round contracts."""

from typing import Dict

import pytest
import torch

ifed = pytest.importorskip("ifed")
from ifed._defaults import RoundResult  # noqa: E402

from opaque.api.engine.clipping.types import FixedClipState  # noqa: E402
from opaque.api.engine.types import ClippedPytree  # noqa: E402
from opaque.federated import Cohort, DataLoader, MinSepSampler, Population, clipped_grad  # noqa: E402


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
        self.iterate_calls = []
        self._count_fn = count_fn

    def iterate(self, state):
        self.iterate_calls.append(state)
        result = RoundResult(
            grads={"w": torch.full((1, 1), 6.0), "b": torch.full((1,), 3.0)},
            count=self._count_fn(),
        )
        return type("FakeRoundHandle", (), {"result": lambda _, timeout: result})()


class FakePlan:
    def __init__(self, count_fn):
        self.count_fn = count_fn
        self.engine = "PYTORCH"
        self.agent_plan = "agent-plan.zip"
        self.server_plan = "server-plan.zip"
        self.config_json = "config.json"
        self.result_cls = RoundResult
        self.work_dir = "."


class FakeClient:
    """Duck-typed ``ifed.Client`` that records native task registration."""

    def __init__(self, count_fn=lambda: 2):
        self.task_calls = []
        self.run = FakeRun(count_fn)

    def create_task(self, task):
        self.task_calls.append(task)
        return self.run


@pytest.fixture
def population():
    return Population(name="/hive")


def _loader(population, rounds=4, batch_size=2, bands=2):
    sampler = MinSepSampler(population, batch_size=batch_size, bands=bands)
    return DataLoader(population, batch_sampler=sampler, rounds=rounds)


def test_eager_compile_at_factory(population, monkeypatch):
    client = FakeClient()
    calls = []
    monkeypatch.setattr(
        ifed.pytorch,
        "compile",
        lambda model, **kwargs: calls.append((model, kwargs)) or FakePlan(client.run._count_fn),
    )
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    assert isinstance(clip_state, FixedClipState)
    assert len(calls) == 1  # EAGER: before any grad_fn call
    model, kwargs = calls[0]
    assert isinstance(model, ifed.FunctionalModel)
    assert callable(kwargs["aggregate"])  # the clipping aggregate
    assert client.task_calls == []  # task NOT created yet


def test_lazy_task_from_cohort_config(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
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
    (task,) = client.task_calls
    assert task.population == ifed.Population(name="/hive", cardinality=2)
    assert task.iterations == 4
    assert task.round_input_timeout == 120
    assert task.policy.assign_separation == ifed.AssignSeparationPolicy(iteration_delta=2)
    assert grad_fn.run is client.run

    grad_fn(PARAMS, next(cohorts), state=clip_state)
    assert len(client.task_calls) == 1  # created exactly once


def test_model_state_and_clipped_output(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    cohort = next(iter(_loader(population, batch_size=2)))
    live_params = {"w": torch.ones(1, 1), "b": torch.ones(1)}
    grads, clip_state2 = grad_fn(live_params, cohort, state=clip_state)

    (sent,) = client.run.iterate_calls
    assert isinstance(sent, ifed.ModelState)
    assert sent.params is live_params

    assert isinstance(grads, ClippedPytree)
    assert grads.max_norm == pytest.approx(1.0 / 2)  # C / k
    assert torch.equal(grads.pytree["w"], torch.full((1, 1), 3.0))  # 6 / k
    assert torch.equal(grads.pytree["b"], torch.full((1,), 1.5))  # 3 / k
    assert clip_state2 is clip_state  # state threads through


def test_normalize_by_override(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris, normalize_by=4.0
    )
    cohort = next(iter(_loader(population, batch_size=2)))
    grads, _ = grad_fn(PARAMS, cohort, state=clip_state)
    assert grads.max_norm == pytest.approx(0.25)
    assert torch.equal(grads.pytree["w"], torch.full((1, 1), 1.5))


def test_rejects_raw_cohort(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    with pytest.raises(ValueError, match="DataLoader"):
        grad_fn(PARAMS, Cohort(round=0, size=2), state=clip_state)


def test_rejects_cohort_from_other_loader(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)
    with pytest.raises(ValueError, match="different DataLoader"):
        grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)


def test_rejects_out_of_order_round(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    cohorts = iter(_loader(population, rounds=4))
    grad_fn(PARAMS, next(cohorts), state=clip_state)
    next(cohorts)  # skip round 1
    with pytest.raises(ValueError, match="out-of-order"):
        grad_fn(PARAMS, next(cohorts), state=clip_state)


def test_preserves_native_policy(population, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    policy = ifed.ComputationPolicy(requirements=ifed.Requirements(gpu=False))
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        client,
        clipping_norm=1.0,
        params=PARAMS,
        data=Iris,
        policy=policy,
    )
    grad_fn(PARAMS, next(iter(_loader(population))), state=clip_state)
    assert client.task_calls[0].policy.requirements == policy.requirements


def test_rejects_wrong_contribution_count(population, monkeypatch):
    client = FakeClient(count_fn=lambda: 1)  # one short
    monkeypatch.setattr(ifed.pytorch, "compile", lambda *_args, **_kwargs: FakePlan(client.run._count_fn))
    grad_fn, clip_state = clipped_grad(
        loss_fn, client, clipping_norm=1.0, params=PARAMS, data=Iris
    )
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        grad_fn(PARAMS, next(iter(_loader(population, batch_size=2))), state=clip_state)


def test_requires_dataset(population):
    with pytest.raises(TypeError, match="pass data="):
        clipped_grad(loss_fn, FakeClient(), clipping_norm=1.0, params=PARAMS)
