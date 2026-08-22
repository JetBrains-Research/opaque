"""DP-SGD mechanisms annotate their step in the Torch provider's trace.

Annotating is not a capability a mechanism negotiates: it hands the label
to the engine, which drops it under a provider that cannot trace. These
tests pin the labels a profiled run sees. That an untraceable provider
still runs the step belongs to the engine's dispatch contract, and is
tested there.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import clipped


@pytest.fixture
def trace_labels(monkeypatch) -> list[str]:
    """Record the labels the Torch provider opens a record_function for."""
    labels: list[str] = []

    @contextmanager
    def record_function(label: str):
        labels.append(label)
        yield

    monkeypatch.setattr(torch.autograd.profiler, "record_function", record_function)
    return labels


def test_gaussian_noise_annotates_its_step(trace_labels: list[str]) -> None:
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(17))

    noise_fn(clipped({"w": torch.zeros(4)}, max_norm=1.0), state)

    assert trace_labels == ["opaque::gaussian_noise"]


def test_adaptive_clipped_grad_annotates_its_step(trace_labels: list[str]) -> None:
    def loss_fn(params, batch):
        return ((batch @ params) ** 2).mean()

    grad_fn, state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=1.0,
        key=key(17),
        batch_argnums=1,
    )

    grad_fn(torch.ones(2), torch.ones((3, 2)), state=state)

    assert trace_labels[0] == "opaque::adaptive_clipped_grad"
