"""DP-FTRL mechanisms annotate their step in the Torch provider's trace.

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

from opaque.dpftrl.noise import blt_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.types import SecondMomentClippingOutput, clipped


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


def _grads() -> dict[str, torch.Tensor]:
    return {"w": torch.zeros(4)}


def test_mf_gaussian_noise_annotates_its_step(trace_labels: list[str]) -> None:
    noise_fn, state = mf_gaussian_noise(
        _grads(),
        blt_strategy(momentum=0.9),
        n_steps=8,
        min_sep=8,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(42),
    )

    noise_fn(clipped(_grads(), max_norm=1.0), state)

    assert trace_labels == ["opaque::mf_gaussian_noise"]


def test_second_moment_mf_noise_annotates_its_step(trace_labels: list[str]) -> None:
    noise_fn, state = mf_gaussian_noise(
        _grads(),
        blt_strategy(momentum=0.9),
        n_steps=8,
        min_sep=8,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(42),
        second_moment_strategy=blt_strategy(momentum=0.99),
    )
    paired = SecondMomentClippingOutput(
        grads=clipped(_grads(), max_norm=1.0),
        squared_grads=clipped(_grads(), max_norm=1.0),
    )

    noise_fn(paired, state)

    assert trace_labels == ["opaque::mf_gaussian_noise"]
