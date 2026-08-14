"""Tests for opaque.optimizers._sgd."""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import sgd
from opaque.types import (
    SecondMomentNoiseOutput,
    clipped,
    noised,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestSGD:
    def test_raw_pytree_matches_torchopt(self, params, grads):
        step, state = sgd(params, lr=1e-2, momentum=0.9, weight_decay=0.01)
        ref = torchopt.sgd(lr=1e-2, momentum=0.9, weight_decay=0.01)
        ref_state = ref.init(params)

        updates, state = step(grads, state, params=params)
        ref_updates, ref_state = ref.update(grads, ref_state, params=params)

        for name in updates:
            torch.testing.assert_close(updates[name], ref_updates[name])

    def test_noisy_pytree_unwraps_without_warning(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        updates, _ = step(
            noised(grads, max_norm=1.0, noise_stddev=0.25),
            state,
            params=params,
        )
        for name in updates:
            assert updates[name].shape == params[name].shape

    def test_clipped_updates_are_rejected(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            step(clipped(grads, max_norm=1.0), state, params=params)

    def test_explicit_metadata_kwargs_are_rejected(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        with pytest.raises(TypeError, match="noise_stddev"):
            step(grads, state, params=params, noise_stddev=0.5)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(grads, state, params=params, noisy_squared_grads={})

    def test_second_moment_output_uses_first_stream_silently(self, params, grads):
        """SGD has no second-moment path, so a SecondMomentNoiseOutput
        falls back to the noisy_grads stream silently."""
        step, state = sgd(params, lr=1e-2)
        sq = {name: value.square() for name, value in grads.items()}
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for name in updates:
            assert updates[name].shape == params[name].shape
