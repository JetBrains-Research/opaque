"""Checkpoint continuation contracts for RNG-bearing MF noise state."""

from __future__ import annotations

import torch

from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped


def _make_band_mf_noise(seed: int):
    template = {"w": torch.zeros(4)}
    noise_fn, state = mf_gaussian_noise(
        template,
        band_mf_strategy(bands=2, momentum=0.8),
        n_steps=4,
        min_sep=1,
        max_participations=4,
        noise_multiplier=1.0,
        key=key(seed),
    )
    return noise_fn, state


def test_band_mf_state_dict_continues_with_saved_streaming_state():
    grads = clipped({"w": torch.zeros(4)}, max_norm=1.0)
    noise_fn, state = _make_band_mf_noise(seed=42)

    _, state = noise_fn(grads, state)
    _, state = noise_fn(grads, state)
    snapshot = state_dict(state)

    _, restore_template = _make_band_mf_noise(seed=99)
    restored = from_state_dict(restore_template, snapshot)

    expected, expected_state = noise_fn(grads, state)
    actual, actual_state = noise_fn(grads, restored)

    torch.testing.assert_close(actual.pytree["w"], expected.pytree["w"])
    assert actual_state._step_counter == expected_state._step_counter
