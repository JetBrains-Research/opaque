"""Torch-native λ-CGD invariants that depend on global torch RNG state."""

from __future__ import annotations

import torch

from opaque.dpftrl.noise import lambda_cgd_strategy, mf_gaussian_noise
from opaque.random import key
from opaque.types import clipped


def _make_noise(seed: int = 42):
    return mf_gaussian_noise(
        {"w": torch.zeros(10)},
        lambda_cgd_strategy(lambda_=0.9, normalized=True),
        n_steps=100,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def test_keyed_noise_ignores_global_torch_rng_draws() -> None:
    noise_fn, state = _make_noise()
    expected, _ = noise_fn(clipped({"w": torch.zeros(10)}, max_norm=1.0), state)

    torch.manual_seed(999)
    torch.randn(1000)

    noise_fn, state = _make_noise()
    actual, _ = noise_fn(clipped({"w": torch.zeros(10)}, max_norm=1.0), state)
    torch.testing.assert_close(actual.pytree["w"], expected.pytree["w"])
