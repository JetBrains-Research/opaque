"""Round-trip tests for ``GaussianNoiseState.state_dict``."""

from __future__ import annotations

import json

import torch

from opaque.dpsgd.noise.gaussian import GaussianNoiseState, gaussian_noise
from opaque.random import RngKey, key


class TestGaussianNoiseStateStateDict:
    def test_initial_state_roundtrip(self):
        _, state = gaussian_noise(stddev=1.0, key=key(123))
        encoded = json.dumps(state.state_dict())
        restored = GaussianNoiseState.from_state_dict(json.loads(encoded))
        assert restored == state

    def test_advanced_state_roundtrip(self):
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(7))
        grads = {"w": torch.zeros(4)}
        for _ in range(3):
            _, state = noise_fn(grads, state)
        assert state._step_counter == 3

        encoded = json.dumps(state.state_dict())
        restored = GaussianNoiseState.from_state_dict(json.loads(encoded))
        assert restored == state

    def test_resume_produces_identical_noise(self):
        """After load_state_dict, subsequent noise samples match the original run."""
        noise_fn, state = gaussian_noise(stddev=1.0, key=key(99))
        grads = {"w": torch.zeros(8)}

        for _ in range(2):
            noisy_a, state = noise_fn(grads, state)

        # Snapshot mid-run.
        snapshot = state.state_dict()

        # Continue original run.
        noisy_a_continued, state = noise_fn(grads, state)

        # Restore from snapshot and produce next sample.
        state_restored = GaussianNoiseState.from_state_dict(snapshot)
        noisy_b_continued, _ = noise_fn(grads, state_restored)

        assert torch.equal(noisy_a_continued["w"], noisy_b_continued["w"])

    def test_custom_rng_key_impl(self):
        state = GaussianNoiseState(
            _step_counter=10,
            _rng_key=RngKey(seed=1234567890, impl="custom_impl"),
        )
        restored = GaussianNoiseState.from_state_dict(state.state_dict())
        assert restored == state
