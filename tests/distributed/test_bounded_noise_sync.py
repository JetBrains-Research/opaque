"""Tests for bounded Gaussian noise distributed sync helpers.

Verifies that sync_rectified_noise_state() and sync_bounded_noise_state()
behave as no-ops when torch.distributed is not initialized and are properly
exported from opaque.noise.distributed.
"""

import pytest
import torch

from opaque.noise import (
    bounded_gaussian_noise,
    gaussian_noise,
    rectified_gaussian_noise,
)
from opaque.noise.distributed import (
    sync_bounded_noise_state,
    sync_gaussian_noise_state,
    sync_mf_noise_state,
    sync_rectified_noise_state,
)
from opaque.noise.gaussian_noise import GaussianNoiseState
from opaque.random import key


class TestSyncRectifiedNoiseState:
    """Tests for sync_rectified_noise_state()."""

    def test_noop_when_not_distributed(self):
        """Returns state unchanged when distributed is not initialized."""
        _, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
        result = sync_rectified_noise_state(state)
        assert result is state

    def test_returns_same_state_object(self):
        """Returns the exact same state (identity, not copy)."""
        state = GaussianNoiseState(step_counter=7, rng_key=key(99))
        result = sync_rectified_noise_state(state)
        assert result is state

    def test_works_after_noise_steps(self):
        """Works correctly after advancing the noise state."""
        noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
        grads = torch.zeros(10)
        _, state = noise_fn(grads, state)
        _, state = noise_fn(grads, state)
        assert state.step_counter == 2

        result = sync_rectified_noise_state(state)
        assert result is state
        assert result.step_counter == 2

    def test_accepts_gaussian_noise_state(self):
        """Accepts any GaussianNoiseState (shared type across noise fns)."""
        _, state = gaussian_noise(stddev=1.0, key=key(42))
        result = sync_rectified_noise_state(state)
        assert result is state


class TestSyncBoundedNoiseState:
    """Tests for sync_bounded_noise_state()."""

    def test_noop_when_not_distributed(self):
        """Returns state unchanged when distributed is not initialized."""
        _, state = bounded_gaussian_noise(stddev=1.0, bounds=(-3.0, 3.0), key=key(42))
        result = sync_bounded_noise_state(state)
        assert result is state

    def test_returns_same_state_object(self):
        """Returns the exact same state (identity, not copy)."""
        state = GaussianNoiseState(step_counter=3, rng_key=key(77))
        result = sync_bounded_noise_state(state)
        assert result is state

    def test_works_after_noise_steps(self):
        """Works correctly after advancing the noise state."""
        noise_fn, state = bounded_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=key(42)
        )
        grads = torch.zeros(10)
        _, state = noise_fn(grads, state)
        _, state = noise_fn(grads, state)
        _, state = noise_fn(grads, state)
        assert state.step_counter == 3

        result = sync_bounded_noise_state(state)
        assert result is state
        assert result.step_counter == 3

    def test_accepts_gaussian_noise_state(self):
        """Accepts any GaussianNoiseState (shared type across noise fns)."""
        _, state = gaussian_noise(stddev=1.0, key=key(42))
        result = sync_bounded_noise_state(state)
        assert result is state


class TestSyncHelpersExports:
    """Verify all sync helpers are exported from opaque.noise.distributed."""

    @pytest.mark.parametrize(
        "name",
        [
            "sync_gaussian_noise_state",
            "sync_rectified_noise_state",
            "sync_bounded_noise_state",
            "sync_mf_noise_state",
        ],
    )
    def test_exported(self, name):
        """Each sync helper is importable."""
        from opaque.noise import distributed as noise_dist

        assert hasattr(noise_dist, name)
        assert callable(getattr(noise_dist, name))

    def test_all_list_complete(self):
        """__all__ lists every sync helper."""
        from opaque.noise import distributed as noise_dist

        expected = {
            "sync_gaussian_noise_state",
            "sync_rectified_noise_state",
            "sync_bounded_noise_state",
            "sync_mf_noise_state",
        }
        assert expected == set(noise_dist.__all__)


class TestBoundedNoiseKeyDeterminism:
    """Cross-verify that bounded noise functions produce identical output
    when initialized with the same key — the property that makes the
    sync helpers sufficient for DDP correctness.
    """

    def test_rectified_same_key_same_noise(self):
        """Two rectified noise fns with same key produce identical output."""
        noise_fn1, st1 = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
        noise_fn2, st2 = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
        grads = torch.randn(50)

        noisy1, st1 = noise_fn1(grads, st1)
        noisy2, st2 = noise_fn2(grads, st2)
        assert torch.allclose(noisy1, noisy2)
        assert st1.step_counter == st2.step_counter
        assert st1.rng_key.seed == st2.rng_key.seed

    def test_bounded_same_key_same_noise(self):
        """Two bounded noise fns with same key produce identical output."""
        noise_fn1, st1 = bounded_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(42)
        )
        noise_fn2, st2 = bounded_gaussian_noise(
            stddev=1.0, bounds=(-3.0, 3.0), key=key(42)
        )
        grads = torch.randn(50)

        noisy1, st1 = noise_fn1(grads, st1)
        noisy2, st2 = noise_fn2(grads, st2)
        assert torch.allclose(noisy1, noisy2)

    def test_rectified_different_key_different_noise(self):
        """Different keys produce different noise (fold_in per rank)."""
        from opaque.random import fold_in

        noise_fn1, st1 = rectified_gaussian_noise(
            stddev=1.0, radius=5.0, key=fold_in(key(42), 0)
        )
        noise_fn2, st2 = rectified_gaussian_noise(
            stddev=1.0, radius=5.0, key=fold_in(key(42), 1)
        )
        grads = torch.zeros(50)

        noisy1, _ = noise_fn1(grads, st1)
        noisy2, _ = noise_fn2(grads, st2)
        assert not torch.allclose(noisy1, noisy2)

    def test_bounded_different_key_different_noise(self):
        """Different keys produce different noise (fold_in per rank)."""
        from opaque.random import fold_in

        noise_fn1, st1 = bounded_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=fold_in(key(42), 0)
        )
        noise_fn2, st2 = bounded_gaussian_noise(
            stddev=1.0, bounds=(-5.0, 5.0), key=fold_in(key(42), 1)
        )
        grads = torch.zeros(50)

        noisy1, _ = noise_fn1(grads, st1)
        noisy2, _ = noise_fn2(grads, st2)
        assert not torch.allclose(noisy1, noisy2)
