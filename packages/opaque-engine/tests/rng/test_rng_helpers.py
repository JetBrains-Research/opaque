"""Engine-only tests for RNG convenience helpers.

DP-SGD-aware integration tests (helpers used with ``gaussian_noise``)
live in ``packages/opaque-dpsgd/tests/rng/test_rng_helpers_integration.py``;
opaque-engine has no dependency on opaque-dpsgd.
"""

import torch

from opaque.random import random_key
from opaque.random.types import RngKey


class TestRandomKey:
    """Test non-deterministic key generation for prototyping."""

    def test_returns_rng_key(self):
        """Should return an RngKey instance."""
        k = random_key()
        assert isinstance(k, RngKey)

    def test_different_on_each_call(self):
        """Should generate different keys on consecutive calls."""
        k1 = random_key()
        k2 = random_key()
        assert k1.seed != k2.seed

    def test_produces_working_generators(self):
        """Generated keys should work with generator_from_key."""
        from opaque.random import generator_from_key

        k = random_key()
        gen = generator_from_key(k)
        assert isinstance(gen, torch.Generator)

        # Should produce different random values
        t1 = torch.randn(10, generator=gen)
        k2 = random_key()
        gen2 = generator_from_key(k2)
        t2 = torch.randn(10, generator=gen2)
        assert not torch.allclose(t1, t2)
