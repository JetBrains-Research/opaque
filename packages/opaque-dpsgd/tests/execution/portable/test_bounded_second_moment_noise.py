"""Provider-neutral bounded second-moment Gaussian-noise behavior.

Paired-stream (second-moment) support for ``gaussian_noise(bound=...)``:
when a :class:`SecondMomentClippingOutput` flows in, allocates the joint
noise budget across the two streams via ``paired_noise_stddevs`` and samples
bounded Gaussian noise on each.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from opaque.api.engine.noise_allocation import paired_noise_stddevs
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import (
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)


def _array(backend_case, value):
    return backend_case.array(value, dtype=backend_case.dtype("float32"))


def _host(backend_case, value):
    return np.asarray(backend_case.to_host(value))


def _paired_input(
    backend_case,
    grads,
    squared,
    *,
    max_norm: float,
    squared_max_norm: float,
) -> SecondMomentClippingOutput:
    return SecondMomentClippingOutput(
        grads=clipped(grads, max_norm=max_norm),
        squared_grads=clipped(squared, max_norm=squared_max_norm),
    )


class TestPairedStreamShape:
    """``noise_fn(SecondMomentClippingOutput, state)`` returns paired noised output."""

    def test_returns_second_moment_noise_output(self, backend_case):
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(10)),
                _array(backend_case, np.zeros(10)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads, NoisedPytree)
        assert isinstance(out.noisy_squared_grads, NoisedPytree)

    def test_streams_match_paired_noise_stddevs(self, backend_case):
        """Output stddevs match ``paired_noise_stddevs`` on the same inputs."""
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(10)),
                _array(backend_case, np.zeros(10)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        first_expected, second_expected = paired_noise_stddevs(
            1.0, first=1.0, second=1.0
        )
        assert out.noisy_grads.noise_stddev == pytest.approx(first_expected)
        assert out.noisy_squared_grads.noise_stddev == pytest.approx(second_expected)
        # With Δ₁ = Δ₂ = 1, S = 2, both streams get σ = sqrt(2).
        assert first_expected == pytest.approx(math.sqrt(2.0))
        assert second_expected == pytest.approx(math.sqrt(2.0))


class TestPairedStreamWithinBound:
    """Bounded noise must stay within the absolute bound for both streams."""

    def test_first_stream_within_bound(self, backend_case):
        bound = 4.0
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=bound,
            key=key(123),
        )
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(2000)),
                _array(backend_case, np.zeros(2000)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        observed = _host(backend_case, out.noisy_grads.pytree)
        assert np.all(np.abs(observed) <= bound + 1e-6)

    def test_second_stream_within_bound(self, backend_case):
        bound = 4.0
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=bound,
            key=key(123),
        )
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(2000)),
                _array(backend_case, np.zeros(2000)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        observed = _host(backend_case, out.noisy_squared_grads.pytree)
        assert np.all(np.abs(observed) <= bound + 1e-6)


class TestPairedStreamCoupling:
    """Squared-stream sensitivity enters S, so it shifts both σ's."""

    def test_squared_max_norm_shifts_both_streams(self, backend_case):
        """The Mahalanobis budget couples both streams: changing Δ² shifts σ¹ and σ²."""
        noise_fn_a, state_a = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        out_a, _ = noise_fn_a(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(10)),
                _array(backend_case, np.zeros(10)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        noise_fn_b, state_b = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        out_b, _ = noise_fn_b(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(10)),
                _array(backend_case, np.zeros(10)),
                max_norm=1.0,
                squared_max_norm=2.0,
            ),
            state_b,
        )
        # Closed form: σ¹ = sqrt(Δ¹ · S), with S = Δ¹ + Δ².
        # Doubling Δ² grows S from 2 → 3, so σ¹ grows by sqrt(3/2) and σ² by
        # sqrt(2 · 3 / (1 · 2)) = sqrt(3).
        assert out_b.noisy_grads.noise_stddev == pytest.approx(
            out_a.noisy_grads.noise_stddev * math.sqrt(3 / 2)
        )
        assert out_b.noisy_squared_grads.noise_stddev == pytest.approx(
            out_a.noisy_squared_grads.noise_stddev * math.sqrt(3.0)
        )


class TestPairedStreamPerGroup:
    """Per-group max_norm on paired streams uses the joint MSE-optimal allocation."""

    def test_per_group_on_both_streams_returns_per_group_stddevs(self, backend_case):
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        first_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        squared_norm = first_norm * first_norm  # per-group squared bounds
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {
                    "a": _array(backend_case, np.zeros(4)),
                    "b": _array(backend_case, np.zeros(4)),
                },
                max_norm=first_norm,
            ),
            squared_grads=clipped(
                {
                    "a": _array(backend_case, np.zeros(4)),
                    "b": _array(backend_case, np.zeros(4)),
                },
                max_norm=squared_norm,
            ),
        )
        out, _ = noise_fn(paired, state)
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        # Per-group bound, per-group stddev keys match the input groups.
        assert out.noisy_grads.noise_stddev.groups == first_norm.groups
        assert out.noisy_squared_grads.noise_stddev.groups == squared_norm.groups

    def test_mismatched_kinds_rejected(self, backend_case):
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        per_group_norm = PerGroup(
            groups={"weight": "g"},
            values={"g": 1.0},
        )
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {"weight": _array(backend_case, np.zeros(4))}, max_norm=per_group_norm
            ),
            squared_grads=clipped(
                {"weight": _array(backend_case, np.zeros(4))}, max_norm=1.0
            ),
        )
        with pytest.raises(TypeError, match="same kind"):
            noise_fn(paired, state)


class TestPairedStreamReproducibility:
    """Same key → same noise; different folds → different noise streams."""

    def test_seeded_runs_match(self, backend_case):
        noise_fn_a, state_a = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(42),
        )
        noise_fn_b, state_b = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(42),
        )
        out_a, _ = noise_fn_a(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(20)),
                _array(backend_case, np.zeros(20)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        out_b, _ = noise_fn_b(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(20)),
                _array(backend_case, np.zeros(20)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_b,
        )
        backend_case.assert_allclose(
            out_a.noisy_grads.pytree, backend_case.to_host(out_b.noisy_grads.pytree)
        )
        backend_case.assert_allclose(
            out_a.noisy_squared_grads.pytree,
            backend_case.to_host(out_b.noisy_squared_grads.pytree),
        )

    def test_first_and_second_streams_have_different_noise(self, backend_case):
        """fold_in(key, 1) vs fold_in(key, 2) namespacing gives independent streams."""
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(42),
        )
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                _array(backend_case, np.zeros(50)),
                _array(backend_case, np.zeros(50)),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        # Streams cover the same domain but with different noise.
        first_host = _host(backend_case, out.noisy_grads.pytree)
        second_host = _host(backend_case, out.noisy_squared_grads.pytree)
        assert not np.array_equal(first_host, second_host)


class TestPairedStreamStateAdvances:
    """Step counter advances exactly once per paired call."""

    def test_state_advances(self, backend_case):
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0,
            bound=5.0,
            key=key(0),
        )
        for expected in (1, 2, 3):
            _, state = noise_fn(
                _paired_input(
                    backend_case,
                    _array(backend_case, np.zeros(4)),
                    _array(backend_case, np.zeros(4)),
                    max_norm=1.0,
                    squared_max_norm=1.0,
                ),
                state,
            )
            assert state._step_counter == expected


class TestZeroNoiseMultiplier:
    """``noise_multiplier=0`` returns the per-stream input clamped to ``bound``.

    Under the absolute-bound interpretation σ=0 means "no noise", and the
    output is ``clamp(input, -bound, bound)`` on each stream — *not* zero.
    """

    def test_zero_noise_paired(self, backend_case):
        bound = 3.0
        noise_fn, state = gaussian_noise(
            noise_multiplier=0.0,
            bound=bound,
            key=key(0),
        )
        grads = _array(backend_case, [5.0, -5.0, 0.5, -0.5])
        squared = _array(backend_case, [10.0, 0.0, 1.0, 2.0])
        out, _ = noise_fn(
            _paired_input(
                backend_case,
                grads,
                squared,
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        expected_grads = np.array([3.0, -3.0, 0.5, -0.5])
        expected_squared = np.array([3.0, 0.0, 1.0, 2.0])
        backend_case.assert_allclose(out.noisy_grads.pytree, expected_grads)
        backend_case.assert_allclose(out.noisy_squared_grads.pytree, expected_squared)
