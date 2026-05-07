"""Paired-stream (second-moment) support for ``truncated_gaussian_noise``.

Mirrors ``gaussian_noise`` second-moment handling: when a
:class:`SecondMomentClippingOutput` flows in, allocates the joint noise
budget across the two streams via :func:`second_moment_stddevs` and adds
truncated Gaussian noise to each.
"""

from __future__ import annotations

import pytest
import torch

from opaque.types import (
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)
from opaque.dpsgd.noise import truncated_gaussian_noise
from opaque.dpsgd.noise._second_moment import (
    DEFAULT_SECOND_MOMENT_OVERHEAD,
    second_moment_stddevs,
)
from opaque.random import key


def _paired_input(
    grads: torch.Tensor,
    squared: torch.Tensor,
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

    def test_returns_second_moment_noise_output(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        assert isinstance(out, SecondMomentNoiseOutput)
        assert isinstance(out.noisy_grads, NoisedPytree)
        assert isinstance(out.noisy_squared_grads, NoisedPytree)

    def test_streams_carry_distinct_stddevs(self):
        """First-stream stddev (with √(3/2) overhead) ≠ second-stream stddev."""
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        first_expected, second_expected = second_moment_stddevs(
            1.0,
            first_max_norm=1.0,
            squared_max_norm=1.0,
            first_moment_overhead=DEFAULT_SECOND_MOMENT_OVERHEAD,
        )
        assert out.noisy_grads.noise_stddev == pytest.approx(first_expected)
        assert out.noisy_squared_grads.noise_stddev == pytest.approx(second_expected)
        # With equal per-record bounds Δ₁ = Δ₂ = 1, the second-stream stddev
        # is σ_first · 1 / sqrt(ρ² − 1) for ρ = sqrt(3/2), which exceeds σ_first.
        assert second_expected > first_expected


class TestPairedStreamNoiseBounded:
    """Truncated noise must stay within ±radius·stddev for both streams."""

    def test_first_stream_within_radius(self):
        radius = 4.0
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=radius,
            key=key(123),
        )
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(2000),
                torch.zeros(2000),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        first_std = out.noisy_grads.noise_stddev
        assert torch.all(out.noisy_grads.pytree.abs() <= radius * first_std + 1e-6)

    def test_second_stream_within_radius(self):
        radius = 4.0
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=radius,
            key=key(123),
        )
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(2000),
                torch.zeros(2000),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        second_std = out.noisy_squared_grads.noise_stddev
        assert torch.all(
            out.noisy_squared_grads.pytree.abs() <= radius * second_std + 1e-6
        )


class TestPairedStreamIndependence:
    """Per-record stream-2 stddev grows with squared_max_norm."""

    def test_squared_max_norm_changes_second_stream_only(self):
        """Doubling `squared_max_norm` doubles the second-stream stddev."""
        noise_fn_a, state_a = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        out_a, _ = noise_fn_a(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        noise_fn_b, state_b = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        out_b, _ = noise_fn_b(
            _paired_input(
                torch.zeros(10),
                torch.zeros(10),
                max_norm=1.0,
                squared_max_norm=2.0,
            ),
            state_b,
        )
        # First stream depends only on first max_norm — unchanged.
        assert out_a.noisy_grads.noise_stddev == pytest.approx(
            out_b.noisy_grads.noise_stddev
        )
        # Second stream stddev scales linearly with squared_max_norm.
        assert out_b.noisy_squared_grads.noise_stddev == pytest.approx(
            2.0 * out_a.noisy_squared_grads.noise_stddev
        )


class TestPairedStreamPerGroup:
    """Per-group max_norm on paired streams uses the joint MSE-optimal allocation."""

    def test_per_group_on_both_streams_returns_per_group_stddevs(self):
        from opaque.types import PerGroup

        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        first_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        squared_norm = first_norm * first_norm  # per-group squared bounds
        paired = SecondMomentClippingOutput(
            grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)}, max_norm=first_norm
            ),
            squared_grads=clipped(
                {"a": torch.zeros(4), "b": torch.zeros(4)},
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

    def test_mismatched_kinds_rejected(self):
        from opaque.types import PerGroup

        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        per_group_norm = PerGroup(
            groups={"weight": "g"},
            values={"g": 1.0},
        )
        paired = SecondMomentClippingOutput(
            grads=clipped({"weight": torch.zeros(4)}, max_norm=per_group_norm),
            squared_grads=clipped({"weight": torch.zeros(4)}, max_norm=1.0),
        )
        with pytest.raises(TypeError, match="matching max_norm kinds"):
            noise_fn(paired, state)


class TestPairedStreamReproducibility:
    """Same key → same noise; different folds → different noise streams."""

    def test_seeded_runs_match(self):
        noise_fn_a, state_a = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(42),
        )
        noise_fn_b, state_b = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(42),
        )
        out_a, _ = noise_fn_a(
            _paired_input(
                torch.zeros(20),
                torch.zeros(20),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_a,
        )
        out_b, _ = noise_fn_b(
            _paired_input(
                torch.zeros(20),
                torch.zeros(20),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state_b,
        )
        assert torch.equal(out_a.noisy_grads.pytree, out_b.noisy_grads.pytree)
        assert torch.equal(
            out_a.noisy_squared_grads.pytree, out_b.noisy_squared_grads.pytree
        )

    def test_first_and_second_streams_have_different_noise(self):
        """fold_in(key, 1) vs fold_in(key, 2) namespacing gives independent streams."""
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(42),
        )
        # Identical inputs and stddevs would only produce identical outputs
        # if the two streams used the same RNG.  Pick equal max_norms so any
        # stddev mismatch is small and the test still distinguishes streams.
        out, _ = noise_fn(
            _paired_input(
                torch.zeros(50),
                torch.zeros(50),
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        # Streams cover the same domain but with different noise.
        assert not torch.equal(out.noisy_grads.pytree, out.noisy_squared_grads.pytree)


class TestPairedStreamStateAdvances:
    """Step counter advances exactly once per paired call."""

    def test_state_advances(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0,
            radius=5.0,
            key=key(0),
        )
        for expected in (1, 2, 3):
            _, state = noise_fn(
                _paired_input(
                    torch.zeros(4),
                    torch.zeros(4),
                    max_norm=1.0,
                    squared_max_norm=1.0,
                ),
                state,
            )
            assert state._step_counter == expected


class TestZeroNoiseMultiplier:
    """``noise_multiplier=0`` collapses both streams to a zero-width support.

    Both the noise stddev *and* the truncation bound (= radius·stddev) go to
    zero, so every leaf is clamped to ±0.  The original input values are not
    preserved — this is the documented "no-noise" path mirroring how
    ``gaussian_noise`` returns the centred deterministic output.
    """

    def test_zero_noise_paired(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=0.0,
            radius=5.0,
            key=key(0),
        )
        out, _ = noise_fn(
            _paired_input(
                torch.ones(8),
                torch.ones(8) * 4,
                max_norm=1.0,
                squared_max_norm=1.0,
            ),
            state,
        )
        # σ=0 → the truncated-noise sample is 0, but the truncation bound
        # also collapses to 0 so values are clamped to ±0.
        assert torch.all(out.noisy_grads.pytree == 0.0)
        assert torch.all(out.noisy_squared_grads.pytree == 0.0)
