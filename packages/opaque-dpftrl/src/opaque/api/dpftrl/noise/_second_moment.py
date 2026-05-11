"""Paired-stream MF noise (first moment + private second moment).

The runtime per-stream stddev allocation lives in
``opaque.api.engine.noise_allocation.paired_noise_stddevs``; this module
carries the joint state and the paired-stream factory that splits the
two strategies into independent noise streams sharing a common joint
Mahalanobis budget.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.engine.noise_allocation import paired_noise_stddevs
from opaque.random import fold_in as rng_fold_in
from opaque.random.types import RngKey
from opaque.types import (
    NoiseState,
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

from ._engine import MFNoiseState, _expect_clipped, _validate_constant_max_norm

if TYPE_CHECKING:
    from .types import MfStrategy


@dataclasses.dataclass(frozen=True)
class SecondMomentMFNoiseState(NoiseState):
    """Internal state for MF noise with private second moments."""

    _first_state: MFNoiseState
    _second_state: MFNoiseState

    @property
    def _step_counter(self) -> int:  # type: ignore[override]
        return self._first_state._step_counter

    @property
    def _rng_key(self) -> RngKey:  # type: ignore[override]
        return self._first_state._rng_key


def make_second_moment_mf_noise(
    grad_template: Any,
    first_strategy: "MfStrategy",
    second_strategy: "MfStrategy",
    *,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None,
) -> tuple[
    Callable[
        [Any, SecondMomentMFNoiseState],
        tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState],
    ],
    SecondMomentMFNoiseState,
]:
    """Build the paired-stream noise function.

    Both strategies are built at the same participation context; the
    joint Mahalanobis budget is calibrated as ``(c1/nm)²`` so the joint
    PLD matches the single-stream MF Gaussian accountant at
    ``(nm, c1)`` where ``c1`` is the first strategy's max column norm.
    """
    # Imported here to avoid a top-level cycle: distributed → engine → ...
    from ._distributed import mf_per_group_sync_fingerprint_for_latch
    from ._mf_gaussian_noise import _make_raw_mf_noise

    first_fn, first_state = _make_raw_mf_noise(
        grad_template,
        first_strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=rng_fold_in(key, 0),
        dtype=dtype,
    )
    second_fn, second_state = _make_raw_mf_noise(
        grad_template,
        second_strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=rng_fold_in(key, 1),
        dtype=dtype,
    )

    init_state = SecondMomentMFNoiseState(
        _first_state=first_state,
        _second_state=second_state,
    )

    def noise_fn(
        clipped_input: Any,
        st: SecondMomentMFNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState]:
        if not isinstance(clipped_input, SecondMomentClippingOutput):
            raise TypeError(
                "mf_gaussian_noise was constructed with `second_moment_strategy` "
                "and expects SecondMomentClippingOutput inputs (paired-stream).  "
                "Build the paired form upstream via "
                "`clipped_grad(..., second_moment=True)`, or rebuild the noise "
                "function without `second_moment_strategy` for single-stream mode."
            )
        first_clipped = _expect_clipped(clipped_input.grads, op="mf_gaussian_noise")
        second_clipped = _expect_clipped(
            clipped_input.squared_grads, op="mf_gaussian_noise (squared stream)"
        )
        max_norm = _validate_constant_max_norm(
            first_clipped, st._first_state._first_max_norm, op="mf_gaussian_noise"
        )
        squared_max_norm = _validate_constant_max_norm(
            second_clipped,
            st._second_state._first_max_norm,
            op="mf_gaussian_noise (squared stream)",
        )
        # Max column norm = single-participation sensitivity at this horizon.
        # Reach through the polymorphic sensitivity surface so each strategy
        # uses its own closed-form (Identity = 1, BLT = ‖C‖_{1→2}, etc.).
        c1 = first_strategy.sensitivity(
            n_steps=n_steps, min_sep=n_steps, max_participations=1
        )
        c2 = second_strategy.sensitivity(
            n_steps=n_steps, min_sep=n_steps, max_participations=1
        )
        first_stddev, second_stddev = paired_noise_stddevs(
            noise_multiplier / c1,
            first=max_norm * c1,
            second=squared_max_norm * c2,
        )
        noisy_grads, new_first = first_fn(
            first_clipped.pytree,
            st._first_state,
            stddev=first_stddev,
        )
        noisy_squared, new_second = second_fn(
            second_clipped.pytree,
            st._second_state,
            stddev=second_stddev,
        )
        sync_fp_first = mf_per_group_sync_fingerprint_for_latch(
            st._first_state, max_norm
        )
        sync_fp_second = mf_per_group_sync_fingerprint_for_latch(
            st._second_state, squared_max_norm
        )
        return (
            SecondMomentNoiseOutput(
                NoisedPytree(
                    pytree=noisy_grads,
                    max_norm=first_clipped.max_norm,
                    noise_stddev=first_stddev,
                ),
                NoisedPytree(
                    pytree=noisy_squared,
                    max_norm=second_clipped.max_norm,
                    noise_stddev=second_stddev,
                ),
            ),
            SecondMomentMFNoiseState(
                _first_state=replace(
                    new_first,
                    _first_max_norm=max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_first,
                ),
                _second_state=replace(
                    new_second,
                    _first_max_norm=squared_max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_second,
                ),
            ),
        )

    return noise_fn, init_state


__all__ = ["SecondMomentMFNoiseState", "make_second_moment_mf_noise"]
