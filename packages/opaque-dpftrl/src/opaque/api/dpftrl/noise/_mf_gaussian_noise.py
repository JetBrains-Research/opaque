"""Public MF noise factory — ``mf_gaussian_noise``.

The strategy + amplification-context tuple selects a streaming matrix
via the polymorphic :meth:`MfStrategy.streaming_matrix` query.  This
file is a thin shell over that polymorphism, the engine's input
validation, and the second-moment dispatch — all heavy lifting lives in
:mod:`_engine`, :mod:`_second_moment`, :mod:`_distributed`, and the
per-strategy files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.random.types import RngKey
from opaque.types import NoisedPytree, PerGroup, SecondMomentClippingOutput

from ._distributed import mf_per_group_sync_fingerprint_for_latch
from ._engine import (
    MFNoiseState,
    _expect_clipped,
    _matrix_factorization_noise,
    _resolve_noise_multiplier,
    _validate_constant_max_norm,
)
from ._lambda_cgd import LambdaCgdStrategy, _make_lambda_cgd_noise
from ._second_moment import SecondMomentMFNoiseState, make_second_moment_mf_noise

if TYPE_CHECKING:
    from .types import MfStrategy


def mf_gaussian_noise(
    grad_template: Any,
    strategy: "MfStrategy",
    *,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    noise_multiplier: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
    second_moment_strategy: "MfStrategy | None" = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState | SecondMomentMFNoiseState]],
    MFNoiseState | SecondMomentMFNoiseState,
]:
    """Create a correlated noise mechanism for the given MF strategy.

    The polymorphic :meth:`MfStrategy.streaming_matrix` query selects the
    appropriate streaming primitive at the supplied amplification
    context; :class:`LambdaCgdStrategy` uses a PRNG-replay path (no
    streaming matrix is materialized).

    The returned ``noise_fn`` dispatches on its input type:

    - ``ClippedPytree`` → ``NoisedPytree`` (single-stream noise).
    - ``SecondMomentClippingOutput`` → ``SecondMomentNoiseOutput``
      (paired-stream noise; only when ``second_moment_strategy`` was
      supplied at construction).

    The paired-stream release uses the sensitivity-proportional joint
    Mahalanobis allocation
    (:func:`~opaque.api.engine.noise_allocation.paired_noise_stddevs`)
    with the MF translation ``nm / ‖C₁‖`` as the joint effective
    multiplier so the joint PLD matches the single-stream MF Gaussian
    accountant at ``(noise_multiplier, ‖C₁‖)``.  ``PerGroup``
    ``max_norm`` is supported on both streams.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        strategy: MF strategy recipe from one of the factory functions.
        n_steps: Total training steps (the amplifier's horizon).
        min_sep: Minimum separation between participations (default 1).
        max_participations: Max participations per example
            (``None`` ⇒ ``n_steps``).
        noise_multiplier: Gaussian noise multiplier.  The clipped-gradient
            ``max_norm`` is read from each ``ClippedPytree`` input.
        key: Explicit RNG key for deterministic randomness.
        dtype: Optional dtype for intermediate noise computation.
        second_moment_strategy: Optional explicit strategy recipe for
            the squared-gradient stream.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    if second_moment_strategy is not None:
        return make_second_moment_mf_noise(
            grad_template,
            strategy,
            second_moment_strategy,
            n_steps=n_steps,
            min_sep=min_sep,
            max_participations=max_participations,
            noise_multiplier=resolved_noise_multiplier,
            key=key,
            dtype=dtype,
        )

    raw_noise_fn, raw_state = _make_raw_mf_noise(
        grad_template,
        strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=key,
        dtype=dtype,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
    ) -> tuple[NoisedPytree, MFNoiseState]:
        if isinstance(clipped_grads, SecondMomentClippingOutput):
            raise TypeError(
                "mf_gaussian_noise was constructed without `second_moment_strategy` "
                "and cannot consume SecondMomentClippingOutput inputs.  Either "
                "pass a single-stream ClippedPytree, or rebuild the noise "
                "function with `second_moment_strategy=...`."
            )
        clipped_grads = _expect_clipped(clipped_grads, op="mf_gaussian_noise")
        max_norm = _validate_constant_max_norm(
            clipped_grads, st._first_max_norm, op="mf_gaussian_noise"
        )
        if isinstance(max_norm, PerGroup):
            base_stddev = per_group_noise_stddev(max_norm, resolved_noise_multiplier)
        else:
            base_stddev = resolved_noise_multiplier * max_norm
        noisy_tree, new_state = raw_noise_fn(
            clipped_grads.pytree,
            st,
            stddev=base_stddev,
        )
        sync_fp = mf_per_group_sync_fingerprint_for_latch(st, max_norm)
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=clipped_grads.max_norm,
                noise_stddev=base_stddev,
            ),
            replace(
                new_state,
                _first_max_norm=max_norm,
                _first_max_norm_sync_fingerprint=sync_fp,
            ),
        )

    return noise_fn, raw_state


def _make_raw_mf_noise(
    grad_template: Any,
    strategy: "MfStrategy",
    *,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    key: RngKey,
    dtype: torch.dtype | None,
) -> tuple[Callable[..., tuple[Any, MFNoiseState]], MFNoiseState]:
    """Build the underlying noise function from the strategy's streaming matrix.

    LambdaCgdStrategy uses PRNG replay rather than a streaming matrix —
    it is the one isinstance check left because the noise primitive
    differs.  Every other strategy goes through the polymorphic
    ``streaming_matrix(...)`` surface.
    """
    if isinstance(strategy, LambdaCgdStrategy):
        return _make_lambda_cgd_noise(
            grad_template,
            strategy,
            n_steps=n_steps,
            key=key,
            dtype=dtype,
        )
    streaming = strategy.streaming_matrix(
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )
    return _matrix_factorization_noise(
        grad_template,
        streaming,
        key=key,
        dtype=dtype,
    )


__all__ = ["mf_gaussian_noise"]
