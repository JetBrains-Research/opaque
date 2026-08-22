"""Public MF noise factory — ``mf_gaussian_noise``.

The strategy + amplification-context tuple selects an immutable execution
plan via the polymorphic :meth:`MfStrategy.execution_plan` query.  This
file is a thin shell over that plan, the engine's input
validation, and the second-moment dispatch — all heavy lifting lives in
:mod:`_engine`, :mod:`_second_moment`, :mod:`_distributed`, and the
per-strategy files.

Realized per-step σ (bug fix): under correlated MF noise the actual
per-coordinate noise variance at step ``t`` is
``base_σ² · ‖row_t(C^-1)‖²``, not ``base_σ²``.  The factory
reads the per-step row-L2 from the host execution plan and publishes
``noise_stddev = base_σ · row_l2(t)`` on each :class:`NoisedPytree`,
so Adam-family optimizers' ``noise_bias_correction`` EMA subtracts the
true variance contribution.  ``row_l2_at`` is a callable returned by
each strategy's noise builder; :class:`IdentityStrategy` returns ``1``
at every step (MF noise reduces to iid DP-SGD), λ-CGD computes its
closed-form column factor, and streaming-matrix strategies precompute
``streaming.row_norms_squared(n_steps).sqrt()`` once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from opaque.api.engine import runtime
from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.random.types import RngKey
from opaque.types import NoisedPytree, PerGroup, SecondMomentClippingOutput

from ._distributed import mf_per_group_sync_fingerprint_for_latch
from ._engine import (
    MFNoiseState,
    _check_mf_horizon,
    _expect_clipped,
    _matrix_factorization_noise,
    _require_positive_int_horizon,
    _resolve_noise_multiplier,
    _validate_constant_max_norm,
)
from ._second_moment import SecondMomentMFNoiseState, make_second_moment_mf_noise

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import MfStrategy


def mf_gaussian_noise(
    grad_template: Any,
    strategy: MfStrategy,
    *,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    noise_multiplier: float,
    key: RngKey,
    compute_dtype: object | None = None,
    second_moment_strategy: MfStrategy | None = None,
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
        compute_dtype: Provider-native dtype used for normal sampling and
            linear-combination arithmetic. Defaults to the active provider's
            ``float32`` — sampling
            Gaussians in bf16/fp16 introduces discretization that distorts
            the noise distribution. Type stability on the public boundary is
            preserved: the input pytree's dtype is matched on output (input
            upcast to ``compute_dtype``, noise added, downcast at return).
        second_moment_strategy: Optional explicit strategy recipe for
            the squared-gradient stream.

    Returns:
        A tuple ``(noise_fn, state)`` for the training loop.
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    n_steps = _require_positive_int_horizon(n_steps)
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
            compute_dtype=compute_dtype,
        )

    raw_noise_fn, raw_state, row_l2_at = _make_raw_mf_noise(
        grad_template,
        strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=key,
        compute_dtype=compute_dtype,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
    ) -> tuple[NoisedPytree, MFNoiseState]:
        with runtime.trace_scope("opaque::mf_gaussian_noise"):
            return _noise_fn_impl(clipped_grads, st)

    def _noise_fn_impl(
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
            # Non-private run: force base σ = 0 before the ``nm * max_norm``
            # product so a disabled clip (max_norm = +inf) gives ``0`` rather
            # than ``0 * inf = NaN``.
            base_stddev = (
                0.0
                if resolved_noise_multiplier == 0.0
                else resolved_noise_multiplier * max_norm
            )
        noisy_tree, new_state = raw_noise_fn(
            clipped_grads.pytree,
            st,
            stddev=base_stddev,
        )
        # Realized per-step σ = base σ · ‖row_t(C^-1)‖.  ``base_stddev``
        # is scalar or :class:`PerGroup`; both broadcast with the scalar
        # ``row_l2`` via ``PerGroup.__mul__`` / float multiplication.
        # Step we just applied noise to is ``st._step_counter`` (the
        # pre-increment value); ``raw_noise_fn`` advances counter inside.
        row_l2 = row_l2_at(st._step_counter)
        realized_stddev = base_stddev * row_l2
        sync_fp = mf_per_group_sync_fingerprint_for_latch(st, max_norm)
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=clipped_grads.max_norm,
                noise_stddev=realized_stddev,
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
    strategy: MfStrategy,
    *,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    key: RngKey,
    compute_dtype: object | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
    Callable[[int], float],
]:
    """Build the underlying noise function + per-step row-L2 lookup.

    Every strategy projects its host recipe into the same portable execution
    plan contract, preventing accounting/runtime coefficient drift.

    The third return value, ``row_l2_at(step) -> float``, exposes the
    per-step L2 norm of ``C^{-1}``'s row so the wrapping factory can
    publish realized σ on :class:`NoisedPytree.noise_stddev`.
    The row norms are precomputed by the host plan at construction time.
    """
    plan = strategy.execution_plan(
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )
    noise_fn, state = _matrix_factorization_noise(
        grad_template,
        plan,
        key=key,
        compute_dtype=compute_dtype,
        n_steps=n_steps,
    )

    def row_l2_at(step: int) -> float:
        _check_mf_horizon(step, n_steps)
        return plan.row_l2[step]

    return noise_fn, state, row_l2_at


__all__ = ["mf_gaussian_noise"]
