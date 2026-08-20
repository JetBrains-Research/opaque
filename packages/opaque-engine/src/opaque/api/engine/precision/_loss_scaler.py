"""Functional fp16 loss scaler.

Operates on the pytree returned by ``vmap(grad(loss_fn))`` rather than on
``nn.Parameter.grad`` — the analog of :class:`torch.amp.GradScaler` for
the functional DP step.

DP-critical invariant
---------------------
The unscale must run **before** the per-example clip-norm. Otherwise the
sensitivity bound the privacy accountant is calibrated to (``C``) is
multiplied by the loss scale, and the realized noise stddev no longer
matches the recorded ``noise_multiplier · C``. Wire :func:`LossScaler.unscale_grads`
as the ``pre_clipping_transform`` of
:func:`opaque.api.engine.clipping.clipped_grad` so the unscale runs inside
``vmap``, per-example, before the clip-norm::

    scaler, scaler_state = loss_scaler()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(...),
        clipping_norm=C,
        pre_clipping_transform=lambda g: scaler.unscale_grads(g, scaler_state),
    )

Skipped steps and the privacy accountant
----------------------------------------
An overflow signal is data-dependent and must not suppress the noised update or
its accounting. Request ``return_stats=True`` from
:func:`opaque.api.engine.clipping.clipped_grad`, run the normal noised update
on every attempted step, and use ``stats.all_finite`` to back off the loss
scale. This module provides scaling and the state machine; the surrounding
loop owns noise, optimization, and accounting.

Factory shape
-------------
``loss_scaler(...) -> (LossScaler, LossScalerState)`` matches the
``(transform, state)`` convention used by other Opaque primitives
(:func:`opaque.dpsgd.noise.gaussian_noise`). The state is a frozen
dataclass — the same convention as ``AdamState`` and friends in
``opaque-optimizers`` — and threads explicitly through the training
loop. No ``init`` slot is needed because the initial state is fully
determined by ``init_scale`` at factory time.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from opaque.api.engine.pytree import tree_flatten
from opaque.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentClippingOutput,
)
from opaque.pytree import tree_map

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LossScaler", "LossScalerState", "all_finite", "loss_scaler"]


@dataclasses.dataclass(frozen=True)
class LossScalerState:
    """Immutable state of the dynamic loss scaler.

    Frozen-dataclass convention matches ``AdamState`` and other Opaque
    optimizer states, so the state pytree serializes through
    :func:`opaque.serialization.state_dict` without bespoke handling.

    Attributes:
        scale: Current loss scale. Multiplied into the loss before
            backward, divided out of the per-example gradients before
            clipping.
        growth_tracker: Number of consecutive clean (all-finite) steps
            observed since the last grow / backoff event.
    """

    scale: float
    growth_tracker: int


class LossScaler(NamedTuple):
    """A bundle of pure functions implementing dynamic fp16 loss scaling.

    Mirrors :class:`torchopt.base.GradientTransformation` (a ``NamedTuple``
    of pure functions, frozen-dataclass state, hyperparameters captured
    in the factory closure) but exposes the pre-backward ``scale_loss``
    slot that a strict gradient transformation cannot carry.

    All members are pure: state is threaded by the caller, no instance
    is mutated in place.
    """

    scale_loss: Callable[[torch.Tensor, LossScalerState], torch.Tensor]
    """Multiply ``loss`` by the current scale. Call inside the loss
    closure (after the autocast forward) so the scaled loss propagates
    to per-example gradients through ``vmap(grad(...))``."""

    unscale_grads: Callable[[Any, LossScalerState], Any]
    """Divide every floating-point leaf of ``grads`` by the current
    scale. Returns a new pytree with the same structure. Shaped to be
    passed (after binding ``state``) as the ``pre_clipping_transform``
    of :func:`opaque.api.engine.clipping.clipped_grad`."""

    update: Callable[[LossScalerState, bool], LossScalerState]
    """Advance the scale schedule. Grow by ``growth_factor`` after
    ``growth_interval`` consecutive clean steps; multiply by
    ``backoff_factor`` and reset the tracker on a non-finite step.
    Returns the new state."""


def loss_scaler(
    *,
    init_scale: float = 2.0**16,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
    enabled: bool = True,
) -> tuple[LossScaler, LossScalerState]:
    """Construct a loss scaler and its initial state.

    Defaults match :class:`torch.amp.GradScaler` so fp16 behavior is
    intuitive for users coming from HuggingFace / native PyTorch.

    Args:
        init_scale: Initial loss scale. Default ``2**16`` (GradScaler default).
        growth_factor: Multiplier applied after ``growth_interval``
            consecutive clean steps. Must be ``> 1.0``.
        backoff_factor: Multiplier applied on a non-finite step. Must
            lie in ``(0.0, 1.0)``.
        growth_interval: Number of consecutive clean steps after which
            the scale grows. Must be ``> 0``.
        enabled: When ``False``, :meth:`LossScaler.scale_loss` and
            :meth:`LossScaler.unscale_grads` are identity and
            :meth:`LossScaler.update` returns the state unchanged.
            Lets callers keep the same wiring for fp16 / bf16 / fp32
            without conditional call sites.

    Returns:
        A ``(transform, state)`` tuple. The ``transform`` is immutable;
        the ``state`` is threaded through training-step calls.
    """
    if growth_factor <= 1.0:
        raise ValueError(f"growth_factor must be > 1.0, got {growth_factor}")
    if not (0.0 < backoff_factor < 1.0):
        raise ValueError(f"backoff_factor must be in (0, 1), got {backoff_factor}")
    if growth_interval <= 0:
        raise ValueError(f"growth_interval must be > 0, got {growth_interval}")

    def scale_loss(loss: torch.Tensor, state: LossScalerState) -> torch.Tensor:
        if not enabled:
            return loss
        return loss * state.scale

    def unscale_grads(updates: Any, state: LossScalerState) -> Any:
        if not enabled:
            return updates
        s = state.scale

        def _unscale(t: Any) -> Any:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t / s
            return t

        return tree_map(_unscale, updates)

    def update(state: LossScalerState, grads_were_finite: bool) -> LossScalerState:
        if not enabled:
            return state
        if grads_were_finite:
            new_tracker = state.growth_tracker + 1
            if new_tracker >= growth_interval:
                return LossScalerState(
                    scale=state.scale * growth_factor, growth_tracker=0
                )
            return dataclasses.replace(state, growth_tracker=new_tracker)
        return LossScalerState(scale=state.scale * backoff_factor, growth_tracker=0)

    transform = LossScaler(
        scale_loss=scale_loss,
        unscale_grads=unscale_grads,
        update=update,
    )
    state = LossScalerState(scale=float(init_scale), growth_tracker=0)
    return transform, state


def all_finite(updates: Any) -> bool:
    """True iff every floating-point leaf of ``updates`` is finite.

    Walks the pytree once. Integer / boolean leaves are ignored — they
    can't carry inf/nan and would raise on :func:`torch.isfinite`.
    Use this only on manually materialized, pre-clipping gradient pytrees.
    ``clipped_grad`` sanitizes non-finite values before returning its
    ``ClippedPytree``; use ``return_stats=True`` for loss scaling instead. The
    helper lives next to :func:`loss_scaler` rather than
    on the ``LossScaler`` NamedTuple because it operates purely on grads, with
    no scaler-state dependency — same as :func:`opaque.pytree.global_norm`.
    """

    def _iter_tensor_containers(value: Any):
        if isinstance(value, torch.Tensor):
            yield value
            return
        if isinstance(value, (ClippedPytree, NoisedPytree)):
            yield from _iter_tensor_containers(value.pytree)
            return
        if isinstance(value, SecondMomentClippingOutput):
            yield from _iter_tensor_containers(value.grads)
            yield from _iter_tensor_containers(value.squared_grads)
            return
        leaves, _ = tree_flatten(value)
        if not leaves:
            return
        for leaf in leaves:
            yield from _iter_tensor_containers(leaf)

    for leaf in _iter_tensor_containers(updates):
        if (
            isinstance(leaf, torch.Tensor)
            and leaf.is_floating_point()
            and not torch.isfinite(leaf).all()
        ):
            return False
    return True
