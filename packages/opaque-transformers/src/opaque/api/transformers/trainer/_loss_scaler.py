"""Stateful fp16 loss-scaler facade over :mod:`opaque.precision`.

DPTrainer wires a single ``self._loss_scaler`` instance into the loss
closure (``scale_loss`` inside the loss fn) and into
``clipped_grad(pre_clipping_transform=...)`` (``unscale_grads`` inside
``vmap``).  The functional primitive in :mod:`opaque.precision` is a
pair ``(LossScaler, LossScalerState)`` of pure functions + immutable
state — explicitly threaded by the caller.  The trainer's loop already
encapsulates that threading inside this object: we hold the state on
``self``, mutate it in :meth:`update`, and expose ``scale_loss`` /
``unscale_grads`` / :meth:`all_finite` / :meth:`update` as the
``GradScaler``-shaped surface the trainer (and its tests) bind to.

All scaling math lives in :func:`opaque.precision.loss_scaler` —
``OpaqueLossScaler`` is the trainer-side adapter that owns the mutable
slot.

DP-critical invariant
---------------------
The unscale must run **before** the per-example clip-norm, otherwise the
sensitivity calibration the privacy accountant relies on is multiplied by
the loss scale and the privacy guarantee breaks.  ``opaque.clipped_grad``
exposes the right hook for this: ``pre_clipping_transform``, applied
inside ``vmap`` and before the clip-norm.  See the lower-level validation
in ``opaque-patches/tests/kernels/test_autocast.py``.

Skipped steps and the privacy accountant
----------------------------------------
On a non-finite step (the loss scaler caught an inf/nan), the trainer
**must** skip noise injection, the optimizer update, and the accountant
update.  Skipped steps consume zero privacy budget — exactly what
``GradScaler.step`` does for the optimizer.  This module only provides
the detection + scale machine; the trainer wires the skip decision in
``training_step``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from opaque.precision import LossScaler, LossScalerState, all_finite, loss_scaler


@dataclass
class OpaqueLossScaler:
    """Stateful ``GradScaler``-shaped wrapper around :mod:`opaque.precision`.

    Defaults match ``torch.amp.GradScaler`` so behavior under fp16 is
    intuitive for HF users.

    Attributes:
        init_scale: Initial loss scale.  Doubled-and-halved by the dynamic
            schedule.  Default 2**16 (the GradScaler default).
        growth_factor: Multiplier applied to the scale after
            ``growth_interval`` clean steps in a row.
        backoff_factor: Multiplier applied to the scale on a non-finite
            step.
        growth_interval: Number of consecutive clean steps after which the
            scale grows.
        enabled: When ``False``, ``scale_loss`` and ``unscale_grads`` are
            no-ops and ``all_finite`` always returns True.  Useful for
            keeping the trainer's wiring uniform across fp16 / bf16 /
            fp32 configurations.
    """

    init_scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True

    _transform: LossScaler = field(init=False, repr=False)
    _state: LossScalerState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ``loss_scaler`` validates ``growth_factor`` / ``backoff_factor`` /
        # ``growth_interval`` and raises ``ValueError`` on bad inputs.
        self._transform, self._state = loss_scaler(
            init_scale=self.init_scale,
            growth_factor=self.growth_factor,
            backoff_factor=self.backoff_factor,
            growth_interval=self.growth_interval,
            enabled=self.enabled,
        )

    @property
    def scale(self) -> float:
        """Current loss scale."""
        return self._state.scale

    @property
    def _growth_tracker(self) -> int:
        """Consecutive clean-step counter — back-compat alias for the
        ``growth_tracker`` field on the underlying ``LossScalerState``.
        """
        return self._state.growth_tracker

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply ``loss`` by the current scale.

        Run inside the loss closure — *after* the autocast forward but
        before returning to ``vmap(grad(...))`` so the scaled loss
        propagates to per-example gradients.
        """
        return self._transform.scale_loss(loss, self._state)

    def unscale_grads(self, grads: Any) -> Any:
        """Divide every floating-point leaf of ``grads`` by the current scale.

        This is the right shape to pass as ``pre_clipping_transform`` to
        :func:`opaque.api.engine.clipping.clipped_grad`: it runs inside ``vmap``,
        before the clip-norm — preserving the DP sensitivity invariant.
        """
        return self._transform.unscale_grads(grads, self._state)

    def all_finite(self, grads: Any) -> bool:
        """True iff every floating-point leaf of ``grads`` is finite.

        Drives the growth/backoff schedule.  Computed *after* the unscale —
        at which point the gradients are at the "real" magnitude — so an
        ``inf`` here is a real forward/backward overflow, not a side
        effect of the loss scale.
        """
        if not self.enabled:
            return True
        return all_finite(grads)

    def update(self, grads_were_finite: bool) -> None:
        """Mirror ``torch.amp.GradScaler.update``.

        After ``growth_interval`` consecutive clean steps, multiply by
        ``growth_factor``.  On a non-finite step, multiply by
        ``backoff_factor`` and reset the tracker.
        """
        self._state = self._transform.update(self._state, grads_were_finite)

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint-friendly snapshot."""
        return {
            "init_scale": self.init_scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "enabled": self.enabled,
            "_scale": self._state.scale,
            "_growth_tracker": self._state.growth_tracker,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""
        self.init_scale = float(state["init_scale"])
        self.growth_factor = float(state["growth_factor"])
        self.backoff_factor = float(state["backoff_factor"])
        self.growth_interval = int(state["growth_interval"])
        self.enabled = bool(state["enabled"])
        # Rebuild the underlying transform so ``growth_factor`` / etc.
        # take effect; restore the persisted ``(scale, growth_tracker)``
        # snapshot afterwards.
        self._transform, _ = loss_scaler(
            init_scale=self.init_scale,
            growth_factor=self.growth_factor,
            backoff_factor=self.backoff_factor,
            growth_interval=self.growth_interval,
            enabled=self.enabled,
        )
        self._state = LossScalerState(
            scale=float(state["_scale"]),
            growth_tracker=int(state["_growth_tracker"]),
        )


__all__ = ["OpaqueLossScaler"]
