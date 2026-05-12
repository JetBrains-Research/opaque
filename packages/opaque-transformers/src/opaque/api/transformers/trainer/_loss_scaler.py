"""Functional loss scaler for fp16 autocast under DP-SGD.

``torch.amp.GradScaler`` was designed to mutate ``param.grad`` on
``nn.Parameter``.  DPTrainer's training step is purely functional: the
gradient is a pytree returned by ``vmap(grad(loss_fn))``, never attached
to the parameters.  ``OpaqueLossScaler`` is the functional analog —
mirroring ``GradScaler``'s scale / unscale / inf-detect / dynamic-scale
state machine, but operating on the gradient pytree.

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

import dataclasses
from typing import Any

import torch

from opaque.pytree import tree_leaves, tree_map


@dataclasses.dataclass
class OpaqueLossScaler:
    """Functional analog of ``torch.amp.GradScaler`` for the DP path.

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

    _scale: float = dataclasses.field(init=False, default=0.0)
    _growth_tracker: int = dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {self.growth_factor}")
        if not (0.0 < self.backoff_factor < 1.0):
            raise ValueError(
                f"backoff_factor must be in (0, 1), got {self.backoff_factor}"
            )
        if self.growth_interval <= 0:
            raise ValueError(f"growth_interval must be > 0, got {self.growth_interval}")
        self._scale = float(self.init_scale)
        self._growth_tracker = 0

    @property
    def scale(self) -> float:
        """Current loss scale."""
        return self._scale

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply ``loss`` by the current scale.

        Run inside the loss closure — *after* the autocast forward but
        before returning to ``vmap(grad(...))`` so the scaled loss
        propagates to per-example gradients.
        """
        if not self.enabled:
            return loss
        return loss * self._scale

    def unscale_grads(self, grads: Any) -> Any:
        """Divide every floating-point leaf of ``grads`` by the current scale.

        This is the right shape to pass as ``pre_clipping_transform`` to
        :func:`opaque.api.engine.clipping.clipped_grad`: it runs inside ``vmap``,
        before the clip-norm — preserving the DP sensitivity invariant.
        """
        if not self.enabled:
            return grads
        scale = self._scale

        def _unscale(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t / scale
            return t

        return tree_map(_unscale, grads)

    def all_finite(self, grads: Any) -> bool:
        """True iff every floating-point leaf of ``grads`` is finite.

        Drives the growth/backoff schedule.  Computed *after* the unscale
        — at which point the gradients are at the "real" magnitude — so
        an inf at this point reflects an actual overflow during the
        forward+backward, not a scale that's too aggressive (a very
        aggressive scale would underflow grads to zero, not overflow).
        """
        if not self.enabled:
            return True
        for leaf in tree_leaves(grads):
            if leaf.is_floating_point() and not torch.isfinite(leaf).all():
                return False
        return True

    def update(self, grads_were_finite: bool) -> None:
        """Mirror ``torch.amp.GradScaler.update``.

        After ``growth_interval`` consecutive clean steps, multiply by
        ``growth_factor``.  On a non-finite step, multiply by
        ``backoff_factor`` and reset the tracker.
        """
        if not self.enabled:
            return
        if grads_were_finite:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0
        else:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint-friendly snapshot."""
        return {
            "init_scale": self.init_scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "enabled": self.enabled,
            "_scale": self._scale,
            "_growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict`."""
        self.init_scale = float(state["init_scale"])
        self.growth_factor = float(state["growth_factor"])
        self.backoff_factor = float(state["backoff_factor"])
        self.growth_interval = int(state["growth_interval"])
        self.enabled = bool(state["enabled"])
        self._scale = float(state["_scale"])
        self._growth_tracker = int(state["_growth_tracker"])


__all__ = ["OpaqueLossScaler"]
