# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Router-load release plumbing for :class:`DPTrainer`.

The mechanism itself lives in :mod:`opaque.api.transformers.moe_load`; this
module holds the trainer-side pieces: the resolved public constants of one
run (:class:`RouterLoadRuntime`), the callback that consumes the noised probe
leaf after clipping and noise and before the optimizer update
(:class:`RouterLoadCallback`), the monitor decision rule, and the checkpoint
sidecar that carries the public post-processing state across a resume.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

from opaque.api.transformers.moe_load import (
    PROBE_NAME,
    RouterLoadState,
    check_resume_compatible,
    decide_trip,
    monitor_value,
    resolve_moe_geometry,
    update,
)
from opaque.exceptions import CheckpointError
from opaque.serialization import from_state_dict as opaque_from_state_dict
from opaque.serialization import state_dict as opaque_state_dict
from transformers.trainer_callback import TrainerCallback

log = logging.getLogger(__name__)

ROUTER_LOAD_STATE_NAME = "router_load_state.pt"
_SIDECAR_VERSION = 1


@dataclasses.dataclass
class RouterLoadRuntime:
    """Resolved public constants of one router-load run.

    Everything here is a public hyperparameter or a function of public
    hyperparameters; the per-step private quantities never leave the
    gradient transform.

    Attributes:
        mode: ``router_load_release``.
        alpha: Configured surrogate coefficient (in force from the start in
            ``"surrogate"``, after the trip in ``"monitor_then_surrogate"``).
        lam: Probe scale from :func:`opaque.api.transformers.moe_load.probe_bounds`.
        ratio: Budget share ``C_h / C_g``.
        num_layers: Number of routed layers ``L``.
        num_experts: Number of experts ``E``.
        top_k: Experts per token ``k``.
        mean_tokens: Public token-count constant ``T_bar`` of ``w_x``.
        max_tokens: Public bound on the valid tokens one protected unit
            contributes (``T_max``, or ``2 T_max`` for a preference pair).
        row_max_tokens: Public bound on the length of one collated row.
        trip: Monitor threshold ``tau``.
        z_loss_coef: Router z-loss coefficient.
        aux: ``"pooled"`` or ``"per_sequence"``.
        probe_name: Key of the probe in the trainable pytree.
        callback: The registered :class:`RouterLoadCallback`.
        target: Device tensor the loss closure reads ``f_tilde`` from.
    """

    mode: str
    alpha: float
    lam: float
    ratio: float
    num_layers: int
    num_experts: int
    top_k: int
    mean_tokens: float
    max_tokens: float
    row_max_tokens: int
    trip: float
    z_loss_coef: float
    aux: str
    probe_name: str = PROBE_NAME
    callback: RouterLoadCallback | None = None
    target: torch.Tensor | None = None

    @property
    def state(self) -> RouterLoadState:
        assert self.callback is not None
        return self.callback.state

    @property
    def alpha_active(self) -> float:
        return float(self.state.alpha_active)


class RouterLoadCallback(TrainerCallback):
    """Consume the noised probe leaf and run the monitor decision rule.

    Registered by :class:`DPTrainer` when ``router_load_release`` is not
    ``"off"``.  On ``on_pre_optimizer_step`` (after clipping, DDP reduction
    and noise, before the optimizer update) it reads the probe leaf of the
    noised pytree, post-processes it into the next public estimate
    ``f_tilde`` through :func:`opaque.api.transformers.moe_load.update`, and
    zeroes the leaf in place so the optimizer update of the probe is exactly
    zero.

    Decision rule: :func:`opaque.api.transformers.moe_load.decide_trip`,
    the one rule every consumer of the state applies, evaluated at the
    logging cadence.  The monitor ``D_t = max_e |f_tilde_e - k/E| / (k/E)``
    of the estimate that enters the surrogate (after the dead zone and the
    shrinkage, so a noise-dominated early estimate reads as zero; with
    ``router_load_shrink=False`` it is the raw ``router_load/D``) above
    ``trip`` on two consecutive logged evaluations sets ``tripped``.  In
    ``"monitor_then_surrogate"`` the trip switches the surrogate
    coefficient from ``0`` to the configured value; the clipping bound, the
    noise and the accountant are untouched (an adaptive choice of the next
    step's loss is post-processing of previous releases).

    The callback stays registered after ``train()`` returns so the final
    state remains readable; a later run without the release hands it a
    pytree without the probe, which it leaves alone.
    """

    def __init__(
        self,
        state: RouterLoadState,
        *,
        mode: str,
        trip: float,
        alpha: float,
        probe_name: str = PROBE_NAME,
    ) -> None:
        self.state = state
        self.mode = mode
        self.trip = float(trip)
        self.alpha = float(alpha)
        self.probe_name = probe_name
        self.consecutive_over_trip = 0

    @property
    def f_tilde(self) -> torch.Tensor:
        return self.state.f_tilde

    @property
    def alpha_active(self) -> float:
        return float(self.state.alpha_active)

    def on_pre_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        *,
        grads: Any = None,
        trainable_params: Any = None,
        **kwargs: Any,
    ) -> Any:
        del args, trainable_params, kwargs
        if grads is None or self.probe_name not in grads.pytree:
            # No probe in this run (the release is off): nothing to consume.
            return control
        leaf = grads.pytree[self.probe_name]
        new_state = update(self.state, leaf)
        # The probe is a public constant zero: its optimizer update must be
        # exactly zero, so the consumed release is removed from the pytree.
        leaf.zero_()
        self.state = self._decide(new_state, state)
        return control

    def _logged_evaluation(self, trainer_state: Any) -> bool:
        """Whether the release just consumed lands on a logged step."""
        every = int(getattr(trainer_state, "logging_steps", 0) or 0)
        if every < 1:
            return True
        step = int(getattr(trainer_state, "global_step", 0)) + 1
        return step % every == 0

    def _decide(
        self, new_state: RouterLoadState, trainer_state: Any
    ) -> RouterLoadState:
        if not self._logged_evaluation(trainer_state):
            return new_state
        decided, self.consecutive_over_trip = decide_trip(
            new_state,
            self.consecutive_over_trip,
            trip=self.trip,
            mode=self.mode,
            alpha=self.alpha,
        )
        if decided.tripped and not new_state.tripped:
            log.info(
                "router_load: monitor D=%.3f exceeded trip=%.3f on two "
                "consecutive logged evaluations at step %d (alpha_active=%g)",
                monitor_value(decided),
                self.trip,
                int(getattr(trainer_state, "global_step", 0)) + 1,
                decided.alpha_active,
            )
        return decided


# ---------------------------------------------------------------------------
# Checkpoint sidecar
# ---------------------------------------------------------------------------


def save_router_load_state(ckpt_dir: str, callback: RouterLoadCallback) -> str:
    """Write the public post-processing state next to the DP runtime bundle."""
    path = Path(ckpt_dir) / ROUTER_LOAD_STATE_NAME
    payload = {
        "version": _SIDECAR_VERSION,
        "state": opaque_state_dict(callback.state),
        "consecutive_over_trip": int(callback.consecutive_over_trip),
    }
    torch.save(payload, str(path))
    return str(path)


def restore_router_load_state(ckpt_dir: str, callback: RouterLoadCallback) -> None:
    """Restore the sidecar onto ``callback`` and reject configuration drift.

    The restored state continues the filter bit for bit; the public
    constants it was built with (``ratio``, filter kind / beta / window,
    ``dead_zone``, ``E``, ``k``, ``L``, ``mean_tokens``, ``max_tokens``, the
    probe scale ``lam``, which encodes the gradient bound ``C_g``, and the
    filter factors ``phi``) must equal the freshly configured ones
    (:func:`opaque.api.transformers.moe_load.check_resume_compatible`),
    otherwise the continued filter would mix releases of two different
    mechanisms.  ``base_noise_std`` is the one value allowed to change: a
    target-epsilon run re-calibrates the noise multiplier for the remaining
    steps, so the restored state carries the freshly calibrated value
    forward (with a warning) and the known noise ``s_t`` of every later
    estimate describes the noise actually added from here on.
    """
    path = Path(ckpt_dir) / ROUTER_LOAD_STATE_NAME
    if not path.exists():
        raise CheckpointError(
            *(
                f"Cannot resume a router_load_release run from {ckpt_dir}: the "
                f"sidecar {ROUTER_LOAD_STATE_NAME} is missing.  The checkpoint "
                "was written without the router-load release; start a fresh "
                "run or set router_load_release='off'.",
            )
        )
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    version = payload.get("version") if isinstance(payload, dict) else None
    if version != _SIDECAR_VERSION:
        raise CheckpointError(
            *(
                f"{ROUTER_LOAD_STATE_NAME} has version {version!r}; this trainer "
                f"reads version {_SIDECAR_VERSION}.",
            )
        )
    restored = opaque_from_state_dict(callback.state, payload["state"])
    check_resume_compatible(restored, callback.state)
    current_base = float(callback.state.base_noise_std)
    if restored.base_noise_std != current_base:
        log.warning(
            "router_load_release resume: the saved release noise std (%g) "
            "differs from the freshly calibrated one (%g); the restored filter "
            "continues with the new value, so the known noise of every later "
            "estimate describes the noise added from here on.",
            restored.base_noise_std,
            current_base,
        )
        restored = dataclasses.replace(restored, base_noise_std=current_base)
    callback.state = restored
    callback.consecutive_over_trip = int(payload.get("consecutive_over_trip", 0))


__all__ = [
    "ROUTER_LOAD_STATE_NAME",
    "RouterLoadCallback",
    "RouterLoadRuntime",
    "resolve_moe_geometry",
    "restore_router_load_state",
    "save_router_load_state",
]
