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
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.transformers.moe_load import (
    PROBE_NAME,
    RouterLoadState,
    update,
)
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.serialization import from_state_dict as opaque_from_state_dict
from opaque.serialization import state_dict as opaque_state_dict
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from torch import nn

log = logging.getLogger(__name__)

ROUTER_LOAD_STATE_NAME = "router_load_state.pt"
_SIDECAR_VERSION = 1
# Fields of :class:`RouterLoadState` that must agree between the saved state
# and the freshly configured run for a resume to be meaningful.
_RESUME_MATCH_FIELDS: tuple[str, ...] = (
    "ratio",
    "kind",
    "beta",
    "window",
    "dead_zone",
    "shrink",
    "num_experts",
    "top_k",
    "num_layers",
    "mean_tokens",
    "max_tokens",
    "lam",
)
_RESUME_RELATIVE_TOLERANCE = 1e-9
_ROUTER_ATTRS = ("top_k", "num_experts", "weight")


def _is_router_module(module: nn.Module) -> bool:
    if "TopKRouter" in type(module).__name__:
        return True
    return all(hasattr(module, attr) for attr in _ROUTER_ATTRS) and hasattr(
        module, "norm_topk_prob"
    )


def resolve_moe_geometry(model: nn.Module) -> tuple[int, int, int, float]:
    """``(num_layers, num_experts, top_k, config_aux_coef)`` of a MoE model.

    ``num_layers`` counts the router modules the backbone records logits for
    (dense layers of a mixed model do not count); ``num_experts`` and
    ``top_k`` come from the model config.  Raises
    :class:`~opaque.exceptions.ConfigurationError` for a family without a
    router.
    """
    config = getattr(model, "config", None)
    num_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    num_layers = sum(1 for module in model.modules() if _is_router_module(module))
    if num_experts is None or top_k is None or num_layers < 1:
        raise ConfigurationError(
            *(
                "router_load_release needs a mixture-of-experts model whose "
                "backbone records router logits (config.num_experts, "
                "config.num_experts_per_tok and a top-k router module); "
                f"{type(model).__name__} has none. Set router_load_release='off' "
                "for a dense model.",
            )
        )
    aux_coef = float(getattr(config, "router_aux_loss_coef", 0.0) or 0.0)
    return int(num_layers), int(num_experts), int(top_k), aux_coef


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

    Decision rule: the monitor ``D_t = max_e |f_tilde_e - k/E| / (k/E)`` of
    the estimate that enters the surrogate (after the dead zone and the
    shrinkage, so a noise-dominated early estimate reads as zero; with
    ``router_load_shrink=False`` it is the raw ``router_load/D``) is evaluated
    at the logging cadence; ``D_t > trip`` on two consecutive logged
    evaluations sets ``tripped``.  In ``"monitor_then_surrogate"`` the trip
    switches the surrogate coefficient from ``0`` to the configured value;
    the clipping bound, the noise and the accountant are untouched (an
    adaptive choice of the next step's loss is post-processing of previous
    releases).
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
        if grads is None:
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
        share = new_state.top_k / new_state.num_experts
        monitor = float((new_state.f_tilde - share).abs().max() / share)
        if monitor > self.trip:
            self.consecutive_over_trip += 1
        else:
            self.consecutive_over_trip = 0
        if self.consecutive_over_trip < 2 or new_state.tripped:  # noqa: PLR2004
            return new_state
        alpha_active = new_state.alpha_active
        if self.mode == "monitor_then_surrogate":
            alpha_active = self.alpha
        log.info(
            "router_load: shrunk monitor D=%.3f exceeded trip=%.3f on two "
            "consecutive logged evaluations at step %d (alpha_active=%g)",
            monitor,
            self.trip,
            int(getattr(trainer_state, "global_step", 0)) + 1,
            alpha_active,
        )
        return dataclasses.replace(new_state, tripped=True, alpha_active=alpha_active)


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


def _values_differ(saved: Any, current: Any) -> bool:
    if isinstance(saved, float) or isinstance(current, float):
        saved_f, current_f = float(saved), float(current)
        scale = max(abs(saved_f), abs(current_f), 1e-300)
        return abs(saved_f - current_f) > _RESUME_RELATIVE_TOLERANCE * scale
    return saved != current


def restore_router_load_state(ckpt_dir: str, callback: RouterLoadCallback) -> None:
    """Restore the sidecar onto ``callback`` and reject configuration drift.

    The restored state continues the filter bit for bit; the public
    constants it was built with (``ratio``, filter kind / beta / window,
    ``dead_zone``, ``E``, ``k``, ``L``, ``mean_tokens``, ``max_tokens`` and
    the probe scale ``lam``, which encodes the gradient bound ``C_g``) must
    equal the freshly configured ones, otherwise the continued filter would
    mix releases of two different mechanisms.
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
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    version = payload.get("version") if isinstance(payload, dict) else None
    if version != _SIDECAR_VERSION:
        raise CheckpointError(
            *(
                f"{ROUTER_LOAD_STATE_NAME} has version {version!r}; this trainer "
                f"reads version {_SIDECAR_VERSION}.",
            )
        )
    restored = opaque_from_state_dict(callback.state, payload["state"])
    mismatched = [
        f"{name}: saved={getattr(restored, name)!r}, "
        f"current={getattr(callback.state, name)!r}"
        for name in _RESUME_MATCH_FIELDS
        if _values_differ(getattr(restored, name), getattr(callback.state, name))
    ]
    if mismatched:
        raise CheckpointError(
            *(
                "router_load_release configuration drift on resume; the saved "
                "post-processing state was built for a different release: "
                + "; ".join(mismatched)
                + ". Restart from scratch to change these settings.",
            )
        )
    if _values_differ(restored.base_noise_std, callback.state.base_noise_std):
        log.warning(
            "router_load_release resume: the saved release noise std (%g) "
            "differs from the freshly calibrated one (%g); the restored filter "
            "keeps the saved value.",
            restored.base_noise_std,
            callback.state.base_noise_std,
        )
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
