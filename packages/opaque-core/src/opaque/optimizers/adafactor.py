"""Adafactor optimizer (Shazeer & Stern, 2018).

Memory-efficient Adam variant: the second moment is **factored** for
tensors of rank ≥ 2 (one ``v_row`` of size ``rows`` and one ``v_col``
of size ``cols`` instead of a full ``rows × cols`` ``v``).  For
tensors of rank < 2 (vectors, scalars) Adafactor falls back to a
scalar second moment.

Reference:
    Shazeer & Stern, "Adafactor: Adaptive Learning Rates with Sublinear
    Memory Cost", arXiv:1804.04235.

The factored estimator approximates the full ``v_t`` matrix as the
outer product ``v_row · v_col / mean(v_row)``.  This saves
``rows·cols − rows − cols`` floats of state per matrix parameter, which
matters at LM scale.

Scope.  Vanilla + decoupled / L2 weight decay, with the paper's RMS
update clip (threshold 1.0 by default).  DP-aware modes (``noise_stddev``
φ-EMA, ``noisy_squared_grads`` private second moments) are **not offered** yet — the
per-axis bias derivation for the factored ``v̂`` needs to be written
down before they can land.  Because the row and column factors are
means, a homogeneous Gaussian noise contribution adds ``(1 − β₂) · σ²``
to each factor; the remaining work is deriving the right factored
post-processing path for non-homogeneous per-axis noise.  Until then ``adafactor``'s
moment scaler does not consume the DP metadata wrappers; passing raw
per-step metadata kwargs raises ``TypeError`` immediately, while
``NoisyPytree`` values are unwrapped with a warning.

Skipped (orthogonal knobs, can be added later):

- ``relative_step``: LR derived from ``min(1e-2, 1/√t)``.  Compose
  with ``opaque.scheduling`` instead.
- ``scale_parameter``: LR scaled by ``RMS(params)``.  Independent
  enhancement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

import optree

from opaque.optimizers._chain import make_optimizer_chain


_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class AdafactorState:
    """State for Adafactor moment scaling.

    Attributes:
        m: Optional first-moment EMA (pytree matching params, or None
            when β₁ == 0 — most Adafactor configs).
        v_flat: Per-leaf second-moment state.  For each leaf, a tuple:

            - ``(v_row, v_col)`` for tensors of rank ≥ 2 (factored),
            - ``(v,)`` for tensors of rank < 2 (scalar).
        treespec: Frozen tree spec from ``optree`` so updates can be
            re-packed in the same shape.
        step: Number of completed updates.
    """

    m: Any
    v_flat: tuple
    treespec: Any
    step: int


def _init_v_for_leaf(leaf: torch.Tensor) -> tuple:
    """Allocate row/col or scalar second-moment state for one leaf."""
    if leaf.dim() >= 2:
        # Factored: one v_row per "row" axis (-2) and one v_col per
        # "col" axis (-1).  Higher-rank tensors collapse the leading
        # dims into rows: shape ``(*leading, rows, cols)`` → row state
        # of shape ``(*leading, rows)``, col state of shape
        # ``(*leading, cols)``.
        v_row = torch.zeros(leaf.shape[:-1], dtype=leaf.dtype, device=leaf.device)
        v_col = torch.zeros(
            (*leaf.shape[:-2], leaf.shape[-1]),
            dtype=leaf.dtype,
            device=leaf.device,
        )
        return (v_row, v_col)
    return (torch.zeros_like(leaf),)


def _approx_v_hat(
    v_row: torch.Tensor, v_col: torch.Tensor, eps_root: float
) -> torch.Tensor:
    """Factored ``v̂`` approximation::

        v̂ ≈ v_row[..., :, None] · v_col[..., None, :] / mean(v_row, dim=-1)

    Following Shazeer & Stern (2018, Algorithm 4).  The mean (rather
    than sum) keeps the factored estimate scale-invariant so the
    approximation matches the full second moment on rank-1 inputs.
    """
    r_mean = v_row.mean(dim=-1, keepdim=True).clamp(min=eps_root)
    return (v_row / r_mean).unsqueeze(-1) * v_col.unsqueeze(-2)


def _scale_by_adafactor(
    b1: float,
    b2_decay: float,
    eps_grad: float,
    eps_root: float,
    update_rms_clip: float,
) -> GradientTransformation:
    """Adafactor moment scaling.

    ``b2_decay`` is the paper's ``c`` exponent in the time-varying
    β₂_t = 1 − t^c (default c = −0.8).
    """
    use_first_moment = b1 > 0.0

    def init_fn(params: Any) -> AdafactorState:
        flat, treespec = optree.tree_flatten(params)
        v_flat = tuple(_init_v_for_leaf(leaf) for leaf in flat)
        m = None
        if use_first_moment:
            m_flat = tuple(torch.zeros_like(leaf) for leaf in flat)
            m = optree.tree_unflatten(treespec, list(m_flat))
        return AdafactorState(m=m, v_flat=v_flat, treespec=treespec, step=0)

    def update_fn(
        updates: Any,
        state: AdafactorState,
        *,
        params: Any = None,  # noqa: ARG001
        inplace: bool = False,  # noqa: ARG001
    ) -> tuple[Any, AdafactorState]:
        t = state.step + 1
        # Time-varying β₂_t per the paper: β₂_t = 1 − t^c.
        beta2_t = 1.0 - (float(t) ** b2_decay)

        flat_grads, _ = optree.tree_flatten(updates)
        if len(flat_grads) != len(state.v_flat):
            raise ValueError(
                f"updates pytree has {len(flat_grads)} leaves, "
                f"but state has {len(state.v_flat)} — params/grads "
                "shape mismatch."
            )

        new_v_flat: list[tuple] = []
        new_grads: list[torch.Tensor] = []

        for g, v_state in zip(flat_grads, state.v_flat, strict=True):
            g_sq = g.pow(2) + eps_grad
            if len(v_state) == 2:
                v_row, v_col = v_state
                # Factored update.
                new_v_row = beta2_t * v_row + (1.0 - beta2_t) * g_sq.mean(dim=-1)
                new_v_col = beta2_t * v_col + (1.0 - beta2_t) * g_sq.mean(dim=-2)
                v_hat = _approx_v_hat(new_v_row, new_v_col, eps_root)
                update = g / v_hat.sqrt().clamp(min=eps_root)
                new_v_flat.append((new_v_row, new_v_col))
            else:
                (v,) = v_state
                new_v = beta2_t * v + (1.0 - beta2_t) * g_sq
                update = g / new_v.sqrt().clamp(min=eps_root)
                new_v_flat.append((new_v,))

            # RMS clip (Adafactor's "update clipping").
            rms = update.pow(2).mean().sqrt()
            scale = torch.clamp(rms / update_rms_clip, min=1.0)
            new_grads.append(update / scale)

        # Optional first moment β₁.
        if use_first_moment:
            assert state.m is not None
            flat_old_m, _ = optree.tree_flatten(state.m)
            new_flat_m = [
                b1 * old_m + (1.0 - b1) * g for old_m, g in zip(flat_old_m, new_grads)
            ]
            updates_unflat = optree.tree_unflatten(state.treespec, new_flat_m)
            new_m = optree.tree_unflatten(state.treespec, new_flat_m)
        else:
            updates_unflat = optree.tree_unflatten(state.treespec, new_grads)
            new_m = None

        return updates_unflat, AdafactorState(
            m=new_m,
            v_flat=tuple(new_v_flat),
            treespec=state.treespec,
            step=t,
        )

    return GradientTransformation(init_fn, update_fn)


def adafactor(
    lr: _LR = 1e-3,
    beta1: float = 0.0,
    decay_rate: float = -0.8,
    eps_grad: float = 1e-30,
    eps_root: float = 1e-3,
    weight_decay: float = 0.0,
    update_rms_clip: float = 1.0,
    *,
    decoupled_weight_decay: bool = True,
) -> GradientTransformation:
    """Create an Adafactor optimizer (Phase A: vanilla + WD only).

    Args:
        lr: Learning rate, scalar or schedule.  In the paper this is
            often left at 1.0 with ``relative_step``; we don't ship
            ``relative_step`` yet, so set an explicit LR.
        beta1: First-moment EMA coefficient.  ``0.0`` (default) disables
            the first moment, matching the paper's most memory-efficient
            configuration.
        decay_rate: Exponent ``c`` in β₂_t = 1 − t^c.  ``-0.8`` is the
            paper's default; smaller magnitudes (closer to 0) make β₂_t
            decay slower with step.
        eps_grad: Stability constant added to ``g²`` before factoring;
            paper default 1e-30.
        eps_root: Stability constant inside the ``√v̂`` denominator;
            paper default 1e-3.
        weight_decay: Decoupled weight-decay coefficient by default.
        update_rms_clip: RMS clip threshold on the moment-scaled update;
            paper default 1.0 (Adafactor bakes this in).
        decoupled_weight_decay: Same semantics as
            :func:`opaque.optimizers.adamw`.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    The factory does not accept ``noise_stddev`` or ``noisy_squared_grads``
    at update time — passing either raises ``TypeError`` from the
    moment-scaler signature.  Use :func:`opaque.optimizers.adamw` for
    DP-aware modes until the per-axis Adafactor derivation lands.
    """
    if decay_rate >= 0:
        raise ValueError(f"decay_rate must be negative, got {decay_rate}")
    if eps_grad <= 0 or eps_root <= 0:
        raise ValueError(
            f"eps_grad and eps_root must be positive, got {eps_grad}, {eps_root}"
        )
    if not 0 <= beta1 < 1:
        raise ValueError(f"beta1 must satisfy 0 <= beta1 < 1, got {beta1}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if update_rms_clip <= 0:
        raise ValueError(f"update_rms_clip must be positive, got {update_rms_clip}")

    moment = _scale_by_adafactor(
        b1=beta1,
        b2_decay=decay_rate,
        eps_grad=eps_grad,
        eps_root=eps_root,
        update_rms_clip=update_rms_clip,
    )
    # NB: do not stack the chain-level ``update_rms_clip`` on top of
    # Adafactor's built-in RMS clip — it's already applied inside the
    # moment scaler.
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=None,
    )


__all__ = ["adafactor", "AdafactorState"]
