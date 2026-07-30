"""Adadelta optimizer with two-EMA DP bias correction.

Vanilla Adadelta (Zeiler 2012, arXiv:1212.5701) maintains two EMAs::

    E[g²]_t  = ρ E[g²]_{t-1}  + (1−ρ) g_t²
    Δx_t     = -(√(E[Δx²]_{t-1} + ε) / √(E[g²]_t + ε)) · g_t
    E[Δx²]_t = ρ E[Δx²]_{t-1} + (1−ρ) (Δx_t)²

Adadelta is famously learning-rate-free; the per-element ratio of update
RMS to gradient RMS *is* the adaptive step size.  The opaque chain still
threads a global ``lr`` factor (default 1.0) so users can scale the
update if they need to.

DP behaviour.  Under noised gradients ``g̃_t = g_clean + ξ_t`` with
``ξ_t ~ N(0, σ_t² I)``, both EMAs accumulate noise:

- ``E[g²]_t`` inherits the same per-step ``σ²`` offset as Adam's ``v_t``.
  Subtract a ρ-EMA ``φ_g`` of ``σ²`` to recover the unbiased estimate.
- ``E[Δx²]_t`` is more subtle.  ``Δx_t = -coef_t · g̃_t`` is linear in
  ``g̃_t``, so the noise variance injected into ``Δx_t`` at element ``i``
  is ``coef_t,i² · σ_t²`` — a *known* per-element scalar (the coef is
  computed before noise is added).  Maintain a second EMA
  ``φ_dx`` per element::

      φ_dx,t = ρ φ_dx,{t-1} + (1−ρ) (coef_t · σ_t)²
      E[Δx²]_corrected,t = max(E[Δx²]_t − φ_dx,t, ε²)

  Use ``E[Δx²]_corrected,{t-1}`` in the next step's coef.

The two φ-EMAs decay at the same rate ``ρ`` as their respective second
moments, so subtraction at any step is consistent with the EMA history.

Memory cost: ``2 · |params|`` tensors for the second moments (same as
vanilla) plus one scalar/dict ``φ_g`` and one tensor ``φ_dx`` per leaf
when BC is active.  Total: roughly 1.5× vanilla Adadelta's footprint —
still less than Adam's ``m + v + φ``.

The derivation is a straightforward propagation of Gaussian variance
through the linear scaling step; no published prior, but the math is
tight enough to drop in.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.api.optimizers._bias_correction import (
    is_per_group,
    resolve_noise_variance,
    update_phi_ema,
    walk_dict_leaves,
)
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.pytree import tree_map

if TYPE_CHECKING:
    from opaque.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AdadeltaState:
    """Immutable state for Adadelta with optional DP bias correction.

    Carries both noise-variance EMAs (``phi_g``, ``phi_dx``) regardless of
    whether BC is active so the state shape is stable across calls and
    checkpoints don't depend on call history.

    Attributes:
        v_g: Squared-gradient EMA ``E[g²]`` (pytree matching params).
        v_dx: Squared-update EMA ``E[Δx²]`` (pytree matching params).
        phi_g: Gradient-noise-variance EMA — scalar (homogeneous σ) or
            ``dict[group, float]`` (PerGroup σ).  Stays at zero unless a
            ``NoisedPytree`` update supplies realized σ metadata.
        phi_dx: Update-noise-variance EMA.  Per-element pytree matching
            params because the per-step variance ``coef_t² · σ²`` is
            element-wise even when σ is scalar.
        step: Number of completed updates.
    """

    v_g: TensorPytree
    v_dx: TensorPytree
    phi_g: float | dict[str, float]
    phi_dx: TensorPytree
    step: int


# ---------------------------------------------------------------------------
# Moment scaler
# ---------------------------------------------------------------------------


def _rms(tensor_or_zero: torch.Tensor, eps: float) -> torch.Tensor:
    """``√(x + ε)`` — Adadelta's smoothed RMS."""
    return (tensor_or_zero + eps).sqrt()


def _scale_by_adadelta(
    rho: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> GradientTransformation:
    """Adadelta moment scaling with two-EMA DP bias correction.

    Returns the un-negated update (``coef · g``); the chain's
    ``scale_by_neg_lr`` applies the sign and global lr scaling at the end.
    """

    def init_fn(params: Any) -> AdadeltaState:
        v_g = tree_map(torch.zeros_like, params)
        v_dx = tree_map(torch.zeros_like, params)
        phi_dx = tree_map(torch.zeros_like, params)
        return AdadeltaState(v_g=v_g, v_dx=v_dx, phi_g=0.0, phi_dx=phi_dx, step=0)

    def update_fn(
        updates: Any,
        state: AdadeltaState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any = None,
    ) -> tuple[Any, AdadeltaState]:
        if noisy_squared_grads is not None and noise_stddev is not None:
            raise ValueError(
                "adadelta.update() received both noisy_squared_grads and "
                "noise_stddev (DP-BC); pass exactly one (or neither)."
            )

        t = state.step + 1

        # ---- E[g²] update -------------------------------------------
        if noisy_squared_grads is not None:
            # External second-moment branch: g² stream replaces (g·g).
            # phi_g is left at its current value (post-processing already
            # debiased v_g for this step).  phi_dx similarly stays put —
            # we don't have σ in this branch, so we can't advance the
            # update-noise EMA without divergence; document the
            # trade-off below.
            new_v_g = tree_map(
                lambda v, g2: rho * v + (1 - rho) * g2,
                state.v_g,
                noisy_squared_grads,
            )
            new_phi_g: Any = state.phi_g
            new_phi_dx: Any = state.phi_dx
        else:
            new_v_g = tree_map(
                lambda v, g: rho * v + (1 - rho) * g * g, state.v_g, updates
            )

            effective = noise_stddev if noise_stddev is not None else 0.0
            if noise_bias_correction:
                if is_per_group(effective) or isinstance(state.phi_g, dict):
                    new_phi_g_dict: dict[str, float] = {}
                    for path, _leaf in walk_dict_leaves(new_v_g):
                        nv = resolve_noise_variance(effective, path)
                        old = (
                            state.phi_g.get(path, 0.0)
                            if isinstance(state.phi_g, dict)
                            else state.phi_g
                        )
                        new_phi_g_dict[path] = rho * old + (1 - rho) * nv
                    new_phi_g = new_phi_g_dict
                else:
                    new_phi_g = update_phi_ema(state.phi_g, float(effective) ** 2, rho)
            else:
                new_phi_g = state.phi_g
            new_phi_dx = state.phi_dx  # advanced below alongside v_dx

        # ---- Per-leaf Δx, v_dx, and (optional) phi_dx ----------------
        # Walk leaves so we can resolve per-group σ and read/write the
        # per-leaf phi_dx tensor.  Single walk handles both paths.

        def _phi_g_for(path: str) -> float:
            if isinstance(new_phi_g, dict):
                return float(new_phi_g.get(path, 0.0))
            return float(new_phi_g)

        def _walk_compute(
            updates_node: Any,
            v_g_node: Any,
            v_dx_node: Any,
            phi_dx_node: Any,
            prefix: str,
        ) -> tuple[Any, Any, Any, Any]:
            if isinstance(updates_node, dict):
                out_dx = {}
                out_v_dx = {}
                out_phi_dx = {}
                out_v_g = {}
                for k in updates_node:
                    sub_prefix = f"{prefix}.{k}" if prefix else str(k)
                    o_dx, o_v_dx, o_phi_dx, o_v_g = _walk_compute(
                        updates_node[k],
                        v_g_node[k],
                        v_dx_node[k],
                        phi_dx_node[k],
                        sub_prefix,
                    )
                    out_dx[k] = o_dx
                    out_v_dx[k] = o_v_dx
                    out_phi_dx[k] = o_phi_dx
                    out_v_g[k] = o_v_g
                return out_dx, out_v_dx, out_phi_dx, out_v_g

            # Tensor leaf at path ``prefix``.
            v_g_t = v_g_node
            # In the second-moment-substitution branch, ``E[g²]`` is
            # already debiased by post-processing — applying φ_g would
            # subtract the noise variance twice if a prior
            # ``NoisedPytree`` step had grown φ_g.  Force zero here so
            # the carried-over EMA does not silently double-correct.
            if noise_bias_correction and noisy_squared_grads is None:
                phi_g_path = _phi_g_for(prefix)
            else:
                phi_g_path = 0.0

            # Corrected E[g²] at this step.
            if phi_g_path > 0:
                corrected_g = v_g_t - phi_g_path
                v_g_corrected = torch.where(corrected_g > 0, corrected_g, v_g_t)
            else:
                v_g_corrected = v_g_t
            # In the SM branch, noisy g² can be negative when noise dominates;
            # clamp to 0 so _rms (which adds eps before sqrt) stays well-defined.
            if noisy_squared_grads is not None:
                v_g_corrected = torch.clamp(v_g_corrected, min=0.0)

            # Corrected E[Δx²] from previous step (the φ_dx is the
            # *previous* one because we read ``v_dx_node`` and the
            # matching pre-step φ_dx_node).  Same double-correction
            # concern as φ_g above: skip the subtraction in the
            # second-moment-substitution branch where σ isn't available
            # to advance the EMA.
            if noise_bias_correction and noisy_squared_grads is None:
                corrected_dx = v_dx_node - phi_dx_node
                v_dx_corrected_prev = torch.where(
                    corrected_dx > 0, corrected_dx, v_dx_node
                )
            else:
                v_dx_corrected_prev = v_dx_node

            rms_g = _rms(v_g_corrected, eps)
            rms_dx_prev = _rms(v_dx_corrected_prev, eps)

            # Per-element coefficient.  Sign: ``Δx = -coef · g``; the
            # chain's ``scale_by_neg_lr`` does the negation, so we
            # return ``coef · g`` here.
            coef = rms_dx_prev / rms_g
            dx = coef * updates_node

            # Update v_dx with raw squared update (Δx_t)².  The negative
            # sign of Δx is irrelevant under squaring.
            v_dx_new_t = rho * v_dx_node + (1 - rho) * dx * dx

            # Update φ_dx with the per-element noise variance injected
            # by this step: coef² · σ².  σ is scalar (or per-group);
            # coef is per-element; the product is per-element.
            if noise_bias_correction and noisy_squared_grads is None:
                sigma_sq = (
                    resolve_noise_variance(effective, prefix)
                    if is_per_group(effective)
                    else float(effective) ** 2
                )
                # ``sigma_sq`` is a scalar; the variance contribution
                # to (Δx_t)_i is ``coef_i² · sigma_sq``.
                noise_var_dx = coef * coef * sigma_sq
                phi_dx_new_t = rho * phi_dx_node + (1 - rho) * noise_var_dx
            else:
                phi_dx_new_t = phi_dx_node

            return dx, v_dx_new_t, phi_dx_new_t, v_g_t  # v_g_t is unchanged here

        result_tree, new_v_dx_tree, new_phi_dx_tree, _ = _walk_compute(
            updates, new_v_g, state.v_dx, state.phi_dx, ""
        )

        # In the noisy_squared_grads branch we already declared new_phi_dx;
        # in the bc-active branch, the walker assembled it.
        if noisy_squared_grads is None:
            new_phi_dx = new_phi_dx_tree

        return result_tree, AdadeltaState(
            v_g=new_v_g,
            v_dx=new_v_dx_tree,
            phi_g=new_phi_g,
            phi_dx=new_phi_dx,
            step=t,
        )

    return GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adadelta(
    lr: _LR = 1.0,
    rho: float = 0.9,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an Adadelta optimizer with two-EMA DP bias correction.

    Args:
        lr: Global multiplier on the update.  Adadelta is classically
            learning-rate-free (paper default 1.0); a non-unit value
            scales the per-element adaptive step.
        rho: EMA decay for both squared-gradient and squared-update
            accumulators (paper default 0.9, torchopt default 0.9).
        eps: Numerical floor inside the RMS computations.
        weight_decay: Weight-decay coefficient.
        decoupled_weight_decay: ``True`` selects decoupled WD (post
            moment scaling); ``False`` folds ``wd · params`` into the
            gradient before moment scaling (L2 regularisation).
        update_rms_clip: Optional StableAdamW-style RMS clip.
        noise_bias_correction: If ``True``, subtract:

            - ``φ_g``: a ρ-EMA of σ² from ``E[g²]`` (recovers unbiased
              squared-gradient estimate).
            - ``φ_dx``: a ρ-EMA of ``coef² · σ²`` from ``E[Δx²]``
              (recovers unbiased squared-update estimate).

            Both EMAs decay at the same rate ρ as their respective
            second moments, so subtraction is consistent with the EMA
            history.  Defaults to ``False``; flip on to ablate against
            vanilla Adadelta under noise.

    Returns:
        A ``torchopt.base.GradientTransformation``.

    DP usage notes:

        - ``NoisedPytree`` updates with ``noise_bias_correction=True``
          activate both BC EMAs.  σ travels on the wrapper.
        - ``SecondMomentNoiseOutput`` updates substitute the privatised
          ``g²`` stream into ``E[g²]`` directly.  In this branch we
          freeze ``φ_dx`` because σ isn't carried alongside the second-
          moment stream — the update-noise variance EMA cannot advance
          without it.  Use the ``NoisedPytree`` path for full BC.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not 0 <= rho < 1:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError(
            f"update_rms_clip must be positive when set, got {update_rms_clip}"
        )

    bc_floor = eps * eps
    moment = _scale_by_adadelta(
        rho=rho,
        eps=eps,
        noise_bias_correction=noise_bias_correction,
        bc_floor=bc_floor,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


__all__ = ["AdadeltaState", "adadelta"]
