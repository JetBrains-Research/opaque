"""Trainer-independent helper for the MoE router-load release.

A mixture-of-experts checkpoint trained with a Switch-style load-balancing
loss needs the batch load vector ``f(B)`` (the fraction of tokens routed to
each expert) in its objective.  That statistic is not per-example separable,
so a per-example DP pipeline replaces it by a public estimate ``f_tilde``
obtained as a differentially private release of the per-example, token
weighted, centred load ``lam * w_x * (h^{(L,E)}(x) - k/E)`` with
``w_x = T_x / mean_tokens``.  The release rides on the same clipped pytree as
the gradient through a zero ``(L, E)`` probe parameter that forms its own
:class:`~opaque.types.PerGroup` group with a structural (never active)
bound, so the joint release is one Gaussian or matrix mechanism and the
privacy accountant is unchanged; the whole price is a ``sqrt(1 + ratio)``
inflation of the gradient noise.

This module owns the four seams a training loop needs:

1. :func:`attach_probe` registers the zero probe parameter on the model
   before :func:`opaque.functional.make_functional` partitions it.
2. :func:`probe_bounds` builds the two-group ``PerGroup`` clipping bound and
   the probe scale ``lam``.
3. :func:`router_load_terms` augments the per-example loss inside the
   gradient transform (value-neutral: the loss value is unchanged).
4. :func:`initial_state`, :func:`filter_factors` and :func:`update` turn the
   noised probe leaf into the public estimate ``f_tilde`` and the imbalance
   monitor; :func:`summary` and :func:`telemetry_without_probe` expose the
   public curves and the probe-free gradient norms.

The consumer-side pieces every loop needs the same way are here as well:
:func:`resolve_moe_geometry` reads ``(L, E, k)`` off the model,
:func:`decide_trip` is the one monitor decision rule, and
:func:`check_resume_compatible` rejects a sidecar written for a different
release.

Every quantity computed from private examples stays inside the gradient
transform; only the noised probe leaf and its post-processing are public.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.api.engine.types import PerGroup
from opaque.exceptions import CheckpointError, ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import nn

PROBE_NAME = "router_load_probe"
"""Default parameter name and ``PerGroup`` group name of the load probe."""

_FILTER_KINDS = ("ema", "window")
_MIN_EXPERTS = 2
# Stabilisation check of the tabulated filter factors: needs at least this many
# entries and flags a relative spread above the tolerance over the last tenth.
_STABILISATION_MIN_ROWS = 20
_STABILISATION_TOL = 1e-2


# ---------------------------------------------------------------------------
# Seam 1: the probe parameter
# ---------------------------------------------------------------------------


def attach_probe(
    model: nn.Module,
    *,
    num_layers: int,
    num_experts: int,
    name: str = PROBE_NAME,
) -> str:
    """Register the zero ``(num_layers, num_experts)`` probe parameter.

    The probe is an fp32 ``nn.Parameter`` with ``requires_grad=True`` and
    value zero.  It must be attached before
    :func:`opaque.functional.make_functional` is called with
    ``partition_trainable=True`` so that it lands in the trainable pytree
    under ``name``.  Calling the function twice is a no-op when a probe of
    the right shape is already attached.

    Args:
        model: Module the probe is registered on.  The parameter is placed
            on the device of the model's first parameter.
        num_layers: Number of MoE layers ``L``.
        num_experts: Number of experts ``E``.
        name: Parameter name; also the ``PerGroup`` group name.

    Returns:
        The parameter name (the key of the probe in the trainable pytree).

    Raises:
        ConfigurationError: if a parameter of that name exists with a
            different shape, or if the shape arguments are not positive.
    """
    if num_layers < 1 or num_experts < 1:
        raise ConfigurationError(
            *(
                "attach_probe requires num_layers >= 1 and num_experts >= 1, "
                f"got num_layers={num_layers}, num_experts={num_experts}.",
            )
        )
    shape = (num_layers, num_experts)
    existing = dict(model.named_parameters()).get(name)
    if existing is not None:
        if tuple(existing.shape) != shape:
            raise ConfigurationError(
                *(
                    f"model already has a parameter {name!r} of shape "
                    f"{tuple(existing.shape)}; the router-load probe needs "
                    f"{shape}.",
                )
            )
        return name
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    probe = torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.float32, device=device), requires_grad=True
    )
    model.register_parameter(name, probe)
    return name


# ---------------------------------------------------------------------------
# Seam 2: the two-group clipping bound
# ---------------------------------------------------------------------------


def load_bound(
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    mean_tokens: float,
    max_tokens: float,
) -> float:
    """Structural L2 bound ``Delta_L`` of the per-layer centred load vector.

    For one example ``0 <= h^l_e <= 1`` and ``sum_e h^l_e = k`` on every
    layer, so ``||h^l - k/E||^2 <= k (1 - k/E)``; over ``L`` layers and with
    the token weight ``w_x <= max_tokens / mean_tokens``::

        Delta_L = (max_tokens / mean_tokens) * sqrt(k * L * (1 - k / E))
    """
    if not 1 <= top_k < num_experts:
        raise ConfigurationError(
            *(
                "router-load release requires 1 <= top_k < num_experts, "
                f"got top_k={top_k}, num_experts={num_experts}.",
            )
        )
    if num_layers < 1:
        raise ConfigurationError(*(f"num_layers must be >= 1, got {num_layers}.",))
    if mean_tokens <= 0 or max_tokens <= 0:
        raise ConfigurationError(
            *(
                "mean_tokens and max_tokens must be positive, got "
                f"mean_tokens={mean_tokens}, max_tokens={max_tokens}.",
            )
        )
    return (max_tokens / mean_tokens) * math.sqrt(
        top_k * num_layers * (1.0 - top_k / num_experts)
    )


def _gradient_groups(
    clipping_norm: float | Mapping[str, float] | PerGroup,
    non_probe: dict[str, torch.Tensor],
) -> PerGroup:
    """The user's ``PerGroup`` over the gradient leaves only."""
    from opaque.api.engine.clipping._per_group import per_group

    if isinstance(clipping_norm, PerGroup):
        expected = {(k,) for k in non_probe}
        actual = set(clipping_norm.groups)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ConfigurationError(
                *(
                    "clipping_norm PerGroup must cover exactly the gradient "
                    "leaves (the trainable pytree minus the probe); "
                    f"missing={missing[:3]}, unexpected={extra[:3]}.",
                )
            )
        return clipping_norm
    if isinstance(clipping_norm, (int, float)):
        if clipping_norm <= 0:
            raise ConfigurationError(
                *(f"clipping_norm must be positive, got {clipping_norm}.",)
            )
        return per_group(non_probe, fallback=float(clipping_norm))
    try:
        patterns = dict(clipping_norm)
    except TypeError as exc:
        raise InputTypeError(
            *(
                "clipping_norm must be a float, a {pattern: value} mapping or "
                f"a PerGroup, got {type(clipping_norm).__name__}.",
            )
        ) from exc
    fallback = patterns.pop("fallback", None)
    return per_group(non_probe, patterns=patterns, fallback=fallback)


def probe_bounds(
    clipping_norm: float | Mapping[str, float] | PerGroup,
    trainable: Mapping[str, torch.Tensor],
    *,
    ratio: float,
    num_layers: int,
    num_experts: int,
    top_k: int,
    mean_tokens: float,
    max_tokens: float,
    guard: float = 1e-3,
    name: str = PROBE_NAME,
) -> tuple[PerGroup, float]:
    """Build the two-group clipping bound and the probe scale ``lam``.

    The user's clipping configuration is compiled over the trainable pytree
    minus the probe (a scalar becomes the single group ``"fallback"``; a
    mapping is passed to :func:`opaque.dpsgd.clipping.per_group` as
    substring patterns, with an optional ``"fallback"`` entry; a
    :class:`~opaque.types.PerGroup` must already cover exactly those
    leaves).  The probe is then added by direct construction, never by a
    substring pattern, so a user pattern such as ``"router"`` can coexist
    with the probe.

    With ``Delta_L`` from :func:`load_bound`::

        lam = ratio * C_g / Delta_L
        C_h = lam * Delta_L * (1 + guard) = ratio * C_g * (1 + guard)

    ``C_g`` is the ``"fallback"`` bound when that group exists and otherwise
    the largest gradient-group bound.  ``ratio`` is the share of the
    clipping budget given to the load group relative to the dominant
    gradient group; with several gradient groups the gradient-noise
    inflation is ``sqrt(1 + C_h / sum_g C_g) <= sqrt(1 + ratio)``.

    The probe's per-record contribution is ``lam * w_x * ||d^{(L,E)}(x)||
    <= lam * Delta_L``, strictly below ``C_h``; the guard absorbs the
    relative shrink the clipper applies to every ratio for floating-point
    safety, so the probe group is a bound that never clips and the release
    is unbiased.

    Args:
        clipping_norm: Gradient clipping configuration (see above).
        trainable: Trainable pytree (flat ``{name: tensor}``) including the
            probe under ``name``.
        ratio: Budget share ``C_h / C_g``.
        num_layers: Number of MoE layers ``L``.
        num_experts: Number of experts ``E``.
        top_k: Experts per token ``k``.
        mean_tokens: Public token-count constant ``T_bar`` of ``w_x``.
        max_tokens: Public row length ``T_max`` (``T_x <= T_max``).
        guard: Relative headroom on the probe bound.  The clipper shrinks
            every scale factor by a device-dependent round-off margin so a
            norm exactly at its bound is never scaled above one; the
            default ``1e-3`` exceeds the measured shrink on every backend
            (about ``2e-7`` on CPU / CUDA, ``1.3e-4`` on MPS) by at least
            an order of magnitude, which is what keeps the probe group a
            bound that never clips.
        name: Probe parameter and group name.

    Returns:
        ``(max_norm, lam)``: the ``PerGroup`` to hand to ``clipped_grad``
        and the probe scale.

    Raises:
        ConfigurationError: on an invalid configuration, when the probe is
            not in ``trainable``, or when no gradient leaf remains.
    """
    if name not in trainable:
        raise ConfigurationError(
            *(
                f"probe {name!r} is not in the trainable pytree; call "
                "attach_probe before make_functional(partition_trainable=True).",
            )
        )
    if ratio <= 0:
        raise ConfigurationError(*(f"ratio must be positive, got {ratio}.",))
    if guard < 0:
        raise ConfigurationError(*(f"guard must be non-negative, got {guard}.",))
    non_probe = {k: v for k, v in trainable.items() if k != name}
    if not non_probe:
        raise ConfigurationError(
            *("the trainable pytree has no gradient leaf besides the probe.",)
        )
    gradient_groups = _gradient_groups(clipping_norm, non_probe)
    if name in gradient_groups.values:
        raise ConfigurationError(
            *(
                f"the gradient clipping groups already use the name {name!r}; "
                "the probe group is reserved.",
            )
        )
    values = gradient_groups.values
    c_g = values["fallback"] if "fallback" in values else max(values.values())
    delta_l = load_bound(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        mean_tokens=mean_tokens,
        max_tokens=max_tokens,
    )
    lam = ratio * c_g / delta_l
    c_h = lam * delta_l * (1.0 + guard)
    max_norm = PerGroup(
        groups={**gradient_groups.groups, (name,): name},
        values={**values, name: c_h},
    )
    return max_norm, lam


# ---------------------------------------------------------------------------
# Seam 3: the per-example loss augmentation
# ---------------------------------------------------------------------------


def router_load_terms(
    loss: torch.Tensor,
    router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None,
    params: Mapping[str, torch.Tensor],
    *,
    f_tilde: torch.Tensor,
    alpha: float,
    lam: float,
    top_k: int,
    num_layers: int,
    num_experts: int,
    mean_tokens: float,
    probe_name: str = PROBE_NAME,
    z_loss_coef: float = 0.0,
) -> torch.Tensor:
    """Augment one example's loss with the surrogate and the probe term.

    Returns::

        loss + alpha * (S - sg[S]) [+ z_loss_coef * (Z - sg[Z])]
             + sum(params[probe] * sg[lam * w * (h_layers - k/E)])

    with ``w = T_x / mean_tokens``, ``S`` the load-balancing surrogate
    ``E * w * <f_tilde - k/E, P>`` and ``Z`` the router z-loss.  The added
    terms are value-neutral (each is exactly zero in value) and contribute
    only gradients: ``alpha * grad S`` (plus ``z_loss_coef * grad Z``) on
    the model parameters and ``lam * w * (h_layers - k/E)`` on the probe.
    A fully masked example has ``w = 0`` and contributes nothing.

    Args:
        loss: The example's scalar loss (for instance its token-mean CE).
        router_logits: Sequence of ``num_layers`` router logit tensors of
            shape ``(T, E)`` for this example.
        attention_mask: The example's attention mask, or ``None`` when every
            position counts.
        params: Parameter pytree containing the probe under ``probe_name``.
        f_tilde: Public load estimate ``(E,)`` for this step.
        alpha: Surrogate coefficient (the router aux-loss coefficient).
        lam: Probe scale from :func:`probe_bounds`.
        top_k: Experts per token ``k``.
        num_layers: Number of MoE layers ``L``.
        num_experts: Number of experts ``E``.
        mean_tokens: Public token-count constant ``T_bar``.
        probe_name: Key of the probe in ``params``.
        z_loss_coef: Router z-loss coefficient (``0`` disables the term).
    """
    from opaque.api.patches.transformers.components.moe_stats import (
        centred_load,
        load_balancing_surrogate,
        router_load_and_probs,
        router_z_loss,
    )

    h_layers, probs, n_tokens = router_load_and_probs(
        router_logits, attention_mask, top_k=top_k, num_layers=num_layers
    )
    weight = n_tokens / mean_tokens
    surrogate = load_balancing_surrogate(
        probs, f_tilde.to(probs), weight, num_experts=num_experts, top_k=top_k
    )
    out = loss + alpha * (surrogate - surrogate.detach())
    if z_loss_coef != 0.0:
        z_loss = router_z_loss(router_logits, attention_mask)
        out = out + z_loss_coef * (z_loss - z_loss.detach())
    probe = params[probe_name]
    target = (lam * weight * centred_load(h_layers, top_k=top_k)).detach()
    return out + (probe * target.to(probe.dtype)).sum()


# ---------------------------------------------------------------------------
# Seam 4: post-processing state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterLoadState:
    """Public post-processing state of the router-load release.

    All tensors are fp32.  The dataclass round-trips through
    :func:`opaque.serialization.state_dict` /
    :func:`opaque.serialization.from_state_dict` (tensors and primitives
    only).  ``tripped`` and ``alpha_active`` are owned by the consumer's
    decision rule; :func:`update` carries them through unchanged.

    Attributes:
        m_layers: Per-layer filter accumulator ``(L, E)`` of the projected
            per-layer releases (monitor only).
        m: Pooled filter accumulator ``(E,)``; the EMA numerator or the
            window sum.
        f_tilde: Public load estimate ``(E,)`` for the next step.
        noise_std: Known noise standard deviation ``s_t`` of the pooled
            bias-corrected estimate.
        phi: Filter factors from :func:`filter_factors`; ``phi[t - 1]`` is
            the factor of the ``t``-th release.
        step: Number of releases consumed.
        tripped: Monitor decision flag (consumer-owned).
        alpha_active: Surrogate coefficient in force (consumer-owned).
        kind: ``"ema"`` or ``"window"``.
        beta: EMA coefficient.
        window: Window length ``W``.
        ratio: Budget share ``C_h / C_g``.
        dead_zone: Dead-zone constant ``c``.
        shrink: Whether the dead zone and James-Stein shrinkage apply.
        top_k: Experts per token ``k``.
        num_experts: Number of experts ``E``.
        num_layers: Number of MoE layers ``L``.
        mean_tokens: Public token-count constant ``T_bar``.
        max_tokens: Public row length ``T_max``.
        lam: Probe scale.
        base_noise_std: Per-entry noise std of one pooled release before
            filtering, ``sigma_h / (lam * sqrt(L))``.
        buffer: Window ring buffer ``(W, E)`` (``(0,)`` for the EMA).
        buffer_layers: Per-layer window ring buffer ``(W, L, E)`` (``(0,)``
            for the EMA).
    """

    m_layers: torch.Tensor
    m: torch.Tensor
    f_tilde: torch.Tensor
    noise_std: float
    phi: torch.Tensor
    step: int
    tripped: bool
    alpha_active: float
    kind: str
    beta: float
    window: int
    ratio: float
    dead_zone: float
    shrink: bool
    top_k: int
    num_experts: int
    num_layers: int
    mean_tokens: float
    max_tokens: float
    lam: float
    base_noise_std: float
    buffer: torch.Tensor
    buffer_layers: torch.Tensor


def _filter_coefficients(
    n: int, *, kind: str, beta: float, window: int
) -> torch.Tensor:
    """Lower-triangular Toeplitz coefficients of the un-normalised filter.

    EMA: ``(1 - beta) * beta**j``; window: ``1`` for ``j < window``.  The
    normalisation (``1 - beta**t`` or ``min(t, W)``) is applied by
    :func:`_correction` on the estimate and its noise alike.
    """
    if kind == "ema":
        return (1.0 - beta) * beta ** torch.arange(n, dtype=torch.float64)
    coef = torch.zeros(n, dtype=torch.float64)
    coef[:window] = 1.0
    return coef


def _correction(kind: str, beta: float, window: int, t: int) -> float:
    """Bias-correction divisor of the ``t``-th filtered estimate."""
    if kind == "ema":
        return 1.0 - beta**t
    return float(min(t, window))


def _validate_filter(kind: str, beta: float, window: int) -> None:
    if kind not in _FILTER_KINDS:
        raise ConfigurationError(
            *(f"filter kind must be one of {_FILTER_KINDS}, got {kind!r}.",)
        )
    if kind == "ema" and not 0.0 < beta < 1.0:
        raise ConfigurationError(*(f"beta must be in (0, 1), got {beta}.",))
    if kind == "window" and window < 1:
        raise ConfigurationError(*(f"window must be >= 1, got {window}.",))


_CONSISTENCY_ROWS = 256


def _inverse_coefficients(strategy: Any, n_steps: int, n: int) -> torch.Tensor:
    """Toeplitz coefficients of ``C^{-1}`` for lags ``0..n-1``.

    Obtained from the strategy's ``coefficients(n_steps)`` by the triangular
    recursion (the streaming inverse), never by a dense solve.  The result
    is checked against the strategy's own noise operator on the first rows
    so a strategy whose noise is not the Toeplitz inverse of its
    coefficients is rejected rather than silently mis-filtered.
    """
    from opaque.api.dpftrl.noise._toeplitz import inverse_coef

    try:
        coef = strategy.coefficients(n_steps=n_steps)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
    except NotImplementedError as exc:
        raise ConfigurationError(
            *(
                f"filter_factors needs a Toeplitz strategy exposing "
                f"coefficients() and streaming_matrix(); {type(strategy).__name__} "
                "does not.",
            )
        ) from exc
    coef = torch.as_tensor(coef, dtype=torch.float64).detach().cpu()
    inv = inverse_coef(coef, n).detach().cpu().to(torch.float64)
    rows = min(n, _CONSISTENCY_ROWS)
    expected = streaming.row_norms_squared(rows).detach().cpu().to(torch.float64)
    actual = torch.cumsum(inv[:rows] ** 2, dim=0)
    if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-12):
        raise ConfigurationError(
            *(
                f"{type(strategy).__name__}: the noise operator's row norms do "
                "not match the Toeplitz inverse of coefficients(); filter_factors "
                "cannot describe its noise.",
            )
        )
    return inv


def filter_factors(
    strategy: Any,
    *,
    n_steps: int,
    kind: str,
    beta: float,
    window: int,
    num_experts: int,
    n_phi: int = 2048,
) -> torch.Tensor:
    """Noise factors ``phi_t`` of the filtered, projected release.

    The filtered estimate is a fixed linear filter ``F`` of the released
    stream ``d + sigma * C^{-1} Z``, so its noise std per entry is exactly
    ``sigma * ||row_t(F C^{-1})||``.  Both ``F`` and ``C^{-1}`` are lower
    triangular Toeplitz, so their product is Toeplitz with coefficients
    ``g = conv(f, c_inv)`` and::

        phi_t^2 = (E - 1) / E * sum_{j < t} g_j^2

    where the ``(E - 1) / E`` factor is the sum-zero projection.  ``f`` is
    the un-normalised filter (``(1 - beta) beta^j`` for the EMA, ones for
    the window); the bias correction ``1 - beta^t`` or ``min(t, W)`` is
    applied by :func:`update` to the estimate and to ``phi_t`` alike.

    With ``strategy=None`` (DP-SGD, ``C = I``) this is the closed-form
    recursion ``phi_t^2 = beta^2 phi_{t-1}^2 + (1 - beta)^2 (E - 1) / E``
    for the EMA and ``phi_t^2 = min(t, W) (E - 1) / E`` for the window.
    With an MF strategy the inverse coefficients come from the strategy's
    streaming Toeplitz inverse (never a dense ``n x n`` solve).

    Args:
        strategy: MF strategy recipe, or ``None`` for DP-SGD.
        n_steps: Training horizon (the amplifier's ``n_steps``).
        kind: ``"ema"`` or ``"window"``.
        beta: EMA coefficient.
        window: Window length ``W``.
        num_experts: Number of experts ``E``.
        n_phi: Number of factors to tabulate; ``phi[-1]`` is reused past it.

    Returns:
        Float64 tensor of length ``min(n_steps, n_phi)``; ``phi[t - 1]`` is
        the factor of the ``t``-th release.
    """
    _validate_filter(kind, beta, window)
    if n_steps < 1 or n_phi < 1:
        raise ConfigurationError(
            *(f"n_steps and n_phi must be >= 1, got {n_steps} and {n_phi}.",)
        )
    if num_experts < _MIN_EXPERTS:
        raise ConfigurationError(*(f"num_experts must be >= 2, got {num_experts}.",))
    n = min(n_steps, n_phi)
    f = _filter_coefficients(n, kind=kind, beta=beta, window=window)
    if strategy is None:
        g = f
    else:
        c_inv = _inverse_coefficients(strategy, n_steps, n)
        g = torch.from_numpy(np.convolve(f.numpy(), c_inv.numpy())[:n])
    phi = torch.sqrt(torch.cumsum(g**2, dim=0) * (num_experts - 1) / num_experts)
    if n_steps > n >= _STABILISATION_MIN_ROWS:
        tail = phi[-n // 10 :]
        spread = float((tail.max() - tail.min()) / tail[-1].clamp_min(1e-300))
        if spread > _STABILISATION_TOL:
            warnings.warn(
                f"filter_factors: phi has not stabilised by n_phi={n_phi} "
                f"(relative spread {spread:.2e} over the last {len(tail)} "
                f"entries); phi[-1] is reused for steps beyond it.",
                stacklevel=2,
            )
    return phi


def initial_state(  # noqa: PLR0913 - the fixed helper contract
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    ratio: float,
    lam: float,
    max_norm: PerGroup,
    noise_multiplier: float,
    kind: str = "ema",
    beta: float = 0.99,
    window: int = 256,
    dead_zone: float = 2.0,
    shrink: bool = True,
    alpha: float,
    mean_tokens: float,
    max_tokens: float,
    phi: torch.Tensor,
    name: str = PROBE_NAME,
) -> RouterLoadState:
    """Build the state before the first release (``f_tilde_0 = k/E``).

    ``max_norm`` is the per-step bound the clipped pytree carries
    (``ClippedPytree.max_norm``), that is the ``PerGroup`` returned by
    :func:`probe_bounds` divided by the ``normalize_by`` of ``clipped_grad``
    (the expected batch size).  The base noise std of one pooled release
    is ``per_group_noise_stddev(max_norm, noise_multiplier)[name] / (lam *
    sqrt(L))``.
    """
    _validate_filter(kind, beta, window)
    if name not in max_norm.values:
        raise ConfigurationError(
            *(f"max_norm has no group {name!r}; build it with probe_bounds.",)
        )
    if lam <= 0:
        raise ConfigurationError(*(f"lam must be positive, got {lam}.",))
    if dead_zone < 0:
        raise ConfigurationError(*(f"dead_zone must be >= 0, got {dead_zone}.",))
    if not 1 <= top_k < num_experts:
        raise ConfigurationError(
            *(f"need 1 <= top_k < num_experts, got {top_k} and {num_experts}.",)
        )
    sigma_h = per_group_noise_stddev(max_norm, noise_multiplier).values[name]
    base_noise_std = sigma_h / (lam * math.sqrt(num_layers))
    phi = torch.as_tensor(phi, dtype=torch.float64).detach().cpu().clone()
    if phi.ndim != 1 or phi.numel() < 1:
        raise ConfigurationError(*("phi must be a non-empty 1-D tensor.",))
    width = window if kind == "window" else 0
    return RouterLoadState(
        m_layers=torch.zeros(num_layers, num_experts),
        m=torch.zeros(num_experts),
        f_tilde=torch.full((num_experts,), top_k / num_experts),
        noise_std=0.0,
        phi=phi,
        step=0,
        tripped=False,
        alpha_active=float(alpha),
        kind=kind,
        beta=float(beta),
        window=int(window),
        ratio=float(ratio),
        dead_zone=float(dead_zone),
        shrink=bool(shrink),
        top_k=int(top_k),
        num_experts=int(num_experts),
        num_layers=int(num_layers),
        mean_tokens=float(mean_tokens),
        max_tokens=float(max_tokens),
        lam=float(lam),
        base_noise_std=float(base_noise_std),
        buffer=torch.zeros(width, num_experts) if width else torch.zeros(0),
        buffer_layers=(
            torch.zeros(width, num_layers, num_experts) if width else torch.zeros(0)
        ),
    )


def _shrink(
    d_tilde: torch.Tensor, noise_std: float, *, num_experts: int, c: float
) -> tuple[torch.Tensor, float, bool]:
    """Dead zone then positive-part James-Stein shrinkage toward balance.

    Returns ``(d_plus, factor, dead)``: ``d_plus = 0`` when ``||d||^2 < c E
    s^2`` (dead zone), otherwise ``d * (1 - E s^2 / ||d||^2)``.
    """
    n2 = float(d_tilde.double().pow(2).sum())
    threshold = c * num_experts * noise_std * noise_std
    if n2 == 0.0 or n2 < threshold:
        return torch.zeros_like(d_tilde), 0.0, True
    factor = 1.0 - num_experts * noise_std * noise_std / n2
    return d_tilde * factor, factor, False


def update(state: RouterLoadState, noised_probe_leaf: torch.Tensor) -> RouterLoadState:
    """Consume one noised probe leaf and produce ``f_tilde`` for the next step.

    Post-processing of the public release ``y_t`` (already divided by the
    expected batch size by the clipper):

    1. ``d_hat^{(L,E)} = y_t / lam``;
    2. pooled ``d_hat = mean_l d_hat^{(L,E)}``;
    3. sum-zero projection ``d_hat -= mean_e d_hat`` (per layer likewise
       for the monitor);
    4. filter: EMA ``m = beta m + (1 - beta) d_hat`` with bias correction
       ``d_tilde = m / (1 - beta^t)``, or window sum with ``d_tilde = m /
       min(t, W)``; known noise std ``s_t = base_noise_std * phi_t /
       correction``;
    5. dead zone ``||d_tilde||^2 < dead_zone * E * s_t^2 -> 0``, otherwise
       the James-Stein factor ``1 - E s_t^2 / ||d_tilde||^2`` (skipped when
       ``shrink`` is off);
    6. ``f_tilde = clamp(k/E + d_plus, 0, 1)``;
    7. the per-layer accumulator feeds the layer monitor in :func:`summary`.

    The consumer zeroes the probe leaf of the noised pytree after reading
    it so the optimizer's update of the probe is exactly zero.
    """
    L, E = state.num_layers, state.num_experts
    y = noised_probe_leaf.detach().to(device=state.m.device, dtype=torch.float32)
    if tuple(y.shape) != (L, E):
        raise ConfigurationError(
            *(f"noised probe leaf has shape {tuple(y.shape)}, expected {(L, E)}.",)
        )
    t = state.step + 1
    d_hat_layers = y / state.lam
    d_hat = d_hat_layers.mean(0)
    d_hat = d_hat - d_hat.mean()
    d_hat_layers = d_hat_layers - d_hat_layers.mean(-1, keepdim=True)
    if state.kind == "ema":
        m = state.beta * state.m + (1.0 - state.beta) * d_hat
        m_layers = state.beta * state.m_layers + (1.0 - state.beta) * d_hat_layers
        buffer, buffer_layers = state.buffer, state.buffer_layers
    else:
        slot = (t - 1) % state.window
        buffer = state.buffer.clone()
        buffer_layers = state.buffer_layers.clone()
        buffer[slot] = d_hat
        buffer_layers[slot] = d_hat_layers
        m = buffer.sum(0)
        m_layers = buffer_layers.sum(0)
    correction = _correction(state.kind, state.beta, state.window, t)
    d_tilde = m / correction
    phi_t = float(state.phi[min(t, state.phi.numel()) - 1])
    noise_std = state.base_noise_std * phi_t / correction
    if state.shrink:
        d_plus, _, _ = _shrink(d_tilde, noise_std, num_experts=E, c=state.dead_zone)
    else:
        d_plus = d_tilde
    f_tilde = torch.clamp(state.top_k / E + d_plus, 0.0, 1.0)
    return replace(
        state,
        m_layers=m_layers,
        m=m,
        f_tilde=f_tilde,
        noise_std=float(noise_std),
        step=t,
        buffer=buffer,
        buffer_layers=buffer_layers,
    )


def summary(state: RouterLoadState) -> dict[str, float]:
    """Public monitor curves of the current state.

    ``router_load/D`` is ``max_e |d_tilde_e| / (k/E)`` from the pooled
    bias-corrected estimate, ``router_load/D_layer_max`` the same over the
    per-layer estimates, ``router_load/entropy`` the entropy (nats) of
    ``f_tilde`` normalised to a distribution, ``router_load/shrink`` the
    James-Stein factor (``0`` inside the dead zone, ``1`` when shrinkage is
    off), ``router_load/dead_zone`` the dead-zone flag,
    ``router_load/noise_std`` ``s_t`` and ``router_load/tripped`` the
    consumer's flag.
    """
    E = state.num_experts
    share = state.top_k / E
    if state.step == 0:
        d_tilde = torch.zeros_like(state.m)
        d_layers = torch.zeros_like(state.m_layers)
    else:
        correction = _correction(state.kind, state.beta, state.window, state.step)
        d_tilde = state.m / correction
        d_layers = state.m_layers / correction
    if state.shrink:
        _, factor, dead = _shrink(
            d_tilde, state.noise_std, num_experts=E, c=state.dead_zone
        )
    else:
        factor, dead = 1.0, False
    f = state.f_tilde.double()
    total = float(f.sum())
    if total > 0:
        p = f / total
        entropy = float(-(p * torch.log(p.clamp_min(1e-300))).sum())
    else:
        entropy = 0.0
    return {
        "router_load/D": float(d_tilde.abs().max() / share),
        "router_load/D_layer_max": float(d_layers.abs().max() / share),
        "router_load/f_min": float(f.min()),
        "router_load/f_max": float(f.max()),
        "router_load/entropy": entropy,
        "router_load/shrink": float(factor),
        "router_load/dead_zone": float(dead),
        "router_load/noise_std": float(state.noise_std),
        "router_load/tripped": float(state.tripped),
    }


def telemetry_without_probe(
    aux: Any, max_norm: PerGroup, probe_name: str = PROBE_NAME
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example gradient norms over the non-probe groups.

    ``grad_norm`` is ``sqrt(sum_{g != probe} group_norms_g^2)`` and
    ``clipped_grad_norm`` is ``sqrt(sum_{g != probe} min(group_norms_g,
    C_g)^2)`` with ``C_g`` the group's bound in ``max_norm``.  Both are
    computed from the per-group norms of the gradient leaves alone, so
    they are bit-identical between runs that differ only in the probe
    scale: the engine's total ``clipped_grad_norms`` is a floating-point
    norm over the whole pytree, probe included, and subtracting the probe
    afterwards would leave its rounding behind.  The clipper's relative
    guard shrink makes this an over-estimate of the clipped norm by less
    than one part in ``1e6``, which is irrelevant for telemetry.

    Args:
        aux: ``ClippedGradAux`` from ``clipped_grad(..., return_aux=True)``
            with per-group norms (``PerGroup`` clipping).
        max_norm: The ``PerGroup`` handed to ``clipped_grad`` (the
            un-normalised bounds returned by :func:`probe_bounds`).
        probe_name: Group name of the probe.

    Returns:
        ``(grad_norm, clipped_grad_norm)`` per-example tensors.
    """
    group_norms = getattr(aux, "group_norms", None)
    if not group_norms or probe_name not in group_norms:
        raise ConfigurationError(
            *(
                "telemetry_without_probe needs ClippedGradAux.group_norms with a "
                f"{probe_name!r} group (per-group clipping).",
            )
        )
    bounds = getattr(max_norm, "values", None)
    if bounds is None:
        raise InputTypeError(
            *(
                "telemetry_without_probe needs the PerGroup bound handed to "
                f"clipped_grad, got {type(max_norm).__name__}.",
            )
        )
    missing = sorted(k for k in group_norms if k != probe_name and k not in bounds)
    if missing:
        raise ConfigurationError(
            *(f"max_norm has no bound for the gradient groups {missing[:3]}.",)
        )
    others = {k: v.double() for k, v in group_norms.items() if k != probe_name}
    raw = torch.stack(list(others.values()), 0)
    clipped_groups = torch.stack(
        [torch.clamp_max(v, float(bounds[k])) for k, v in others.items()], 0
    )
    grad_norm = torch.sqrt(raw.pow(2).sum(0))
    clipped = torch.sqrt(clipped_groups.pow(2).sum(0))
    dtype = aux.clipped_grad_norms.dtype
    return grad_norm.to(dtype), clipped.to(dtype)


# ---------------------------------------------------------------------------
# Shared consumer helpers: geometry, decision rule, resume compatibility
# ---------------------------------------------------------------------------


def is_router_module(module: nn.Module) -> bool:
    """Whether ``module`` is a top-k router the backbone records logits for.

    The single router predicate of
    :mod:`opaque.api.patches.transformers.components.router` (class name
    containing ``"TopKRouter"`` or the stock router attributes), shared with
    the fp32-router installer so every consumer counts the same modules.
    """
    from opaque.api.patches.transformers.components.router import (
        is_router_module as _is_router,
    )

    return _is_router(module)


def resolve_moe_geometry(model: nn.Module) -> tuple[int, int, int, float]:
    """``(num_layers, num_experts, top_k, config_aux_coef)`` of a MoE model.

    ``num_layers`` counts the router modules the backbone records logits
    for (dense layers of a mixed model do not count); ``num_experts`` and
    ``top_k`` come from ``model.config`` (``num_experts`` /
    ``num_experts_per_tok``) or, when the config lacks them, from the
    routers themselves.  ``config_aux_coef`` is the checkpoint's own
    ``router_aux_loss_coef`` (``0`` when absent).

    Raises:
        ConfigurationError: for a family without a router, or when the
            routers disagree on ``(num_experts, top_k)``.
    """
    config = getattr(model, "config", None)
    routers = [module for module in model.modules() if is_router_module(module)]
    num_experts = getattr(config, "num_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if routers and (num_experts is None or top_k is None):
        shapes = {
            (int(getattr(r, "num_experts", 0)), int(getattr(r, "top_k", 0)))
            for r in routers
        }
        if len(shapes) != 1:
            raise ConfigurationError(
                *(f"routers disagree on (num_experts, top_k): {sorted(shapes)}.",)
            )
        num_experts, top_k = next(iter(shapes))
    if not routers or not num_experts or not top_k:
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
    return len(routers), int(num_experts), int(top_k), aux_coef


def monitor_value(state: RouterLoadState) -> float:
    """``D_t = max_e |f_tilde_e - k/E| / (k/E)`` of the estimate in force.

    This is the deviation of the estimate that enters the surrogate: after
    the dead zone and the shrinkage when ``state.shrink`` is on (so a
    noise-dominated early estimate reads as ``0``), the raw bias-corrected
    ``router_load/D`` otherwise.
    """
    share = state.top_k / state.num_experts
    return float((state.f_tilde - share).abs().max() / share)


def decide_trip(
    state: RouterLoadState,
    streak: int,
    *,
    trip: float,
    mode: str,
    alpha: float,
) -> tuple[RouterLoadState, int]:
    """The monitor decision rule, shared by every consumer of the state.

    Evaluates :func:`monitor_value` once (call it at the logging cadence):
    ``D > trip`` extends the streak of consecutive evaluations over the
    threshold, anything else resets it.  Two consecutive evaluations over
    the threshold set ``tripped``; in ``"monitor_then_surrogate"`` the trip
    also switches ``alpha_active`` to ``alpha``.  A state that already
    tripped is returned unchanged apart from the streak.  Post-processing
    of previous releases only: the bound, the noise and the accountant are
    untouched.

    Args:
        state: State after the release just consumed.
        streak: Consecutive logged evaluations over the threshold so far.
        trip: Threshold ``tau`` on ``D``.
        mode: ``"monitor"``, ``"surrogate"`` or ``"monitor_then_surrogate"``.
        alpha: Surrogate coefficient to switch on at the trip.

    Returns:
        ``(state, streak)`` after the evaluation.
    """
    streak = streak + 1 if monitor_value(state) > trip else 0
    if streak < 2 or state.tripped:  # noqa: PLR2004 - two consecutive evaluations
        return state, streak
    alpha_active = alpha if mode == "monitor_then_surrogate" else state.alpha_active
    return replace(state, tripped=True, alpha_active=alpha_active), streak


RESUME_MATCH_FIELDS: tuple[str, ...] = (
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
"""Fields of :class:`RouterLoadState` that must agree across a resume."""

_RESUME_RELATIVE_TOLERANCE = 1e-9


def _values_differ(saved: Any, current: Any) -> bool:
    if isinstance(saved, float) or isinstance(current, float):
        saved_f, current_f = float(saved), float(current)
        scale = max(abs(saved_f), abs(current_f), 1e-300)
        return abs(saved_f - current_f) > _RESUME_RELATIVE_TOLERANCE * scale
    return saved != current


def check_resume_compatible(saved: RouterLoadState, current: RouterLoadState) -> None:
    """Reject a saved state that describes a different release than ``current``.

    The public constants of :data:`RESUME_MATCH_FIELDS` (``ratio``, the
    filter, the dead zone, ``E``, ``k``, ``L``, ``mean_tokens``,
    ``max_tokens`` and the probe scale ``lam``, which encodes the gradient
    bound) and the filter factors ``phi`` (which encode the noise operator
    and, under a matrix mechanism, the horizon) must match, otherwise the
    continued filter would mix releases of two different mechanisms.
    ``base_noise_std`` is deliberately not compared: a target-epsilon run
    re-calibrates the remaining steps on resume, and the consumer decides
    how to carry the new value.

    Raises:
        CheckpointError: naming every mismatched field.
    """
    mismatched = [
        f"{name}: saved={getattr(saved, name)!r}, current={getattr(current, name)!r}"
        for name in RESUME_MATCH_FIELDS
        if _values_differ(getattr(saved, name), getattr(current, name))
    ]
    if tuple(saved.phi.shape) != tuple(current.phi.shape) or not torch.allclose(
        saved.phi.double(), current.phi.double(), rtol=1e-6, atol=1e-12
    ):
        mismatched.append(
            "phi: the filter factors differ (a different noise operator or "
            "matrix-mechanism horizon)"
        )
    if mismatched:
        raise CheckpointError(
            *(
                "router_load_release configuration drift on resume; the saved "
                "post-processing state was built for a different release: "
                + "; ".join(mismatched)
                + ". Restart from scratch to change these settings.",
            )
        )


__all__ = [
    "PROBE_NAME",
    "RESUME_MATCH_FIELDS",
    "RouterLoadState",
    "attach_probe",
    "check_resume_compatible",
    "decide_trip",
    "filter_factors",
    "initial_state",
    "is_router_module",
    "load_bound",
    "monitor_value",
    "probe_bounds",
    "resolve_moe_geometry",
    "router_load_terms",
    "summary",
    "telemetry_without_probe",
    "update",
]
