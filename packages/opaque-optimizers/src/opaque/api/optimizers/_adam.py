"""Backend-neutral Adam / AdamW factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from opaque.api.engine import ops
from opaque.api.engine.pytree import ParamPath, tree_flatten_with_paths, tree_unflatten
from opaque.api.optimizers._chain import make_optimizer_chain
from opaque.api.optimizers.types import AdamState
from opaque.pytree import tree_map
from opaque.types import PerGroup

_LR = float | Callable[[int], float]


def _init_per_group_phi(params: Any) -> dict[ParamPath, float]:
    paths, _leaves, _treedef = tree_flatten_with_paths(params)
    del _leaves, _treedef
    return dict.fromkeys(paths, 0.0)


def _resolve_noise_variance(
    noise_stddev: float | PerGroup,
    path: ParamPath | None = None,
) -> float:
    if isinstance(noise_stddev, PerGroup):
        if path is None:
            raise ValueError(
                "resolve_noise_variance requires `path` for PerGroup noise_stddev"
            )
        return float(noise_stddev.for_path(path)) ** 2
    return float(noise_stddev) ** 2


def _is_per_group(noise_stddev: float | PerGroup) -> bool:
    return isinstance(noise_stddev, PerGroup)


def _map_leaves_with_path(
    fn: Callable[..., Any],
    tree: Any,
    *others: Any,
) -> Any:
    """Apply ``fn(path, leaf, *other_leaves)`` and rebuild ``tree``'s structure."""
    paths, leaves, treedef = tree_flatten_with_paths(tree)
    other_flat = [tree_flatten_with_paths(t) for t in others]
    for i, (other_paths, other_leaves, _) in enumerate(other_flat):
        if other_paths != paths:
            raise ValueError(
                f"pytree ParamPath mismatch for argument {i}: "
                f"primary paths {paths!r}, got {other_paths!r}."
            )
        if len(other_leaves) != len(leaves):
            raise ValueError(
                f"pytree leaf count mismatch: primary has {len(leaves)}, "
                f"argument {i} has {len(other_leaves)}"
            )
    out_leaves = []
    for j, path in enumerate(paths):
        args = [leaves[j], *[flat[1][j] for flat in other_flat]]
        out_leaves.append(fn(path, *args))
    return tree_unflatten(treedef, out_leaves)


def _scale_by_adam(
    b1: float,
    b2: float,
    eps: float,
    noise_bias_correction: bool,
    bc_floor: float,
) -> Callable[..., tuple[Any, AdamState]]:
    """Adam moment scaling with optional DP bias correction or private second moments."""

    def step(
        updates: Any,
        state: AdamState,
        params: Any,
        noise_stddev: float | PerGroup | None = None,
        noisy_squared_grads: Any | None = None,
    ) -> tuple[Any, AdamState]:
        del params
        # Second-moment stream takes precedence; a NoisedPytree on the
        # first stream may still carry noise_stddev, but it is ignored
        # when a private second-moment stream is supplied.

        t = state.step + 1
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t

        new_mu = tree_map(
            lambda m, g: ops.add(ops.multiply(m, b1), ops.multiply(g, 1.0 - b1)),
            state.mu,
            updates,
        )

        if noisy_squared_grads is not None:
            new_nu = tree_map(
                lambda v, g2: ops.add(ops.multiply(v, b2), ops.multiply(g2, 1.0 - b2)),
                state.nu,
                noisy_squared_grads,
            )
            new_phi = state.phi

            def _compute_sm(m: Any, v: Any) -> Any:
                m_hat = ops.divide(m, bc1)
                v_hat = ops.divide(v, bc2)
                v_eff = ops.where(
                    ops.greater(v_hat, 0.0),
                    v_hat,
                    ops.square(m_hat),
                )
                v_eff = ops.clamp(v_eff, lo=bc_floor)
                return ops.divide(m_hat, ops.add(ops.sqrt(v_eff), eps))

            result = tree_map(_compute_sm, new_mu, new_nu)
            return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

        new_nu = tree_map(
            lambda v, g: ops.add(
                ops.multiply(v, b2),
                ops.multiply(ops.square(g), 1.0 - b2),
            ),
            state.nu,
            updates,
        )

        effective_stddev = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:

            def _vanilla(m: Any, v: Any) -> Any:
                return ops.divide(
                    ops.divide(m, bc1),
                    ops.add(ops.sqrt(ops.divide(v, bc2)), eps),
                )

            result = tree_map(_vanilla, new_mu, new_nu)
            return result, AdamState(mu=new_mu, nu=new_nu, phi=state.phi, step=t)

        per_group = _is_per_group(effective_stddev) or isinstance(state.phi, dict)

        if per_group:
            new_phi: dict = {}

            def _bc_leaf(path: ParamPath, mu_node: Any, nu_node: Any) -> Any:
                nv = _resolve_noise_variance(effective_stddev, path)
                old_phi_k = (
                    state.phi.get(path, 0.0)  # type: ignore[union-attr]
                    if isinstance(state.phi, dict)
                    else state.phi
                )
                new_phi_k = b2 * old_phi_k + (1.0 - b2) * nv
                new_phi[path] = new_phi_k
                m_hat = ops.divide(mu_node, bc1)
                phi_hat = ops.divide(ops.scalar(new_phi_k, like=nu_node), bc2)
                v_raw = ops.divide(nu_node, bc2)
                v_hat = ops.where(
                    ops.greater(phi_hat, 0.0),
                    ops.where(
                        ops.greater(ops.subtract(v_raw, phi_hat), 0.0),
                        ops.subtract(v_raw, phi_hat),
                        v_raw,
                    ),
                    v_raw,
                )
                return ops.divide(m_hat, ops.add(ops.sqrt(v_hat), eps))

            result = _map_leaves_with_path(_bc_leaf, new_mu, new_nu)
        else:
            scalar_var = float(effective_stddev) ** 2
            if isinstance(state.phi, dict):
                raise TypeError(
                    "phi is per-group dict but noise_stddev is scalar; "
                    "either both must be per-group or both must be scalar."
                )
            new_phi_value = b2 * state.phi + (1.0 - b2) * scalar_var
            phi_hat = new_phi_value / bc2

            if phi_hat > 0.0:

                def _bc_scalar(m: Any, v: Any) -> Any:
                    v_hat = ops.divide(v, bc2)
                    corrected = ops.subtract(v_hat, phi_hat)
                    denom = ops.add(
                        ops.sqrt(
                            ops.where(ops.greater(corrected, 0.0), corrected, v_hat)
                        ),
                        eps,
                    )
                    return ops.divide(ops.divide(m, bc1), denom)

            else:

                def _bc_scalar(m: Any, v: Any) -> Any:
                    return ops.divide(
                        ops.divide(m, bc1),
                        ops.add(ops.sqrt(ops.divide(v, bc2)), eps),
                    )

            result = tree_map(_bc_scalar, new_mu, new_nu)
            new_phi = new_phi_value

        return result, AdamState(mu=new_mu, nu=new_nu, phi=new_phi, step=t)

    return step


def adam(
    params: Any,
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdamState]], AdamState]:
    """Create an Adam optimizer (L2 weight decay) with Opaque's wrapper-aware API."""
    return adamw(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        decoupled_weight_decay=False,
        update_rms_clip=update_rms_clip,
        noise_bias_correction=noise_bias_correction,
    )


def adamw(
    params: Any,
    lr: _LR = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
    noise_bias_correction: bool = False,
) -> tuple[Callable[..., tuple[Any, AdamState]], AdamState]:
    """Universal Adam / AdamW factory.

    Args:
        params: Parameter pytree used to initialise optimizer state.
        lr: Learning rate, scalar or ``step -> float`` callable schedule.
        betas: ``(β₁, β₂)`` coefficients for first / second moment EMAs.
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient (decoupled by default).
        decoupled_weight_decay: ``True`` selects AdamW (decoupled WD).
            ``False`` selects the original Adam, where ``weight_decay * params``
            is folded into the gradient before moment scaling.
        update_rms_clip: When not ``None``, divides the moment-scaled
            update by ``max(1, rms / threshold)`` (StableAdamW).
        noise_bias_correction: If ``True``, subtract a β₂-EMA of the
            realized noise variance from the second moment when
            ``NoisedPytree`` updates are passed (DP-AdamW-BC).

    Returns:
        ``(step_fn, AdamState)``.
    """
    _validate_adam(eps, betas, weight_decay, update_rms_clip)

    mu = tree_map(lambda p: ops.zeros_like(p), params)
    nu = tree_map(lambda p: ops.zeros_like(p), params)
    phi: float | dict[ParamPath, float] = (
        _init_per_group_phi(params) if noise_bias_correction else 0.0
    )
    init_state = AdamState(mu=mu, nu=nu, phi=phi, step=0)

    bc_floor = eps * eps
    moment_step = _scale_by_adam(
        b1=betas[0],
        b2=betas[1],
        eps=eps,
        noise_bias_correction=noise_bias_correction,
        bc_floor=bc_floor,
    )

    return make_optimizer_chain(
        moment_step=moment_step,
        moment_init_state=init_state,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=update_rms_clip,
    )


def _validate_adam(
    eps: float,
    betas: tuple[float, float],
    weight_decay: float,
    update_rms_clip: float | None,
) -> None:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if len(betas) != 2:
        raise ValueError(f"betas must contain exactly two values, got {betas}")
    b1, b2 = betas
    if not 0 <= b1 < 1:
        raise ValueError(f"beta_1 must satisfy 0 <= beta_1 < 1, got {b1}")
    if not 0 <= b2 < 1:
        raise ValueError(f"beta_2 must satisfy 0 <= beta_2 < 1, got {b2}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if update_rms_clip is not None and update_rms_clip <= 0:
        raise ValueError(
            f"update_rms_clip must be positive when set, got {update_rms_clip}"
        )


__all__ = ["adam", "adamw"]
