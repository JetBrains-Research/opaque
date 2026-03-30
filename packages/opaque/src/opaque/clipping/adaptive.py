"""Adaptive gradient clipping with explicit state-passing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from opaque.clipping.clipped_fun import ClippedFunAux
from opaque.clipping.clipped_grad import clipped_grad
from opaque.clipping.types import ClipState
from opaque.random import RngKey, fold_in, generator_from_key

_DEFAULT_FRACTION_NOISE_STD = 0.05


@dataclass(frozen=True)
class AdaptiveClippedGradAux(ClippedFunAux):
    """Diagnostic outputs from adaptive_clipped_grad.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use
    ``ClipState.sensitivity`` for noise calibration.

    Inherits all fields from :class:`ClippedFunAux`.
    """


@dataclass(frozen=True)
class AdaptiveClipState(ClipState):
    """Immutable state for adaptive gradient clipping.

    This state is passed explicitly to the clipping function and returned
    as part of the output, enabling pure functional composition.

    Public attributes (for monitoring and noise calibration):
        clipping_norm: Raw clipping threshold used at the current step.
        normalize_by: Divisor applied to the clipped gradient sum
            (1.0 = no averaging).  Controls gradient sensitivity:
            ``sensitivity = clipping_norm / normalize_by``.  Also used as
            the denominator when computing the clipping fraction
            (when > 1); pass the same value to
            ``acc.adaclip(expected_batch_size=...)``.
        sensitivity: ``clipping_norm / normalize_by`` (property from
            :class:`ClipState`).
        next_clipping_norm: Clipping threshold for the *next* step C_{t+1}.
        step: Step counter.

    Internal attributes (carry config/counts for distributed sync —
    do not read directly):
        _key, _fraction_noise_std, _learning_rate, _target_quantile,
        _clipping_norm_min, _clipping_norm_max: Config constants replicated so
        that ``sync_adaptive_clip_state`` can recompute ``next_clipping_norm``
        from globally aggregated counts.
        _num_clipped, _batch_size: Raw local counts summed across ranks
        during distributed aggregation.
    """

    # -- public --
    clipping_norm: float
    normalize_by: float
    next_clipping_norm: float
    step: int

    # -- internal (config carried for distributed sync) --
    _key: RngKey
    _fraction_noise_std: float
    _learning_rate: float
    _target_quantile: float
    _clipping_norm_min: float
    _clipping_norm_max: float

    # -- internal (per-step counts for distributed aggregation) --
    _num_clipped: float
    _batch_size: float

    def __post_init__(self):
        """Validate state values."""
        if self.next_clipping_norm <= 0:
            raise ValueError(f"next_clipping_norm must be positive, got {self.next_clipping_norm}")
        if self.normalize_by <= 0:
            raise ValueError(f"normalize_by must be positive, got {self.normalize_by}")
        if self._fraction_noise_std <= 0:
            raise ValueError(
                "fraction_noise_std must be > 0, "
                f"got {self._fraction_noise_std}"
            )


def _sample_noisy_clipping_rate(
    clipping_rate: float,
    *,
    key: RngKey,
    step: int,
    fraction_noise_std: float,
) -> float:
    """Add DP Gaussian noise to clipping rate using step-folded RNG key."""
    step_key = fold_in(key, step)
    generator = generator_from_key(step_key)
    noise = torch.randn(1, generator=generator).item() * fraction_noise_std
    return clipping_rate + noise


def _adaptive_clipping_norm_update(
    *,
    base_clipping_norm: float,
    noisy_clipping_rate: float,
    target_quantile: float,
    learning_rate: float,
    clipping_norm_min: float,
    clipping_norm_max: float,
) -> float:
    """Compute geometric adaptive clipping update: C * exp(η * (ρ̃ - γ))."""
    update_factor = torch.exp(
        torch.tensor(learning_rate * (noisy_clipping_rate - target_quantile))
    ).item()
    new_clipping_norm = base_clipping_norm * update_factor
    return float(max(clipping_norm_min, min(clipping_norm_max, new_clipping_norm)))


def adaptive_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    initial_clipping_norm: float = 0.1,
    target_quantile: float = 0.5,
    learning_rate: float = 0.2,
    clipping_norm_min: float = 0.01,
    clipping_norm_max: float = 100.0,
    fraction_noise_std: float = _DEFAULT_FRACTION_NOISE_STD,
    key: RngKey,
    return_aux: bool = False,
    **clipped_grad_kwargs: Any,
) -> tuple[Callable, AdaptiveClipState]:
    """Create function for adaptive gradient clipping with explicit state-passing.

    Returns ``(clipped_grad_fn, initial_state)``.  The returned function
    takes ``state`` as an explicit parameter and returns
    ``(grad, new_state)`` or ``((grad, aux), new_state)``.

    Args:
        loss_fn: Loss function (scalar output). If ``has_aux``, returns
            ``(scalar, loss_aux)``.
        argnums: Which argument(s) to differentiate w.r.t.
        has_aux: If True, ``loss_fn`` returns ``(value, loss_aux)``.
        initial_clipping_norm: Initial clipping threshold C_0.
        target_quantile: Target fraction of clipped gradients.
        learning_rate: Step size for geometric adaptation.
        clipping_norm_min: Minimum allowed clipping threshold.
        clipping_norm_max: Maximum allowed clipping threshold.
        fraction_noise_std: Std of Gaussian noise added to the clipping
            fraction (default 0.05).
        key: RNG key for quantile noise generation.
        return_aux: If True, return per-example aux with loss values,
            gradient norms, and clipping rate.
        **clipped_grad_kwargs: Passed to ``clipped_grad()``
            (``batch_argnums``, ``normalize_by``, etc).  When
            ``normalize_by > 1`` it is also used as the fraction
            denominator.  Pass the same value to
            ``acc.adaclip(expected_batch_size=...)``.

    Returns:
        A tuple of (clipped_grad_fn, initial_state) where:
            - clipped_grad_fn: Function with signature
                (*args, state, **kwargs) -> (grad, new_state) or
                (*args, state, **kwargs) -> ((grad, aux), new_state)
            - initial_state: Initial AdaptiveClipState

    Example (single-device or distributed):
        >>> import torch
        >>> from opaque.clipping import adaptive_clipped_grad
        >>> from opaque.noise import gaussian_noise
        >>> from opaque.random import key
        >>> import torchopt
        >>>
        >>> def loss_fn(params, x, y):
        ...     pred = x @ params
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Create adaptive clipping function with initial state
        >>> # (automatically detects if distributed!)
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     initial_clipping_norm=0.1,
        ...     target_quantile=0.5,
        ...     key=key(0),
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Training loop with explicit state-passing
        >>> params = torch.randn(10, requires_grad=False)
        >>> optimizer = torchopt.adamw(lr=1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1, key=key(1))
        >>> for batch_x, batch_y in dataloader:
        ...     # Compute clipped gradients - state passed explicitly
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...
        ...     # Add DP noise scaled to current threshold
        ...     noisy_grad, noise_state = noise_fn(grad, noise_state)
        ...
        ...     # Optimizer step
        ...     updates, opt_state = optimizer.update(noisy_grad, opt_state, params=params)
        ...     params = torchopt.apply_updates(params, updates)
        ...
        ...     # Monitor adaptation
        ...     if clip_state.step % 100 == 0:
        ...         print(f"Step {clip_state.step}: C={clip_state.clipping_norm:.4f}")

    Example with distributed training (DDP with Poisson sampling):
        >>> import torch.distributed as dist
        >>> from opaque.clipping import adaptive_clipped_grad, sync_adaptive_clip_state
        >>> from opaque.distributed import sum_gradients
        >>> from opaque.random import key
        >>> from opaque.sampling import PoissonSampler
        >>>
        >>> # Initialize distributed
        >>> dist.init_process_group(backend='nccl')
        >>>
        >>> # Create adaptive clipping (local-only function)
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     key=key(0),
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Use Poisson sampling (different batch sizes on each device)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, key=key(42))
        >>>
        >>> for batch_x, batch_y in dataloader:
        ...     # Each device: compute clipped gradients and local adaptive state
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...     # Explicit sync step for adaptive clipping state
        ...     clip_state = sync_adaptive_clip_state(clip_state)
        ...
        ...     # Sum clipped gradients across devices
        ...     grad = sum_gradients(grad)
        ...
        ...     # Add noise and update
        ...     noise_fn, noise_state = gaussian_noise(
        ...         stddev=clip_state.sensitivity * 1.1, key=key(2))
        ...     noisy_grad, noise_state = noise_fn(grad, noise_state)
        ...     # ... optimizer step

    References:
        Andrew et al., "Differentially Private Learning with Adaptive
        Clipping", NeurIPS 2021.
    """
    # Validate parameters
    if initial_clipping_norm <= 0:
        raise ValueError(f"initial_clipping_norm must be positive, got {initial_clipping_norm}")
    if not 0 < target_quantile < 1:
        raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if clipping_norm_min <= 0:
        raise ValueError(f"clipping_norm_min must be positive, got {clipping_norm_min}")
    if clipping_norm_max <= clipping_norm_min:
        raise ValueError(
            f"clipping_norm_max ({clipping_norm_max}) must be > clipping_norm_min ({clipping_norm_min})"
        )
    if fraction_noise_std <= 0:
        raise ValueError(
            "fraction_noise_std must be positive, "
            f"got {fraction_noise_std}"
        )

    # Store config in closure (immutable)
    normalize_by = clipped_grad_kwargs.get("normalize_by", 1.0)

    config = {
        "target_quantile": target_quantile,
        "learning_rate": learning_rate,
        "clipping_norm_min": clipping_norm_min,
        "clipping_norm_max": clipping_norm_max,
        "fraction_noise_std": fraction_noise_std,
    }

    def grad_fn(*args, state: AdaptiveClipState, **kwargs):
        """Compute clipped gradients with adaptive threshold.

        Args:
            *args: Positional arguments to pass to the loss function.
            state: Current AdaptiveClipState.
            **kwargs: Keyword arguments to pass to the loss function.

        Returns:
            If return_aux:
                ((grad, aux), new_state)
            Else:
                (grad, new_state)
        """
        # Compute gradients with current threshold
        # Force grad_norms computation to update the threshold
        user_wants_return_aux = return_aux
        clipped_result = cast(
            tuple[Callable, Any],
            clipped_grad(
                loss_fn,
                argnums=argnums,
                has_aux=has_aux,
                clipping_norm=state.next_clipping_norm,
                return_aux=user_wants_return_aux,
                _force_grad_norms=not user_wants_return_aux,
                **clipped_grad_kwargs,
            ),
        )
        inner_fn, inner_state = clipped_result

        result, _ = inner_fn(*args, state=inner_state, **kwargs)

        # Extract gradients and auxiliary output
        aux = None
        if isinstance(result, tuple):
            grads, aux = result
            grad_norms = aux.grad_norms
        else:
            grads = result
            grad_norms = None

        num_clipped = 0.0
        batch_size = aux.batch_size if aux is not None else 0
        if grad_norms is not None:
            num_clipped = float((grad_norms > state.next_clipping_norm).sum().item())
            clipping_rate = aux.clipping_rate

            noisy_clipping_rate = _sample_noisy_clipping_rate(
                clipping_rate,
                key=state._key,
                step=state.step,
                fraction_noise_std=config["fraction_noise_std"],
            )

            new_clipping_norm = _adaptive_clipping_norm_update(
                base_clipping_norm=state.next_clipping_norm,
                noisy_clipping_rate=noisy_clipping_rate,
                target_quantile=config["target_quantile"],
                learning_rate=config["learning_rate"],
                clipping_norm_min=config["clipping_norm_min"],
                clipping_norm_max=config["clipping_norm_max"],
            )
        else:
            new_clipping_norm = state.next_clipping_norm

        new_state = AdaptiveClipState(
            clipping_norm=state.next_clipping_norm,
            normalize_by=normalize_by,
            next_clipping_norm=new_clipping_norm,
            step=state.step + 1,
            _key=state._key,
            _fraction_noise_std=config["fraction_noise_std"],
            _learning_rate=config["learning_rate"],
            _target_quantile=config["target_quantile"],
            _clipping_norm_min=config["clipping_norm_min"],
            _clipping_norm_max=config["clipping_norm_max"],
            _num_clipped=num_clipped,
            _batch_size=batch_size,
        )

        if user_wants_return_aux:
            adaptive_aux = AdaptiveClippedGradAux(
                loss_values=aux.loss_values if aux is not None else None,
                grad_norms=aux.grad_norms if aux is not None else None,
                clipped_grad_norms=(
                    aux.clipped_grad_norms if aux is not None else None
                ),
                loss_aux=aux.loss_aux if aux is not None else None,
                clipping_rate=aux.clipping_rate if aux is not None else None,
                batch_size=batch_size,
            )
            return (grads, adaptive_aux), new_state

        return grads, new_state

    # Create initial state
    initial_state = AdaptiveClipState(
        clipping_norm=initial_clipping_norm,
        normalize_by=normalize_by,
        next_clipping_norm=initial_clipping_norm,
        step=0,
        _key=key,
        _fraction_noise_std=fraction_noise_std,
        _learning_rate=learning_rate,
        _target_quantile=target_quantile,
        _clipping_norm_min=clipping_norm_min,
        _clipping_norm_max=clipping_norm_max,
        _num_clipped=0.0,
        _batch_size=0,
    )

    return grad_fn, initial_state


__all__ = [
    "adaptive_clipped_grad",
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
]
