"""Adaptive gradient clipping with explicit state-passing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import torch

from opaque.clipping.clipped_grad import clipped_grad
from opaque.clipping.types import ClipState
from opaque.random import RngKey, fold_in, generator_from_key

_DEFAULT_FRACTION_NOISE_STD = 0.05


class AdaptiveClippedGradAux(NamedTuple):
    """Auxiliary outputs from adaptive_clipped_grad extending ClippedGradAux.

    Attributes:
        loss_values: Per-example loss values (if return_aux=True).
        grad_norms: L2 norms of per-example gradients before clipping (if return_aux=True).
        clipped_grad_norms: L2 norms after clipping (if return_aux=True).
        loss_aux: Auxiliary outputs from loss function (if has_aux=True).
        clipping_rate: Fraction of per-example gradients clipped at this step.
    """

    loss_values: Any | None
    grad_norms: Any | None
    clipped_grad_norms: Any | None
    loss_aux: Any | None
    clipping_rate: float | None


@dataclass(frozen=True)
class AdaptiveClipState(ClipState):
    """Immutable state for adaptive gradient clipping.

    This state is passed explicitly to the clipping function and returned
    as part of the output, enabling pure functional composition.

    Attributes:
        clip_norm: Raw clipping threshold used at the current step.
        normalize_by: Divisor applied to the clipped gradient sum
            (1.0 = no averaging).  Controls gradient sensitivity:
            ``sensitivity = clip_norm / normalize_by``.  Also used as
            the denominator when computing the clipping fraction
            (when > 1); pass the same value to
            ``acc.adaclip(expected_batch_size=...)``.
        next_clip_norm: Clipping threshold for the *next* step C_{t+1}.
        clipping_rate: Fraction of gradients clipped in last call (for monitoring).
        key: RNG key for quantile noise (None if no noise).
        step: Step counter for key derivation.
        batch_size: Actual number of examples processed in the last call.
            In distributed training the synced state holds the *global*
            batch size (sum across ranks).
    """

    clip_norm: float
    normalize_by: float
    next_clip_norm: float
    clipping_rate: float
    key: RngKey
    step: int
    fraction_noise_std: float
    learning_rate: float
    target_quantile: float
    clip_norm_min: float
    clip_norm_max: float
    num_clipped: float
    total: float
    batch_size: float

    def __post_init__(self):
        """Validate state values."""
        if self.next_clip_norm <= 0:
            raise ValueError(f"next_clip_norm must be positive, got {self.next_clip_norm}")
        if self.normalize_by <= 0:
            raise ValueError(f"normalize_by must be positive, got {self.normalize_by}")
        if self.clipping_rate < 0:
            raise ValueError(
                f"clipping_rate must be non-negative, got {self.clipping_rate}"
            )
        if self.fraction_noise_std <= 0:
            raise ValueError(
                "fraction_noise_std must be > 0, "
                f"got {self.fraction_noise_std}"
            )


def _compute_clipping_stats(
    grad_norms: torch.Tensor,
    clip_norm: float,
    normalize_by: float = 1.0,
) -> tuple[float, float, float]:
    """Compute local clipping statistics from per-example gradient norms.

    When *normalize_by* > 1, the clipping fraction is computed as
    ``num_clipped / normalize_by`` — a data-independent denominator
    required for correct privacy accounting under Poisson sampling.
    Otherwise the actual number of examples is used (correct for
    fixed-size batches).
    """
    num_clipped = float((grad_norms > clip_norm).sum().item())
    actual = float(max(1, grad_norms.numel()))
    denominator = normalize_by if normalize_by > 1.0 else actual
    clipping_rate = num_clipped / max(1.0, denominator)
    return num_clipped, actual, clipping_rate


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


def _adaptive_clip_norm_update(
    *,
    base_clip_norm: float,
    noisy_clipping_rate: float,
    target_quantile: float,
    learning_rate: float,
    clip_norm_min: float,
    clip_norm_max: float,
) -> float:
    """Compute geometric adaptive clipping update: C * exp(η * (ρ̃ - γ))."""
    update_factor = torch.exp(
        torch.tensor(learning_rate * (noisy_clipping_rate - target_quantile))
    ).item()
    new_clip_norm = base_clip_norm * update_factor
    return float(max(clip_norm_min, min(clip_norm_max, new_clip_norm)))


def adaptive_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    initial_clip_norm: float = 0.1,
    target_quantile: float = 0.5,
    learning_rate: float = 0.2,
    clip_norm_min: float = 0.01,
    clip_norm_max: float = 100.0,
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
        initial_clip_norm: Initial clipping threshold C_0.
        target_quantile: Target fraction of clipped gradients.
        learning_rate: Step size for geometric adaptation.
        clip_norm_min: Minimum allowed clipping threshold.
        clip_norm_max: Maximum allowed clipping threshold.
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
        ...     initial_clip_norm=0.1,
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
        ...         print(f"Step {clip_state.step}: C={clip_state.clip_norm:.4f}, "
        ...               f"ρ={clip_state.clipping_rate:.2%}")

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
    if initial_clip_norm <= 0:
        raise ValueError(f"initial_clip_norm must be positive, got {initial_clip_norm}")
    if not 0 < target_quantile < 1:
        raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if clip_norm_min <= 0:
        raise ValueError(f"clip_norm_min must be positive, got {clip_norm_min}")
    if clip_norm_max <= clip_norm_min:
        raise ValueError(
            f"clip_norm_max ({clip_norm_max}) must be > clip_norm_min ({clip_norm_min})"
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
        "clip_norm_min": clip_norm_min,
        "clip_norm_max": clip_norm_max,
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
                l2_clip_norm=state.next_clip_norm,
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
        total = 0.0
        batch_size = grad_norms.numel() if grad_norms is not None else 0
        if grad_norms is not None:
            num_clipped, total, clipping_rate = _compute_clipping_stats(
                grad_norms, state.next_clip_norm, normalize_by,
            )

            noisy_clipping_rate = _sample_noisy_clipping_rate(
                clipping_rate,
                key=state.key,
                step=state.step,
                fraction_noise_std=state.fraction_noise_std,
            )

            new_clip_norm = _adaptive_clip_norm_update(
                base_clip_norm=state.next_clip_norm,
                noisy_clipping_rate=noisy_clipping_rate,
                target_quantile=config["target_quantile"],
                learning_rate=config["learning_rate"],
                clip_norm_min=config["clip_norm_min"],
                clip_norm_max=config["clip_norm_max"],
            )
        else:
            # No norms available (shouldn't happen)
            new_clip_norm = state.next_clip_norm
            clipping_rate = 0.0

        # Create new state (IMMUTABLE) with incremented step
        new_state = AdaptiveClipState(
            clip_norm=state.next_clip_norm,
            normalize_by=normalize_by,
            next_clip_norm=new_clip_norm,
            clipping_rate=clipping_rate,
            key=state.key,
            step=state.step + 1,
            fraction_noise_std=state.fraction_noise_std,
            learning_rate=state.learning_rate,
            target_quantile=state.target_quantile,
            clip_norm_min=state.clip_norm_min,
            clip_norm_max=state.clip_norm_max,
            num_clipped=num_clipped,
            total=total,
            batch_size=batch_size,
        )

        if user_wants_return_aux:
            adaptive_aux = AdaptiveClippedGradAux(
                loss_values=(
                    aux.loss_values
                    if aux is not None and hasattr(aux, "loss_values")
                    else None
                ),
                grad_norms=(
                    aux.grad_norms
                    if aux is not None and hasattr(aux, "grad_norms")
                    else None
                ),
                clipped_grad_norms=(
                    aux.clipped_grad_norms
                    if aux is not None and hasattr(aux, "clipped_grad_norms")
                    else None
                ),
                loss_aux=(
                    aux.loss_aux
                    if aux is not None and hasattr(aux, "loss_aux")
                    else None
                ),
                clipping_rate=new_state.clipping_rate,
            )
            return (grads, adaptive_aux), new_state

        return grads, new_state

    # Create initial state
    initial_state = AdaptiveClipState(
        clip_norm=initial_clip_norm,
        normalize_by=normalize_by,
        next_clip_norm=initial_clip_norm,
        clipping_rate=0.0,
        key=key,
        step=0,
        fraction_noise_std=fraction_noise_std,
        learning_rate=learning_rate,
        target_quantile=target_quantile,
        clip_norm_min=clip_norm_min,
        clip_norm_max=clip_norm_max,
        num_clipped=0.0,
        total=0.0,
        batch_size=0,
    )

    return grad_fn, initial_state


__all__ = [
    "adaptive_clipped_grad",
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
]
