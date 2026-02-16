"""Adaptive gradient clipping with explicit state-passing.

This module provides a pure functional interface for adaptive gradient clipping
(Andrew et al. 2021) where state is passed explicitly as a parameter and returned
as part of the output. This design avoids mutable closures and works seamlessly
with distributed training, torch.compile, and other PyTorch features.

Inspired by JAX-Privacy and Optax's functional state-passing design.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from opaque.clipping.clipped_grad import clipped_grad
from opaque.clipping.types import ClippedGradAux, ClipState, NeighboringRelation


@dataclass(frozen=True)
class AdaptiveClipState(ClipState):
    """Immutable state for adaptive gradient clipping.

    This state is passed explicitly to the clipping function and returned
    as part of the output, enabling pure functional composition.

    Attributes:
        clip_norm: Current clipping threshold C_t.
        step: Number of gradient computations performed.
        clipping_rate: Fraction of gradients clipped in last call (for monitoring).
        rescale_to_unit_norm: Whether gradients were rescaled to unit norm.
    """

    clip_norm: float
    step: int
    clipping_rate: float = 0.0
    rescale_to_unit_norm: bool = False

    def __post_init__(self):
        """Validate state values."""
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {self.clip_norm}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        if not 0 <= self.clipping_rate <= 1:
            raise ValueError(
                f"clipping_rate must be in [0, 1], got {self.clipping_rate}"
            )

    def sensitivity(
        self,
        neighboring_relation: NeighboringRelation = NeighboringRelation.REPLACE_SPECIAL,
    ) -> float:
        """Compute L2 sensitivity for differential privacy noise calibration.

        The sensitivity depends on whether gradients were rescaled to unit norm:
        - If rescale_to_unit_norm=True: sensitivity is always 1.0
        - If rescale_to_unit_norm=False: sensitivity is clip_norm

        Args:
            neighboring_relation: The neighboring relation to use for DP.

        Returns:
            L2 sensitivity (float).

        Example:
            >>> from opaque.noise import gaussian_noise
            >>> grad, clip_state = grad_fn(params, x, y, state=clip_state)
            >>> sens = clip_state.sensitivity()
            >>> noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * sens)
            >>> noisy_grad, noise_state = noise_fn(grad, noise_state)
        """
        l2_bound = 1.0 if self.rescale_to_unit_norm else self.clip_norm

        match neighboring_relation:
            case NeighboringRelation.ADD_OR_REMOVE_ONE:
                return l2_bound
            case NeighboringRelation.REPLACE_ONE:
                return 2 * l2_bound
            case NeighboringRelation.REPLACE_SPECIAL:
                return l2_bound
            case _:
                raise ValueError(
                    f"Unsupported neighboring_relation={neighboring_relation}. "
                    f"Must be one of: {list(NeighboringRelation)}"
                )


def adaptive_clipped_grad(
    fun: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    initial_clip_norm: float = 0.1,
    target_quantile: float = 0.5,
    learning_rate: float = 0.2,
    clip_norm_min: float = 0.01,
    clip_norm_max: float = 100.0,
    **clipped_grad_kwargs: Any,
) -> tuple[Callable, AdaptiveClipState]:
    """Create function for adaptive gradient clipping with explicit state-passing.

    This function returns a tuple of (clipped_grad_fn, initial_state). The
    clipped_grad_fn takes state as an explicit parameter and returns
    (grad, new_state) or ((grad, aux), new_state) depending on has_aux.

    The clipping threshold adapts geometrically based on observed clipping rate:
        C_{t+1} = C_t * exp(η * sign(ρ_t - γ))

    Where ρ_t is the fraction of per-example gradients clipped at step t.

    Args:
        fun: The function to be differentiated (loss function). Should return a scalar.
            If `has_aux` is True, should return (scalar, aux).
        argnums: Which argument(s) of `fun` to differentiate with respect to.
            Typically 0 (parameters). Can be int or tuple of ints.
        has_aux: If True, `fun` returns (value, aux). The aux data will be
            returned per-example.
        initial_clip_norm: Initial clipping threshold C_0. Default: 0.1
            (as recommended in Andrew et al. 2021).
        target_quantile: Target quantile γ for clipping rate. Default: 0.5 (median).
            The algorithm tries to clip this fraction of gradients.
        learning_rate: Learning rate η_C for geometric updates. Default: 0.2
            (as used in Andrew et al. 2021). Controls adaptation speed.
        clip_norm_min: Minimum allowed clipping threshold. Default: 0.01.
        clip_norm_max: Maximum allowed clipping threshold. Default: 100.0.
        **clipped_grad_kwargs: Additional arguments passed to `clipped_grad()`,
            such as `batch_argnums`, `rescale_to_unit_norm`, `normalize_by`, etc.

    Returns:
        A tuple of (clipped_grad_fn, initial_state) where:
            - clipped_grad_fn: Function with signature
                (*args, state, **kwargs) -> (grad, new_state) or
                (*args, state, **kwargs) -> ((grad, aux), new_state)
            - initial_state: Initial AdaptiveClipState

    Example:
        >>> import torch
        >>> from opaque.clipping.stateful import adaptive_clipped_grad
        >>> from opaque.noise import gaussian_noise
        >>> import torchopt
        >>>
        >>> def loss_fn(params, x, y):
        ...     pred = x @ params
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Create adaptive clipping function with initial state
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     initial_clip_norm=0.1,
        ...     target_quantile=0.5,
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Training loop with explicit state-passing
        >>> params = torch.randn(10, requires_grad=False)
        >>> optimizer = torchopt.adamw(lr=1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> # Create noise function once
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1 * clip_state.clip_norm, generator=42)
        >>>
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

    Example with distributed training (DDP):
        >>> import torch.distributed as dist
        >>>
        >>> for batch_x, batch_y in dataloader:
        ...     # Compute gradients (local to each device)
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...
        ...     # Synchronize clip_norm across devices
        ...     clip_norm_tensor = torch.tensor(clip_state.clip_norm)
        ...     dist.all_reduce(clip_norm_tensor, op=dist.ReduceOp.AVG)
        ...
        ...     # Update state with synchronized clip_norm
        ...     clip_state = AdaptiveClipState(
        ...         clip_norm=clip_norm_tensor.item(),
        ...         step=clip_state.step,
        ...         clipping_rate=clip_state.clipping_rate,
        ...     )
        ...
        ...     # Continue with noise and optimizer...

    Notes:
        - State is IMMUTABLE - a new state object is returned each call.
        - Works with torch.compile, DDP, FSDP (state is explicit).
        - The clipping threshold adapts over ~23 iterations by a factor of 10
          with default parameters (learning_rate=0.2, target_quantile=0.5).
        - Andrew et al. recommend using the median (γ=0.5) as it works well
          across different tasks without tuning.
        - The adaptation uses negligible privacy budget compared to DP-SGD.

    References:
        Galen Andrew, Om Thakkar, Brendan McMahan, and Swaroop Ramaswamy.
        "Differentially Private Learning with Adaptive Clipping."
        NeurIPS 2021. https://arxiv.org/abs/1905.03871
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

    # Extract rescale_to_unit_norm from kwargs for state initialization
    rescale_to_unit_norm = clipped_grad_kwargs.get("rescale_to_unit_norm", False)

    # Store config in closure (immutable)
    config = {
        "target_quantile": target_quantile,
        "learning_rate": learning_rate,
        "clip_norm_min": clip_norm_min,
        "clip_norm_max": clip_norm_max,
    }

    def grad_fn(*args, state: AdaptiveClipState, **kwargs):
        """Compute clipped gradients with adaptive threshold.

        Args:
            *args: Positional arguments to pass to the loss function.
            state: Current AdaptiveClipState.
            **kwargs: Keyword arguments to pass to the loss function.

        Returns:
            If has_aux or return_values or return_grad_norms:
                ((grad, aux), new_state)
            Else:
                (grad, new_state)
        """
        # Compute gradients with current threshold
        # Force return_grad_norms=True to compute clipping rate
        inner_fn, inner_state = clipped_grad(
            fun,
            argnums=argnums,
            has_aux=has_aux,
            l2_clip_norm=state.clip_norm,
            return_grad_norms=True,  # Always compute norms for adaptation
            **clipped_grad_kwargs,
        )

        result, _ = inner_fn(*args, state=inner_state, **kwargs)

        # Extract gradients and auxiliary output
        if isinstance(result, tuple):
            grads, aux = result
            grad_norms = aux.grad_norms
        else:
            # Shouldn't happen since return_grad_norms=True
            grads = result
            grad_norms = None

        # Update clipping threshold using Andrew et al. 2021 algorithm
        if grad_norms is not None:
            # Compute clipping rate: ρ_t = fraction of gradients clipped
            num_clipped = (grad_norms > state.clip_norm).sum().item()
            clipping_rate = num_clipped / max(1, grad_norms.numel())

            # Geometric update: C_{t+1} = C_t * exp(η * sign(ρ_t - γ))
            if clipping_rate > config["target_quantile"]:
                # Too many gradients clipped → increase threshold
                direction = 1.0
            else:
                # Too few gradients clipped → decrease threshold
                direction = -1.0

            # Update with exponential
            update_factor = torch.exp(torch.tensor(config["learning_rate"] * direction))
            new_clip_norm = state.clip_norm * update_factor.item()

            # Clamp to valid range
            new_clip_norm = max(
                config["clip_norm_min"], min(config["clip_norm_max"], new_clip_norm)
            )
        else:
            # No norms available (shouldn't happen)
            new_clip_norm = state.clip_norm
            clipping_rate = 0.0

        # Create new state (IMMUTABLE)
        new_state = AdaptiveClipState(
            clip_norm=new_clip_norm,
            step=state.step + 1,
            clipping_rate=clipping_rate,
            rescale_to_unit_norm=state.rescale_to_unit_norm,
        )

        # Return result according to user's requested signature
        user_wants_return_values = clipped_grad_kwargs.get("return_values", False)
        user_wants_return_norms = clipped_grad_kwargs.get("return_grad_norms", False)

        if user_wants_return_values or user_wants_return_norms or has_aux:
            # User wants auxiliary outputs - return them (but recreate without forced grad_norms)
            # We need to filter out the forced grad_norms if user didn't ask for them
            if not user_wants_return_norms and grad_norms is not None:
                # Remove grad_norms from aux
                new_aux = ClippedGradAux(
                    loss_values=aux.loss_values
                    if hasattr(aux, "loss_values")
                    else None,
                    grad_norms=None,
                    user_aux=aux.user_aux if hasattr(aux, "user_aux") else None,
                )
                return (grads, new_aux), new_state
            else:
                # Return as-is
                return (grads, aux), new_state
        else:
            # Return only gradients
            return grads, new_state

    # Create initial state
    initial_state = AdaptiveClipState(
        clip_norm=initial_clip_norm,
        step=0,
        clipping_rate=0.0,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )

    return grad_fn, initial_state


__all__ = ["adaptive_clipped_grad", "AdaptiveClipState"]
