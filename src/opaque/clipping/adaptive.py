"""Adaptive gradient clipping with explicit state-passing.

This module provides a pure functional interface for adaptive gradient clipping
(Andrew et al. 2021) where state is passed explicitly as a parameter and returned
as part of the output. This design avoids mutable closures and works seamlessly
with distributed training, torch.compile, and other PyTorch features.

Inspired by JAX-Privacy and Optax's functional state-passing design.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch

from opaque.clipping.clipped_grad import ClippedGradAux, clipped_grad
from opaque.clipping.types import ClipState, NeighboringRelation


class AdaptiveClippedGradAux(NamedTuple):
    """Auxiliary outputs from adaptive_clipped_grad extending ClippedGradAux.

    Attributes:
        loss_values: Per-example loss values (if return_loss_values=True).
        grad_norms: L2 norms of per-example gradients before clipping (if return_grad_norms=True).
        clipped_grad_norms: L2 norms after clipping (if return_grad_norms=True).
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
        clip_norm: Current clipping threshold C_t.
        clipping_rate: Fraction of gradients clipped in last call (for monitoring).
        rescale_to_unit_norm: Whether gradients were rescaled to unit norm.
    """

    clip_norm: float
    clipping_rate: float = 0.0
    rescale_to_unit_norm: bool = False

    def __post_init__(self):
        """Validate state values."""
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {self.clip_norm}")
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
            >>> grad, clip_state = grad_fn(params, x, y, state=clip_state)
            >>> sens = clip_state.sensitivity()
            >>> noise_fn, ns = gaussian_noise(stddev=noise_multiplier * sens)
            >>> noisy_grad, ns = noise_fn(grad, ns)
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
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    initial_clip_norm: float = 0.1,
    target_quantile: float = 0.5,
    learning_rate: float = 0.2,
    clip_norm_min: float = 0.01,
    clip_norm_max: float = 100.0,
    return_aux: bool = False,
    distributed: Literal["auto"] | bool = "auto",
    **clipped_grad_kwargs: Any,
) -> tuple[Callable, AdaptiveClipState]:
    """Create function for adaptive gradient clipping with explicit state-passing.

    This function returns a tuple of (clipped_grad_fn, initial_state). The
    clipped_grad_fn takes state as an explicit parameter and returns
    (grad, new_state) or ((grad, aux), new_state) depending on return_aux.

    The clipping threshold adapts geometrically based on observed clipping rate:
        C_{t+1} = C_t * exp(η * sign(ρ_t - γ))

    Where ρ_t is the fraction of per-example gradients clipped at step t.

    **Distributed training is automatically detected** - no configuration needed!

    Args:
        loss_fn: The loss function to be differentiated. Should return a scalar.
            If `has_aux` is True, should return (scalar, loss_aux).
        argnums: Which argument(s) of `loss_fn` to differentiate with respect to.
            Typically 0 (parameters). Can be int or tuple of ints.
        has_aux: If True, `loss_fn` returns (value, loss_aux). The loss_aux data will be
            returned per-example.
        initial_clip_norm: Initial clipping threshold C_0. Default: 0.1
            (as recommended in Andrew et al. 2021).
        target_quantile: Target quantile γ for clipping rate. Default: 0.5 (median).
            The algorithm tries to clip this fraction of gradients.
        learning_rate: Learning rate η_C for geometric updates. Default: 0.2
            (as used in Andrew et al. 2021). Controls adaptation speed.
        clip_norm_min: Minimum allowed clipping threshold. Default: 0.01.
        clip_norm_max: Maximum allowed clipping threshold. Default: 100.0.
        return_aux: If True, return a per-example aux NamedTuple with loss values,
            gradient norms, loss aux, and adaptive fields.
        distributed: Distributed handling mode:
            - "auto": enable distributed reductions if torch.distributed is initialized
            - True: require distributed mode and perform internal reductions
            - False: do not perform any distributed reductions
        **clipped_grad_kwargs: Additional arguments passed to `clipped_grad()`,
            such as `batch_argnums`, `rescale_to_unit_norm`, `normalize_by`, etc.

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
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Training loop with explicit state-passing
        >>> params = torch.randn(10, requires_grad=False)
        >>> optimizer = torchopt.adamw(lr=1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> noise_fn, noise_state = gaussian_noise(stddev=1.1)
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
        >>> from opaque.clipping import adaptive_clipped_grad
        >>> from opaque.distributed import sum_gradients
        >>> from opaque.sampling import PoissonSampler
        >>>
        >>> # Initialize distributed
        >>> dist.init_process_group(backend='nccl')
        >>>
        >>> # Create adaptive clipping (auto-detects distributed!)
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Use Poisson sampling (different batch sizes on each device)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, distributed=False)
        >>>
        >>> for batch_x, batch_y in dataloader:
        ...     # Each device: compute clipped gradients on local batch
        ...     # (clip_state.clip_norm is IDENTICAL across devices)
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...
        ...     # Sum clipped gradients across devices
        ...     grad = sum_gradients(grad)
        ...
        ...     # Add noise and update (only on rank 0, or broadcast)
        ...     noisy_grad = gaussian_noise(grad, clip_state.clip_norm * 1.1)
        ...     # ... optimizer step

    Notes:
        - State is IMMUTABLE - a new state object is returned each call.
        - Works with torch.compile, DDP, FSDP (state is explicit).
        - **Distributed mode is automatically detected** via torch.distributed.is_initialized()
        - In distributed mode, per-example norms are gathered from ALL devices
          to compute the clipping rate, ensuring clip_norm is identical everywhere.
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

    def _use_distributed() -> bool:
        if distributed not in ("auto", True, False):
            raise ValueError(
                f"distributed must be one of {{'auto', True, False}}, got {distributed!r}"
            )

        try:
            from opaque.distributed import is_distributed
        except ImportError as err:
            if distributed is True:
                raise RuntimeError(
                    "distributed=True requested but opaque.distributed is unavailable."
                ) from err
            return False

        active = is_distributed()
        if distributed is True and not active:
            raise RuntimeError(
                "distributed=True requested but torch.distributed is not initialized."
            )
        if distributed is False:
            return False
        return active

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
        inner_fn, inner_state = clipped_grad(
            loss_fn,
            argnums=argnums,
            has_aux=has_aux,
            l2_clip_norm=state.clip_norm,
            return_aux=user_wants_return_aux,
            distributed=False,
            _force_grad_norms=not user_wants_return_aux,
            **clipped_grad_kwargs,
        )

        result, _ = inner_fn(*args, state=inner_state, **kwargs)

        # Extract gradients and auxiliary output
        if isinstance(result, tuple):
            grads, aux = result
            grad_norms = aux.grad_norms
        else:
            grads = result
            grad_norms = None

        distributed_active = _use_distributed()
        if distributed_active:
            from opaque.distributed import sum_gradients

            grads = sum_gradients(grads)

        # Update clipping threshold using Andrew et al. 2021 algorithm
        local_num_clipped = 0.0
        local_total = 0.0
        if grad_norms is not None:
            # Compute clipping rate: ρ_t = fraction of gradients clipped
            num_clipped = (grad_norms > state.clip_norm).sum().item()
            total = max(1, grad_norms.numel())
            clipping_rate = num_clipped / total
            local_num_clipped = float(num_clipped)
            local_total = float(total)

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
            clipping_rate=clipping_rate,
            rescale_to_unit_norm=state.rescale_to_unit_norm,
        )

        if distributed_active:
            new_state = sync_adaptive_clip_state(
                new_state,
                local_num_clipped=local_num_clipped,
                local_total=local_total,
            )

        if user_wants_return_aux:
            if distributed_active:
                from opaque.distributed import gather_pytree

                gathered_aux = gather_pytree(
                    {
                        "loss_values": aux.loss_values,
                        "grad_norms": aux.grad_norms,
                        "clipped_grad_norms": aux.clipped_grad_norms,
                        "loss_aux": aux.loss_aux,
                    }
                )
                aux = ClippedGradAux(
                    loss_values=gathered_aux.get("loss_values"),
                    grad_norms=gathered_aux.get("grad_norms"),
                    clipped_grad_norms=gathered_aux.get("clipped_grad_norms"),
                    loss_aux=gathered_aux.get("loss_aux"),
                )

            adaptive_aux = AdaptiveClippedGradAux(
                loss_values=aux.loss_values if hasattr(aux, "loss_values") else None,
                grad_norms=aux.grad_norms if hasattr(aux, "grad_norms") else None,
                clipped_grad_norms=(
                    aux.clipped_grad_norms
                    if hasattr(aux, "clipped_grad_norms")
                    else None
                ),
                loss_aux=aux.loss_aux if hasattr(aux, "loss_aux") else None,
                clipping_rate=new_state.clipping_rate,
            )
            return (grads, adaptive_aux), new_state

        return grads, new_state

    # Create initial state
    initial_state = AdaptiveClipState(
        clip_norm=initial_clip_norm,
        clipping_rate=0.0,
        rescale_to_unit_norm=rescale_to_unit_norm,
    )

    return grad_fn, initial_state


def sync_adaptive_clip_state(
    state: AdaptiveClipState,
    local_num_clipped: float | int,
    local_total: float | int,
) -> AdaptiveClipState:
    """Synchronize adaptive clipping state using global clipped counts.

    This composes mean reduction for clip_norm with a globally consistent
    clipping rate computed from the sum of per-device counts:

        global_rate = sum(local_num_clipped) / max(1, sum(local_total))

    Args:
        state: Adaptive clipping state.
        local_num_clipped: Number of locally clipped examples at this step.
        local_total: Number of local examples considered at this step.

    Returns:
        New synchronized state with globally consistent clipping_rate.
    """
    from opaque.distributed import is_distributed, reduce_scalar

    if not is_distributed():
        return state

    # Mean reduction for clip_norm
    synced_clip_norm = reduce_scalar(state.clip_norm, op="mean")

    # Global clipping rate from summed counts
    global_num_clipped = reduce_scalar(float(local_num_clipped), op="sum")
    global_total = reduce_scalar(float(local_total), op="sum")
    global_rate = global_num_clipped / max(1.0, global_total)

    return AdaptiveClipState(
        clip_norm=synced_clip_norm,
        clipping_rate=float(global_rate),
        rescale_to_unit_norm=state.rescale_to_unit_norm,
    )


__all__ = [
    "adaptive_clipped_grad",
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "sync_adaptive_clip_state",
]
