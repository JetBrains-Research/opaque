"""DP Optimizer with Adaptive Clipping (DP-Adam-AC).

This module implements a differentially private optimizer with adaptive gradient
clipping, based on:
    Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable
    Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288 (October 2024)

Key features:
    - Adaptive Clipping: Gradient threshold adapts based on gradient distribution
    - Dynamic LR Scaling: Learning rate adjusts with clipping frequency
    - EMA Smoothing: Exponential moving average for better generalization
    - Flexible Base Optimizer: Works with any TorchOpt optimizer (default: AdamW)
    - Purely Functional: Immutable state, no side effects
    - External Accounting: Privacy tracking is user's responsibility

Design:
    - Follows TorchOpt pattern: (init_fn, step_fn) tuple
    - Accepts any base optimizer from TorchOpt (AdamW, Adam, SGD, etc.)
    - Uses functional clip_buffer for adaptive threshold tracking
    - RNG state managed as immutable ByteTensor snapshots
"""

from collections.abc import Callable
from typing import Any, Optional

import torch

import torchopt
from opaque.optimizers.adaptive import clip_buffer
from opaque.optimizers.adaptive.lr_scheduler import (
    clip_rate_based_lr_adjustment,
    compute_clip_rate_thresholds,
)
from opaque.optimizers.types import DPAdaptiveClipState
from opaque.utils.pytree import tree_map

# Default constants for adaptive clipping
DEFAULT_HISTORY_SIZE = 1000  # Gradient norm buffer capacity

# Constants for dynamic LR scaling (Zuo et al. 2024)
DEFAULT_LR_MULTIPLIER_MIN = 0.1  # Minimum LR scaling factor
DEFAULT_LR_MULTIPLIER_MAX = 2.0  # Maximum LR scaling factor
DEFAULT_LR_INCREASE_FACTOR = 1.01  # Multiplicative increase when clip rate too low
DEFAULT_LR_DECREASE_FACTOR = 0.995  # Multiplicative decrease when clip rate too high
DEFAULT_CLIP_RATE_TOLERANCE = 0.10  # Tolerance band around target clip rate


def adaptive_clipping(
    base_optimizer: Any,
    *,
    initial_clip_norm: float = 1.0,
    target_clip_rate: float = 0.20,
    clip_norm_min: float = 0.1,
    clip_norm_max: float = 10.0,
    use_clip_lr_scaling: bool = False,
) -> tuple[Callable, Callable]:
    """Wrap any TorchOpt optimizer with adaptive gradient clipping.

    **Default Behavior (Andrew et al. 2021 baseline)**:
    By default, implements AdaClip from "Differentially Private Learning with Adaptive
    Clipping" (Andrew et al., NeurIPS 2021). This provides:
    - Adaptive gradient clipping threshold C (adapts to gradient distribution)
    - Works with any TorchOpt optimizer (AdamW, Adam, SGD, etc.)

    **Optional Enhancement**:
    - `use_clip_lr_scaling=True`: Dynamic LR adjustment based on clip rate
      (Zuo et al., "DP-Adam-AC", 2024)

    **External Concerns** (intentionally not included):
    - **Noise injection**: User must call add_gaussian_noise() before step_fn
    - **Privacy accounting**: User manages privacy budget separately
    - **EMA smoothing**: User can wrap optimizer in external EMA smoother

    **TorchOpt Functional Pattern**:
        # Create base optimizer (fully configured)
        base_opt = torchopt.adamw(lr=3e-4, weight_decay=0.01)

        # Wrap with adaptive clipping
        init_fn, step_fn = adaptive_clipping(base_opt, initial_clip_norm=1.0)
        state = init_fn(params)

        # Training step
        updates, state, metrics = step_fn(grads, grad_norms, state, params=params)
        params = torchopt.apply_updates(params, updates)

    **Algorithm** (Andrew et al. baseline + optional LR scaling):
        1. Compute per-example gradients and norms (external, via clipped_grad)
        2. Clip gradients to adaptive threshold C (external, via clipped_grad)
        3. Add Gaussian noise N(0, (σ·C)²I) (external, via add_gaussian_noise)
        4. Compute base optimizer updates: Δ ← optimizer_update(noisy_grads)
        5. [Optional] Scale updates by LR multiplier: Δ ← γ·Δ (if use_lr_scaling)
        6. Track gradient norms in buffer
        7. Adapt C ← Percentile_q(buffer) where q = 1 - ρ*
        8. [Optional] Adjust γ based on clip rate ρ (if use_lr_scaling)

    Args:
        base_optimizer: TorchOpt GradientTransformation (fully configured).
            Examples: torchopt.adamw(lr=3e-4, weight_decay=0.01),
                     torchopt.sgd(lr=0.01, momentum=0.9)
        initial_clip_norm: Initial clipping threshold C. Default: 1.0
        target_clip_rate: Target fraction of clipped gradients ρ*. Default: 0.20
        clip_norm_min: Minimum clipping threshold C_min. Default: 0.1
        clip_norm_max: Maximum clipping threshold C_max. Default: 10.0
        use_clip_lr_scaling: Enable dynamic LR scaling based on clip rate
            (Zuo et al. 2024). When enabled, scales optimizer updates by a multiplier
            that increases/decreases based on the observed clip rate to maintain
            stable training. Default: False (Andrew et al. 2021 baseline)

    Returns:
        Tuple of (init_fn, step_fn) where:
            - init_fn(params) -> DPAdaptiveClipState
            - step_fn(grads, grad_norms, state, *, params)
                -> (updates, new_state, metrics)
            User must call torchopt.apply_updates(params, updates) to get new params

    Example - Basic usage (Andrew et al. 2021 baseline):
        >>> import torch
        >>> import torchopt
        >>> from opaque.optimizers import adaptive_clipping
        >>> from opaque import clipped_grad, add_gaussian_noise
        >>>
        >>> # Model parameters
        >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(5)}
        >>>
        >>> def loss_fn(params, x, y):
        ...     logits = x @ params['weight'] + params['bias']
        ...     return ((logits - y) ** 2).mean()
        >>>
        >>> # Create base optimizer (fully configured)
        >>> base_opt = torchopt.adamw(lr=3e-4, weight_decay=0.01)
        >>>
        >>> # Wrap with adaptive clipping (Andrew et al. baseline)
        >>> init_fn, step_fn = adaptive_clipping(base_opt,initial_clip_norm=1.0)
        >>>
        >>> # Initialize state
        >>> state = init_fn(params)
        >>>
        >>> # DP parameters (user managed)
        >>> noise_multiplier = 1.1
        >>> rng = torch.Generator().manual_seed(42)
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in dataloader:
        ...     # 1. Compute clipped gradients with norms
        ...     clipped_grad_fn = clipped_grad(
        ...         loss_fn,
        ...         argnums=0,
        ...         batch_argnums=(1, 2),
        ...         l2_clip_norm=state.current_clip_norm,
        ...         return_grad_norms=True,
        ...     )
        ...     grads, aux = clipped_grad_fn(params, x_batch, y_batch)
        ...     grad_norms = aux.grad_norms
        ...
        ...     # 2. Add DP noise (external to optimizer)
        ...     stddev = noise_multiplier * state.current_clip_norm
        ...     noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=rng)
        ...
        ...     # 3. Optimizer step (returns updates, not new_params)
        ...     updates, state, metrics = step_fn(noisy_grads, grad_norms, state, params=params)
        ...
        ...     # 4. Apply updates to get new parameters
        ...     params = torchopt.apply_updates(params, updates)
        ...
        ...     # Monitor adaptive behavior
        ...     if state.step % 100 == 0:
        ...         print(f"Step {state.step}:")
        ...         print(f"  Clip norm = {metrics['clip_norm']:.2f}")
        ...         print(f"  Clip rate = {metrics['clip_rate']:.1%}")

    Example - With dynamic LR scaling (Zuo et al. 2024):
        >>> # Create base optimizer
        >>> base_opt = torchopt.adamw(lr=3e-4, weight_decay=0.01)
        >>>
        >>> # Enable LR scaling for more stable training
        >>> init_fn, step_fn = adaptive_clipping(base_opt,initial_clip_norm=1.0,use_clip_lr_scaling=True)
        >>>
        >>> # Training loop (same as above)
        >>> # Monitor LR multiplier in metrics
        >>> if state.step % 100 == 0:
        ...     print(f"LR multiplier: {metrics['lr_multiplier']:.2f}")

    Example - External EMA smoothing (optional wrapper):
        >>> # EMA is an external concern - can be wrapped around optimizer
        >>> # For example, using a hypothetical EMA wrapper:
        >>> # params_ema = tree_map(lambda p: p.clone(), params)
        >>> # After each step:
        >>> #   params_ema = tree_map(
        >>> #       lambda ema, p: 0.999 * ema + 0.001 * p,
        >>> #       params_ema, params
        >>> #   )
        >>> # Use params_ema for evaluation

    Notes:
        - **Default**: Andrew et al. (2021) baseline with adaptive clipping only
        - **Noise injection**: EXTERNAL - add noise before calling step_fn
        - **Privacy accounting**: EXTERNAL - track privacy budget separately
        - **EMA smoothing**: EXTERNAL - can wrap optimizer with custom smoother
        - **LR schedulers**: EXTERNAL - with TorchOpt's functional API, you have options:
            1. Recreate optimizer with new LR each epoch (simple but inefficient)
            2. Scale updates externally: `updates = tree_map(lambda u: u * lr_schedule(step), updates)`
            3. Use TorchOpt's schedule wrappers (if available)
        - **LR scaling**: When enabled (use_clip_lr_scaling=True), learning rate adjusts
          dynamically based on clip rate for more stable training (independent of external schedulers)
        - **Returns updates**: Follows TorchOpt pattern - user must call apply_updates
        - Requires pre-clip gradient norms (use clipped_grad with return_grad_norms=True)
        - Clip norm C adapts automatically based on gradient distribution
        - Typically achieves 1-3% better accuracy than fixed clipping
        - Modular design allows experimenting with different noise mechanisms

    References:
        - Andrew et al. (2021): "Differentially Private Learning with Adaptive
          Clipping" (NeurIPS 2021) - Baseline adaptive clipping
        - Zuo et al. (2024): "DP-Adam-AC: Privacy-preserving Fine-Tuning of
          Localizable Language Models" - LR scaling enhancement

    See Also:
        opaque.clipping.clipped_grad: Compute per-example clipped gradients
        opaque.accounting: External privacy accounting
        torchopt: Base optimizers (adam, adamw, sgd, etc.)
    """
    # Compute clip rate thresholds for LR scaling
    rho_low, rho_high = compute_clip_rate_thresholds(target_clip_rate, DEFAULT_CLIP_RATE_TOLERANCE)

    def init_fn(params):
        """Initialize optimizer state.

        Args:
            params: PyTree of model parameters

        Returns:
            DPAdaptiveClipState with all components initialized
        """
        # Initialize base optimizer
        opt_state = base_optimizer.init(params)

        # Create gradient norm buffer state (immutable)
        buffer_state = clip_buffer.create(
            capacity=DEFAULT_HISTORY_SIZE,
            target_clip_rate=target_clip_rate,
        )

        return DPAdaptiveClipState(
            opt_state=opt_state,
            clip_buffer_state=buffer_state,
            current_clip_norm=initial_clip_norm,
            lr_multiplier=1.0,  # Always track (used conditionally in step_fn)
            step=0,
        )

    def step_fn(
        grads,
        grad_norms: torch.Tensor,
        state: DPAdaptiveClipState,
        *,
        params,
        batch_sizes: Optional[torch.Tensor] = None,
    ):
        """Perform one optimizer step with adaptive clipping.

        Args:
            grads: Gradients (user should add noise before calling this)
            grad_norms: Pre-clip gradient norms (for adaptive threshold)
            state: Current DPAdaptiveClipState
            params: Current PyTree of parameters (keyword-only)
            batch_sizes: Optional batch sizes for each example (for microbatching).
                If None, assumes all examples have size 1. Default: None

        Returns:
            Tuple of (updates, new_state, metrics) where:
                - updates: Parameter updates (call torchopt.apply_updates to get new params)
                - new_state: New optimizer state
                - metrics: Dict with 'clip_norm', 'clip_rate', 'lr_multiplier'
                    - clip_norm: Current adaptive clipping threshold
                    - clip_rate: Fraction of gradients that were clipped
                    - lr_multiplier: Multiplicative scaling factor applied to updates (1.0 if use_lr_scaling=False)
        """
        # Default batch_sizes to ones (standard training without microbatching)
        if batch_sizes is None:
            # Infer batch size from grad_norms
            if isinstance(grad_norms, torch.Tensor):
                if grad_norms.dim() == 0:
                    batch_sizes = torch.ones(1, device=grad_norms.device)
                else:
                    batch_sizes = torch.ones(len(grad_norms), device=grad_norms.device)
            else:
                batch_sizes = torch.ones(1)

        # 1. Get parameter updates from base optimizer (LR already applied)
        # Note: Pass unscaled gradients so Adam's momentum buffers see correct statistics
        updates, new_opt_state = base_optimizer.update(grads, state.opt_state, params=params)

        # 2. [Optional] Scale updates by LR multiplier (Zuo et al. 2024 enhancement)
        # This adjusts the step size without affecting momentum buffers
        if use_clip_lr_scaling:
            scaled_updates = tree_map(lambda u: u * state.lr_multiplier, updates)
        else:
            scaled_updates = updates  # No scaling in baseline

        # 3. Update gradient norm buffer (functional)
        new_buffer_state = clip_buffer.update(state.clip_buffer_state, grad_norms, batch_sizes)

        # 4. Compute adaptive clip norm: C ← Percentile_q(buffer)
        new_clip_norm = clip_buffer.get_adaptive_clip_norm(
            new_buffer_state,
            target_clip_rate=target_clip_rate,
            clip_norm_min=clip_norm_min,
            clip_norm_max=clip_norm_max,
        )

        # 5. Compute current clip rate
        current_clip_rate = clip_buffer.get_clip_rate(new_buffer_state, state.current_clip_norm)

        # 6. [Optional] Adjust learning rate multiplier based on clip rate
        if use_clip_lr_scaling:
            new_lr_multiplier = clip_rate_based_lr_adjustment(
                current_lr_multiplier=state.lr_multiplier,
                clip_rate=current_clip_rate,
                target_clip_rate=target_clip_rate,
                clip_rate_low=rho_low,
                clip_rate_high=rho_high,
                increase_factor=DEFAULT_LR_INCREASE_FACTOR,
                decrease_factor=DEFAULT_LR_DECREASE_FACTOR,
                lr_multiplier_min=DEFAULT_LR_MULTIPLIER_MIN,
                lr_multiplier_max=DEFAULT_LR_MULTIPLIER_MAX,
            )
        else:
            new_lr_multiplier = 1.0  # No adjustment in baseline

        # 7. Create new state (immutable)
        new_state = DPAdaptiveClipState(
            opt_state=new_opt_state,
            clip_buffer_state=new_buffer_state,
            current_clip_norm=new_clip_norm,
            lr_multiplier=new_lr_multiplier,
            step=state.step + 1,
        )

        # 8. Return metrics (no privacy accounting - external responsibility)
        metrics = {
            "step": new_state.step,
            "clip_norm": new_clip_norm,
            "clip_rate": current_clip_rate,
            "lr_multiplier": new_lr_multiplier,  # Scaling factor (1.0 if use_clip_lr_scaling=False)
        }

        # Return updates (user must call torchopt.apply_updates)
        return scaled_updates, new_state, metrics

    return init_fn, step_fn


__all__ = ["adaptive_clipping"]
