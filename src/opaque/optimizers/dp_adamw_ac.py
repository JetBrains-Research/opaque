"""DP-AdamW-AC: DP-AdamW with Adaptive Clipping.

This module implements DP-AdamW with adaptive clipping, combining:
    - AdamW optimizer (adaptive learning rates + decoupled weight decay)
    - Adaptive clipping (gradient threshold adapts to distribution)
    - Dynamic LR scaling (learning rate adjusts based on clip rate)
    - EMA smoothing (exponential moving average for better generalization)

Based on DP-Adam-AC from:
    Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable
    Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288 (October 2024)

Key improvements over DP-Adam-AC:
    - Uses AdamW instead of Adam for better LLM training
    - Explicit weight decay (not relying on noise as regularizer)
    - Same adaptive clipping and LR scheduling
"""

from collections.abc import Callable
from typing import Optional

import torch

import torchopt
# from opaque.accounting import RDPAccountant  # TODO: Update to functional API
from opaque.adaptive import (
    ClipNormBuffer,
    clip_rate_based_lr_adjustment,
    compute_clip_rate_thresholds,
)
from opaque.noise import add_gaussian_noise
from opaque.optimizers.dp_adam_ac import AdaptiveClipState  # Reuse same state class
from opaque.utils.pytree import tree_map


def dp_adamw_ac(
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    *,
    initial_clip_norm: float = 3.0,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    # Adaptive clipping parameters
    target_clip_rate: float = 0.20,
    history_size: int = 1000,
    clip_norm_min: float = 0.1,
    clip_norm_max: float = 10.0,
    # Dynamic LR parameters
    lr_multiplier_min: float = 0.1,
    lr_multiplier_max: float = 2.0,
    lr_increase_factor: float = 1.01,
    lr_decrease_factor: float = 0.995,
    clip_rate_tolerance: float = 0.10,
    # EMA parameters
    ema_decay: float = 0.999,
    # Other
    accountant_type: str = "rdp",
    seed: int = 42,
) -> tuple[Callable, Callable]:
    """Create DP-AdamW-AC optimizer with adaptive clipping.

    DP-AdamW-AC combines AdamW optimization with adaptive gradient clipping.
    The clipping threshold C adapts based on the distribution of recent
    gradient norms, and the learning rate scales with the empirical clipping
    frequency for stable training.

    Algorithm (adapted from DP-Adam-AC paper):
        1. Compute per-example gradients and norms
        2. Clip gradients to adaptive threshold C
        3. Add Gaussian noise N(0, (σ·C)²I)
        4. Update with scaled AdamW: θ ← θ - γ·η·m̂/(√v̂ + ε) - η·λ·θ
        5. Update EMA parameters: θ̂ ← d·θ̂ + (1-d)·θ
        6. Track gradient norms in buffer
        7. Adapt C ← Percentile_q(buffer) where q = 1 - ρ*
        8. Adjust γ based on observed clip rate ρ

    Args:
        learning_rate: Base learning rate (η_base). Default: 3e-4
        betas: Adam momentum decay rates (β₁, β₂). Default: (0.9, 0.999)
        eps: Numerical stability constant. Default: 1e-8
        weight_decay: Weight decay coefficient (λ). Default: 0.01
        initial_clip_norm: Initial clipping threshold C. Default: 3.0
        noise_multiplier: DP noise scale σ (required)
        sample_rate: Sampling rate q = batch_size / dataset_size (required)
        target_delta: Target δ for (ε, δ)-DP (required)
        target_clip_rate: Target fraction of clipped gradients (ρ*). Default: 0.20
        history_size: Gradient norm buffer size (H). Default: 1000
        clip_norm_min: Minimum C (C_min). Default: 0.1
        clip_norm_max: Maximum C (C_max). Default: 10.0
        lr_multiplier_min: Minimum γ (γ_min). Default: 0.1
        lr_multiplier_max: Maximum γ (γ_max). Default: 2.0
        lr_increase_factor: Multiplicative increase (↑). Default: 1.01
        lr_decrease_factor: Multiplicative decrease (↓). Default: 0.995
        clip_rate_tolerance: Tolerance around ρ* for LR adjustment. Default: 0.10
        ema_decay: EMA smoothing factor (d). Default: 0.999
        accountant_type: "rdp" or "pld". Default: "rdp"
        seed: Random seed for reproducible noise. Default: 42

    Returns:
        (init_fn, step_fn) for DP-AdamW-AC optimization where:
          - init_fn(params) -> AdaptiveClipState
          - step_fn(params, grads, grad_norms, state, batch_sizes=None)
              -> (new_params, new_state, metrics)

    Example:
        >>> from opaque.optimizers import dp_adamw_ac
        >>> from opaque import clipped_grad
        >>> import torch
        >>>
        >>> # Setup
        >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(5)}
        >>>
        >>> def loss_fn(params, x, y):
        ...     logits = x @ params['weight'] + params['bias']
        ...     return ((logits - y) ** 2).mean()
        >>>
        >>> # Create optimizer
        >>> init_fn, step_fn = dp_adamw_ac(
        ...     learning_rate=3e-4,
        ...     weight_decay=0.01,
        ...     initial_clip_norm=3.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ...     target_clip_rate=0.20,
        ... )
        >>>
        >>> # Initialize
        >>> state = init_fn(params)
        >>>
        >>> # Create clipped gradient function with return_grad_norms=True
        >>> clipped_grad_fn = clipped_grad(
        ...     loss_fn,
        ...     argnums=0,
        ...     batch_argnums=(1, 2),
        ...     l2_clip_norm=state.current_clip_norm,  # Initial value
        ...     return_grad_norms=True,
        ... )
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in dataloader:
        ...     # Update clipping threshold to current adaptive value
        ...     clipped_grad_fn.keywords['l2_clip_norm'] = state.current_clip_norm
        ...
        ...     # Compute clipped gradients with pre-clip norms
        ...     grads, aux = clipped_grad_fn(params, x_batch, y_batch)
        ...     grad_norms = aux.grad_norms
        ...
        ...     # DP-AdamW-AC step (batch_sizes optional for standard training)
        ...     params, state, metrics = step_fn(params, grads, grad_norms, state)
        ...
        ...     # Monitor adaptive behavior
        ...     if state.step % 100 == 0:
        ...         print(f"Step {state.step}:")
        ...         print(f"  ε = {metrics['epsilon']:.2f}")
        ...         print(f"  C = {metrics['clip_norm']:.2f}")
        ...         print(f"  ρ = {metrics['clip_rate']:.1%}")
        ...         print(f"  γ = {metrics['lr_multiplier']:.2f}")

    Notes:
        - Requires gradient norms BEFORE clipping (use return_grad_norms=True)
        - Clip norm C adapts automatically based on gradient distribution
        - Learning rate scales with clip rate to maintain stable training
        - EMA parameters θ̂ can be used for evaluation (better generalization)
        - Weight decay is explicit (not relying on noise as implicit regularizer)
        - Typically achieves 1-3% better accuracy than fixed clipping

    Difference from DP-Adam-AC:
        - DP-Adam-AC: No weight decay (noise acts as implicit regularizer)
        - DP-AdamW-AC: Explicit weight decay for better LLM training

    See Also:
        dp_adam_ac: Adam variant without weight decay
        dp_adamw: Standard DP-AdamW with fixed clipping
        ClipNormBuffer: Underlying adaptive clipping logic
    """
    # Create base AdamW optimizer
    base_optimizer = torchopt.adamw(
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    # Create privacy accountant
    if accountant_type == "rdp":
        accountant = None  # RDPAccountant()  # TODO: Update to functional API
    elif accountant_type == "pld":
        # from opaque.accounting import PLDAccountant  # TODO: Update to functional API

        accountant = None  # PLDAccountant()  # TODO: Update to functional API
    else:
        raise ValueError(f"Unknown accountant_type: {accountant_type}")

    # Compute clip rate thresholds
    rho_low, rho_high = compute_clip_rate_thresholds(target_clip_rate, clip_rate_tolerance)

    def init_fn(params):
        """Initialize DP-AdamW-AC state.

        Args:
            params: PyTree of model parameters

        Returns:
            AdaptiveClipState with all components initialized
        """
        # Initialize base AdamW optimizer
        opt_state = base_optimizer.init(params)

        # Create noise generator
        noise_gen = torch.Generator().manual_seed(seed)

        # Create gradient norm buffer
        clip_buffer = ClipNormBuffer(
            capacity=history_size,
            target_clip_rate=target_clip_rate,
        )

        # Initialize EMA parameters (copy of initial params)
        ema_params = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, params)

        return AdaptiveClipState(
            opt_state=opt_state,
            accountant=accountant,
            noise_gen=noise_gen,
            clip_buffer=clip_buffer,
            current_clip_norm=initial_clip_norm,
            lr_multiplier=1.0,  # Start at neutral
            ema_params=ema_params,
            step=0,
        )

    def step_fn(params, grads, grad_norms, state, batch_sizes: Optional[torch.Tensor] = None):
        """Perform one DP-AdamW-AC step with adaptive clipping.

        Args:
            params: Current PyTree of parameters
            grads: Clipped gradients (already clipped to current C)
            grad_norms: Pre-clip gradient norms (for adaptive threshold)
            state: Current AdaptiveClipState
            batch_sizes: Optional batch sizes for each example (for microbatching).
                If None, assumes all examples have size 1. Default: None

        Returns:
            Tuple of (new_params, new_state, metrics)
        """
        # Default batch_sizes to ones (standard training without microbatching)
        if batch_sizes is None:
            # Infer batch size from grad_norms
            if isinstance(grad_norms, torch.Tensor):
                if grad_norms.dim() == 0:
                    batch_sizes = torch.ones(1)
                else:
                    batch_sizes = torch.ones(len(grad_norms))
            else:
                batch_sizes = torch.ones(1)

        # 1. Add DP noise to gradients
        stddev = noise_multiplier * state.current_clip_norm
        noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=state.noise_gen)

        # 2. Get parameter updates from AdamW (with LR scaling)
        # Scale updates by γ (LR multiplier)
        scaled_grads = tree_map(lambda g: g * state.lr_multiplier, noisy_grads)

        updates, new_opt_state = base_optimizer.update(
            scaled_grads, state.opt_state, params=params
        )

        # 3. Apply updates to parameters
        new_params = torchopt.apply_updates(params, updates)

        # 4. Update EMA parameters: θ̂ ← d·θ̂ + (1-d)·θ
        new_ema_params = tree_map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            state.ema_params,
            new_params,
        )

        # 5. Update gradient norm buffer
        state.clip_buffer.update(grad_norms, batch_sizes)

        # 6. Compute adaptive clip norm: C ← Percentile_q(buffer)
        new_clip_norm = state.clip_buffer.get_adaptive_clip_norm(
            clip_norm_min=clip_norm_min,
            clip_norm_max=clip_norm_max,
        )

        # 7. Compute current clip rate
        current_clip_rate = state.clip_buffer.get_clip_rate(state.current_clip_norm)

        # 8. Adjust learning rate multiplier based on clip rate
        new_lr_multiplier = clip_rate_based_lr_adjustment(
            current_lr_multiplier=state.lr_multiplier,
            clip_rate=current_clip_rate,
            target_clip_rate=target_clip_rate,
            clip_rate_low=rho_low,
            clip_rate_high=rho_high,
            increase_factor=lr_increase_factor,
            decrease_factor=lr_decrease_factor,
            lr_multiplier_min=lr_multiplier_min,
            lr_multiplier_max=lr_multiplier_max,
        )

        # 9. Track privacy
        new_accountant = state.accountant
        new_accountant.step_poisson(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            num_steps=1,
        )

        # 10. Compute current privacy cost
        epsilon = new_accountant.get_epsilon(target_delta=target_delta)

        # Create new state
        new_state = AdaptiveClipState(
            opt_state=new_opt_state,
            accountant=new_accountant,
            noise_gen=state.noise_gen,
            clip_buffer=state.clip_buffer,  # Mutable, already updated
            current_clip_norm=new_clip_norm,
            lr_multiplier=new_lr_multiplier,
            ema_params=new_ema_params,
            step=state.step + 1,
        )

        # Metrics for monitoring
        metrics = {
            "epsilon": epsilon,
            "delta": target_delta,
            "step": new_state.step,
            "clip_norm": new_clip_norm,
            "clip_rate": current_clip_rate,
            "lr_multiplier": new_lr_multiplier,
            "effective_lr": learning_rate * new_lr_multiplier,
        }

        return new_params, new_state, metrics

    return init_fn, step_fn


__all__ = ["dp_adamw_ac"]
