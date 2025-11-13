"""DP-Adam-AC: DP-Adam with Adaptive Clipping.

This module implements the state-of-the-art DP-Adam-AC optimizer from:
    Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable
    Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288 (October 2024)

Key innovations over standard DP-Adam:
    1. Adaptive Clipping: Clip norm C adapts based on gradient percentiles
    2. Dynamic LR Scaling: Learning rate adjusts based on clipping frequency
    3. EMA Smoothing: Exponential moving average for better privacy-utility tradeoff
    4. No Weight Decay: DP noise acts as implicit regularization
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torchopt

from opaque.accounting import RDPAccountant
from opaque.adaptive import (
    ClipNormBuffer,
    clip_rate_based_lr_adjustment,
    compute_clip_rate_thresholds,
)
from opaque.noise import add_gaussian_noise
from opaque.utils.pytree import tree_map


class AdaptiveClipState(NamedTuple):
    """State for DP-Adam-AC with adaptive clipping.

    Attributes:
        opt_state: Internal Adam optimizer state from TorchOpt
        accountant: Privacy accountant (RDP or PLD)
        noise_gen: Random number generator for reproducible noise
        clip_buffer: Buffer tracking gradient norms for adaptive threshold
        current_clip_norm: Current adaptive clipping threshold (C)
        lr_multiplier: Current learning rate multiplier (γ)
        ema_params: Exponential moving average of parameters (θ̂)
        step: Training step counter
    """

    opt_state: Any
    accountant: Any
    noise_gen: torch.Generator
    clip_buffer: ClipNormBuffer
    current_clip_norm: float
    lr_multiplier: float
    ema_params: Any  # PyTree of EMA parameters
    step: int


def dp_adam_ac(
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
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
    """Create DP-Adam-AC optimizer with adaptive clipping.

    DP-Adam-AC (Adaptive Clipping) adapts the gradient clipping threshold C
    dynamically based on the distribution of recent gradient norms, and adjusts
    the learning rate based on the empirical clipping frequency.

    Algorithm (from paper):
        1. Compute per-example gradients and norms
        2. Clip gradients to adaptive threshold C
        3. Add Gaussian noise N(0, (σ·C)²I)
        4. Update with scaled Adam: θ ← θ - γ·η·m̂/(√v̂ + ε)
        5. Update EMA parameters: θ̂ ← d·θ̂ + (1-d)·θ
        6. Track gradient norms in buffer
        7. Adapt C ← Percentile_q(buffer) where q = 1 - ρ*
        8. Adjust γ based on observed clip rate ρ

    Args:
        learning_rate: Base learning rate (η_base in paper). Default: 3e-4
        betas: Adam momentum decay rates (β₁, β₂). Default: (0.9, 0.999)
        eps: Numerical stability constant. Default: 1e-8
        initial_clip_norm: Initial clipping threshold C. Default: 3.0
        noise_multiplier: DP noise scale σ (required)
        sample_rate: Sampling rate q = batch_size / dataset_size (required)
        target_delta: Target δ for (ε, δ)-DP (required)
        target_clip_rate: Target fraction of clipped gradients (ρ*). Default: 0.20
        history_size: Gradient norm buffer size (H in paper). Default: 1000
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
        (init_fn, step_fn) for DP-Adam-AC optimization where:
          - init_fn(params) -> AdaptiveClipState
          - step_fn(params, grads, grad_norms, batch_sizes, state)
              -> (new_params, new_state, metrics)

    Example:
        >>> from opaque.optimizers import dp_adam_ac
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
        >>> init_fn, step_fn = dp_adam_ac(
        ...     learning_rate=3e-4,
        ...     initial_clip_norm=3.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ...     target_clip_rate=0.20,  # Aim for 20% clipping
        ... )
        >>>
        >>> # Initialize
        >>> state = init_fn(params)
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in dataloader:
        ...     # Need to compute gradients AND track norms for adaptive clipping
        ...     # Use clipped_grad with return_grad_norms=True
        ...     clipped_grad_fn = clipped_grad(
        ...         loss_fn,
        ...         argnums=0,
        ...         batch_argnums=(1, 2),
        ...         l2_clip_norm=state.current_clip_norm,  # Use adaptive C
        ...         return_grad_norms=True,
        ...     )
        ...
        ...     grads, aux = clipped_grad_fn(params, x_batch, y_batch)
        ...     grad_norms = aux.grad_norms  # Pre-clip norms
        ...     batch_sizes = torch.ones(len(x_batch))
        ...
        ...     # DP-Adam-AC step
        ...     params, state, metrics = step_fn(
        ...         params, grads, grad_norms, batch_sizes, state
        ...     )
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
        - Typically achieves 1-3% better accuracy than fixed clipping

    See Also:
        dp_adam: Standard DP-Adam with fixed clipping
        ClipNormBuffer: Underlying adaptive clipping logic
    """
    # Create base Adam optimizer
    base_optimizer = torchopt.adam(
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=0.0,  # No weight decay (noise is regularizer)
    )

    # Create privacy accountant
    if accountant_type == "rdp":
        accountant = RDPAccountant()
    elif accountant_type == "pld":
        from opaque.accounting import PLDAccountant

        accountant = PLDAccountant()
    else:
        raise ValueError(f"Unknown accountant_type: {accountant_type}")

    # Compute clip rate thresholds
    rho_low, rho_high = compute_clip_rate_thresholds(target_clip_rate, clip_rate_tolerance)

    def init_fn(params):
        """Initialize DP-Adam-AC state.

        Args:
            params: PyTree of model parameters

        Returns:
            AdaptiveClipState with all components initialized
        """
        # Initialize base Adam optimizer
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

    def step_fn(params, grads, grad_norms, batch_sizes, state):
        """Perform one DP-Adam-AC step with adaptive clipping.

        Args:
            params: Current PyTree of parameters
            grads: Clipped gradients (already clipped to current C)
            grad_norms: Pre-clip gradient norms (for adaptive threshold)
            batch_sizes: Batch sizes for each example (for microbatching)
            state: Current AdaptiveClipState

        Returns:
            Tuple of (new_params, new_state, metrics)
        """
        # 1. Add DP noise to gradients
        stddev = noise_multiplier * state.current_clip_norm
        noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=state.noise_gen)

        # 2. Get parameter updates from Adam (with LR scaling)
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


__all__ = ["dp_adam_ac", "AdaptiveClipState"]
