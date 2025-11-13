"""DP-SGD: Differentially Private Stochastic Gradient Descent.

This module implements DP-SGD using TorchOpt's functional SGD optimizer
with integrated gradient clipping, noise injection, and privacy accounting.

Reference:
    Abadi et al., "Deep Learning with Differential Privacy", CCS 2016
    https://arxiv.org/abs/1607.00133
"""

from collections.abc import Callable

import torchopt

from opaque.optimizers.base import make_dp_optimizer


def dp_sgd(
    learning_rate: float,
    momentum: float = 0.0,
    dampening: float = 0.0,
    nesterov: bool = False,
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
    seed: int = 42,
) -> tuple[Callable, Callable]:
    """Create a DP-SGD optimizer using TorchOpt.

    DP-SGD (Differentially Private Stochastic Gradient Descent) adds noise
    to clipped gradients to provide formal (ε, δ)-differential privacy guarantees.

    The algorithm:
        1. Clip per-example gradients to max L2 norm C
        2. Sum clipped gradients
        3. Add Gaussian noise N(0, σ²C²I)
        4. Apply SGD update with optional momentum

    Args:
        learning_rate: Learning rate for SGD (η)
        momentum: Momentum coefficient for SGD (default: 0.0 = no momentum)
        dampening: Dampening for momentum (default: 0.0)
        nesterov: Use Nesterov momentum (default: False)
        l2_clip_norm: Maximum L2 norm for gradient clipping (C)
        noise_multiplier: Noise scale relative to clip norm (σ)
        sample_rate: Sampling rate q = batch_size / dataset_size
        target_delta: Target δ for (ε, δ)-DP
        accountant_type: "rdp" or "pld" for privacy accounting
        seed: Random seed for reproducible noise generation

    Returns:
        (init_fn, step_fn) for DP-SGD optimization where:
          - init_fn(params) -> DPOptimizerState
          - step_fn(params, clipped_grads, state) -> (new_params, new_state, metrics)

    Example:
        >>> from opaque.optimizers import dp_sgd
        >>> from opaque import clipped_grad
        >>> import torch
        >>>
        >>> # Model and loss
        >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(5)}
        >>> def loss_fn(params, x, y):
        ...     logits = x @ params['weight'] + params['bias']
        ...     return ((logits - y) ** 2).mean()
        >>>
        >>> # Create clipped gradient function
        >>> clipped_grad_fn = clipped_grad(
        ...     loss_fn,
        ...     argnums=0,
        ...     batch_argnums=(1, 2),
        ...     l2_clip_norm=1.0,
        ... )
        >>>
        >>> # Create DP-SGD optimizer
        >>> init_fn, step_fn = dp_sgd(
        ...     learning_rate=0.1,
        ...     momentum=0.9,
        ...     l2_clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ... )
        >>>
        >>> # Initialize
        >>> state = init_fn(params)
        >>>
        >>> # Training loop
        >>> for x_batch, y_batch in dataloader:
        ...     # Compute clipped gradients
        ...     grads = clipped_grad_fn(params, x_batch, y_batch)
        ...
        ...     # DP-SGD step (adds noise + updates)
        ...     params, state, metrics = step_fn(params, grads, state)
        ...
        ...     if state.step % 100 == 0:
        ...         print(f"Step {state.step}: ε={metrics['epsilon']:.2f}")

    Notes:
        - Gradients must be pre-clipped before passing to step_fn
        - Use opaque.clipped_grad() to compute clipped gradients
        - Privacy accounting happens automatically per step
        - For per-epoch accounting, accumulate steps manually
    """
    # Create base TorchOpt SGD optimizer
    base_optimizer = torchopt.sgd(
        lr=learning_rate,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        weight_decay=0.0,  # No weight decay in DP-SGD (noise acts as regularizer)
    )

    # Wrap with DP functionality
    return make_dp_optimizer(
        base_optimizer,
        l2_clip_norm=l2_clip_norm,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        target_delta=target_delta,
        accountant_type=accountant_type,
        seed=seed,
    )


__all__ = ["dp_sgd"]
