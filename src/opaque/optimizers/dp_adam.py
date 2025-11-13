"""DP-Adam: Differentially Private Adam Optimizer.

This module implements DP-Adam using TorchOpt's functional Adam optimizer
with integrated gradient clipping, noise injection, and privacy accounting.

DP-Adam combines the adaptive learning rate benefits of Adam with formal
privacy guarantees through gradient clipping and noise injection.
"""

from collections.abc import Callable

import torchopt

from opaque.optimizers.base import make_dp_optimizer


def dp_adam(
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
    seed: int = 42,
) -> tuple[Callable, Callable]:
    """Create a DP-Adam optimizer using TorchOpt.

    DP-Adam applies the Adam optimizer with differential privacy guarantees
    by adding noise to clipped gradients before computing adaptive moments.

    The algorithm:
        1. Clip per-example gradients to max L2 norm C
        2. Sum clipped gradients
        3. Add Gaussian noise N(0, σ²C²I)
        4. Apply Adam update with first/second moments

    Args:
        learning_rate: Learning rate for Adam (default: 1e-3)
        betas: Coefficients for computing running averages of gradient
            and its square (β₁, β₂). Default: (0.9, 0.999)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        l2_clip_norm: Maximum L2 norm for gradient clipping (C)
        noise_multiplier: Noise scale relative to clip norm (σ)
        sample_rate: Sampling rate q = batch_size / dataset_size
        target_delta: Target δ for (ε, δ)-DP
        accountant_type: "rdp" or "pld" for privacy accounting
        seed: Random seed for reproducible noise generation

    Returns:
        (init_fn, step_fn) for DP-Adam optimization where:
          - init_fn(params) -> DPOptimizerState
          - step_fn(params, clipped_grads, state) -> (new_params, new_state, metrics)

    Example:
        >>> from opaque.optimizers import dp_adam
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
        >>> # Create DP-Adam optimizer
        >>> init_fn, step_fn = dp_adam(
        ...     learning_rate=1e-3,
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
        ...     # DP-Adam step (adds noise + adaptive updates)
        ...     params, state, metrics = step_fn(params, grads, state)
        ...
        ...     if state.step % 100 == 0:
        ...         print(f"Step {state.step}: ε={metrics['epsilon']:.2f}")

    Notes:
        - Gradients must be pre-clipped before passing to step_fn
        - Use opaque.clipped_grad() to compute clipped gradients
        - Adam's adaptive learning rate helps with varying gradient scales
        - No weight decay: DP noise acts as implicit regularization
        - For fine-tuning LLMs, consider DP-Adam-AC (adaptive clipping)

    See Also:
        dp_adam_ac: Adaptive clipping variant with better performance
    """
    # Create base TorchOpt Adam optimizer
    # Note: weight_decay=0 because DP noise acts as regularizer
    base_optimizer = torchopt.adam(
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
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


__all__ = ["dp_adam"]
