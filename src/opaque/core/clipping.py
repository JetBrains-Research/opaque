"""Per-example gradient clipping for differential privacy.

This module implements gradient clipping mechanisms required for DP-SGD training.
The core function `clipped_grad` computes per-example gradients and clips each to
a maximum L2 norm before summing.

NOTE: Implementation stubs only - to be implemented following TDD workflow.
"""

import torch


def clip_pytree(
    pytree: dict[str, torch.Tensor],
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
    nan_safe: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip PyTree of tensors to maximum L2 norm.

    Computes the global L2 norm across all tensors in the PyTree and scales
    them proportionally to satisfy the norm constraint. If the original norm
    is already below clip_norm, tensors are returned unchanged.

    Args:
        pytree: Dictionary of tensors (e.g., model gradients)
        clip_norm: Maximum L2 norm allowed
        rescale_to_unit_norm: If True, scale output to unit norm instead of clip_norm
        nan_safe: If True, replace NaNs/Infs with zeros before clipping

    Returns:
        Tuple of (clipped_pytree, original_norm)

    Raises:
        ValueError: If clip_norm is negative or NaN

    Example:
        >>> grads = {'weight': torch.tensor([3.0, 4.0]), 'bias': torch.tensor([0.0])}
        >>> clipped, norm = clip_pytree(grads, clip_norm=1.0)
        >>> # Original norm was 5.0, so gradients are scaled by 1.0/5.0
        >>> torch.isclose(clipped['weight'], torch.tensor([0.6, 0.8]))
        tensor(True)

    References:
        Abadi et al. 2016, "Deep Learning with Differential Privacy"
        https://arxiv.org/abs/1607.00133
    """
    raise NotImplementedError("To be implemented following TDD workflow - see CONTRIBUTING.md")


def clipped_grad(
    fun,
    argnums: int | tuple[int, ...] = 0,
    *,
    l2_clip_norm: float,
    batch_argnums: int | tuple[int, ...] = 1,
    rescale_to_unit_norm: bool = False,
    normalize_by: float = 1.0,
    keep_batch_dim: bool = True,
    microbatch_size: int | None = None,
    return_grad_norms: bool = False,
):
    """Return function computing sum of clipped per-example gradients.

    Transforms a scalar loss function to compute per-example gradients,
    clip each to max L2 norm, and sum the clipped gradients. This is the
    core operation required for DP-SGD training.

    Note: This function does NOT add noise. Noise injection is handled
    separately (see opaque.core.noise module).

    Args:
        fun: Scalar loss function with signature (params, data) -> loss
        argnums: Which argument positions to differentiate w.r.t. (typically params)
        l2_clip_norm: Maximum gradient norm per example
        batch_argnums: Which argument positions have batch dimension (typically data)
        rescale_to_unit_norm: If True, sensitivity = 1.0; else sensitivity = clip_norm
        normalize_by: Divide result by this value (typically batch_size for averaging)
        keep_batch_dim: If True, pass data with batch dim to loss; else squeeze
        microbatch_size: Process batch in chunks of this size (None = no microbatching)
        return_grad_norms: If True, also return per-example norms before clipping

    Returns:
        Callable that computes clipped gradients. If return_grad_norms=True,
        returns tuple (clipped_grads, per_example_norms).

    Example:
        >>> # Loss for single example
        >>> def loss_fn(param, data):
        ...     return 0.5 * ((data - param) ** 2).mean()
        >>>
        >>> # Create clipped gradient function
        >>> clipped_grad_fn = clipped_grad(
        ...     loss_fn,
        ...     l2_clip_norm=1.0,
        ...     normalize_by=3.0,  # batch size
        ... )
        >>>
        >>> param = torch.tensor(3.0, requires_grad=True)
        >>> data = torch.tensor([0.0, 7.0, -2.0])
        >>> grad = clipped_grad_fn(param, data)
        >>> # Each per-example gradient is clipped to norm 1.0, then summed and averaged

    References:
        This implements the gradient clipping step of DP-SGD:
        Abadi et al. 2016, "Deep Learning with Differential Privacy"

        Functional API design inspired by JAX-Privacy:
        https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/src/experimental/clipping.py
    """
    # TODO: Implementation coming in Phase 1
    # This is a stub to enable test writing
    raise NotImplementedError(
        "clipped_grad implementation is in progress (Stage 1). "
        "This stub enables test-driven development. "
        "See docs/development/stage1-plan.md for details."
    )
