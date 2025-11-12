"""Utilities for converting PyTorch modules to functional form.

This module provides helpers for working with PyTorch's functional API,
particularly for converting stateful nn.Module objects to functional form
compatible with torch.func transformations.
"""

import copy
from typing import Callable, Tuple

import torch
import torch.nn as nn


def make_functional(
    mod: nn.Module,
    disable_autograd_tracking: bool = False,
) -> Tuple[Callable, Tuple[torch.Tensor, ...]]:
    """Convert a PyTorch module to functional form.

    This helper mimics the behavior of the deprecated `functorch.make_functional()`.
    It creates a stateless version of the module that can be called with external
    parameters, enabling use with `torch.func` transformations like `vmap` and `grad`.

    The returned functional model can be used with per-example gradient computation,
    which is essential for differential privacy.

    Args:
        mod: PyTorch module to convert to functional form.
        disable_autograd_tracking: If True, detach parameters from autograd graph.
            Useful when you only need to compute gradients via torch.func.grad,
            not through standard .backward(). Default: False.

    Returns:
        A tuple containing:
        - fmodel: Functional version of the module. Takes parameters as first argument,
          followed by the module's normal forward arguments.
        - params: Tuple of the module's parameter tensors.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from opaque.functional_utils import make_functional
        >>>
        >>> # Create a simple model
        >>> model = nn.Linear(10, 1)
        >>>
        >>> # Convert to functional form
        >>> fmodel, params = make_functional(model)
        >>>
        >>> # Use with new parameters
        >>> x = torch.randn(5, 10)
        >>> output = fmodel(params, x)
        >>> print(output.shape)
        torch.Size([5, 1])
        >>>
        >>> # Works with torch.func transformations
        >>> from torch.func import grad, vmap
        >>>
        >>> def loss_fn(params, x, y):
        ...     pred = fmodel(params, x)
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Compute gradient w.r.t. parameters
        >>> x_batch = torch.randn(3, 10)
        >>> y_batch = torch.randn(3, 1)
        >>> grads = grad(loss_fn)(params, x_batch, y_batch)

    Example with per-example gradients:
        >>> # Per-example loss function
        >>> def loss_single(params, x, y):
        ...     pred = fmodel(params, x.unsqueeze(0))
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Compute per-example gradients with vmap
        >>> per_example_grads = vmap(grad(loss_single), in_dims=(None, 0, 0))(
        ...     params, x_batch, y_batch
        ... )
        >>> # per_example_grads is a tuple with shape (batch_size, *param_shape) for each param

    Note:
        This function creates a deep copy of the module and moves it to the "meta" device,
        which means the copied module has no actual parameter storage. The functional model
        uses the provided parameters during forward passes.

        If the module has an `allow_grad_accumulation()` method (used in some advanced
        modules), it will be called on the stateless copy.

    See Also:
        - PyTorch migration guide: https://pytorch.org/docs/master/func.migrating.html
        - torch.func.functional_call: The underlying primitive used by this function
    """
    # Extract parameters as ordered dict
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    # Create stateless version of module (parameters on "meta" device)
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to("meta")

    # Allow gradient accumulation if the module supports it
    # (some advanced modules like FSDP-wrapped models use this)
    if hasattr(stateless_mod, "allow_grad_accumulation"):
        stateless_mod.allow_grad_accumulation()

    def fmodel(new_params_values, *args, **kwargs):
        """Functional version of the module.

        Args:
            new_params_values: Tuple of parameter tensors.
            *args: Positional arguments for the module's forward method.
            **kwargs: Keyword arguments for the module's forward method.

        Returns:
            Output of the module's forward pass.
        """
        # Reconstruct parameter dict from tuple
        new_params_dict = {
            name: value for name, value in zip(params_names, new_params_values)
        }
        # Call module with external parameters
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)

    # Optionally detach parameters from autograd graph
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)

    return fmodel, params_values
