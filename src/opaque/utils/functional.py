"""Functional utilities for PyTorch models.

This module provides helpers for working with PyTorch's functional API,
particularly for converting stateful nn.Module objects to functional form
compatible with torch.func transformations.
"""

from collections.abc import Callable  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def make_functional(
    mod: nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> tuple[Callable, tuple[torch.Tensor, ...]] | tuple[Callable, dict, dict]:
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
        partition_trainable: If True, return (fmodel, trainable_params, frozen_params)
            as dicts instead of (fmodel, params) as tuple. Partitioning is based on
            the `requires_grad` attribute of the original module's parameters.
            This is ideal for LoRA-style fine-tuning. Default: False.

    Returns:
        If partition_trainable=False:
            A tuple containing:
            - fmodel: Functional version of the module. Takes parameters as first argument,
              followed by the module's normal forward arguments.
            - params: Tuple of the module's parameter tensors.

        If partition_trainable=True:
            A tuple containing:
            - fmodel: Functional version of the module. Takes parameters dict as first argument.
            - trainable_params: Dict of parameters where requires_grad=True.
            - frozen_params: Dict of parameters where requires_grad=False.

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

    Example with LoRA partitioning:
        >>> import torch.nn as nn
        >>> from opaque import dp_adam, clipped_grad
        >>> from opaque.utils import make_functional, merge
        >>>
        >>> # Create model and freeze backbone
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 5, bias=False),  # Pretrained backbone
        ...     nn.Linear(5, 1, bias=False),   # Pretrained backbone
        ... )
        >>>
        >>> # Freeze first layer, keep second trainable
        >>> model[0].weight.requires_grad = False  # Frozen
        >>> model[1].weight.requires_grad = True   # Trainable
        >>>
        >>> # Convert with automatic partitioning
        >>> fmodel, trainable, frozen = make_functional(
        ...     model, partition_trainable=True
        ... )
        >>>
        >>> # Only optimize trainable parameters
        >>> init_fn, step_fn = dp_adam(
        ...     learning_rate=1e-3,
        ...     l2_clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ... )
        >>> state = init_fn(trainable)
        >>>
        >>> # Training loop
        >>> def loss_fn(train_params, x, y):
        ...     all_params = merge(frozen, train_params)
        ...     pred = fmodel(all_params, x)
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> clipped_grad_fn = clipped_grad(
        ...     loss_fn, argnums=0, batch_argnums=(1, 2), l2_clip_norm=1.0
        ... )
        >>>
        >>> for x_batch, y_batch in dataloader:
        ...     grads = clipped_grad_fn(trainable, x_batch, y_batch)
        ...     trainable, state, metrics = step_fn(trainable, grads, state)

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
    # Extract parameters with their requires_grad status
    params_dict = dict(mod.named_parameters())

    # For functional_call, we don't actually need to create a stateless copy
    # The original module works fine, and functional_call will replace parameters
    # Creating a meta device copy breaks some models (like HuggingFace transformers)
    # that have buffers or special initialization logic
    stateless_mod = mod

    # Optionally detach parameters from autograd graph
    if disable_autograd_tracking:
        params_dict = {name: param.detach() for name, param in params_dict.items()}

    if partition_trainable:
        # Partition based on requires_grad (BEFORE detaching!)
        # We need to check the original parameters
        original_params = dict(mod.named_parameters())
        trainable_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if param.requires_grad
        }
        frozen_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if not param.requires_grad
        }

        def fmodel_dict(params_dict_input, *args, **kwargs):
            """Functional version that takes dict parameters.

            Args:
                params_dict_input: Dict of parameter tensors.
                *args: Positional arguments for the module's forward method.
                **kwargs: Keyword arguments for the module's forward method.

            Returns:
                Output of the module's forward pass.
            """
            return torch.func.functional_call(
                stateless_mod, params_dict_input, args, kwargs
            )

        return fmodel_dict, trainable_params, frozen_params

    else:
        # Original behavior: return tuple
        params_names = list(params_dict.keys())
        params_values = tuple(params_dict.values())

        def fmodel_tuple(new_params_values, *args, **kwargs):
            """Functional version that takes tuple parameters.

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
            return torch.func.functional_call(
                stateless_mod, new_params_dict, args, kwargs
            )

        return fmodel_tuple, params_values


__all__ = ["make_functional"]
