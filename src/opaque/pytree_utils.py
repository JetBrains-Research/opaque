"""PyTree utilities for working with nested parameter structures.

This module provides utilities for working with PyTrees, i.e. nested Python
containers (dicts, lists, tuples, …) whose leaves are tensors. We implement a
thin wrapper on top of `optree` to provide a stable, fast pytree API that does
not depend on PyTorch's private modules.

Key functions:
- `tree_leaves(tree)`: return the list of tensor leaves in a PyTree.
- `tree_map(fn, *trees)`: apply a function to corresponding leaves of one or
  more PyTrees and rebuild the structure.
- `global_norm(tree)`: compute the global L2 norm across all tensor leaves:
  sqrt(sum(||leaf||^2)).

Notes
- Only tensor leaves are considered for `tree_leaves` and `global_norm`.
- `tree_map` applies `fn` to all leaves, passing through non-tensor leaves
  unchanged unless `fn` handles them explicitly.
- Supports mixed dtypes and complex numbers, returning complex norm for complex tensors.
- Handles empty trees gracefully, returning 0.0 for empty PyTrees.
- Supports PyTree parameters with nested dictionaries of tensors.
- Handles microbatching for memory efficiency in gradient clipping.
"""

from collections.abc import Callable
from typing import Any

import optree as _ot
import torch


def tree_leaves(tree: Any) -> list[torch.Tensor]:
    """Extract all leaf tensors from a PyTree.

    Args:
        tree: Nested structure (dict/list/tuple/…) containing tensors as leaves.

    Returns:
        List of all tensor leaves in the tree (non-tensor leaves are ignored).

    Example:
        >>> tree = {'a': torch.tensor([1, 2]), 'b': {'c': torch.tensor([3])}}
        >>> leaves = tree_leaves(tree)
        >>> len(leaves)
        2
    """
    flat, _ = _ot.tree_flatten(tree)
    return [x for x in flat if isinstance(x, torch.Tensor)]


def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    """Apply function to all leaves of one or more PyTrees.

    Args:
        fn: Function to apply to each leaf (or set of leaves from multiple trees).
        *trees: One or more PyTrees with matching structure.

    Returns:
        PyTree with same structure as inputs, with `fn` applied to leaves.

    Example:
        >>> tree = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([3.0])}
        >>> doubled = tree_map(lambda x: x * 2, tree)
        >>> doubled['a']
        tensor([2., 4.])
    """
    return _ot.tree_map(fn, *trees)


def global_norm(tree: Any) -> torch.Tensor:
    """Compute global L2 norm across all tensors in a PyTree.

    The global norm is the square root of the sum of squared norms of all
    leaf tensors:
        global_norm = sqrt(sum(||leaf||^2 for leaf in tree))

    Args:
        tree: PyTree of tensors (e.g., parameters or gradients)

    Returns:
        Scalar tensor containing the global L2 norm on the device of the first
        tensor leaf (or CPU if the tree is empty).

    Example:
        >>> tree = {'w': torch.tensor([3.0, 4.0]), 'b': torch.tensor([0.0, 12.0])}
        >>> norm = global_norm(tree)
        >>> # norm = sqrt(3^2 + 4^2 + 0^2 + 12^2) = sqrt(169) = 13
        >>> torch.isclose(norm, torch.tensor(13.0))
        True

    References:
        This function is commonly used in gradient clipping for deep learning.
        See: Pascanu et al. 2013, "On the difficulty of training RNNs"
    """
    leaves = tree_leaves(tree)
    if not leaves:
        return torch.tensor(0.0)

    # Determine common floating dtype for accumulation.
    # If all leaves are integer/bool, promote to float32.
    dtypes = [t.dtype for t in leaves]
    float_dtypes = [dt for dt in dtypes if torch.is_floating_point(torch.empty((), dtype=dt))]
    complex_dtypes = [dt for dt in dtypes if torch.is_complex(torch.empty((), dtype=dt))]

    if complex_dtypes:
        acc_dtype = torch.complex64
        for dt in complex_dtypes:
            acc_dtype = torch.promote_types(acc_dtype, dt)
        # For complex, we'll accumulate real magnitudes in the corresponding real dtype
        real_acc_dtype = torch.promote_types(
            torch.float32, torch.tensor(0, dtype=acc_dtype).real.dtype
        )
    elif float_dtypes:
        acc_dtype = torch.float32
        for dt in float_dtypes:
            acc_dtype = torch.promote_types(acc_dtype, dt)
        real_acc_dtype = acc_dtype
    else:
        # All integer/bool → accumulate in float32
        acc_dtype = torch.float32
        real_acc_dtype = torch.float32

    device = leaves[0].device
    total = torch.zeros((), device=device, dtype=real_acc_dtype)

    for leaf in leaves:
        if torch.is_complex(leaf):
            # ||z||^2 = (real^2 + imag^2)
            real = leaf.real.to(real_acc_dtype)
            imag = leaf.imag.to(real_acc_dtype)
            total = (
                total
                + (real * real).sum(dtype=real_acc_dtype)
                + (imag * imag).sum(dtype=real_acc_dtype)
            )
        else:
            x = leaf.to(real_acc_dtype)
            total = total + (x * x).sum(dtype=real_acc_dtype)

    return torch.sqrt(total)


__all__ = ["tree_leaves", "tree_map", "global_norm"]
