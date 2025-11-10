"""Shared test utilities for Opaque tests."""

import torch


def compute_norm(pytree: dict) -> torch.Tensor:
    """Compute global L2 norm of a PyTree (for test assertions).

    Args:
        pytree: Dictionary of tensors

    Returns:
        Global L2 norm as scalar tensor
    """
    from opaque.core.pytree_utils import global_norm

    return global_norm(pytree)


def assert_pytrees_close(
    actual: dict,
    expected: dict,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    msg: str = "",
):
    """Assert two PyTrees have numerically close values.

    Args:
        actual: Actual PyTree
        expected: Expected PyTree
        atol: Absolute tolerance
        rtol: Relative tolerance
        msg: Optional message to display on failure
    """
    from opaque.core.pytree_utils import tree_leaves

    actual_leaves = tree_leaves(actual)
    expected_leaves = tree_leaves(expected)

    assert len(actual_leaves) == len(expected_leaves), f"PyTree structure mismatch. {msg}"

    for i, (a, e) in enumerate(zip(actual_leaves, expected_leaves)):
        assert torch.allclose(a, e, atol=atol, rtol=rtol), (
            f"Leaf {i} not close:\nActual: {a}\nExpected: {e}\n{msg}"
        )


def create_random_pytree(
    keys: list[str], shapes: list[tuple[int, ...]], seed: int = 42
) -> dict[str, torch.Tensor]:
    """Create random PyTree for testing.

    Args:
        keys: List of keys for the PyTree
        shapes: List of shapes corresponding to each key
        seed: Random seed

    Returns:
        Dictionary of random tensors
    """
    torch.manual_seed(seed)
    return {key: torch.randn(shape) for key, shape in zip(keys, shapes)}


def jax_to_torch(jax_array):
    """Convert JAX array to PyTorch tensor.

    Args:
        jax_array: JAX numpy array

    Returns:
        PyTorch tensor with same values
    """
    try:
        import jax.numpy as jnp

        return torch.from_numpy(jnp.asarray(jax_array))
    except ImportError:
        raise ImportError("JAX not installed. Install with 'uv sync --group jax-validation'")


def torch_to_jax(torch_tensor):
    """Convert PyTorch tensor to JAX array.

    Args:
        torch_tensor: PyTorch tensor

    Returns:
        JAX numpy array with same values
    """
    try:
        import jax.numpy as jnp

        return jnp.asarray(torch_tensor.detach().cpu().numpy())
    except ImportError:
        raise ImportError("JAX not installed. Install with 'uv sync --group jax-validation'")
