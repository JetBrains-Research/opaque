import pytest
import torch

from opaque.bounded import BoundedPytree, NoisyPytree
from opaque.clipping.per_group import PerGroup


def test_scalar_multiplication_scales_tensor_leaves_and_bound(device):
    bounded = BoundedPytree(
        {"w": torch.tensor([1.0, -2.0], device=device), "meta": None},
        bound=3.0,
    )

    scaled = -2.0 * bounded

    torch.testing.assert_close(
        scaled.pytree["w"], torch.tensor([-2.0, 4.0], device=device)
    )
    assert scaled.pytree["meta"] is None
    assert scaled.bound == pytest.approx(6.0)
    assert scaled.sensitivity == pytest.approx(6.0)


def test_division_scales_per_group_bound(device):
    bound = PerGroup(
        groups={"layer.weight": "layer", "head.weight": "head"},
        values={"layer": 2.0, "head": 4.0},
    )
    bounded = BoundedPytree(
        {"layer.weight": torch.ones(2, device=device)},
        bound=bound,
    )

    scaled = bounded / 2.0

    torch.testing.assert_close(
        scaled.pytree["layer.weight"], torch.full((2,), 0.5, device=device)
    )
    assert isinstance(scaled.bound, PerGroup)
    assert scaled.bound.groups == bound.groups
    assert scaled.bound.values == {"layer": 1.0, "head": 2.0}
    assert scaled.sensitivity == pytest.approx((1.0**2 + 2.0**2) ** 0.5)


def test_noisy_pytree_scales_noise_stddev(device):
    noisy = NoisyPytree(
        {"w": torch.tensor([1.0], device=device)},
        bound=2.0,
        noise_stddev=0.5,
    )

    scaled = noisy * -3.0

    assert isinstance(scaled, NoisyPytree)
    torch.testing.assert_close(scaled.pytree["w"], torch.tensor([-3.0], device=device))
    assert scaled.bound == pytest.approx(6.0)
    assert scaled.noise_stddev == pytest.approx(1.5)


def test_clone_detach_to_preserve_metadata(device):
    tensor = torch.tensor([1.0], device=device, requires_grad=True)
    bounded = BoundedPytree({"w": tensor}, bound=1.0)

    cloned = bounded.clone()
    detached = bounded.detach()
    moved = bounded.to(dtype=torch.float32)

    assert cloned.bound == bounded.bound
    assert cloned.pytree["w"] is not tensor
    assert detached.bound == bounded.bound
    assert detached.pytree["w"].requires_grad is False
    assert moved.bound == bounded.bound
    assert moved.pytree["w"].dtype == torch.float32


def test_unsupported_operations_raise_helpful_error(device):
    bounded = BoundedPytree({"w": torch.ones(1, device=device)}, bound=1.0)

    with pytest.raises(TypeError, match="reconstruct the bounded value"):
        _ = bounded + bounded
    with pytest.raises(TypeError, match="reconstruct the bounded value"):
        _ = bounded + 1.0
    with pytest.raises(TypeError, match="reconstruct the bounded value"):
        _ = 1.0 / bounded
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = bounded * bounded
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = bounded * torch.tensor(2.0, device=device)
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = bounded * True
    with pytest.raises(ZeroDivisionError):
        _ = bounded / 0.0
