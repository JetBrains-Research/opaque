"""Torch-native device-placement behavior for PyTree mapping."""

from __future__ import annotations

import torch

from opaque import pytree


def test_tree_map_preserves_the_torch_device(device) -> None:
    tree = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        "bias": torch.tensor([0.5, -0.5], device=device),
    }

    doubled = pytree.tree_map(lambda value: value * 2, tree)

    assert doubled["weight"].device.type == device.type
    torch.testing.assert_close(doubled["weight"], tree["weight"] * 2)
    torch.testing.assert_close(doubled["bias"], tree["bias"] * 2)
