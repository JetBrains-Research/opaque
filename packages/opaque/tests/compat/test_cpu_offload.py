# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for vmap-compatible save_on_cpu (Patch 5 in _checkpoint_patches)."""

import pytest
import torch
import torch.nn as nn
from torch.func import grad, vmap
from torch.utils.checkpoint import checkpoint

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="save_on_cpu requires CUDA"
)


class CheckpointedMLP(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(self._layer_forward, layer, x, use_reentrant=False)
        return x.sum()

    @staticmethod
    def _layer_forward(layer, x):
        return layer(x).relu()


class TestSaveOnCpuVmap:
    def test_basic_vmap_grad(self):
        """save_on_cpu works with vmap(grad(...))."""
        import opaque  # noqa: F401 — applies patches

        def loss_fn(x):
            return (x**2).sum()

        x = torch.randn(4, 8, device="cuda")
        grads_vanilla = vmap(grad(loss_fn))(x)
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            grads_offload = vmap(grad(loss_fn))(x)
        torch.testing.assert_close(grads_offload, grads_vanilla)

    def test_with_checkpoint(self):
        """save_on_cpu composes with gradient checkpointing under vmap."""
        import opaque  # noqa: F401 — applies patches

        def loss_fn(x):
            def block(h):
                return (h * 2).relu()

            h = checkpoint(block, x, use_reentrant=False)
            h = checkpoint(block, h, use_reentrant=False)
            return h.sum()

        x = torch.randn(4, 8, device="cuda")
        grads_vanilla = vmap(grad(loss_fn))(x)
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            grads_offload = vmap(grad(loss_fn))(x)
        torch.testing.assert_close(grads_offload, grads_vanilla)

    def test_no_pin_memory(self):
        """save_on_cpu(pin_memory=False) uses .cpu() path."""
        import opaque  # noqa: F401 — applies patches

        x = torch.randn(4, 8, device="cuda")
        grads_vanilla = vmap(grad(lambda x: (x**2).sum()))(x)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            grads_offload = vmap(grad(lambda x: (x**2).sum()))(x)
        torch.testing.assert_close(grads_offload, grads_vanilla)
