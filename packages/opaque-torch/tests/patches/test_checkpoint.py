# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Gradient checkpointing under vmap(grad(...)).

Verifies that the Torch provider's checkpoint patches are applied and that
``torch.utils.checkpoint`` then composes with ``vmap``/``grad``/``clipped_grad``
and saves the memory it is meant to save. The Hugging Face half of this concern
is exercised where it lives, beside the Hugging Face patches.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap
from torch.utils.checkpoint import checkpoint

from opaque.torch import apply_runtime_patches
from opaque.torch.checkpoint import is_checkpoint_patched

apply_runtime_patches(vmap_checkpointing=True)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Checkpoint patch tests require CUDA",
    ),
]


def _peak_memory(fn) -> int:
    """Peak CUDA allocation of a second, warmed-up call to ``fn``."""
    import gc

    fn()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated()


def _assert_savings(mem_plain: int, mem_ckpt: int) -> None:
    savings = (mem_plain - mem_ckpt) / mem_plain
    assert savings > 0.15, (
        f"Expected >15% memory savings from checkpoint, got {savings:.1%} "
        f"(plain={mem_plain / 1e6:.0f}MB, ckpt={mem_ckpt / 1e6:.0f}MB)"
    )


class TestCheckpointPatches:
    """Verify checkpoint patches are applied and functional."""

    def test_patches_applied(self):
        assert is_checkpoint_patched()

    def test_vmap_grad_with_checkpoint(self, device):
        """vmap(grad(f)) works when f uses checkpoint internally."""
        W = torch.randn(32, 32, device=device)

        def f_ckpt(x):
            h = checkpoint(lambda x: F.gelu(x @ W), x, use_reentrant=False)
            h = checkpoint(lambda x: F.gelu(x @ W), h, use_reentrant=False)
            return h.sum()

        def f_ref(x):
            return F.gelu(F.gelu(x @ W) @ W).sum()

        x = torch.randn(4, 32, device=device)
        g = vmap(grad(f_ckpt))(x)
        g_ref = vmap(grad(f_ref))(x)
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-5)

    def test_functional_call_param_context_protocol(self, device):
        """functional_call + checkpoint: recomputation sees the correct params.

        Verifies that checkpoint recomputation reads the functional_call
        parameters, which the patch re-binds for the backward recompute.
        """
        model = torch.nn.Linear(32, 32, bias=False).to(device)

        new_weight = torch.randn(32, 32, device=device)
        params = {"weight": new_weight}

        def f(x):
            h = checkpoint(
                lambda x: F.gelu(torch.func.functional_call(model, params, (x,))),
                x,
                use_reentrant=False,
            )
            return h.sum()

        x = torch.randn(4, 32, device=device)
        g = vmap(grad(f))(x)

        # Reference: no checkpoint
        def f_ref(x):
            return F.gelu(torch.func.functional_call(model, params, (x,))).sum()

        g_ref = vmap(grad(f_ref))(x)
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-5)

    def test_checkpoint_saves_memory(self, device):
        """Checkpoint with patches actually reduces peak GPU memory."""
        if device.type != "cuda":
            pytest.skip("Memory measurement requires CUDA")

        d, n_layers = 512, 8
        Ws = [torch.randn(d, d, device=device) for _ in range(n_layers)]

        def f_plain(x):
            for W in Ws:
                x = F.gelu(x @ W)
            return x.sum()

        def f_ckpt(x):
            for s in range(0, n_layers, 4):

                def block(x, s=s):
                    for j in range(s, min(s + 4, n_layers)):
                        x = F.gelu(x @ Ws[j])
                    return x

                x = checkpoint(block, x, use_reentrant=False)
            return x.sum()

        B, N = 4, 1024
        with torch.no_grad():
            mem_plain = _peak_memory(
                lambda: vmap(grad(f_plain))(torch.randn(B, N, d, device=device))
            )
            mem_ckpt = _peak_memory(
                lambda: vmap(grad(f_ckpt))(torch.randn(B, N, d, device=device))
            )

        _assert_savings(mem_plain, mem_ckpt)

    def test_clipped_grad_saves_memory(self, device):
        """clipped_grad preserves checkpoint memory savings."""
        if device.type != "cuda":
            pytest.skip("Memory measurement requires CUDA")

        from opaque.api.engine.clipping import clipped_grad

        d, n_layers = 512, 8
        params = {
            "weights": [torch.randn(d, d, device=device) for _ in range(n_layers)]
        }

        def loss_plain(p, x):
            for weight in p["weights"]:
                x = F.gelu(x @ weight)
            return x.sum()

        def loss_ckpt(p, x):
            for s in range(0, n_layers, 4):

                def block(x, s=s):
                    for j in range(s, min(s + 4, n_layers)):
                        x = F.gelu(x @ p["weights"][j])
                    return x

                x = checkpoint(block, x, use_reentrant=False)
            return x.sum()

        grad_plain, state_plain = clipped_grad(loss_plain, clipping_norm=1.0)
        grad_ckpt, state_ckpt = clipped_grad(loss_ckpt, clipping_norm=1.0)

        B, N = 4, 1024
        mem_plain = _peak_memory(
            lambda: grad_plain(
                params, torch.randn(B, N, d, device=device), state=state_plain
            )
        )
        mem_ckpt = _peak_memory(
            lambda: grad_ckpt(
                params, torch.randn(B, N, d, device=device), state=state_ckpt
            )
        )

        _assert_savings(mem_plain, mem_ckpt)
