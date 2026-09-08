# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-implementation validation.

The per-family suites run under ``sdpa`` (the transformers default). This file
keeps an explicit cross-check that representative families — one dense (llama),
one MoE (mellum) — work under DP ``vmap(grad)`` on BOTH ``eager`` (the
O(N²) reference) and ``sdpa``.
"""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_no_grad,
    assert_vmap_grad,
    build_moe_model,
    get_tiny_config_kwargs,
)

IMPLS = ["eager", "sdpa"]


def _build_llama(device, impl="sdpa"):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from opaque.patches import apply_model_patches

    config = LlamaConfig(**get_tiny_config_kwargs())
    config._attn_implementation = impl
    model = LlamaForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def _sdpa_backends():
    from torch.nn.attention import SDPBackend

    return [
        pytest.param(SDPBackend.MATH, id="math"),
        pytest.param(SDPBackend.EFFICIENT_ATTENTION, id="efficient"),
        pytest.param(SDPBackend.CUDNN_ATTENTION, id="cudnn"),
        pytest.param(SDPBackend.FLASH_ATTENTION, id="flash"),
    ]


def _backward_operator(backend):
    from torch.nn.attention import SDPBackend

    return {
        SDPBackend.EFFICIENT_ATTENTION: (
            "aten::_scaled_dot_product_efficient_attention_backward"
        ),
        SDPBackend.CUDNN_ATTENTION: "aten::_scaled_dot_product_cudnn_attention_backward",
        SDPBackend.FLASH_ATTENTION: "aten::_scaled_dot_product_flash_attention_backward",
    }.get(backend)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SDPA backends need CUDA")
@pytest.mark.parametrize("backend", _sdpa_backends())
def test_sdpa_backends_under_vmap(backend, device):
    """Fused SDPA backward batches the physical DP examples in one dispatch."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile

    # The fused SDPA kernels (efficient/cudnn/flash) only provide a bf16 path on
    # Ampere+ (sm>=80); on older GPUs (e.g. Turing/T4, sm_75) they raise
    # "No available kernel". The MATH backend is pure-PyTorch and runs anywhere.
    if backend is not SDPBackend.MATH and device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            pytest.skip(
                "fused bf16 SDPA (efficient/cudnn/flash) requires CUDA sm>=80; "
                f"this GPU is sm_{major}x"
            )

    model = _build_llama(device, "sdpa")
    with (
        sdpa_kernel([backend]),
        profile(activities=[ProfilerActivity.CPU]) as profiler,
    ):
        assert_vmap_grad(model, device, dtype=torch.bfloat16)

    backward_operator = _backward_operator(backend)
    if backward_operator is not None:
        selected = {
            event.key: event.count
            for event in profiler.key_averages()
            if event.key.startswith("aten::_scaled_dot_product_")
            and event.key.endswith("_backward")
        }
        assert backward_operator in selected, f"selected SDPA operators: {selected}"
        # Python batching records the outer dispatch and the merged redispatch.
        assert selected[backward_operator] <= 2 * model.config.num_key_value_heads


@pytest.mark.parametrize("impl", IMPLS)
def test_llama_attention_impl(impl, device):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from opaque.patches import apply_model_patches

    config = LlamaConfig(**get_tiny_config_kwargs())
    config._attn_implementation = impl
    model = LlamaForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    assert_forward_no_grad(model, device)
    assert_vmap_grad(model, device)


@pytest.mark.parametrize("impl", IMPLS)
def test_mellum_attention_impl(impl, device):
    model, _ = build_moe_model(
        "mellum",
        device,
        attn_impl=impl,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    assert_forward_no_grad(model, device)
    assert_vmap_grad(model, device)
