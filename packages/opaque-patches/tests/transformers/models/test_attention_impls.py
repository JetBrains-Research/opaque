# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-implementation validation.

The per-family suites run under ``sdpa`` (the transformers default). This file
keeps an explicit cross-check that representative families — one dense (llama),
one MoE (mellum) — work under DP ``vmap(grad)`` on BOTH ``eager`` (the
O(N²) reference) and ``sdpa``.
"""

import os
import sys

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, os.path.dirname(__file__))
from _test_utils import (  # noqa: E402
    build_moe_model,
    get_tiny_config_kwargs,
    assert_forward_no_grad,
    assert_vmap_grad,
)

IMPLS = ["eager", "sdpa"]


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
        "mellum", device, attn_impl=impl,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64,
    )
    assert_forward_no_grad(model, device)
    assert_vmap_grad(model, device)
