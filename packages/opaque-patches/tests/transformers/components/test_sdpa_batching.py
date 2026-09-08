# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from opaque.api.patches.transformers.components import sdpa_batching


@pytest.mark.parametrize("bdim", [0, 1])
def test_merge_vmap_and_model_batch(bdim):
    tensor = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    expected = tensor.movedim(bdim, 0).flatten(0, 1)

    actual = sdpa_batching._merge_vmap_and_model_batch(tensor, bdim, tensor.shape[bdim])

    torch.testing.assert_close(actual, expected)


def test_merge_shared_tensor_expands_vmap_batch():
    tensor = torch.arange(2 * 3).reshape(2, 3)

    actual = sdpa_batching._merge_vmap_and_model_batch(tensor, None, 4)

    assert actual.shape == (8, 3)
    torch.testing.assert_close(actual.unflatten(0, (4, 2))[0], tensor)
    torch.testing.assert_close(actual.unflatten(0, (4, 2))[3], tensor)


def test_flash_rule_merges_physical_and_model_batches(monkeypatch):
    vmap_size, model_batch, heads, sequence, head_dim = 3, 2, 4, 5, 8
    query = torch.randn(vmap_size, model_batch, heads, sequence, head_dim)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seen_shapes = []

    def fake_backward(
        grad_out, merged_query, merged_key, merged_value, *args, **kwargs
    ):
        seen_shapes.append(merged_query.shape)
        return merged_query, merged_key, merged_value

    monkeypatch.setattr(
        torch.ops.aten._scaled_dot_product_flash_attention_backward,
        "default",
        fake_backward,
    )
    auxiliary = torch.empty(0, dtype=torch.int64)

    def backward(q, k, v):
        logsumexp = q[..., 0]
        return sdpa_batching._flash_attention_backward_batch_rule(
            q,
            q,
            k,
            v,
            q,
            logsumexp,
            auxiliary,
            auxiliary,
            sequence,
            sequence,
            0.0,
            False,
            auxiliary,
            auxiliary,
        )

    actual = torch.vmap(backward)(query, key, value)

    assert seen_shapes == [(vmap_size * model_batch, heads, sequence, head_dim)]
    for gradient, expected in zip(actual, (query, key, value), strict=True):
        torch.testing.assert_close(gradient, expected)


def test_flash_rule_supports_a_batched_cotangent(monkeypatch):
    vmap_size, model_batch, heads, sequence, head_dim = 3, 2, 4, 5, 8
    query = torch.randn(model_batch, heads, sequence, head_dim)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad_outputs = torch.randn(vmap_size, *query.shape)
    seen_shapes = []

    def fake_backward(
        grad_out, merged_query, merged_key, merged_value, *args, **kwargs
    ):
        seen_shapes.append((grad_out.shape, merged_query.shape))
        return merged_query, merged_key, merged_value

    monkeypatch.setattr(
        torch.ops.aten._scaled_dot_product_flash_attention_backward,
        "default",
        fake_backward,
    )
    auxiliary = torch.empty(0, dtype=torch.int64)
    logsumexp = query[..., 0]

    def backward(grad_out):
        return sdpa_batching._flash_attention_backward_batch_rule(
            grad_out,
            query,
            key,
            value,
            query,
            logsumexp,
            auxiliary,
            auxiliary,
            sequence,
            sequence,
            0.0,
            False,
            auxiliary,
            auxiliary,
        )

    actual = torch.vmap(backward)(grad_outputs)

    merged_shape = (vmap_size * model_batch, heads, sequence, head_dim)
    assert seen_shapes == [(merged_shape, merged_shape)]
    for gradient, expected in zip(actual, (query, key, value), strict=True):
        torch.testing.assert_close(
            gradient, expected.expand(vmap_size, *expected.shape)
        )


def test_batch_helpers_preserve_nested_vmap_dimensions():
    definition = torch.library.Library("opaque_sdpa_batching_test", "DEF")
    definition.define("identity(Tensor tensor) -> Tensor")
    implementation = torch.library.Library("opaque_sdpa_batching_test", "IMPL")
    implementation.impl("identity", lambda tensor: tensor, "CPU")
    batching = torch.library.Library(
        "opaque_sdpa_batching_test", "IMPL", "FuncTorchBatched"
    )

    def batch_rule(tensor):
        level = sdpa_batching._active_vmap_level(tensor)
        tensor, bdim = sdpa_batching._unwrap(tensor, level)
        batch_size = sdpa_batching._batch_size((tensor, bdim))
        merged = sdpa_batching._merge_vmap_and_model_batch(tensor, bdim, batch_size)
        output = torch.ops.opaque_sdpa_batching_test.identity(merged)
        return sdpa_batching._restore_vmap_batch(output, batch_size, level)

    batching.impl("identity", batch_rule)
    inputs = torch.randn(2, 3, 1, 4)

    actual = torch.vmap(
        torch.vmap(torch.ops.opaque_sdpa_batching_test.identity.default)
    )(inputs)

    torch.testing.assert_close(actual, inputs)


def test_supported_sdpa_schemas_are_recognized():
    assert all(
        sdpa_batching._has_expected_schema(operator)
        for operator in sdpa_batching._EXPECTED_SCHEMAS
    )


def test_install_fused_sdpa_batching_rules_is_idempotent():
    sdpa_batching.install_fused_sdpa_batching_rules()
    sdpa_batching.install_fused_sdpa_batching_rules()

    assert all(
        sdpa_batching._has_native_batch_rule(operator)
        for operator in sdpa_batching._EXPECTED_SCHEMAS
    )


def test_installer_defers_when_pytorch_has_native_rules(monkeypatch):
    monkeypatch.setattr(sdpa_batching, "_RULES_INSTALLED", False)
    monkeypatch.setattr(sdpa_batching, "_has_native_batch_rule", lambda _operator: True)
    monkeypatch.setattr(
        torch.library,
        "Library",
        lambda *_args, **_kwargs: pytest.fail("native rules must not be replaced"),
    )

    sdpa_batching.install_fused_sdpa_batching_rules()


def test_installer_keeps_fallback_for_incompatible_schemas(monkeypatch):
    monkeypatch.setattr(sdpa_batching, "_RULES_INSTALLED", False)
    monkeypatch.setattr(
        sdpa_batching, "_has_native_batch_rule", lambda _operator: False
    )
    monkeypatch.setattr(sdpa_batching, "_has_expected_schema", lambda _operator: False)
    monkeypatch.setattr(
        torch.library,
        "Library",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible operators must keep the fallback"
        ),
    )

    sdpa_batching.install_fused_sdpa_batching_rules()
