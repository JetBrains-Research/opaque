"""CUDA tests for private bounded-memory sliding-window attention."""

import math

import pytest
import torch
from torch.func import grad, vmap

pytest.importorskip("triton")

from opaque.api.patches.kernels import (
    _can_use_triton_sliding_window_attention,
    _triton_sliding_window_attention,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _dense_reference(query, key, value, padding_mask, window, scale, softcap):
    heads_q = query.shape[-3]
    heads_kv = key.shape[-3]
    repeats = heads_q // heads_kv
    key = key.repeat_interleave(repeats, dim=-3)
    value = value.repeat_interleave(repeats, dim=-3)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    seq_len = query.shape[-2]
    row = torch.arange(seq_len, device=query.device)[:, None]
    col = torch.arange(seq_len, device=query.device)[None, :]
    valid = (col <= row) & (col > row - window)
    valid = valid.expand(*query.shape[:-3], heads_q, seq_len, seq_len)
    if padding_mask is not None:
        valid = valid & padding_mask.bool()[..., None, None, :]

    scores = scores.masked_fill(~valid, -torch.inf)
    row_has_value = valid.any(dim=-1, keepdim=True)
    scores = torch.where(row_has_value, scores, torch.zeros_like(scores))
    probabilities = torch.where(
        row_has_value,
        torch.softmax(scores, dim=-1),
        torch.zeros_like(scores),
    )
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def _run_parity_case(
    *,
    shape,
    kv_heads,
    window,
    dtype,
    padding_mask=None,
    softcap=None,
):
    torch.manual_seed(963)
    *leading, _heads_q, seq_len, head_dim = shape
    query = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    kv_shape = (*leading, kv_heads, seq_len, head_dim)
    key = torch.randn(kv_shape, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(kv_shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = 1.0 / math.sqrt(head_dim)
    upstream = torch.randn_like(query)

    reference = _dense_reference(
        query, key, value, padding_mask, window, scale, softcap
    )
    reference.backward(upstream)
    expected_grads = (query.grad.clone(), key.grad.clone(), value.grad.clone())
    query.grad = key.grad = value.grad = None

    actual = _triton_sliding_window_attention(
        query, key, value, padding_mask, window, scale, softcap
    )
    actual.backward(upstream)

    torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
    for actual_grad, expected_grad in zip(
        (query.grad, key.grad, value.grad), expected_grads, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_forward_backward_no_mask(dtype):
    _run_parity_case(
        shape=(2, 4, 64, 32),
        kv_heads=4,
        window=17,
        dtype=dtype,
    )


def test_forward_backward_padded_gqa_odd_shape():
    mask = torch.ones((2, 47), device="cuda", dtype=torch.float32)
    mask[0, :9] = False
    mask[1, 38:] = False
    mask[:, 23] = False
    _run_parity_case(
        shape=(2, 8, 47, 40),
        kv_heads=2,
        window=19,
        dtype=torch.float16,
        padding_mask=mask,
    )


def test_forward_backward_softcap():
    _run_parity_case(
        shape=(1, 6, 35, 24),
        kv_heads=3,
        window=13,
        dtype=torch.bfloat16,
        softcap=7.0,
    )


def test_fully_masked_rows_are_zero_with_zero_gradients():
    query = torch.randn(
        (2, 4, 33, 32), device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    mask = torch.zeros((2, 33), device="cuda", dtype=torch.int64)

    output = _triton_sliding_window_attention(
        query, key, value, mask, 11, 32**-0.5, None
    )
    output.sum().backward()

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    for tensor in (query, key, value):
        torch.testing.assert_close(
            tensor.grad, torch.zeros_like(tensor), rtol=0, atol=0
        )


def test_vmap_forward_and_grad_match_dense_reference():
    torch.manual_seed(12)
    mapped_batch, heads_q, heads_kv, seq_len, head_dim = 3, 4, 2, 37, 32
    query = torch.randn(
        mapped_batch,
        heads_q,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        mapped_batch,
        heads_kv,
        seq_len,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    mask = torch.ones((mapped_batch, seq_len), device="cuda", dtype=torch.bool)
    mask[0, :5] = False
    mask[1, 11:15] = False
    mask[2, 30:] = False
    window = 15
    scale = head_dim**-0.5

    def opaque(q, k, v, m):
        return _triton_sliding_window_attention(q, k, v, m, window, scale, 5.0)

    def reference(q, k, v, m):
        return _dense_reference(q, k, v, m, window, scale, 5.0)

    actual = vmap(opaque)(query, key, value, mask)
    expected = vmap(reference)(query, key, value, mask)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    def opaque_loss(q, k, v, m):
        return opaque(q, k, v, m).float().square().mean()

    def reference_loss(q, k, v, m):
        return reference(q, k, v, m).float().square().mean()

    actual_grads = vmap(grad(opaque_loss, argnums=(0, 1, 2)))(query, key, value, mask)
    expected_grads = vmap(grad(reference_loss, argnums=(0, 1, 2)))(
        query, key, value, mask
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


def test_autograd_state_is_linear_in_sequence_length():
    seq_len = 97
    query = torch.randn(
        (1, 4, seq_len, 32),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    output = _triton_sliding_window_attention(
        query, key, value, None, 31, 32**-0.5, None
    )

    saved = output.grad_fn.saved_tensors
    assert all(tensor.numel() <= query.numel() for tensor in saved)
    assert not any(
        tensor.ndim >= 2 and tensor.shape[-2:] == (seq_len, seq_len) for tensor in saved
    )


def test_eligibility_rejects_unsupported_inputs_without_launching():
    query = torch.randn((1, 4, 16, 32), device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    assert _can_use_triton_sliding_window_attention(
        query, key, value, None, 8, 32**-0.5, None
    )
    assert not _can_use_triton_sliding_window_attention(
        query.float(), key.float(), value.float(), None, 8, 32**-0.5, None
    )
    assert not _can_use_triton_sliding_window_attention(
        query, key[..., :-1, :], value[..., :-1, :], None, 8, 32**-0.5, None
    )
    assert not _can_use_triton_sliding_window_attention(
        query,
        key,
        value,
        torch.ones((1, 15), device="cuda", dtype=torch.bool),
        8,
        32**-0.5,
        None,
    )
