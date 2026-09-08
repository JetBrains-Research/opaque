# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MATRIX_DIMENSIONS = 2


def _register_projection_pack(module, name: str, components: Sequence[str]) -> None:
    """Register non-persistent storage used by an internal projection pack."""
    keys = getattr(module, "_opaque_projection_pack_keys", None)
    if keys is None:
        keys = {}
        module._opaque_projection_pack_keys = keys
    for component in components:
        buffer_name = f"_opaque_{name}_{component}_pack"
        if buffer_name not in module._buffers:
            module.register_buffer(buffer_name, torch.empty(0), persistent=False)
        keys.setdefault(buffer_name, None)


def _source_tensor(tensor: torch.Tensor) -> torch.Tensor:
    base = getattr(tensor, "_base", None)
    return base if base is not None else tensor


def _source_key(tensor: torch.Tensor) -> tuple:
    source = _source_tensor(tensor)
    return (
        id(source),
        source.untyped_storage().data_ptr(),
        source._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.device,
        tensor.dtype,
    )


def _can_mutate_cache(tensors: Sequence[torch.Tensor]) -> bool:
    # functional_call replaces Parameters with ordinary tensors. Do not mutate
    # module buffers while such an external parameter set is installed.
    return all(
        not torch._C._functorch.is_functorch_wrapped_tensor(tensor)
        and isinstance(_source_tensor(tensor), torch.nn.Parameter)
        for tensor in tensors
    )


def _cached_pack(
    module,
    buffer_name: str,
    sources: Sequence[torch.Tensor],
    extra_key: tuple,
    build: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if not _can_mutate_cache(sources):
        return build()

    key = tuple(_source_key(tensor) for tensor in sources) + extra_key
    keys = module._opaque_projection_pack_keys
    cached = getattr(module, buffer_name)
    if keys.get(buffer_name) == key and cached.numel() != 0:
        return cached

    packed = build()
    setattr(module, buffer_name, packed.detach())
    keys[buffer_name] = key
    return getattr(module, buffer_name)


def _cached_adapter_pack(
    module,
    buffer_names: tuple[str, str],
    sources: Sequence[torch.Tensor],
    extra_key: tuple,
    build: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_mutate_cache(sources):
        return build()

    key = tuple(_source_key(tensor) for tensor in sources) + extra_key
    keys = module._opaque_projection_pack_keys
    cached = tuple(getattr(module, name) for name in buffer_names)
    if all(keys.get(name) == key for name in buffer_names) and all(
        tensor.numel() != 0 for tensor in cached
    ):
        return cached

    packed = build()
    for name, tensor in zip(buffer_names, packed, strict=True):
        setattr(module, name, tensor.detach())
        keys[name] = key
    return tuple(getattr(module, name) for name in buffer_names)


def _adapters_are_packable(
    adapters: Sequence[tuple[torch.Tensor | None, torch.Tensor | None, float]],
) -> bool:
    if not adapters or any(A is None or B is None for A, B, _ in adapters):
        return False
    first_A, first_B, _ = adapters[0]
    assert first_A is not None
    assert first_B is not None
    rank = first_A.shape[1]
    device = first_A.device
    dtype = first_A.dtype
    in_features = first_A.shape[0]
    return all(
        A is not None
        and B is not None
        and A.ndim == _MATRIX_DIMENSIONS
        and B.ndim == _MATRIX_DIMENSIONS
        and A.shape == (in_features, rank)
        and B.shape[0] == rank
        and A.device == device
        and B.device == device
        and A.dtype == dtype
        and B.dtype == dtype
        for A, B, _ in adapters
    )


def _build_adapter_pack(
    adapters: Sequence[tuple[torch.Tensor | None, torch.Tensor | None, float]],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not _adapters_are_packable(adapters):
        return None, None

    As = [A for A, _, _ in adapters]
    Bs = [B for _, B, _ in adapters]
    assert all(A is not None for A in As)
    assert all(B is not None for B in Bs)
    typed_As = [A for A in As if A is not None]
    typed_Bs = [B for B in Bs if B is not None]
    rank = typed_As[0].shape[1]
    out_features = [B.shape[1] for B in typed_Bs]
    packed_A = torch.cat(typed_As, dim=1)
    packed_B = typed_Bs[0].new_zeros((rank * len(typed_Bs), sum(out_features)))
    row = 0
    column = 0
    for B, (_, _, scaling) in zip(typed_Bs, adapters, strict=True):
        packed_B[row : row + rank, column : column + B.shape[1]] = B * scaling
        row += rank
        column += B.shape[1]
    return packed_A, packed_B


def _projection_packs(
    module,
    name: str,
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor | None] | None,
    adapters: Sequence[tuple[torch.Tensor | None, torch.Tensor | None, float]],
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return refreshed base/bias/adapter packs for same-input projections."""
    weight_name = f"_opaque_{name}_weight_pack"
    packed_weight = _cached_pack(
        module,
        weight_name,
        weights,
        (),
        lambda: torch.cat(weights, dim=0),
    )

    packed_bias = None
    if biases is not None and any(bias is not None for bias in biases):
        bias_sources = [bias for bias in biases if bias is not None]
        bias_name = f"_opaque_{name}_bias_pack"

        def build_bias():
            return torch.cat(
                [
                    bias if bias is not None else weight.new_zeros(weight.shape[0])
                    for weight, bias in zip(weights, biases, strict=True)
                ]
            )

        packed_bias = _cached_pack(
            module,
            bias_name,
            bias_sources,
            tuple(None if bias is None else index for index, bias in enumerate(biases)),
            build_bias,
        )

    packed_A = packed_B = None
    if _adapters_are_packable(adapters):
        adapter_sources = [
            tensor for A, B, _ in adapters for tensor in (A, B) if tensor is not None
        ]
        adapter_key = tuple(float(scaling) for _, _, scaling in adapters)
        A_name = f"_opaque_{name}_adapter_a_pack"
        B_name = f"_opaque_{name}_adapter_b_pack"

        def build_adapter_pack() -> tuple[torch.Tensor, torch.Tensor]:
            adapter_A, adapter_B = _build_adapter_pack(adapters)
            assert adapter_A is not None
            assert adapter_B is not None
            return adapter_A, adapter_B

        packed_A, packed_B = _cached_adapter_pack(
            module,
            (A_name, B_name),
            adapter_sources,
            adapter_key,
            build_adapter_pack,
        )

    return packed_weight, packed_bias, packed_A, packed_B
