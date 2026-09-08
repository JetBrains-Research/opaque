# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Selective saved-tensor CPU offload for functional transforms."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ._saved_tensors import ensure_saved_tensor_hooks_guard

_DEFAULT_MIN_BYTES = 1 << 20
_DEFAULT_MAX_PINNED_BYTES = 1 << 30
_MAX_PENDING_TRANSFERS = 2
_RECURSIVE_CONTEXT = "save_on_cpu contexts cannot be entered recursively"
_CONTEXT_NOT_ENTERED = "save_on_cpu context was not entered"
_PIN_MEMORY_TYPE = "pin_memory must be a bool"
_MIN_BYTES_VALUE = "min_bytes must be a non-negative integer"
_MAX_PINNED_BYTES_VALUE = "max_pinned_bytes must be a non-negative integer"
_STATS_TYPE = "stats must be a SaveOnCpuStats instance or None"
_MODIFIED_SAVED_TENSOR = (
    "a tensor needed for gradient computation was modified after it was saved"
)


@dataclass
class SaveOnCpuStats:
    """Cumulative statistics produced by :func:`save_on_cpu`."""

    selected_tensors: int = 0
    selected_bytes: int = 0
    pinned_bytes: int = 0
    pageable_bytes: int = 0
    skipped_small_tensors: int = 0
    skipped_small_bytes: int = 0
    skipped_protected_tensors: int = 0
    skipped_protected_bytes: int = 0
    skipped_device_tensors: int = 0
    skipped_device_bytes: int = 0
    peak_pinned_bytes: int = 0
    max_pending_transfers: int = 0
    d2h_seconds: float = 0.0

    def to_dict(self, prefix: str = "activation_offload_") -> dict[str, int | float]:
        """Return flat logging metrics."""
        return {
            f"{prefix}selected_tensors": self.selected_tensors,
            f"{prefix}selected_bytes": self.selected_bytes,
            f"{prefix}pinned_bytes": self.pinned_bytes,
            f"{prefix}pageable_bytes": self.pageable_bytes,
            f"{prefix}skipped_small_tensors": self.skipped_small_tensors,
            f"{prefix}skipped_small_bytes": self.skipped_small_bytes,
            f"{prefix}skipped_protected_tensors": self.skipped_protected_tensors,
            f"{prefix}skipped_protected_bytes": self.skipped_protected_bytes,
            f"{prefix}skipped_device_tensors": self.skipped_device_tensors,
            f"{prefix}skipped_device_bytes": self.skipped_device_bytes,
            f"{prefix}peak_pinned_bytes": self.peak_pinned_bytes,
            f"{prefix}max_pending_transfers": self.max_pending_transfers,
            f"{prefix}d2h_seconds": self.d2h_seconds,
        }


@dataclass
class _PendingTransfer:
    started: Any
    completed: Any


@dataclass
class _OnDeviceTensor:
    tensor: Tensor
    version: int


@dataclass
class _PackedTensor:
    device: torch.device
    tensor: Tensor
    pinned: bool
    ready: Any | None = None


def _iter_tensors(values: Any) -> Iterable[Tensor]:
    if isinstance(values, Tensor):
        yield values
    elif isinstance(values, Mapping):
        for value in values.values():
            yield from _iter_tensors(value)
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        for value in values:
            yield from _iter_tensors(value)


def _physical_tensor(tensor: Tensor) -> Tensor:
    functorch = torch._C._functorch
    current = tensor
    while functorch.is_functorch_wrapped_tensor(current):
        current = functorch.get_unwrapped(current)
    return current


def _num_bytes(tensor: Tensor) -> int:
    physical = _physical_tensor(tensor)
    return physical.numel() * physical.element_size()


def _storage_key(tensor: Tensor) -> tuple[str, int | None, int] | None:
    try:
        physical = _physical_tensor(tensor)
        device = physical.device
        return (device.type, device.index, physical.untyped_storage().data_ptr())
    except (AttributeError, NotImplementedError, RuntimeError):
        return None


class _SaveOnCpu(AbstractContextManager["_SaveOnCpu"]):
    def __init__(
        self,
        *,
        pin_memory: bool,
        device_type: str,
        min_bytes: int,
        protected_tensors: Any,
        max_pinned_bytes: int,
        stats: SaveOnCpuStats,
    ) -> None:
        self.pin_memory = pin_memory
        self.device_type = device_type
        self._device_module = getattr(torch, device_type, torch.cuda)
        self.min_bytes = min_bytes
        self.max_pinned_bytes = max_pinned_bytes
        self.stats = stats
        self._protected_tensors = tuple(_iter_tensors(protected_tensors))
        self._protected_storages: set[tuple[str, int | None, int]] = set()
        self._hooks: Any | None = None
        self._streams: dict[torch.device, Any] = {}
        self._pending: deque[_PendingTransfer] = deque()
        self._current_pinned_bytes = 0

    def __enter__(self) -> _SaveOnCpu:
        if self._hooks is not None:
            raise RuntimeError(_RECURSIVE_CONTEXT)
        self._protected_storages = {
            key
            for tensor in self._protected_tensors
            if (key := _storage_key(tensor)) is not None
        }
        self._current_pinned_bytes = 0
        ensure_saved_tensor_hooks_guard()
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        hooks = self._hooks
        if hooks is None:
            raise RuntimeError(_CONTEXT_NOT_ENTERED)
        try:
            return hooks.__exit__(exc_type, exc_value, traceback)
        finally:
            self._drain_all()
            self._hooks = None
            self._protected_storages.clear()
            self._current_pinned_bytes = 0

    def _pack(self, tensor: Tensor) -> _OnDeviceTensor | _PackedTensor:
        num_bytes = _num_bytes(tensor)
        if tensor.device.type == "cpu":
            self.stats.skipped_device_tensors += 1
            self.stats.skipped_device_bytes += num_bytes
            return _OnDeviceTensor(tensor.detach(), _physical_tensor(tensor)._version)
        if num_bytes == 0 or num_bytes < self.min_bytes:
            self.stats.skipped_small_tensors += 1
            self.stats.skipped_small_bytes += num_bytes
            return _OnDeviceTensor(tensor.detach(), _physical_tensor(tensor)._version)
        key = _storage_key(tensor)
        if key is not None and key in self._protected_storages:
            self.stats.skipped_protected_tensors += 1
            self.stats.skipped_protected_bytes += num_bytes
            return _OnDeviceTensor(tensor.detach(), _physical_tensor(tensor)._version)

        self.stats.selected_tensors += 1
        self.stats.selected_bytes += num_bytes
        can_pin = (
            self.pin_memory
            and self.device_type == "cuda"
            and tensor.device.type == self.device_type
            and self._device_module.is_available()
            and tensor.layout == torch.strided
            and self._current_pinned_bytes + num_bytes <= self.max_pinned_bytes
        )
        if not can_pin:
            started = time.perf_counter()
            packed = tensor.to("cpu")
            self.stats.d2h_seconds += time.perf_counter() - started
            self.stats.pageable_bytes += num_bytes
            return _PackedTensor(tensor.device, packed, pinned=False)

        self._reap_completed()
        if len(self._pending) >= _MAX_PENDING_TRANSFERS:
            self._drain_one()
        stream = self._streams.get(tensor.device)
        if stream is None:
            stream = self._streams[tensor.device] = torch.cuda.Stream(
                device=tensor.device
            )
        producer = torch.cuda.current_stream(tensor.device)
        snapshot = tensor.detach().clone(memory_format=torch.preserve_format)
        stream.wait_stream(producer)
        started = torch.cuda.Event(enable_timing=True)
        completed = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            started.record(stream)
            packed = torch.empty_like(snapshot, device="cpu", pin_memory=True)
            packed.copy_(snapshot, non_blocking=True)
            _physical_tensor(snapshot).record_stream(stream)
            completed.record(stream)
        self._pending.append(_PendingTransfer(started, completed))
        self.stats.max_pending_transfers = max(
            self.stats.max_pending_transfers, len(self._pending)
        )
        self._current_pinned_bytes += num_bytes
        self.stats.pinned_bytes += num_bytes
        self.stats.peak_pinned_bytes = max(
            self.stats.peak_pinned_bytes, self._current_pinned_bytes
        )
        return _PackedTensor(tensor.device, packed, pinned=True, ready=completed)

    def _unpack(self, packed: _OnDeviceTensor | _PackedTensor) -> Tensor:
        if isinstance(packed, _OnDeviceTensor):
            if _physical_tensor(packed.tensor)._version != packed.version:
                raise RuntimeError(_MODIFIED_SAVED_TENSOR)
            return packed.tensor
        if not packed.pinned:
            return packed.tensor.to(packed.device)
        current = torch.cuda.current_stream(packed.device)
        current.wait_event(packed.ready)
        return packed.tensor.to(packed.device, non_blocking=True)

    def _reap_completed(self) -> None:
        while self._pending and self._pending[0].completed.query():
            self._finish_transfer(self._pending.popleft())

    def _drain_one(self) -> None:
        pending = self._pending.popleft()
        pending.completed.synchronize()
        self._finish_transfer(pending)

    def _drain_all(self) -> None:
        while self._pending:
            self._drain_one()

    def _finish_transfer(self, pending: _PendingTransfer) -> None:
        self.stats.d2h_seconds += pending.started.elapsed_time(pending.completed) / 1000


def save_on_cpu(
    pin_memory: bool = False,
    device_type: str = "cuda",
    *,
    min_bytes: int = _DEFAULT_MIN_BYTES,
    protected_tensors: Any = (),
    max_pinned_bytes: int = _DEFAULT_MAX_PINNED_BYTES,
    stats: SaveOnCpuStats | None = None,
) -> AbstractContextManager:
    """Selectively save tensors needed by backward on CPU.

    This is shaped like :class:`torch.autograd.graph.save_on_cpu`, but skips
    tensors smaller than ``min_bytes`` and tensors sharing storage with
    ``protected_tensors``. Pinned CUDA transfers use a bounded transfer queue;
    allocations beyond ``max_pinned_bytes`` fall back to pageable CPU memory.

    Args:
        pin_memory: Use pinned host memory and a CUDA transfer stream when the
            pinned-byte budget permits. The pageable path is the safe default.
        device_type: Accelerator type used to determine whether pinned transfer
            support is available, matching PyTorch's argument.
        min_bytes: Minimum physical tensor size selected for offload.
        protected_tensors: A tensor, nested mapping, or iterable whose storage
            aliases must remain on device. Pass current functional parameters.
        max_pinned_bytes: Maximum pinned bytes allocated by one context entry.
            Selected tensors beyond the budget use pageable host memory.
        stats: Optional cumulative statistics object shared across entries.

    Returns:
        A reusable saved-tensor-hooks context manager. Its ``stats`` attribute
        is the supplied or newly-created :class:`SaveOnCpuStats` instance.

    Note:
        Create a fresh context after functional optimizers replace parameter
        tensors so ``protected_tensors`` describes the current storages.
    """
    if not isinstance(pin_memory, bool):
        raise TypeError(_PIN_MEMORY_TYPE)
    if isinstance(min_bytes, bool) or not isinstance(min_bytes, int) or min_bytes < 0:
        raise ValueError(_MIN_BYTES_VALUE)
    if (
        isinstance(max_pinned_bytes, bool)
        or not isinstance(max_pinned_bytes, int)
        or max_pinned_bytes < 0
    ):
        raise ValueError(_MAX_PINNED_BYTES_VALUE)
    if stats is not None and not isinstance(stats, SaveOnCpuStats):
        raise TypeError(_STATS_TYPE)
    return _SaveOnCpu(
        pin_memory=pin_memory,
        device_type=device_type,
        min_bytes=min_bytes,
        protected_tensors=protected_tensors,
        max_pinned_bytes=max_pinned_bytes,
        stats=stats if stats is not None else SaveOnCpuStats(),
    )
