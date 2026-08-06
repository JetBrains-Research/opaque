from __future__ import annotations

import contextlib
import dataclasses
import gc
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from benchmarks.core import exact_metric

if TYPE_CHECKING:
    from benchmarks.core import RunOptions


@dataclass(frozen=True)
class TimingMeasurement:
    samples_ms: list[float]
    baseline_device_bytes: int | None
    peak_device_bytes: int | None
    peak_kind: str | None

    @property
    def incremental_peak_device_bytes(self) -> int | None:
        if self.baseline_device_bytes is None or self.peak_device_bytes is None:
            return None
        return max(self.peak_device_bytes - self.baseline_device_bytes, 0)


def device_memory_metrics(timing: TimingMeasurement) -> dict[str, dict[str, Any]]:
    if timing.peak_device_bytes is None:
        return {}
    peak = exact_metric(timing.peak_device_bytes, "byte")
    incremental = exact_metric(timing.incremental_peak_device_bytes or 0, "byte")
    if timing.peak_kind is not None:
        peak["method"] = timing.peak_kind
        incremental["method"] = timing.peak_kind
    return {
        "baseline_device_bytes": exact_metric(
            timing.baseline_device_bytes or 0, "byte"
        ),
        "peak_device_bytes": peak,
        "incremental_peak_device_bytes": incremental,
    }


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _prepare_memory_window(device: str) -> tuple[int | None, str | None]:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated(), "allocated"
    if device == "mps":
        torch.mps.empty_cache()
        synchronize(device)
        return torch.mps.driver_allocated_memory(), "sampled_driver_allocated"
    return None, None


def _sample_mps_memory(stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(0.0005):
        with contextlib.suppress(RuntimeError):
            samples.append(torch.mps.driver_allocated_memory())


def _peak_device_bytes(device: str, mps_samples: list[int]) -> int | None:
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    if device == "mps":
        with contextlib.suppress(RuntimeError):
            mps_samples.append(torch.mps.driver_allocated_memory())
        return max(mps_samples) if mps_samples else None
    return None


def measure_callable(fn: Callable[[], Any], options: RunOptions) -> TimingMeasurement:
    for _ in range(options.warmup):
        result = fn()
        synchronize(options.device)
        del result

    baseline, peak_kind = _prepare_memory_window(options.device)
    mps_samples = [baseline] if options.device == "mps" and baseline is not None else []
    stop = threading.Event()
    sampler = None
    if options.device == "mps":
        sampler = threading.Thread(
            target=_sample_mps_memory,
            args=(stop, mps_samples),
            daemon=True,
        )
        sampler.start()
    samples: list[float] = []
    try:
        for _ in range(options.repeats):
            synchronize(options.device)
            start = time.perf_counter_ns()
            result = fn()
            synchronize(options.device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            del result
    finally:
        stop.set()
        if sampler is not None:
            sampler.join()
    return TimingMeasurement(
        samples_ms=samples,
        baseline_device_bytes=baseline,
        peak_device_bytes=_peak_device_bytes(options.device, mps_samples),
        peak_kind=peak_kind,
    )


def _walk_tensors(value: Any, seen_objects: set[int]):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    identity = id(value)
    if identity in seen_objects:
        return
    seen_objects.add(identity)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk_tensors(getattr(value, field.name), seen_objects)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_tensors(key, seen_objects)
            yield from _walk_tensors(child, seen_objects)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for child in value:
            yield from _walk_tensors(child, seen_objects)


def tensor_storage_bytes(value: Any) -> int:
    storage_keys: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in _walk_tensors(value, set()):
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), size)
        if key not in storage_keys:
            storage_keys.add(key)
            total += size
    return total


__all__ = [
    "TimingMeasurement",
    "device_memory_metrics",
    "measure_callable",
    "synchronize",
    "tensor_storage_bytes",
]
