"""Thin wrappers around optional provider collectives.

Provides ``is_distributed``, ``get_rank``, ``get_world_size``, ``all_reduce``,
and ``barrier``. All are safe to call outside a
process group; non-distributed contexts fall through to sensible defaults
(rank 0, world_size 1, no-op barrier).

Backend activation is context-local and driven by the values a call
receives, so a query about the process group can arrive before any
provider is active. Rather than report the single-process defaults while
a group is live, these helpers consult the probes providers register at
import (:func:`register_distributed_probe`), and — when no probe has been
registered yet — import the provider package of every framework this
process has already loaded so it can register one
(:func:`_discover_process_group`). A provider-free install has nothing to
import and keeps the defaults, so the engine stays backend-neutral and
never imports a runtime to answer the question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.api.engine import runtime
from opaque.api.engine.backend import (
    BackendNotSelectedError,
    _registry,
    active_backend,
    set_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.engine.backend import Backend, KnownBackend

    Probe = Callable[[], tuple[int, int] | None]
    BackendFactory = Callable[[], Backend]

# Provider-supplied views of a live process group, consulted only when no
# backend is active. Each probe returns ``(rank, world_size)`` for a live
# group or ``None``, and is paired with the factory that builds the backend
# owning that group. Registration order breaks ties between providers.
_DISTRIBUTED_PROBES: list[tuple[Probe, BackendFactory]] = []

# Backends whose provider package this process has already tried to import
# on the discovery path. Recorded before the attempt, so a provider that
# queries the process group while importing cannot recurse, and retained
# afterwards so a provider that is not installed is not re-attempted on
# every query.
_DISCOVERY_ATTEMPTED: set[KnownBackend] = set()


def register_distributed_probe(
    probe: Probe,
    *,
    backend_factory: BackendFactory,
) -> None:
    """Register a provider view of its process group.

    Providers call this at import so the engine can answer
    ``is_distributed`` / ``get_rank`` / ``get_world_size`` before any
    value has selected a backend. ``probe`` must not raise and must
    return ``None`` when that provider has no live group.

    ``backend_factory`` builds the backend that owns the group ``probe``
    reports. A live group belongs to exactly one runtime, so a probe that
    answers identifies its provider as precisely as a value from that
    runtime does. The engine activates it for the collectives that follow
    the query, which carry Python scalars and identify nothing on their
    own.
    """
    if not any(registered is probe for registered, _ in _DISTRIBUTED_PROBES):
        _DISTRIBUTED_PROBES.append((probe, backend_factory))


def _probe_process_group() -> tuple[tuple[int, int], BackendFactory] | None:
    """Return the group view and provider factory of the first live group."""
    for probe, backend_factory in tuple(_DISTRIBUTED_PROBES):
        result = probe()
        if result is not None:
            return result, backend_factory
    return None


def _discover_process_group() -> tuple[tuple[int, int], BackendFactory] | None:
    """Import candidate providers so their probes can answer, then re-ask.

    A provider registers its probe when its package is imported, so a
    process that created a group without ever importing the provider
    façade has no probe to consult. Only backends whose framework is
    already in ``sys.modules`` are candidates: a process group is created
    by calling into the framework, so a framework this process never
    imported cannot have one live. That makes discovery free — and
    provably inert — in a build where the framework is not installed at
    all, and keeps the engine from ever being the reason a runtime gets
    imported.
    """
    for kind in _registry.loaded_runtime_backends():
        if kind in _DISCOVERY_ATTEMPTED:
            continue
        _DISCOVERY_ATTEMPTED.add(kind)
        if not _registry.import_provider_package(kind):
            continue
        found = _probe_process_group()
        if found is not None:
            return found
    return None


def _process_group_view() -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` for a live group, or ``None``.

    Activates the provider that owns the group when the context has no
    backend yet, so a caller that acts on a positive answer — a rank-gated
    collective, a scalar assertion — has a backend to dispatch on.
    """
    found = _probe_process_group()
    if found is None:
        found = _discover_process_group()
    if found is None:
        return None
    view, backend_factory = found
    if active_backend() is None:
        set_backend(backend_factory())
    return view


def is_distributed() -> bool:
    """Return whether a distributed process group is live.

    True for any initialized group, including a single-rank one; callers
    that need multi-rank behavior compare :func:`get_world_size` instead.
    Safe to call before any backend is active: registered provider probes
    answer for a live group, and with no provider at all there is no group
    to report.
    """
    try:
        return bool(runtime.distributed_initialized())
    except BackendNotSelectedError:
        return _process_group_view() is not None


def get_rank() -> int:
    """Return the current rank (0 if not distributed).

    Safe to call before any backend is active: registered provider probes
    answer for a live group, and with no provider at all the single-process
    default applies.
    """
    try:
        return runtime.distributed_rank()
    except BackendNotSelectedError:
        probed = _process_group_view()
        return 0 if probed is None else probed[0]


def get_world_size() -> int:
    """Return the world size (1 if not distributed).

    Safe to call before any backend is active: registered provider probes
    answer for a live group, and with no provider at all the single-process
    default applies.
    """
    try:
        return runtime.distributed_world_size()
    except BackendNotSelectedError:
        probed = _process_group_view()
        return 1 if probed is None else probed[1]


def all_reduce(tensor: Any, op: str = "sum") -> Any:
    """Return a reduced value; input is unchanged."""
    return runtime.distributed_all_reduce(tensor, op=runtime.ReduceOp(op))


def barrier() -> None:
    """Block until every rank reaches this call (no-op if not distributed)."""
    try:
        runtime.distributed_barrier()
    except BackendNotSelectedError:
        if _process_group_view() is None:
            return
        # Discovery activated the provider that owns the live group; the
        # barrier carries no value that could have identified it.
        runtime.distributed_barrier()


def is_main_process() -> bool:
    """Return True on the rank-0 process (always True if not distributed)."""
    return get_rank() == 0


def num_processes() -> int:
    """Return the world size (1 if not distributed). Alias of ``get_world_size``."""
    return get_world_size()


def process_index() -> int:
    """Return the current rank (0 if not distributed). Alias of ``get_rank``."""
    return get_rank()


def wait_for_everyone() -> None:
    """Block until every rank reaches this call (no-op if not distributed).

    Module-level alias of :func:`barrier`, matching the
    ``accelerator.wait_for_everyone()`` idiom that callers port from.
    """
    barrier()


def gather_for_metrics(tensor: Any) -> Any:
    """All-gather ``tensor`` across ranks and concatenate along dim 0.

    In a non-distributed context returns ``tensor`` unchanged. Intended for
    metric aggregation (e.g. detached KL means, reference-logprob shards),
    where duplicate samples from Poisson sampling are not a correctness
    concern. This is **not** a gradient primitive — do not use it inside the
    clipped/noised per-example gradient path.

    All ranks must pass arrays with the same dtype, rank, and trailing shape;
    the leading dimension may vary by rank.
    """
    return runtime.distributed_all_gather(tensor, axis=0)


__all__ = [
    "all_reduce",
    "barrier",
    "gather_for_metrics",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "num_processes",
    "process_index",
    "wait_for_everyone",
]
