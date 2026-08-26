"""Explicit run and prefix types for whole-horizon DP processes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core._pld_cache import pld_cache


def _new_run_id() -> str:
    """Return a serialization-safe identity for one deployed horizon run."""
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class HorizonPrefix(DpProcess):
    """The first ``steps`` releases of one deployed horizon process.

    ``run_id`` records deployment lineage, not a numerical PLD input. Two
    equal-configured processes with different run IDs are accounted as
    distinct deployments; advancing an existing run is handled explicitly by
    :meth:`~opaque.accounting.types.Accountant.advance`.
    """

    process: DpHorizonProcess
    steps: int
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.process, DpHorizonProcess):
            raise TypeError(
                "HorizonPrefix requires a DpHorizonProcess, got "
                f"{type(self.process).__name__}."
            )
        if self.steps < 1:
            raise ValueError(f"steps ({self.steps}) must be >= 1")
        if self.steps > self.process.n_steps:
            raise ValueError(
                f"steps ({self.steps}) exceeds n_steps ({self.process.n_steps}); "
                f"{type(self.process).__name__} is undefined beyond its "
                "declared horizon."
            )
        if not isinstance(self.run_id, str):
            raise TypeError("run_id must be a string")
        if not self.run_id:
            raise ValueError("run_id must be non-empty")

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        # Deployment lineage changes how prefixes may be advanced, but not the
        # PLD of an already materialized K-prefix. Excluding run_id lets equal
        # mechanisms share the canonical numerical cache safely.
        del n_steps
        return (
            "HorizonPrefix",
            self.steps,
            self.process._pld_cache_key(n_steps=self.steps),
        )

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.process.pld_at(
            self.steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

    def advanced(self, count: int = 1) -> HorizonPrefix:
        """Return the later prefix of this same deployed run."""
        if count < 1:
            raise ValueError(f"count ({count}) must be >= 1")
        return HorizonPrefix(
            process=self.process,
            steps=self.steps + count,
            run_id=self.run_id,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        del (
            count,
            discretization,
            log_x_mass_truncation_bound,
            max_grid_size,
            max_conv_grid,
            seed,
            mc_resolution,
            mc_failure_probability,
        )
        raise TypeError(
            "A HorizonPrefix is one deployed transcript and cannot be "
            "self-composed. Create a fresh per_step(process) run for an "
            "independent deployment."
        )

    @property
    def run(self) -> PerStep:
        """Return a run handle bound to this prefix's deployment lineage."""
        return PerStep(process=self.process, run_id=self.run_id)

    def __mul__(self, count: int) -> DpProcess:
        raise TypeError(
            "A HorizonPrefix is one deployed transcript and cannot be repeated "
            "implicitly. Create a fresh per_step(process) run for an independent "
            "deployment."
        )

    def __rmul__(self, count: int) -> DpProcess:
        return self.__mul__(count)


@dataclass(frozen=True, slots=True)
class PerStep(DpProcess):
    """Handle for advancing one deployed :class:`DpHorizonProcess`.

    The class remains a ``DpProcess`` subclass for source and checkpoint
    compatibility, but it is not an ordinary composable event. Use it with an
    :class:`~opaque.accounting.types.Accountant` or multiply it to obtain an
    explicit :class:`HorizonPrefix`.
    """

    process: DpHorizonProcess
    run_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.process, DpHorizonProcess):
            raise TypeError(
                f"PerStep requires a DpHorizonProcess, got {type(self.process).__name__}."
            )
        if not isinstance(self.run_id, str):
            raise TypeError("run_id must be a string")
        if not self.run_id:
            object.__setattr__(self, "run_id", _new_run_id())

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        return (
            "PerStep",
            self.process._pld_cache_key(n_steps=1 if n_steps is None else n_steps),
        )

    def prefix(self, steps: int) -> HorizonPrefix:
        """Describe the first ``steps`` releases of this deployed run."""
        return HorizonPrefix(process=self.process, steps=steps, run_id=self.run_id)

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        return self.process.pld_at(
            1,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        # Compatibility for callers that invoked this method directly. New
        # process trees contain HorizonPrefix rather than Repeated(PerStep).
        return self.prefix(count).pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

    def __mul__(self, count: int) -> HorizonPrefix:
        return self.prefix(count)

    def __rmul__(self, count: int) -> HorizonPrefix:
        return self.__mul__(count)


# Lifecycle-oriented name for new code; ``PerStep`` remains the compatibility
# spelling used by the existing public factory and serialized checkpoints.
HorizonRun = PerStep


def _same_horizon_process(
    left: DpHorizonProcess,
    right: DpHorizonProcess,
) -> bool:
    """Compare every input affecting a deployed horizon's privacy PLDs.

    Dataclass equality alone may intentionally ignore callable schedules. The
    canonical full-horizon PLD fingerprint materializes those recipes and is
    therefore part of deployment compatibility. Identity is the training-loop
    fast path.
    """
    if left is right:
        return True
    return (
        type(left) is type(right)
        and left == right
        and left.n_steps == right.n_steps
        and left._pld_cache_key(n_steps=left.n_steps)
        == right._pld_cache_key(n_steps=right.n_steps)
    )


def _split_horizon_frontier(
    process: DpProcess,
) -> tuple[DpProcess, HorizonPrefix | None]:
    """Split the rightmost active horizon prefix from a process tree.

    Cache nodes are transparent only to this explicit lifecycle operation.
    Their closed portion remains cached; the active prefix is returned as a
    symbolic frontier because a materialized K-prefix cannot derive K+1.
    The walk is iterative so checkpoint depth is heap-bounded.
    """
    from opaque.api.accounting.core.composition._cached import CachedProcess
    from opaque.api.accounting.core.composition._composed import Composed
    from opaque.api.accounting.core.mechanisms.types import Identity

    frames: list[tuple[str, DpProcess | None]] = []
    node = process
    while True:
        if isinstance(node, HorizonPrefix):
            active = node
            break
        if isinstance(node, PerStep):
            # Programmatic compatibility for Accountant(prefix=step).
            active = node.prefix(1)
            break
        if isinstance(node, CachedProcess):
            frames.append(("cached", None))
            node = node.inner
            continue
        if isinstance(node, Composed):
            frames.append(("composed", node.left))
            node = node.right
            continue
        return process, None

    closed: DpProcess = Identity()
    for kind, payload in reversed(frames):
        if kind == "composed":
            assert payload is not None
            closed = (
                payload if isinstance(closed, Identity) else Composed(payload, closed)
            )
        elif not isinstance(closed, Identity | CachedProcess):
            closed = CachedProcess(closed)
    return closed, active


def _join_horizon_frontier(closed: DpProcess, active: HorizonPrefix) -> DpProcess:
    """Rebuild a process from the closed prefix and active frontier."""
    from opaque.api.accounting.core.composition._composed import Composed
    from opaque.api.accounting.core.mechanisms.types import Identity

    return active if isinstance(closed, Identity) else Composed(closed, active)


def _contains_horizon_run(process: DpProcess, run_id: str) -> bool:
    """Return whether ``run_id`` occurs anywhere in a process tree."""
    from opaque.api.accounting.core.composition._cached import CachedProcess
    from opaque.api.accounting.core.composition._composed import Composed
    from opaque.api.accounting.core.composition._repeated import Repeated

    stack = [process]
    while stack:
        node = stack.pop()
        if isinstance(node, (HorizonPrefix, PerStep)) and node.run_id == run_id:
            return True
        if isinstance(node, CachedProcess | Repeated):
            stack.append(node.inner)
        elif isinstance(node, Composed):
            stack.extend((node.left, node.right))
    return False


def _horizon_run_ids(process: DpProcess) -> set[str]:
    """Return all deployment IDs in a process tree, iteratively."""
    from opaque.api.accounting.core.composition._cached import CachedProcess
    from opaque.api.accounting.core.composition._composed import Composed
    from opaque.api.accounting.core.composition._repeated import Repeated

    found: set[str] = set()
    stack = [process]
    while stack:
        node = stack.pop()
        if isinstance(node, (HorizonPrefix, PerStep)):
            found.add(node.run_id)
        if isinstance(node, CachedProcess | Repeated):
            stack.append(node.inner)
        elif isinstance(node, Composed):
            stack.extend((node.left, node.right))
    return found


def _normalize_legacy_horizon_process(process: DpProcess) -> DpProcess:
    """Upgrade legacy ``Repeated(PerStep, K)`` trees to explicit prefixes.

    Old checkpoints carry no deployment ID. Loading their ``PerStep`` leaf
    creates one fresh ID, which is retained for that homogeneous group. We do
    not infer that separately serialized equal-config groups share lineage.
    """
    from opaque.api.accounting.core.composition._cached import CachedProcess
    from opaque.api.accounting.core.composition._composed import Composed
    from opaque.api.accounting.core.composition._repeated import Repeated

    # Values are (rebuilt node, still-a-legacy-run-handle).  Keeping the marker
    # through CachedProcess lets Repeated(CachedProcess(PerStep), K) migrate as
    # one K-prefix without conflating an explicit Repeated(HorizonPrefix, K).
    built: dict[int, tuple[DpProcess, bool]] = {}
    stack: list[tuple[DpProcess, bool]] = [(process, False)]
    while stack:
        node, expanded = stack.pop()
        if isinstance(node, Composed):
            if not expanded:
                stack.append((node, True))
                stack.append((node.right, False))
                stack.append((node.left, False))
                continue
            left, left_handle = built[id(node.left)]
            right, right_handle = built[id(node.right)]
            if left_handle:
                assert isinstance(left, PerStep)
                left = left.prefix(1)
            if right_handle:
                assert isinstance(right, PerStep)
                right = right.prefix(1)
            built[id(node)] = (Composed(left, right), False)
            continue
        if isinstance(node, Repeated):
            if not expanded:
                stack.append((node, True))
                stack.append((node.inner, False))
                continue
            inner, legacy_handle = built[id(node.inner)]
            if legacy_handle:
                assert isinstance(inner, PerStep)
                built[id(node)] = (inner.prefix(node.count), False)
            else:
                built[id(node)] = (Repeated(inner, node.count), False)
            continue
        if isinstance(node, CachedProcess):
            if not expanded:
                stack.append((node, True))
                stack.append((node.inner, False))
                continue
            inner, legacy_handle = built[id(node.inner)]
            built[id(node)] = (
                inner if legacy_handle else CachedProcess(inner),
                legacy_handle,
            )
            continue
        built[id(node)] = (node, isinstance(node, PerStep))

    normalized, legacy_handle = built[id(process)]
    if legacy_handle:
        assert isinstance(normalized, PerStep)
        return normalized.prefix(1)
    return normalized


def _validate_legacy_horizon_state(state: dict[str, Any]) -> None:
    """Reject legacy trees whose missing lineage makes recovery ambiguous.

    A single old ``PerStep`` leaf (possibly below ``Repeated`` and cache
    wrappers) unambiguously denotes one deployed run. Multiple equal-configured
    leaves might instead be cache fragments of that run *or* separate
    deployments. Structural equality cannot distinguish those histories, so a
    loader must not guess at a privacy bound.
    """
    legacy_processes: list[dict[str, Any]] = []
    stack: list[object] = [state]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("type") == "PerStep" and not value.get("run_id"):
                process_state = value.get("process")
                if not isinstance(process_state, dict):
                    raise ValueError(
                        "Legacy PerStep checkpoint is missing its process state."
                    )
                if any(process_state == prior for prior in legacy_processes):
                    raise ValueError(
                        "Legacy accountant contains multiple equal-configured "
                        "horizon groups without deployment lineage. They may "
                        "be fragments of one correlated run or independent "
                        "runs, so the checkpoint cannot be resumed soundly."
                    )
                legacy_processes.append(process_state)
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def per_step(process: DpHorizonProcess) -> PerStep:
    """Create a fresh run handle for step-wise horizon accounting."""
    if not isinstance(process, DpHorizonProcess):
        raise TypeError(
            f"per_step() requires a DpHorizonProcess, got {type(process).__name__}."
        )
    return PerStep(process=process)


__all__ = ["HorizonPrefix", "HorizonRun", "PerStep", "per_step"]
