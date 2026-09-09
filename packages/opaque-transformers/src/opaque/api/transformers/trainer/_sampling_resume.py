"""Sampling-law checks for trainer checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from opaque.accounting.types import (
    CachedProcess,
    Composed,
    DpProcess,
    Identity,
    Repeated,
)
from opaque.dpsgd.accounting.amplification.types import KOutOfT, Poisson
from opaque.exceptions import CheckpointError


def _exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Read an integer checkpoint field without accepting coercible values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointError(
            *(f"Checkpoint field {name!r} must be an integer; got {value!r}.",)
        )
    if minimum is not None and value < minimum:
        raise CheckpointError(
            *(f"Checkpoint field {name!r} must be >= {minimum}; got {value!r}.",)
        )
    return value


def _finite_number(value: Any, name: str) -> float:
    """Read a finite numeric checkpoint field without string/bool coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(
            *(f"Checkpoint field {name!r} must be numeric; got {value!r}.",)
        )
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointError(
            *(f"Checkpoint field {name!r} must be finite; got {value!r}.",)
        )
    return result


def _state_value(state: Mapping[str, Any] | None, key: str, law: str) -> Any:
    if not isinstance(state, Mapping):
        raise CheckpointError(*(f"Checkpoint has no recognizable {law} state.",))
    try:
        return state[key]
    except KeyError:
        raise CheckpointError(
            *(f"Checkpoint {law} state is missing required field {key!r}.",)
        ) from None


@dataclass(frozen=True, slots=True)
class PoissonSamplingLaw:
    """Parameters that determine a Poisson sampler's inclusion law."""

    sample_rate: float
    truncated_batch_size: int | None
    num_samples: int

    def __post_init__(self) -> None:
        rate = _finite_number(self.sample_rate, "sample_rate")
        if not 0.0 < rate <= 1.0:
            raise CheckpointError(*(f"Invalid Poisson sample rate {rate!r}.",))
        if self.truncated_batch_size is not None:
            _exact_int(
                self.truncated_batch_size,
                "truncated_batch_size",
                minimum=1,
            )
        _exact_int(self.num_samples, "num_samples", minimum=1)

    @classmethod
    def from_sampler_state(cls, state: Mapping[str, Any] | None) -> PoissonSamplingLaw:
        """Read the law restored by ``PoissonSampler`` without coercion."""
        rate = _finite_number(
            _state_value(state, "sample_rate", "Poisson"),
            "sample_rate",
        )
        cap_value = _state_value(state, "truncated_batch_size", "Poisson")
        cap = (
            None
            if cap_value is None
            else _exact_int(cap_value, "truncated_batch_size", minimum=1)
        )
        num_samples = _exact_int(
            _state_value(state, "num_samples", "Poisson"),
            "num_samples",
            minimum=1,
        )
        return cls(rate, cap, num_samples)


@dataclass(frozen=True, slots=True)
class KOutOfTSamplingLaw:
    """Parameters that determine a K-out-of-T allocation."""

    k: int
    t: int
    allocation: str
    num_samples: int

    def __post_init__(self) -> None:
        k = _exact_int(self.k, "k", minimum=1)
        t = _exact_int(self.t, "t", minimum=1)
        if k > t:
            raise CheckpointError(*(f"Invalid K-out-of-T parameters k={k}, t={t}.",))
        if not isinstance(self.allocation, str):
            raise CheckpointError(
                *(
                    "Checkpoint field 'allocation' must be a string; "
                    f"got {self.allocation!r}.",
                )
            )
        if self.allocation not in ("block", "total"):
            raise CheckpointError(
                *(f"Invalid K-out-of-T allocation {self.allocation!r}.",)
            )
        _exact_int(self.num_samples, "num_samples", minimum=1)

    @classmethod
    def from_sampler_state(cls, state: Mapping[str, Any] | None) -> KOutOfTSamplingLaw:
        """Read the law restored by ``KOutOfTSampler`` without coercion."""
        allocation = _state_value(state, "allocation", "K-out-of-T")
        if not isinstance(allocation, str):
            raise CheckpointError(
                *(
                    "Checkpoint field 'allocation' must be a string; "
                    f"got {allocation!r}.",
                )
            )
        return cls(
            k=_exact_int(_state_value(state, "k", "K-out-of-T"), "k", minimum=1),
            t=_exact_int(_state_value(state, "t", "K-out-of-T"), "t", minimum=1),
            allocation=allocation,
            num_samples=_exact_int(
                _state_value(state, "num_samples", "K-out-of-T"),
                "num_samples",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class _AccountedPoissonLaw:
    sample_rate: float
    truncated_batch_size: int | None
    dataset_size: int | None


@dataclass(frozen=True, slots=True)
class _AccountedKOutOfTLaw:
    k: int
    t: int
    allocation: str


def _iter_accounted_leaves(process: DpProcess) -> Iterator[tuple[DpProcess, int]]:
    """Project only the stable composition wrappers used by saved accountants."""
    stack: list[tuple[DpProcess, int]] = [(process, 1)]
    while stack:
        node, count = stack.pop()
        if isinstance(node, Identity):
            continue
        if isinstance(node, CachedProcess):
            stack.append((node.inner, count))
            continue
        if isinstance(node, Repeated):
            repeated = _exact_int(node.count, "accountant repeat count", minimum=1)
            stack.append((node.inner, count * repeated))
            continue
        if isinstance(node, Composed):
            stack.append((node.right, count))
            stack.append((node.left, count))
            continue
        yield node, count


def _accounted_poisson_laws(
    process: DpProcess,
) -> tuple[set[_AccountedPoissonLaw], int]:
    laws: set[_AccountedPoissonLaw] = set()
    releases = 0
    for node, count in _iter_accounted_leaves(process):
        if not isinstance(node, Poisson):
            raise CheckpointError(
                *(
                    "Checkpoint accountant is not a standard per-step Poisson "
                    f"history: found {type(node).__name__}.",
                )
            )
        rate = _finite_number(node.sample_rate, "accountant sample_rate")
        cap = node.truncated_batch_size
        if cap is not None:
            cap = _exact_int(cap, "accountant truncated_batch_size", minimum=1)
        dataset_size = node.dataset_size
        if dataset_size is not None:
            dataset_size = _exact_int(
                dataset_size, "accountant dataset_size", minimum=1
            )
        laws.add(_AccountedPoissonLaw(rate, cap, dataset_size))
        releases += count
    return laws, releases


def _accounted_k_out_of_t_laws(
    process: DpProcess,
) -> tuple[set[_AccountedKOutOfTLaw], int]:
    laws: set[_AccountedKOutOfTLaw] = set()
    logical_processes = 0
    for node, count in _iter_accounted_leaves(process):
        if not isinstance(node, KOutOfT):
            raise CheckpointError(
                *(
                    "Checkpoint accountant is not a K-out-of-T horizon: "
                    f"found {type(node).__name__}.",
                )
            )
        allocation = node.allocation
        if not isinstance(allocation, str) or allocation not in ("block", "total"):
            raise CheckpointError(
                *(
                    "Checkpoint accountant contains an invalid K-out-of-T "
                    f"allocation: {allocation!r}.",
                )
            )
        laws.add(
            _AccountedKOutOfTLaw(
                k=_exact_int(node.k, "accountant k", minimum=1),
                t=_exact_int(node.t, "accountant t", minimum=1),
                allocation=allocation,
            )
        )
        logical_processes += count
    return laws, logical_processes


def _validate_release_count(releases: int, global_step: int) -> None:
    step = _exact_int(global_step, "global_step", minimum=0)
    if releases != step:
        raise CheckpointError(
            *(
                "Checkpoint accountant release count does not match global_step: "
                f"accounted={releases}, global_step={step}.",
            )
        )


def validate_poisson_checkpoint(
    *,
    sampler_state: Mapping[str, Any] | None,
    runtime_sample_rate: float,
    current_law: PoissonSamplingLaw,
    accounted_process: DpProcess,
    global_step: int,
) -> None:
    """Require sampler, runtime, and every executed accountant leaf to agree."""
    saved_law = PoissonSamplingLaw.from_sampler_state(sampler_state)
    runtime_rate = _finite_number(runtime_sample_rate, "runtime sample_rate")
    if runtime_rate != saved_law.sample_rate:
        raise CheckpointError(
            *(
                "Checkpoint Poisson state disagrees with its runtime sample rate: "
                f"sampler={saved_law.sample_rate!r}, runtime={runtime_rate!r}.",
            )
        )
    if saved_law != current_law:
        raise CheckpointError(
            *(
                "Poisson sampler does not match the resolved sampling law: "
                f"sampler={saved_law!r}, resolved={current_law!r}.",
            )
        )

    laws, releases = _accounted_poisson_laws(accounted_process)
    expected = _AccountedPoissonLaw(
        sample_rate=saved_law.sample_rate,
        truncated_batch_size=saved_law.truncated_batch_size,
        dataset_size=(
            saved_law.num_samples
            if saved_law.truncated_batch_size is not None
            else None
        ),
    )
    if laws and laws != {expected}:
        raise CheckpointError(
            *(
                "Checkpoint accountant contains a different Poisson sampling law: "
                f"saved={expected!r}, accounted={sorted(map(repr, laws))!r}.",
            )
        )
    _validate_release_count(releases, global_step)


def validate_distributed_resume(
    *,
    saved_world_size: int,
    current_world_size: int,
) -> None:
    """Reject topology drift and shared-state distributed replay."""
    saved = _exact_int(saved_world_size, "world_size", minimum=1)
    current = _exact_int(current_world_size, "current world_size", minimum=1)
    if saved != current:
        raise CheckpointError(
            *(
                "Distributed topology changed across resume: "
                f"saved={saved}, current={current}.",
            )
        )
    if current > 1:
        raise CheckpointError(
            *(
                "Distributed resume requires rank-specific sampler state; this "
                "checkpoint contains only rank 0's stream.",
            )
        )


def validate_dataset_schedule(
    *,
    saved_identity: str | None,
    current_identity: str | None,
) -> None:
    """Require stable record-to-position identity for horizon samplers."""
    if (
        not isinstance(saved_identity, str)
        or not saved_identity.strip()
        or not isinstance(current_identity, str)
        or not current_identity.strip()
    ):
        raise CheckpointError(
            *(
                "Horizon resume requires a non-empty dataset_schedule_id in both "
                "the checkpoint and current arguments.",
            )
        )
    if saved_identity != current_identity:
        raise CheckpointError(
            *(
                "Dataset schedule changed across resume: "
                f"saved={saved_identity!r}, current={current_identity!r}.",
            )
        )


def validate_k_out_of_t_checkpoint(
    *,
    sampler_state: Mapping[str, Any] | None,
    current_law: KOutOfTSamplingLaw,
    accounted_process: DpProcess,
) -> None:
    """Require sampler and whole-horizon accountant to describe one K-out-of-T law."""
    saved_law = KOutOfTSamplingLaw.from_sampler_state(sampler_state)
    if saved_law != current_law:
        raise CheckpointError(
            *(
                "K-out-of-T sampler does not match the resolved sampling law: "
                f"sampler={saved_law!r}, resolved={current_law!r}.",
            )
        )

    laws, logical_processes = _accounted_k_out_of_t_laws(accounted_process)
    expected = _AccountedKOutOfTLaw(
        k=saved_law.k,
        t=saved_law.t,
        allocation=saved_law.allocation,
    )
    if laws != {expected}:
        raise CheckpointError(
            *(
                "Checkpoint accountant contains a different K-out-of-T law: "
                f"saved={expected!r}, accounted={sorted(map(repr, laws))!r}.",
            )
        )
    if logical_processes != 1:
        raise CheckpointError(
            *(
                "Checkpoint accountant must contain exactly one K-out-of-T "
                f"horizon process; found {logical_processes}.",
            )
        )


__all__ = [
    "KOutOfTSamplingLaw",
    "PoissonSamplingLaw",
    "validate_dataset_schedule",
    "validate_distributed_resume",
    "validate_k_out_of_t_checkpoint",
    "validate_poisson_checkpoint",
]
