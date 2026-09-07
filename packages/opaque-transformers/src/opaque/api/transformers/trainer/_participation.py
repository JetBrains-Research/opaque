"""Resolved sampling contracts shared by Trainer accounting and execution."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from opaque.exceptions import CheckpointError, ConfigurationError

PARTICIPATION_PLAN_VERSION = 1

_CONTRACT_BY_PAIR: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "gaussian": MappingProxyType(
            {
                "poisson": "whole_dataset_poisson",
                "k_out_of_t": "k_out_of_t",
            }
        ),
        "mf_identity": MappingProxyType(
            {
                "poisson": "whole_dataset_poisson",
                "balls_in_bins": "balls_in_bins",
            }
        ),
        "mf_band": MappingProxyType({"b_min_sep": "b_min_sep"}),
        "mf_blt": MappingProxyType({"balls_in_bins": "balls_in_bins"}),
        "mf_bisr": MappingProxyType({"balls_in_bins": "balls_in_bins"}),
        "mf_bsr": MappingProxyType({"balls_in_bins": "balls_in_bins"}),
        "mf_lambda_cgd": MappingProxyType({"balls_in_bins": "balls_in_bins"}),
    }
)

ALLOWED_SAMPLERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {mechanism: frozenset(modes) for mechanism, modes in _CONTRACT_BY_PAIR.items()}
)
SAMPLING_MODES: frozenset[str] = frozenset(
    mode for modes in ALLOWED_SAMPLERS.values() for mode in modes
)

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "mechanism_kind",
        "sampling_mode",
        "contract_id",
        "sampling_kwargs",
        "population_size",
        "expected_batch_size",
        "sample_rate",
        "total_steps",
        "num_bins",
        "world_size",
    }
)
_SAMPLING_KWARG_ITEM_LENGTH = 2


def _pair_contract(mechanism_kind: str, sampling_mode: str) -> str:
    if type(mechanism_kind) is not str or type(sampling_mode) is not str:
        raise ConfigurationError(
            *(
                "mechanism_kind and sampling_mode must be strings; "
                f"got {mechanism_kind!r} and {sampling_mode!r}.",
            )
        )
    if mechanism_kind == "mf_band" and sampling_mode == "poisson":
        raise ConfigurationError(
            *(
                "sampling_mode='poisson' is incompatible with "
                "privacy_noise_mechanism='mf_band': plain whole-dataset Poisson "
                "does not realize BandMF's participation process. Use "
                "sampling_mode='b_min_sep', or use "
                "privacy_noise_mechanism='mf_identity' for Poisson sampling.",
            )
        )
    try:
        return _CONTRACT_BY_PAIR[mechanism_kind][sampling_mode]
    except KeyError as exc:
        allowed = sorted(ALLOWED_SAMPLERS.get(mechanism_kind, ()))
        raise ConfigurationError(
            *(
                f"sampling_mode={sampling_mode!r} is not valid for "
                f"privacy_noise_mechanism={mechanism_kind!r}; allowed: {allowed}.",
            )
        ) from exc


def _require_int(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(
            *(f"{label} must be an int >= {minimum}, got {value!r}.",)
        )
    return value


def canonical_sampling_kwargs(
    sampling_mode: str,
    sampling_kwargs: Mapping[str, Any] | None,
) -> tuple[tuple[str, int | str], ...]:
    if sampling_kwargs is not None and not isinstance(sampling_kwargs, Mapping):
        raise ConfigurationError(
            *(
                "sampling_kwargs must be a mapping; "
                f"got {type(sampling_kwargs).__name__}.",
            )
        )
    raw = dict(sampling_kwargs or {})
    if sampling_mode == "poisson":
        allowed = {"truncated_batch_size", "max_batch_size"}
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigurationError(
                *(
                    "sampling_mode='poisson' received unsupported sampling_kwargs "
                    f"{sorted(unknown, key=repr)}; supported: {sorted(allowed)}.",
                )
            )
        if allowed <= set(raw):
            raise ConfigurationError(
                *(
                    "sampling_kwargs may not set both 'truncated_batch_size' and "
                    "'max_batch_size'; use 'truncated_batch_size'.",
                )
            )
        cap = raw.get("truncated_batch_size", raw.get("max_batch_size"))
        if cap is None:
            return ()
        return (
            (
                "truncated_batch_size",
                _require_int(cap, label="truncated_batch_size", minimum=1),
            ),
        )

    if sampling_mode == "k_out_of_t":
        required = {"k", "allocation"}
        if set(raw) != required:
            raise ConfigurationError(
                *(
                    "sampling_mode='k_out_of_t' requires exactly sampling_kwargs "
                    f"{sorted(required)}; missing={sorted(required - set(raw))}, "
                    f"unexpected={sorted(set(raw) - required, key=repr)}.",
                )
            )
        allocation = raw["allocation"]
        if type(allocation) is not str or allocation not in ("block", "total"):
            raise ConfigurationError(
                *(
                    "sampling_kwargs['allocation'] must be 'block' or 'total', "
                    f"got {allocation!r}.",
                )
            )
        return (
            ("allocation", allocation),
            ("k", _require_int(raw["k"], label="sampling_kwargs['k']", minimum=1)),
        )

    if raw:
        raise ConfigurationError(
            *(
                f"sampling_mode={sampling_mode!r} does not accept sampling_kwargs; "
                f"got {sorted(raw, key=repr)}.",
            )
        )
    return ()


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedParticipationPlan:
    """Immutable contract consumed by accounting, sampling, and checkpoints."""

    schema_version: int
    mechanism_kind: str
    sampling_mode: str
    contract_id: str
    sampling_kwargs: tuple[tuple[str, int | str], ...]
    population_size: int
    expected_batch_size: int
    sample_rate: float
    total_steps: int
    num_bins: int
    world_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != PARTICIPATION_PLAN_VERSION
        ):
            raise ConfigurationError(
                *(
                    f"unsupported participation-plan version {self.schema_version!r}; "
                    f"expected {PARTICIPATION_PLAN_VERSION}.",
                )
            )
        expected_contract = _pair_contract(self.mechanism_kind, self.sampling_mode)
        if type(self.contract_id) is not str or self.contract_id != expected_contract:
            raise ConfigurationError(
                *(
                    "participation contract does not match the mechanism/sampler "
                    f"pair: saved={self.contract_id!r}, expected={expected_contract!r}.",
                )
            )
        if not isinstance(self.sampling_kwargs, tuple) or any(
            not isinstance(item, tuple) or len(item) != _SAMPLING_KWARG_ITEM_LENGTH
            for item in self.sampling_kwargs
        ):
            raise ConfigurationError(
                *("participation sampling_kwargs must be canonical key/value pairs.",)
            )
        kwargs = dict(self.sampling_kwargs)
        if len(kwargs) != len(self.sampling_kwargs) or tuple(
            sorted(kwargs.items())
        ) != (self.sampling_kwargs):
            raise ConfigurationError(
                *("participation sampling_kwargs must be unique and sorted.",)
            )
        canonical = canonical_sampling_kwargs(self.sampling_mode, kwargs)
        if canonical != self.sampling_kwargs:
            raise ConfigurationError(
                *("participation sampling_kwargs are not canonical.",)
            )

        _require_int(self.population_size, label="population_size", minimum=1)
        _require_int(
            self.expected_batch_size,
            label="expected_batch_size",
            minimum=1,
        )
        _require_int(self.total_steps, label="total_steps", minimum=1)
        _require_int(self.num_bins, label="num_bins", minimum=1)
        _require_int(self.world_size, label="world_size", minimum=1)
        if (
            self.world_size > 1
            and self.sampling_mode == "poisson"
            and dict(self.sampling_kwargs).get("truncated_batch_size") is not None
        ):
            raise ConfigurationError(
                *(
                    "truncated Poisson sampling is not supported with distributed "
                    "training: the configured cap would be applied independently "
                    "to each rank-local shard, which is not the globally truncated "
                    "process used by the accountant. Remove "
                    "sampling_kwargs['truncated_batch_size'] or run with "
                    "world_size=1.",
                )
            )
        if self.population_size % self.world_size:
            raise ConfigurationError(
                *(
                    "population_size must be divisible by world_size; the Trainer "
                    "does not data-dependently trim records because that is not "
                    "stable under add/remove adjacency; "
                    f"got {self.population_size} and {self.world_size}.",
                )
            )
        if self.expected_batch_size % self.world_size:
            raise ConfigurationError(
                *(
                    "expected_batch_size must be divisible by world_size; "
                    f"got {self.expected_batch_size} and {self.world_size}.",
                )
            )
        if (
            type(self.sample_rate) is not float
            or not math.isfinite(self.sample_rate)
            or not 0 < self.sample_rate <= 1
        ):
            raise ConfigurationError(
                *(
                    f"sample_rate must be a finite float in (0, 1], got {self.sample_rate!r}.",
                )
            )
        expected_rate = self.expected_batch_size / self.population_size
        if not math.isclose(
            self.sample_rate,
            expected_rate,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ConfigurationError(
                *(
                    "sample_rate does not match expected_batch_size/population_size: "
                    f"{self.sample_rate!r} != {expected_rate!r}.",
                )
            )

    @classmethod
    def resolve(
        cls,
        *,
        mechanism_kind: str,
        sampling_mode: str,
        sampling_kwargs: Mapping[str, Any] | None,
        population_size: int,
        expected_batch_size: int,
        sample_rate: float,
        total_steps: int,
        num_bins: int,
        world_size: int,
    ) -> ResolvedParticipationPlan:
        """Revalidate mutable arguments and capture their realized values."""
        if sampling_kwargs is not None and not isinstance(sampling_kwargs, Mapping):
            raise ConfigurationError(
                *(
                    "sampling_kwargs must be a mapping when training starts; "
                    f"got {type(sampling_kwargs).__name__}.",
                )
            )
        contract = _pair_contract(mechanism_kind, sampling_mode)
        return cls(
            schema_version=PARTICIPATION_PLAN_VERSION,
            mechanism_kind=mechanism_kind,
            sampling_mode=sampling_mode,
            contract_id=contract,
            sampling_kwargs=canonical_sampling_kwargs(
                sampling_mode,
                sampling_kwargs,
            ),
            population_size=population_size,
            expected_batch_size=expected_batch_size,
            sample_rate=float(sample_rate),
            total_steps=total_steps,
            num_bins=num_bins,
            world_size=world_size,
        )

    def sampler_kwargs_dict(self) -> dict[str, int | str]:
        """Return a disposable mutable copy for legacy sampler constructors."""
        return dict(self.sampling_kwargs)

    @property
    def local_population_size(self) -> int:
        """Population represented by each rank-local sampler."""
        return self.population_size // self.world_size

    def to_state_dict(self) -> dict[str, Any]:
        """Return canonical, data-only checkpoint provenance."""
        return {
            "schema_version": self.schema_version,
            "mechanism_kind": self.mechanism_kind,
            "sampling_mode": self.sampling_mode,
            "contract_id": self.contract_id,
            "sampling_kwargs": self.sampler_kwargs_dict(),
            "population_size": self.population_size,
            "expected_batch_size": self.expected_batch_size,
            "sample_rate": self.sample_rate,
            "total_steps": self.total_steps,
            "num_bins": self.num_bins,
            "world_size": self.world_size,
        }

    @classmethod
    def from_state_dict(cls, saved: Mapping[str, Any]) -> ResolvedParticipationPlan:
        """Validate and restore checkpoint provenance strictly."""
        if not isinstance(saved, Mapping):
            raise CheckpointError(*("checkpoint participation plan is invalid.",))
        try:
            saved_fields = frozenset(saved)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                *("checkpoint participation plan is invalid.",)
            ) from exc
        if saved_fields != _PLAN_FIELDS:
            raise CheckpointError(
                *(
                    "participation-plan fields do not match the current schema: "
                    f"missing={sorted(_PLAN_FIELDS - saved_fields, key=repr)}, "
                    f"unexpected={sorted(saved_fields - _PLAN_FIELDS, key=repr)}.",
                )
            )
        kwargs = saved["sampling_kwargs"]
        if not isinstance(kwargs, Mapping):
            raise CheckpointError(*("participation-plan sampling_kwargs is invalid.",))
        try:
            return cls(
                schema_version=saved["schema_version"],
                mechanism_kind=saved["mechanism_kind"],
                sampling_mode=saved["sampling_mode"],
                contract_id=saved["contract_id"],
                sampling_kwargs=tuple(
                    sorted(kwargs.items(), key=lambda item: repr(item[0]))
                ),
                population_size=saved["population_size"],
                expected_batch_size=saved["expected_batch_size"],
                sample_rate=saved["sample_rate"],
                total_steps=saved["total_steps"],
                num_bins=saved["num_bins"],
                world_size=saved["world_size"],
            )
        except (ConfigurationError, TypeError, ValueError) as exc:
            raise CheckpointError(
                *("checkpoint contains an invalid participation plan.",)
            ) from exc


__all__ = [
    "ALLOWED_SAMPLERS",
    "PARTICIPATION_PLAN_VERSION",
    "ResolvedParticipationPlan",
    "SAMPLING_MODES",
    "canonical_sampling_kwargs",
]
