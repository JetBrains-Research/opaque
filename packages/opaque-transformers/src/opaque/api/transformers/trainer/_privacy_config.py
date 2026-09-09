"""Validation and resolution for trainer privacy configuration."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from opaque.exceptions import ConfigurationError, InputTypeError

PRIVACY_DICT_FIELDS: frozenset[str] = frozenset(
    {
        "clipping_kwargs",
        "sampling_kwargs",
        "noise_calibration_kwargs",
        "privacy_noise_mechanism_kwargs",
    }
)

MECHANISMS_DPFTRL: frozenset[str] = frozenset(
    {"mf_band", "mf_blt", "mf_bisr", "mf_bsr", "mf_lambda_cgd", "mf_identity"}
)
MECHANISMS: frozenset[str] = frozenset({"gaussian", *MECHANISMS_DPFTRL})

SAMPLING_MODES: frozenset[str] = frozenset(
    {
        "poisson",
        "k_out_of_t",
        "b_min_sep",
        "balls_in_bins",
        "cyclic_poisson",
        "sequential",
    }
)
SAMPLER_BY_MECHANISM: dict[str, str] = {
    "gaussian": "poisson",
    "mf_identity": "poisson",
    "mf_band": "cyclic_poisson",
    "mf_blt": "balls_in_bins",
    "mf_bisr": "balls_in_bins",
    "mf_bsr": "balls_in_bins",
    "mf_lambda_cgd": "balls_in_bins",
}
ALLOWED_SAMPLERS: dict[str, frozenset[str]] = {
    "gaussian": frozenset({"poisson", "k_out_of_t"}),
    "mf_identity": frozenset({"poisson", "balls_in_bins"}),
    "mf_band": frozenset({"cyclic_poisson", "b_min_sep"}),
    "mf_blt": frozenset({"balls_in_bins"}),
    "mf_bisr": frozenset({"balls_in_bins"}),
    "mf_bsr": frozenset({"balls_in_bins"}),
    "mf_lambda_cgd": frozenset({"balls_in_bins"}),
}
CURSOR_FREE_SAMPLING_MODES: frozenset[str] = frozenset({"poisson"})

# Defaults materialized on the public arguments for compatibility.
MECHANISM_DEFAULTS: dict[str, dict[str, Any]] = {
    "mf_band": {"bands": 16},
    "mf_blt": {"max_buffers": 10},
    "mf_bisr": {"bandwidth": 4},
    "mf_bsr": {"bandwidth": 8, "alpha": 1.0, "beta": 0.9},
    "mf_lambda_cgd": {"lambda_": 0.5},
    "mf_identity": {},
}

CLIPPING_SCHEMAS: dict[str, frozenset[str]] = {
    "fixed": frozenset(),
    "adaptive": frozenset(
        {
            "target_quantile",
            "target_clipping_rate",
            "clipping_norm_max",
            "norm_max",
        }
    ),
    "auto": frozenset({"gamma"}),
}
SAMPLING_SCHEMAS: dict[str, frozenset[str]] = {
    "poisson": frozenset({"truncated_batch_size", "max_batch_size"}),
    "k_out_of_t": frozenset({"k", "allocation"}),
    "b_min_sep": frozenset(),
    "balls_in_bins": frozenset(),
    "cyclic_poisson": frozenset(),
    "sequential": frozenset(),
}
MECHANISM_SCHEMAS: dict[str, frozenset[str]] = {
    "gaussian": frozenset({"compute_dtype"}),
    "mf_band": frozenset({"bands", "momentum", "lr_schedule"}),
    "mf_blt": frozenset({"max_buffers", "momentum", "lr_schedule"}),
    "mf_bisr": frozenset({"bandwidth", "normalized", "momentum", "inv_coefficients"}),
    "mf_bsr": frozenset({"bandwidth", "alpha", "beta"}),
    "mf_lambda_cgd": frozenset({"lambda_", "normalized"}),
    "mf_identity": frozenset(),
}
CALIBRATION_SCHEMA: frozenset[str] = frozenset(
    {"param_min", "min", "param_max", "max", "tolerance"}
)

CALIBRATION_DEFAULTS = {"min": 0.11, "max": 10.0, "tolerance": 1e-3}
_HISTORICAL_CALIBRATION_DEFAULTS = (
    CALIBRATION_DEFAULTS,
    {"min": 0.01, "max": 10.0, "tolerance": 1e-3},
)
_ADAPTIVE_CLIPPING_NORM_MIN = 0.01
_MAX_BLT_BUFFERS = 15
_MIN_BISR_LEADING_COEFFICIENT = 1e-30


@dataclasses.dataclass(frozen=True, slots=True)
class ClippingConfig:
    """Resolved clipping mode and effective parameters."""

    mode: str
    norm: float | tuple[tuple[str, float], ...]
    target_quantile: float | None = None
    clipping_norm_max: float | None = None
    gamma: float | None = None

    def norm_value(self) -> float | dict[str, float]:
        if isinstance(self.norm, tuple):
            return dict(self.norm)
        return self.norm


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Resolved sampler mode and effective parameters."""

    mode: str
    truncated_batch_size: int | None = None
    k: int | None = None
    allocation: str | None = None

    def as_kwargs(self) -> dict[str, Any]:
        if self.mode == "poisson" and self.truncated_batch_size is not None:
            return {"truncated_batch_size": self.truncated_batch_size}
        if self.mode == "k_out_of_t":
            return {"k": self.k, "allocation": self.allocation}
        return {}


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Resolved noise-calibration search parameters."""

    param_min: float
    param_max: float
    tolerance: float


@dataclasses.dataclass(frozen=True, slots=True)
class MechanismConfig:
    """Resolved noise mechanism and canonical factory arguments."""

    kind: str
    kwargs: tuple[tuple[str, Any], ...]

    def as_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedPrivacyConfig:
    """Run-scoped resolved privacy configuration.

    The dataclass prevents field reassignment, but callable schedule values are
    retained by reference for compatibility and are not deeply immutable.
    """

    clipping: ClippingConfig
    sampling: SamplingConfig
    calibration: CalibrationConfig | None
    mechanism: MechanismConfig
    noise_multiplier: float | None
    target_epsilon: float | None
    target_delta: float | None


def _validate_keys(
    field_name: str,
    selector_name: str,
    selector: str,
    kwargs: Mapping[Any, Any],
    allowed: frozenset[str],
) -> None:
    non_string = [key for key in kwargs if not isinstance(key, str)]
    if non_string:
        raise InputTypeError(
            *(f"{field_name} keys must be strings; got {non_string!r}.",)
        )
    unknown = set(kwargs) - allowed
    if not unknown:
        return
    accepted = sorted(allowed)
    suffix = repr(accepted) if accepted else "<none>"
    raise ConfigurationError(
        *(
            f"Unsupported {field_name} for {selector_name}={selector!r}: "
            f"{sorted(unknown)}. Accepted keys: {suffix}.",
        )
    )


def _kwargs(args: Any, field_name: str) -> dict[Any, Any]:
    value = getattr(args, field_name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InputTypeError(
            *(
                f"{field_name} must be a mapping after normalization; "
                f"got {type(value).__name__}.",
            )
        )
    return dict(value)


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise InputTypeError(*(f"{name} must be a finite number, not bool.",))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputTypeError(
            *(f"{name} must be a finite number; got {value!r}.",)
        ) from error
    if not math.isfinite(result):
        raise ConfigurationError(*(f"{name} must be finite; got {value!r}.",))
    return result


def _as_clipping_norm(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise InputTypeError(*(f"{name} must be numeric, not bool.",))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputTypeError(
            *(f"{name} must be a positive number; got {value!r}.",)
        ) from error
    if math.isnan(result) or result <= 0.0:
        raise ConfigurationError(*(f"{name} must be > 0; got {value!r}.",))
    return result


def _as_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise InputTypeError(*(f"{name} must be an integer, not bool.",))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputTypeError(
            *(f"{name} must be an integer; got {value!r}.",)
        ) from error
    try:
        exact = bool(result == value)
    except Exception:
        exact = False
    if not exact:
        raise InputTypeError(*(f"{name} must be an integer; got {value!r}.",))
    if result < minimum:
        raise ConfigurationError(*(f"{name} must be >= {minimum}; got {result}.",))
    return result


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InputTypeError(*(f"{name} must be a bool; got {value!r}.",))
    return value


def _alias_value(
    kwargs: Mapping[str, Any],
    canonical: str,
    aliases: tuple[str, ...],
    *,
    default: Any,
    convert,
) -> Any:
    names = (canonical, *aliases)
    present = [(name, convert(kwargs[name], name)) for name in names if name in kwargs]
    if not present:
        return convert(default, canonical)
    value = present[0][1]
    conflicts = [
        (name, candidate) for name, candidate in present[1:] if candidate != value
    ]
    if conflicts:
        supplied = dict(present)
        raise ConfigurationError(
            *(f"Conflicting aliases for {canonical!r}: {supplied!r}.",)
        )
    return value


def _resolve_budget(args: Any) -> tuple[float | None, float | None, float | None]:
    noise_multiplier = (
        None
        if args.privacy_noise_multiplier is None
        else _as_finite_float(args.privacy_noise_multiplier, "privacy_noise_multiplier")
    )
    target_epsilon = (
        None
        if args.privacy_target_epsilon is None
        else _as_finite_float(args.privacy_target_epsilon, "privacy_target_epsilon")
    )
    target_delta = (
        None
        if args.privacy_target_delta is None
        else _as_finite_float(args.privacy_target_delta, "privacy_target_delta")
    )

    if noise_multiplier is None and target_epsilon is None:
        raise ConfigurationError(
            *(
                "Set either privacy_noise_multiplier (use 0.0 for non-private "
                "training) or privacy_target_epsilon (to calibrate noise to a "
                "budget); neither was provided.",
            )
        )
    if noise_multiplier is not None and noise_multiplier < 0.0:
        raise ConfigurationError(
            *(f"privacy_noise_multiplier must be >= 0; got {noise_multiplier!r}.",)
        )
    if target_epsilon is not None and target_epsilon <= 0.0:
        raise ConfigurationError(
            *(
                "privacy_target_epsilon must be > 0 when provided; got "
                f"{target_epsilon!r}.",
            )
        )
    if noise_multiplier == 0.0 and target_epsilon is not None:
        raise ConfigurationError(
            *(
                "privacy_noise_multiplier=0.0 is the non-private path; "
                "privacy_target_epsilon is meaningless there. Drop the target "
                "or set a positive noise multiplier.",
            )
        )
    if target_delta is not None and not 0.0 < target_delta < 1.0:
        raise ConfigurationError(
            *(f"privacy_target_delta must lie in (0, 1); got {target_delta!r}.",)
        )
    return noise_multiplier, target_epsilon, target_delta


def _resolve_clipping_norm(
    raw_norm: Any, *, noise_multiplier: float | None
) -> float | tuple[tuple[str, float], ...]:
    if isinstance(raw_norm, Mapping):
        non_string = [key for key in raw_norm if not isinstance(key, str)]
        if non_string:
            raise InputTypeError(
                *(f"clipping_norm keys must be strings; got {non_string!r}.",)
            )
        if "fallback" not in raw_norm:
            raise ConfigurationError(
                *(
                    "clipping_norm must include a 'fallback' key with the "
                    "default per-example clip bound.",
                )
            )
        resolved = tuple(
            sorted(
                (key, _as_clipping_norm(value, f"clipping_norm[{key!r}]"))
                for key, value in raw_norm.items()
            )
        )
        norm: float | tuple[tuple[str, float], ...]
        norm = resolved[0][1] if len(resolved) == 1 else resolved
    else:
        norm = _as_clipping_norm(raw_norm, "clipping_norm")

    values = (norm,) if isinstance(norm, float) else tuple(value for _, value in norm)
    if any(math.isinf(value) for value in values) and noise_multiplier != 0.0:
        raise ConfigurationError(
            *(
                "Disabling clipping (clipping_norm=math.inf) is only valid for "
                "a non-private baseline (privacy_noise_multiplier=0.0).",
            )
        )
    return norm


def _resolve_clipping(
    args: Any, mechanism: str, noise_multiplier: float | None
) -> ClippingConfig:
    mode = args.clipping_mode
    if not isinstance(mode, str):
        raise InputTypeError(
            *(f"clipping_mode must be a string; got {type(mode).__name__}.",)
        )
    if mode not in CLIPPING_SCHEMAS:
        raise ConfigurationError(
            *(f"clipping_mode={mode!r}; expected one of {sorted(CLIPPING_SCHEMAS)}.",)
        )
    if mechanism in MECHANISMS_DPFTRL and mode == "adaptive":
        mode = "fixed"

    kwargs = _kwargs(args, "clipping_kwargs")
    _validate_keys(
        "clipping_kwargs", "clipping_mode", mode, kwargs, CLIPPING_SCHEMAS[mode]
    )

    norm = _resolve_clipping_norm(args.clipping_norm, noise_multiplier=noise_multiplier)

    if mode == "adaptive":
        target_quantile = _alias_value(
            kwargs,
            "target_quantile",
            ("target_clipping_rate",),
            default=0.5,
            convert=_as_finite_float,
        )
        clipping_norm_max = _alias_value(
            kwargs,
            "clipping_norm_max",
            ("norm_max",),
            default=10.0,
            convert=_as_finite_float,
        )
        if not 0.0 < target_quantile < 1.0:
            raise ConfigurationError(
                *(f"target_quantile must be in (0, 1); got {target_quantile}.",)
            )
        if clipping_norm_max <= _ADAPTIVE_CLIPPING_NORM_MIN:
            raise ConfigurationError(
                *(
                    "clipping_norm_max must be greater than the adaptive minimum "
                    f"{_ADAPTIVE_CLIPPING_NORM_MIN}; got {clipping_norm_max}.",
                )
            )
        return ClippingConfig(
            mode=mode,
            norm=norm,
            target_quantile=target_quantile,
            clipping_norm_max=clipping_norm_max,
        )
    if mode == "auto":
        gamma = _as_finite_float(kwargs.get("gamma", 0.01), "gamma")
        if gamma <= 0.0:
            raise ConfigurationError(*(f"gamma must be > 0; got {gamma}.",))
        return ClippingConfig(mode=mode, norm=norm, gamma=gamma)
    return ClippingConfig(mode=mode, norm=norm)


def _resolve_sampling_mode(mechanism: str, requested_mode: Any) -> str:
    mode = requested_mode
    if not isinstance(mode, str):
        raise InputTypeError(
            *(f"sampling_mode must be a string; got {type(mode).__name__}.",)
        )
    if mode == "random_allocation":
        raise ConfigurationError(
            *(
                "sampling_mode='random_allocation' was replaced by "
                "sampling_mode='k_out_of_t' with allocation='total'.",
            )
        )
    if mode == "auto":
        mode = SAMPLER_BY_MECHANISM[mechanism]
    elif mode not in SAMPLING_MODES:
        raise ConfigurationError(
            *(
                f"sampling_mode={mode!r}; expected 'auto' or one of "
                f"{sorted(SAMPLING_MODES)}.",
            )
        )
    if mode not in ALLOWED_SAMPLERS[mechanism]:
        if mechanism == "mf_band" and mode == "poisson":
            raise ConfigurationError(
                *(
                    "sampling_mode='poisson' is incompatible with "
                    "privacy_noise_mechanism='mf_band'; use 'cyclic_poisson' "
                    "or 'b_min_sep', or use privacy_noise_mechanism='mf_identity' "
                    "for ordinary Poisson sampling.",
                )
            )
        raise ConfigurationError(
            *(
                f"sampling_mode={mode!r} is not valid for "
                f"privacy_noise_mechanism={mechanism!r}; allowed: "
                f"{sorted(ALLOWED_SAMPLERS[mechanism])}.",
            )
        )

    return mode


def _resolve_sampling(args: Any, mechanism: str) -> SamplingConfig:
    mode = _resolve_sampling_mode(mechanism, args.sampling_mode)
    kwargs = _kwargs(args, "sampling_kwargs")
    if "total_participations" in kwargs:
        raise ConfigurationError(
            *(
                "sampling_kwargs['total_participations'] was replaced by 'k' "
                "with allocation='total'.",
            )
        )
    _validate_keys(
        "sampling_kwargs", "sampling_mode", mode, kwargs, SAMPLING_SCHEMAS[mode]
    )
    if mode == "poisson":
        cap = _alias_value(
            kwargs,
            "truncated_batch_size",
            ("max_batch_size",),
            default=None,
            convert=lambda value, name: (
                None if value is None else _as_positive_int(value, name)
            ),
        )
        return SamplingConfig(mode=mode, truncated_batch_size=cap)
    if mode == "k_out_of_t":
        missing = {"k", "allocation"} - kwargs.keys()
        if missing:
            raise ConfigurationError(
                *(
                    "sampling_mode='k_out_of_t' requires sampling_kwargs with "
                    f"{sorted(missing)}.",
                )
            )
        k = _as_positive_int(kwargs["k"], "k")
        allocation = kwargs["allocation"]
        if not isinstance(allocation, str):
            raise InputTypeError(
                *(
                    "sampling_kwargs['allocation'] must be a string; "
                    f"got {type(allocation).__name__}.",
                )
            )
        if allocation not in ("block", "total"):
            raise ConfigurationError(
                *(
                    "sampling_kwargs['allocation'] must be 'block' or 'total'; "
                    f"got {allocation!r}.",
                )
            )
        return SamplingConfig(mode=mode, k=k, allocation=allocation)
    return SamplingConfig(mode=mode)


def _calibration_user_keys(args: Any, kwargs: Mapping[str, Any]) -> frozenset[str]:
    source = getattr(args, "noise_calibration_kwargs", None)
    recorded = getattr(source, "explicit_keys", None)
    if recorded is None:
        if any(
            dict(kwargs) == defaults for defaults in _HISTORICAL_CALIBRATION_DEFAULTS
        ):
            return frozenset()
        return frozenset(kwargs)

    explicit = set(recorded) & set(kwargs)
    for key, value in kwargs.items():
        if key in explicit:
            continue
        if key not in CALIBRATION_DEFAULTS or value != CALIBRATION_DEFAULTS[key]:
            explicit.add(key)
    return frozenset(explicit)


def _resolve_calibration(
    args: Any, noise_multiplier: float | None
) -> CalibrationConfig | None:
    kwargs = _kwargs(args, "noise_calibration_kwargs")
    _validate_keys(
        "noise_calibration_kwargs",
        "privacy_noise_multiplier",
        "calibrated" if noise_multiplier is None else "fixed",
        kwargs,
        CALIBRATION_SCHEMA,
    )
    explicit = _calibration_user_keys(args, kwargs)
    if noise_multiplier is not None:
        if explicit:
            raise ConfigurationError(
                *(
                    "noise_calibration_kwargs is inactive when "
                    "privacy_noise_multiplier is fixed; remove explicit keys "
                    f"{sorted(explicit)}.",
                )
            )
        return None

    param_min = _alias_value(
        kwargs,
        "param_min",
        ("min",),
        default=CALIBRATION_DEFAULTS["min"],
        convert=_as_finite_float,
    )
    param_max = _alias_value(
        kwargs,
        "param_max",
        ("max",),
        default=CALIBRATION_DEFAULTS["max"],
        convert=_as_finite_float,
    )
    tolerance = _as_finite_float(
        kwargs.get("tolerance", CALIBRATION_DEFAULTS["tolerance"]), "tolerance"
    )
    if param_min <= 0.0:
        raise ConfigurationError(*(f"param_min must be > 0; got {param_min}.",))
    if param_min >= param_max:
        raise ConfigurationError(
            *(f"param_min ({param_min}) must be < param_max ({param_max}).",)
        )
    if tolerance <= 0.0:
        raise ConfigurationError(*(f"tolerance must be > 0; got {tolerance}.",))
    return CalibrationConfig(param_min, param_max, tolerance)


def _resolve_compute_dtype(value: Any) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "float64": torch.float64,
        "torch.float64": torch.float64,
    }
    if isinstance(value, str):
        value = aliases.get(value.strip().lower(), value)
    if value is not torch.float32 and value is not torch.float64:
        raise ConfigurationError(
            *(f"compute_dtype must be float32 or float64; got {value!r}.",)
        )
    return value


def _as_coefficients(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise InputTypeError(
            *(f"inv_coefficients must be a sequence of numbers; got {value!r}.",)
        )
    return tuple(
        _as_finite_float(coefficient, f"inv_coefficients[{index}]")
        for index, coefficient in enumerate(value)
    )


def _resolve_mechanism(args: Any, mechanism: str) -> MechanismConfig:
    supplied = _kwargs(args, "privacy_noise_mechanism_kwargs")
    _validate_keys(
        "privacy_noise_mechanism_kwargs",
        "privacy_noise_mechanism",
        mechanism,
        supplied,
        MECHANISM_SCHEMAS[mechanism],
    )
    if mechanism == "gaussian":
        dtype = _resolve_compute_dtype(supplied.get("compute_dtype", torch.float32))
        return MechanismConfig(mechanism, (("compute_dtype", dtype),))

    kwargs = {**MECHANISM_DEFAULTS[mechanism], **supplied}
    if mechanism == "mf_band":
        kwargs["bands"] = _as_positive_int(kwargs["bands"], "bands")
        kwargs["momentum"] = _as_finite_float(kwargs.get("momentum", 1.0), "momentum")
        if kwargs["momentum"] < 0.0:
            raise ConfigurationError(
                *(f"momentum must be >= 0; got {kwargs['momentum']}.",)
            )
    elif mechanism == "mf_blt":
        kwargs["max_buffers"] = _as_positive_int(kwargs["max_buffers"], "max_buffers")
        if kwargs["max_buffers"] > _MAX_BLT_BUFFERS:
            raise ConfigurationError(
                *(
                    f"max_buffers must be <= {_MAX_BLT_BUFFERS}; "
                    f"got {kwargs['max_buffers']}.",
                )
            )
        kwargs["momentum"] = _as_finite_float(kwargs.get("momentum", 1.0), "momentum")
        if kwargs["momentum"] < 0.0:
            raise ConfigurationError(
                *(f"momentum must be >= 0; got {kwargs['momentum']}.",)
            )
    elif mechanism == "mf_bisr":
        kwargs["bandwidth"] = _as_positive_int(
            kwargs["bandwidth"], "bandwidth", minimum=2
        )
        kwargs["normalized"] = _as_bool(kwargs.get("normalized", True), "normalized")
        kwargs["momentum"] = _as_finite_float(kwargs.get("momentum", 0.0), "momentum")
        if not 0.0 <= kwargs["momentum"] < 1.0:
            raise ConfigurationError(
                *(f"momentum must be in [0, 1); got {kwargs['momentum']}.",)
            )
        if "inv_coefficients" in kwargs:
            kwargs["inv_coefficients"] = _as_coefficients(kwargs["inv_coefficients"])
            if (
                kwargs["inv_coefficients"] is not None
                and len(kwargs["inv_coefficients"]) != kwargs["bandwidth"]
            ):
                raise ConfigurationError(
                    *(
                        "inv_coefficients length must equal bandwidth; got "
                        f"{len(kwargs['inv_coefficients'])} and {kwargs['bandwidth']}.",
                    )
                )
            if (
                kwargs["inv_coefficients"] is not None
                and abs(kwargs["inv_coefficients"][0]) < _MIN_BISR_LEADING_COEFFICIENT
            ):
                raise ConfigurationError(
                    *(
                        "inv_coefficients[0] must have magnitude >= "
                        f"{_MIN_BISR_LEADING_COEFFICIENT:.0e}.",
                    )
                )
    elif mechanism == "mf_bsr":
        kwargs["bandwidth"] = _as_positive_int(kwargs["bandwidth"], "bandwidth")
        kwargs["alpha"] = _as_finite_float(kwargs["alpha"], "alpha")
        kwargs["beta"] = _as_finite_float(kwargs["beta"], "beta")
        if not 0.0 < kwargs["alpha"] <= 1.0:
            raise ConfigurationError(
                *(f"alpha must be in (0, 1]; got {kwargs['alpha']}.",)
            )
        if not 0.0 <= kwargs["beta"] < 1.0:
            raise ConfigurationError(
                *(f"beta must be in [0, 1); got {kwargs['beta']}.",)
            )
        if kwargs["alpha"] <= kwargs["beta"]:
            raise ConfigurationError(
                *(
                    f"alpha must be greater than beta; got {kwargs['alpha']} <= {kwargs['beta']}.",
                )
            )
    elif mechanism == "mf_lambda_cgd":
        kwargs["lambda_"] = _as_finite_float(kwargs["lambda_"], "lambda_")
        kwargs["normalized"] = _as_bool(kwargs.get("normalized", True), "normalized")
        if not 0.0 <= kwargs["lambda_"] < 1.0:
            raise ConfigurationError(
                *(f"lambda_ must be in [0, 1); got {kwargs['lambda_']}.",)
            )

    if (
        "lr_schedule" in kwargs
        and kwargs["lr_schedule"] is not None
        and not callable(kwargs["lr_schedule"])
    ):
        raise InputTypeError(
            *(
                "lr_schedule must be callable or None; "
                f"got {type(kwargs['lr_schedule']).__name__}.",
            )
        )
    return MechanismConfig(mechanism, tuple(sorted(kwargs.items())))


def resolve_privacy_config(args: Any) -> ResolvedPrivacyConfig:
    """Validate and resolve the privacy fields on ``args``."""
    mechanism = args.privacy_noise_mechanism
    if not isinstance(mechanism, str):
        raise InputTypeError(
            *(
                "privacy_noise_mechanism must be a string; got "
                f"{type(mechanism).__name__}.",
            )
        )
    if mechanism not in MECHANISMS:
        raise ConfigurationError(
            *(
                f"privacy_noise_mechanism={mechanism!r}; expected one of "
                f"{sorted(MECHANISMS)}.",
            )
        )

    noise_multiplier, target_epsilon, target_delta = _resolve_budget(args)
    clipping = _resolve_clipping(args, mechanism, noise_multiplier)
    sampling = _resolve_sampling(args, mechanism)
    if (
        noise_multiplier is not None
        and target_epsilon is not None
        and (mechanism != "gaussian" or sampling.mode == "k_out_of_t")
    ):
        raise ConfigurationError(
            *(
                "privacy_target_epsilon cannot be combined with a fixed "
                "privacy_noise_multiplier for whole-horizon mechanisms; "
                "incomplete-horizon privacy accounting and early stopping are "
                "unsupported. Set only privacy_target_epsilon to calibrate the "
                "complete horizon, or set only privacy_noise_multiplier.",
            )
        )
    calibration = _resolve_calibration(args, noise_multiplier)
    mechanism_config = _resolve_mechanism(args, mechanism)
    return ResolvedPrivacyConfig(
        clipping=clipping,
        sampling=sampling,
        calibration=calibration,
        mechanism=mechanism_config,
        noise_multiplier=noise_multiplier,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )


__all__ = [
    "ALLOWED_SAMPLERS",
    "CALIBRATION_DEFAULTS",
    "CALIBRATION_SCHEMA",
    "CLIPPING_SCHEMAS",
    "CURSOR_FREE_SAMPLING_MODES",
    "MECHANISMS",
    "MECHANISMS_DPFTRL",
    "MECHANISM_DEFAULTS",
    "MECHANISM_SCHEMAS",
    "PRIVACY_DICT_FIELDS",
    "ResolvedPrivacyConfig",
    "SAMPLER_BY_MECHANISM",
    "SAMPLING_MODES",
    "SAMPLING_SCHEMAS",
    "resolve_privacy_config",
]
