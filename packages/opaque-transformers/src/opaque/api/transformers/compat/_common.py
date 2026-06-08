"""Shared compat-conversion helpers — DP overrides, dispatcher, error formatters.

The HF and TRL converters call into the same dispatcher (`apply_manifest`)
with their own DIRECT/RENAME/TRANSFORM/REJECT/DROP manifests. This module
holds the dispatcher plus the DP-knob normalization that both
``from_hf`` and ``from_trl`` accept.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any, Callable, TypedDict


class DPOverrides(TypedDict, total=False):
    """The privacy / DP-mechanism kwargs every ``from_hf`` / ``from_trl`` accepts.

    Either ``privacy_noise_multiplier`` or ``privacy_target_epsilon`` MUST
    be set (matches opaque's runtime requirement). All other fields have
    safe defaults.
    """

    privacy_noise_multiplier: float | None
    privacy_target_epsilon: float | None
    privacy_target_delta: float | None
    clipping_norm: float | dict[str, float]
    privacy_noise_mechanism: str
    privacy_noise_radius: float
    clipping_mode: str
    clipping_kwargs: dict[str, Any]
    sampling_mode: str
    sampling_kwargs: dict[str, Any]
    noise_calibration_kwargs: dict[str, Any]
    microbatch_size: int | None
    auto_find_microbatch_size: bool
    activation_offloading: bool
    use_compat_patches: bool
    use_performance_kernels: bool
    performance_kernels_config: dict[str, Any] | None


def normalize_dp_overrides(
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a dict of opaque-side DP fields from kwargs.

    At least one of ``privacy_noise_multiplier`` or ``privacy_target_epsilon``
    must be set — opaque's runtime would reject the config otherwise, and
    surfacing the error here gives a clearer message tied to the converter
    call site.
    """
    noise_mult = overrides.get("privacy_noise_multiplier")
    target_eps = overrides.get("privacy_target_epsilon")
    if noise_mult is None and target_eps is None:
        raise ValueError(
            "Converting to an opaque config requires a DP knob: pass either "
            "``privacy_noise_multiplier=<float>`` (fixed-noise mode) or "
            "``privacy_target_epsilon=<float>`` (calibrated-noise mode) as a "
            "keyword argument to the converter."
        )
    # Pass through every override verbatim — opaque's TrainingArguments
    # __post_init__ does the cross-field validation.
    return dict(overrides)


def _is_default(value: Any, default: Any) -> bool:
    """``True`` if ``value`` matches ``default`` for a dataclass field.

    Handles the awkward cases the HF surface raises: dataclass instances
    (like ``AcceleratorConfig``) whose ``__eq__`` may not be defined, and
    pairs where ``default_factory`` constructs a fresh instance that
    isn't ``==`` to the value the user got.
    """
    if isinstance(default, dataclasses.Field):
        if default.default is not dataclasses.MISSING:
            default_value = default.default
        elif default.default_factory is not dataclasses.MISSING:
            default_value = default.default_factory()
        else:
            return False
    else:
        default_value = default

    # Fast path: identity match.
    if value is default_value:
        return True

    # Standard equality.
    try:
        if value == default_value:
            return True
    except Exception:  # pragma: no cover — pathological __eq__
        pass

    # Both dataclass instances? Compare via field dicts (handles
    # ``AcceleratorConfig`` and other HF nested dataclasses).
    if dataclasses.is_dataclass(value) and dataclasses.is_dataclass(default_value):
        try:
            return dataclasses.asdict(value) == dataclasses.asdict(default_value)
        except Exception:  # pragma: no cover
            pass

    return False


def get_dataclass_field_values(
    obj: Any,
) -> dict[str, Any]:
    """Return a name→value dict for every field of a dataclass instance."""
    if not dataclasses.is_dataclass(obj):
        raise TypeError(
            f"Expected a dataclass instance, got {type(obj).__name__}. "
            "The HF/TRL converters accept dataclass instances only."
        )
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def get_dataclass_field_defaults(
    cls: type,
) -> dict[str, Any]:
    """Return a name→default-value dict for the dataclass type ``cls``."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            out[f.name] = f.default_factory()
        else:
            out[f.name] = None
    return out


def apply_manifest(
    *,
    source_values: Mapping[str, Any],
    source_defaults: Mapping[str, Any],
    direct: frozenset[str],
    rename: Mapping[str, str],
    transform: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
    reject: Mapping[str, Callable[[Any], str | None]],
    drop: Mapping[str, str],
    source_label: str,
    strict: bool,
) -> dict[str, Any]:
    """Apply the bucketed manifest to ``source_values`` and return an opaque dict.

    ``reject`` maps field name → callable that returns an error message
    string if the value is unsupported, or ``None`` if the value is
    benign (e.g. ``packing=False`` is fine, only ``packing=True`` is
    rejected). The callable receives the raw source value.

    ``drop`` maps field name → reason string surfaced in the
    ``RuntimeWarning`` when the field is non-default.

    ``transform`` maps field name → callable that receives the full
    ``source_values`` dict and returns a partial opaque-side dict to
    merge in. Used for multi-field transforms like the batch collapse.
    """
    opaque: dict[str, Any] = {}
    errors: list[str] = []

    for name, value in source_values.items():
        # Layer 1: REJECT_IF_SET — if non-default, run the rejector callable.
        # If the value IS the default, treat the field as a silent drop
        # (the user did not set it; we have nothing to complain about).
        # If the callable returns ``None`` on a non-default value, the value
        # is benign for this REJECT rule — fall through to the next bucket
        # (e.g. ``optim="lion"`` is non-default + non-paged, so the
        # paged-optim rejector returns None and we want the TRANSFORM /
        # DIRECT pass to handle it).
        if name in reject:
            if _is_default(value, source_defaults.get(name)):
                continue  # default value → silent drop
            message = reject[name](value)
            if message is not None:
                errors.append(f"  - {source_label}.{name}={value!r}: {message}")
                continue
            # message is None → benign; fall through to other buckets.

        # Layer 2: DROP_WITH_WARN — drop and (optionally) warn.
        if name in drop:
            if not _is_default(value, source_defaults.get(name)):
                _warn_drop(source_label, name, value, drop[name], strict)
            continue

        # Layer 3: TRANSFORM — multi-field derivation.
        if name in transform:
            # Transforms run once; the dispatcher invokes each transform
            # callable exactly once, regardless of how many source fields
            # it inspects.
            continue

        # Layer 4: RENAME — name swap, value preserved.
        if name in rename:
            opaque[rename[name]] = value
            continue

        # Layer 5: DIRECT — copy as-is.
        if name in direct:
            opaque[name] = value
            continue

        # Unbucketed field: surface as a hard error so the canary test can
        # catch upstream additions before they bite a user.
        errors.append(
            f"  - {source_label}.{name}={value!r}: field is not classified by "
            f"the opaque compat manifest (neither DIRECT, RENAME, TRANSFORM, "
            f"REJECT, nor DROP). This usually means upstream HF/TRL added a "
            f"new field on a version the opaque manifest hasn't been updated "
            f"for. File a bug or pin a supported version."
        )

    if errors:
        raise ValueError(
            f"Converting {source_label} to opaque failed:\n" + "\n".join(errors)
        )

    # Run all transforms — they see the full source dict.
    for name, transform_fn in transform.items():
        if name in source_values:
            opaque.update(transform_fn(dict(source_values)))

    return opaque


def _warn_drop(
    source_label: str, name: str, value: Any, reason: str, strict: bool
) -> None:
    """Emit a Python warning for a dropped non-default field."""
    msg = (
        f"opaque compat: dropping {source_label}.{name}={value!r} — {reason}. "
        f"This field has no opaque equivalent and is being discarded."
    )
    if strict:
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
    # When ``strict=False``, drop silently.
