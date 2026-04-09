"""Per-group values for per-group clipping and noise.

The ``PerGroup`` type carries pre-resolved parameter-to-group assignments
and per-group values (clipping norms, sensitivities, noise stddevs).  It
flows through the entire DP-SGD pipeline unchanged and supports arithmetic
so that training-loop code like ``noise_multiplier * clip_state.sensitivity``
works without modification.

The ``per_group()`` factory constructs a ``PerGroup`` from a parameter dict
and substring patterns::

    from opaque import per_group
    pg = per_group(params, self_attn=1.0, mlp=2.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _extract_keys(params) -> list[str]:
    """Extract all leaf keys from a parameter dict.

    For flat dicts (``make_functional`` output): returns dict keys directly.
    For nested dicts: returns dotted paths to leaf tensors.
    """
    keys: list[str] = []

    def _recurse(d, prefix):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                _recurse(v, full_key)
            else:
                keys.append(full_key)

    _recurse(params, "")
    return keys


@dataclass(frozen=True)
class PerGroup:
    """Per-group values with pre-resolved parameter-to-group assignment.

    Flows through the entire DP-SGD pipeline as clipping_norm, sensitivity,
    and noise stddev. Supports arithmetic so training loop code is unchanged::

        noise_multiplier * clip_state.sensitivity  # returns PerGroup

    Attributes:
        groups: Mapping from parameter key to group name (pre-resolved).
        values: Mapping from group name to the per-group value.
    """

    groups: dict[str, str]
    values: dict[str, float]

    @property
    def effective(self) -> float:
        """sqrt(sum v**2) -- effective global value for accounting."""
        return math.sqrt(sum(v**2 for v in self.values.values()))

    def __rmul__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: scalar * v for k, v in self.values.items()})

    def __mul__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: v * scalar for k, v in self.values.items()})

    def __truediv__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: v / scalar for k, v in self.values.items()})

    def for_key(self, key: str) -> float:
        """Look up the per-group value for a parameter key."""
        return self.values[self.groups[key]]


def per_group(params, /, patterns=None, **kwargs) -> PerGroup:
    """Construct PerGroup from parameter keys and substring patterns.

    Each pattern is a substring matched against parameter keys.  Params
    whose key contains the substring are assigned to that group.  Every
    parameter must match exactly one pattern (error on 0 or 2+).

    The ``patterns`` dict arg merges with ``**kwargs`` to support keys
    containing dots (which cannot be used as keyword arguments)::

        per_group(params, **{"layers.0": 0.5, "layers.1": 1.0})
        # equivalent to:
        per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})

    The special pattern ``"other"`` acts as a catch-all: any parameter that
    does not match an explicit pattern is assigned to the ``"other"`` group.
    Without ``"other"``, unmatched parameters raise ``ValueError``.

    Args:
        params: Parameter dict (flat or nested).  Keys are matched against
            patterns.
        patterns: Optional dict of ``{pattern: value}`` pairs.  Merged with
            ``**kwargs``.
        **kwargs: Pattern-value pairs where the kwarg name is the substring
            pattern and the value is the per-group value (e.g. clipping norm).

    Returns:
        PerGroup with pre-resolved parameter-to-group assignments.

    Raises:
        ValueError: If no patterns are provided, if a parameter matches zero
            patterns (and no ``"other"`` fallback is given), if a parameter
            matches multiple patterns, or if any value is not positive.

    Examples:
        >>> per_group(params, self_attn=1.0, mlp=2.0)
        >>> per_group(params, self_attn=1.0, other=0.5)  # catch-all fallback
        >>> per_group(params, q_proj=0.5, k_proj=0.5, v_proj=0.5, o_proj=0.8,
        ...           gate_proj=1.0, up_proj=1.0, down_proj=1.0)
        >>> per_group(params, **{f'layers.{i}': norms[i] for i in range(32)})
    """
    all_patterns: dict[str, float] = {}
    if patterns is not None:
        all_patterns.update(patterns)
    all_patterns.update(kwargs)

    if not all_patterns:
        raise ValueError("At least one pattern must be provided.")

    for pat, val in all_patterns.items():
        if val <= 0:
            raise ValueError(
                f"Per-group value must be positive, got {val} for pattern '{pat}'."
            )

    has_other = "other" in all_patterns
    # 'other' is a catch-all, not a substring pattern.
    explicit_patterns = {k: v for k, v in all_patterns.items() if k != "other"}

    param_keys = _extract_keys(params)

    groups: dict[str, str] = {}
    for param_key in param_keys:
        matches = [pat for pat in explicit_patterns if pat in param_key]
        if len(matches) == 0:
            if has_other:
                groups[param_key] = "other"
                continue
            raise ValueError(
                f"Parameter '{param_key}' did not match any pattern. "
                f"Available patterns: {list(all_patterns.keys())}. "
                f"Add 'other=<value>' to catch unmatched parameters."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Parameter '{param_key}' matched multiple patterns: {matches}. "
                f"Each parameter must match exactly one pattern."
            )
        groups[param_key] = matches[0]

    used_groups = set(groups.values())
    values = {
        pat: float(val)
        for pat, val in all_patterns.items()
        if pat in used_groups
    }
    return PerGroup(groups=groups, values=values)


__all__ = ["PerGroup", "per_group"]
