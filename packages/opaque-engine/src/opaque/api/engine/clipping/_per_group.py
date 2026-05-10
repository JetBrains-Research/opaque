"""Build a :class:`opaque.types.PerGroup` from params + substring patterns::

from opaque.dpsgd.clipping import per_group
pg = per_group(params, self_attn=1.0, mlp=2.0)
pg = per_group(params, q_proj=1.0, fallback=0.5)  # catch-all
"""

from __future__ import annotations

from opaque.api.engine.types import PerGroup as _PerGroup


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


def per_group(
    params: dict,
    /,
    patterns: dict[str, float] | None = None,
    *,
    fallback: float | None = None,
    **kwargs: float,
) -> _PerGroup:
    """Construct PerGroup from parameter keys and substring patterns.

    Each pattern is a substring matched against parameter keys.  Params
    whose key contains the substring are assigned to that group.  Every
    parameter must match exactly one pattern (error on 0 or 2+).

    The ``patterns`` dict arg merges with ``**kwargs`` to support keys
    containing dots (which cannot be used as keyword arguments)::

        per_group(params, **{"layers.0": 0.5, "layers.1": 1.0})
        # equivalent to:
        per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})

    The ``fallback`` parameter assigns a value to any parameter that does
    not match an explicit pattern.  Without ``fallback``, unmatched
    parameters raise ``ValueError``.

    Args:
        params: Parameter dict (flat or nested).  Keys are matched against
            patterns.
        patterns: Optional dict of ``{pattern: value}`` pairs.  Merged with
            ``**kwargs``.
        fallback: Optional value for parameters that don't match any
            explicit pattern.  Unmatched params are assigned to a group
            named ``"fallback"``.
        **kwargs: Pattern-value pairs where the kwarg name is the substring
            pattern and the value is the per-group value (e.g. clipping norm).

    Returns:
        :class:`~opaque.types.PerGroup` with pre-resolved
        parameter-to-group assignments.

    See also:
        :mod:`opaque.serialization` — persist or restore a ``PerGroup`` with
        ``state_dict`` / ``from_state_dict``.  Call :func:`per_group` again
        when parameter keys or grouping patterns change.

    Raises:
        ValueError: If no patterns are provided, if a parameter matches zero
            patterns (and no ``fallback`` is given), if a parameter
            matches multiple patterns, or if any value is not positive.

    Examples:
        >>> per_group(params, self_attn=1.0, mlp=2.0)
        >>> per_group(params, self_attn=1.0, fallback=0.5)  # catch-all
        >>> per_group(params, q_proj=0.5, k_proj=0.5, v_proj=0.5, o_proj=0.8,
        ...           gate_proj=1.0, up_proj=1.0, down_proj=1.0)
        >>> per_group(params, **{f'layers.{i}': norms[i] for i in range(32)})
    """
    all_patterns: dict[str, float] = {}
    if patterns is not None:
        all_patterns.update(patterns)
    all_patterns.update(kwargs)

    if not all_patterns and fallback is None:
        raise ValueError("At least one pattern must be provided.")

    for pat, val in all_patterns.items():
        if val <= 0:
            raise ValueError(
                f"Per-group value must be positive, got {val} for pattern '{pat}'."
            )

    if fallback is not None:
        if fallback <= 0:
            raise ValueError(f"Fallback value must be positive, got {fallback}.")

    param_keys = _extract_keys(params)

    groups: dict[str, str] = {}
    for param_key in param_keys:
        matches = [pat for pat in all_patterns if pat in param_key]
        if len(matches) == 0:
            if fallback is not None:
                groups[param_key] = "fallback"
                continue
            raise ValueError(
                f"Parameter '{param_key}' did not match any pattern. "
                f"Available patterns: {list(all_patterns.keys())}. "
                f"Use fallback=<value> to catch unmatched parameters."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Parameter '{param_key}' matched multiple patterns: {matches}. "
                f"Each parameter must match exactly one pattern."
            )
        groups[param_key] = matches[0]

    used_groups = set(groups.values())
    values = {
        pat: float(val) for pat, val in all_patterns.items() if pat in used_groups
    }
    if fallback is not None and "fallback" in used_groups:
        values["fallback"] = float(fallback)
    return _PerGroup(groups=groups, values=values)


__all__ = ["per_group"]
