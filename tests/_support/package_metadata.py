"""Assertions shared by package-metadata integration tests."""

from __future__ import annotations

import re
from collections.abc import Iterable

_APPLE_SILICON_MARKER = "platform_system=='Darwin'andplatform_machine=='arm64'"


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\s]", spec, maxsplit=1)[0].split("[", 1)[0].lower()


def _marker(spec: str) -> str:
    _, separator, marker = spec.partition(";")
    return marker.replace('"', "'").replace(" ", "") if separator else ""


def assert_portable_backend_test_matrix(dependencies: Iterable[str]) -> None:
    """Require Torch and the platform-gated MLX provider in a test group."""
    provider_specs = {
        _dependency_name(spec): spec
        for spec in dependencies
        if _dependency_name(spec) in {"opaque-torch", "opaque-mlx"}
    }

    assert set(provider_specs) == {"opaque-torch", "opaque-mlx"}, (
        "portable package tests must declare opaque-torch plus platform-gated "
        f"opaque-mlx; found {sorted(provider_specs) or 'no provider dependencies'}."
    )
    assert _marker(provider_specs["opaque-torch"]) == ""
    assert _marker(provider_specs["opaque-mlx"]) == _APPLE_SILICON_MARKER
