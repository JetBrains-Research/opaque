"""No sub-package ships an initializer at a shared namespace root.

The ``opaque``, ``opaque.api``, and ``opaque.api.accounting`` package roots
are PEP 420 implicit namespaces composed by every installed wheel (ARC-001).
A regular ``__init__.py`` at any of those roots in any wheel would shadow
every other wheel's contribution the moment the packages are installed
together.
"""

from __future__ import annotations

import pathlib

PACKAGES_DIR = pathlib.Path(__file__).resolve().parents[3] / "packages"

_NAMESPACE_ROOTS = (
    "src/opaque/__init__.py",
    "src/opaque/api/__init__.py",
    "src/opaque/api/accounting/__init__.py",
)


def test_no_wheel_ships_a_namespace_root_initializer() -> None:
    violations = [
        str((wheel / root).relative_to(PACKAGES_DIR.parent))
        for wheel in sorted(PACKAGES_DIR.iterdir())
        if wheel.is_dir()
        for root in _NAMESPACE_ROOTS
        if (wheel / root).is_file()
    ]
    assert not violations, (
        "PEP 420 namespace roots must not carry __init__.py:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
