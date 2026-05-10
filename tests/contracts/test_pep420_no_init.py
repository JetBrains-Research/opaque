"""PEP 420 invariant: no wheel ships ``__init__.py`` at namespace roots.

Three roots are pure implicit namespaces because multiple wheels contribute
to them:

- ``opaque/``
- ``opaque/api/``
- ``opaque/api/accounting/``

If any wheel ships an ``__init__.py`` at one of these paths, importing two
sibling wheels produces a hard collision instead of a merged namespace.
"""

from __future__ import annotations

import pathlib

from .conftest import PACKAGES_DIR

FORBIDDEN_INIT_PATHS = (
    "src/opaque/__init__.py",
    "src/opaque/api/__init__.py",
    "src/opaque/api/accounting/__init__.py",
)


def test_no_wheel_ships_namespace_init() -> None:
    violations: list[pathlib.Path] = []
    for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
        if not pkg_dir.is_dir():
            continue
        for forbidden in FORBIDDEN_INIT_PATHS:
            candidate = pkg_dir / forbidden
            if candidate.exists():
                violations.append(candidate)

    assert not violations, (
        "PEP 420 violation: the following ``__init__.py`` files would shadow "
        "the namespace and break sibling-wheel imports:\n"
        + "\n".join(f"  - {v.relative_to(PACKAGES_DIR.parent)}" for v in violations)
    )
