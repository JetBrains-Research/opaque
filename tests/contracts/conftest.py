"""Repo-level contract tests run against the source tree, not installed wheels.

These tests enforce structural invariants of the package split:

- PEP 420 namespace rule (no ``__init__.py`` at namespace roots).
- Façade-vs-impl discipline (façade modules contain only re-exports).
- Cross-cutting accounting façade does not re-export per-stack factories.
- Each wheel's tests respect the dependency cone declared in
  ``pyproject.toml``.

They are intentionally tolerant of the pre-refactor state: a wheel or
namespace that does not yet exist simply has nothing to check. They start
catching violations as soon as the relevant artefacts land.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"
