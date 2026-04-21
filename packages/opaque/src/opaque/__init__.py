"""Opaque — Functional Differential Privacy for PyTorch (umbrella facade).

This meta-package provides a small curated top-level API over the individual
``opaque-*`` distributions. Sub-packages (``opaque.core``, ``opaque.dpsgd``,
``opaque.mf``, ``opaque.auditing``, ``opaque.performance``,
``opaque.huggingface``, ``opaque.accounting``) are independently installable;
the umbrella simply re-exports a curated slice and provides a cross-cutting
``patch_all()`` helper.

The ``opaque`` namespace itself remains a PEP 420 namespace that composes with
sub-packages installed elsewhere on ``sys.path``; this is achieved via
``pkgutil.extend_path`` below.

Environment variables:
    OPAQUE_SKIP_COMPAT_PATCHES:
        Controls which subsystems ``patch_all()`` skips. Accepts
        ``"all"``, ``"huggingface"``, ``"performance"``, or a
        comma-separated combination (e.g. ``"huggingface,performance"``).
"""

from __future__ import annotations

import os
import pkgutil
from typing import Iterable

# ---------------------------------------------------------------------------
# PEP 420 composition contract
# ---------------------------------------------------------------------------
# The umbrella is the ONLY distribution that ships ``src/opaque/__init__.py``.
# Every other ``opaque-*`` distribution installs its modules under the
# ``opaque`` namespace WITHOUT an ``__init__.py`` (a PEP 420 namespace
# package). ``pkgutil.extend_path`` below extends ``__path__`` with every
# other directory named ``opaque/`` found on ``sys.path`` so that:
#
#   * ``import opaque.dpsgd`` works whether or not the umbrella is installed;
#   * ``import opaque`` (with the umbrella installed) still exposes modules
#     from sibling sub-packages that do NOT live in this directory.
#
# See PEP 420 (implicit namespace packages) and ``pkgutil.extend_path`` for
# background. A CI step in ``.github/workflows/ci.yml`` enforces that no
# other package accidentally commits ``src/opaque/__init__.py``, which would
# shadow everything else.
__path__ = pkgutil.extend_path(__path__, __name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    from opaque.core import __version__
except ImportError:  # pragma: no cover - core should always be present
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------------------
# Optional sub-package patch hooks
# ---------------------------------------------------------------------------
try:
    from opaque.performance import patch_all as _patch_performance
except ImportError:  # pragma: no cover
    _patch_performance = None

try:
    from opaque.huggingface import patch_all as _patch_huggingface
except ImportError:  # pragma: no cover
    _patch_huggingface = None


_VALID_SKIP_TOKENS = {"all", "huggingface", "performance"}


def _parse_skip(skip: Iterable[str] | str | None) -> set[str]:
    """Normalize ``skip`` argument + ``OPAQUE_SKIP_COMPAT_PATCHES`` env var."""
    tokens: set[str] = set()
    env = os.environ.get("OPAQUE_SKIP_COMPAT_PATCHES", "").strip()
    if env:
        tokens.update(t.strip().lower() for t in env.split(",") if t.strip())
    if skip is not None:
        if isinstance(skip, str):
            tokens.update(t.strip().lower() for t in skip.split(",") if t.strip())
        else:
            tokens.update(str(t).strip().lower() for t in skip if str(t).strip())
    unknown = tokens - _VALID_SKIP_TOKENS
    if unknown:
        raise ValueError(
            f"Unknown OPAQUE_SKIP_COMPAT_PATCHES token(s): {sorted(unknown)}. "
            f"Valid values are: {sorted(_VALID_SKIP_TOKENS)}"
        )
    return tokens


def patch_all(skip: Iterable[str] | str | None = None) -> None:
    """Apply all available compatibility/performance patches.

    Calls ``opaque.performance.patch_all()`` and ``opaque.huggingface.patch_all()``
    when the respective sub-packages are installed. Opt-out is available via the
    ``skip`` argument or the ``OPAQUE_SKIP_COMPAT_PATCHES`` environment variable;
    valid tokens are ``"all"``, ``"huggingface"``, ``"performance"``.
    """
    tokens = _parse_skip(skip)
    if "all" in tokens:
        return
    if "performance" not in tokens and _patch_performance is not None:
        _patch_performance()
    if "huggingface" not in tokens and _patch_huggingface is not None:
        _patch_huggingface()


__all__ = [
    "__version__",
    "patch_all",
]
