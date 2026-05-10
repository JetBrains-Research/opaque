"""Hugging Face Trainer integration for Opaque.

Skeleton; populated by the parallel HF-trainer branch.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("opaque-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
