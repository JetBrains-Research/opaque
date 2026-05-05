"""Opaque HuggingFace Transformers integration.

Houses the DP-SGD-aware :class:`opaque.transformers.trainer.DPTrainer` shim.
HuggingFace compatibility patches and Triton kernels live in
:mod:`opaque.patches.transformers`; import them via
``opaque.patches.apply_model_patches`` / ``apply_runtime_patches``.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("opaque-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
