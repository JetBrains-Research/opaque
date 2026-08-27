"""Shared exception types for Opaque public APIs."""

from __future__ import annotations

from typing import Never


class OpaqueError(Exception):
    """Base class for failures reported by Opaque public APIs."""

    @classmethod
    def raise_(
        cls,
        message: str,
        *,
        cause: BaseException | None = None,
        suppress_context: bool = False,
    ) -> Never:
        """Raise this category with a diagnostic message."""
        if suppress_context:
            raise cls(message) from None
        if cause is not None:
            raise cls(message) from cause
        raise cls(message)


class ConfigurationError(OpaqueError, ValueError):
    """Raised when an Opaque API receives invalid or incompatible configuration."""


class InputTypeError(OpaqueError, TypeError):
    """Raised when an Opaque API receives an argument of an invalid type."""


class OperationError(OpaqueError, RuntimeError):
    """Raised when an Opaque operation cannot complete its requested work."""


class CalibrationError(ConfigurationError):
    """Raised when privacy calibration cannot validate or satisfy its search."""


class PrivacyBudgetError(ConfigurationError):
    """Raised when a privacy-budget definition or operation is invalid."""


class CheckpointError(OperationError):
    """Raised when an Opaque checkpoint cannot be saved, restored, or resumed."""


__all__ = [
    "CalibrationError",
    "CheckpointError",
    "ConfigurationError",
    "InputTypeError",
    "OpaqueError",
    "OperationError",
    "PrivacyBudgetError",
]
