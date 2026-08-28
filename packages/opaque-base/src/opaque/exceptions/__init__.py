"""Public semantic exception taxonomy for Opaque.

Import these types when application code needs to handle stable categories of
Opaque failures. They preserve familiar Python error relationships:
configuration, calibration, and privacy-budget errors are ``ValueError``
subclasses, while checkpoint failures are ``RuntimeError`` subclasses.
"""

from __future__ import annotations

from opaque.api.base.exceptions import (
    CalibrationError,
    CheckpointError,
    ConfigurationError,
    InputTypeError,
    OpaqueError,
    OperationError,
    PrivacyBudgetError,
)

__all__ = [
    "CalibrationError",
    "CheckpointError",
    "ConfigurationError",
    "InputTypeError",
    "OpaqueError",
    "OperationError",
    "PrivacyBudgetError",
]
