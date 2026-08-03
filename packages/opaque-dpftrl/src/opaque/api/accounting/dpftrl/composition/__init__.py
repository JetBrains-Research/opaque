"""Backward-compatibility shim — dpftrl composition namespace."""

from opaque.api.accounting.core.composition._per_step import (
    PerStep as PerStep,
)
from opaque.api.accounting.core.composition._per_step import (
    per_step as per_step,
)

__all__: list[str] = []
