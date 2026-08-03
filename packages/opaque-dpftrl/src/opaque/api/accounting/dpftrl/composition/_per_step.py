"""Backward-compatibility shim — PerStep and per_step.

The dpftrl-local PerStep and per_step have been unified into
opaque-accounting core (generic over any DpHorizonProcess).
Import the aliases from here for existing code; prefer
opaque.accounting.per_step in new code.
"""

from opaque.api.accounting.core.composition._per_step import PerStep, per_step

__all__ = ["PerStep", "per_step"]
