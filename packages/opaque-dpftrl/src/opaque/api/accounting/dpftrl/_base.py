"""Backward-compatibility shim — DpFtrlProcess → DpHorizonProcess.

The dpftrl-local DpFtrlProcess base class has been unified into
opaque-accounting as DpHorizonProcess.  Import the alias from here
for existing code; prefer opaque.accounting.types.DpHorizonProcess
in new code.
"""

from opaque.api.accounting.core._horizon import DpHorizonProcess as DpFtrlProcess

__all__ = ["DpFtrlProcess"]
