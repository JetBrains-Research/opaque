"""Public type alias for step-indexed scalar schedules."""

from __future__ import annotations

from typing import Callable

__all__ = ["Schedule"]


Schedule = Callable[[int], float]
