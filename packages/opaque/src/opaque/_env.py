"""Internal environment variable helpers."""

from __future__ import annotations

import os


def parse_skip_env(name: str) -> set[str]:
    """Parse comma-separated env var values into normalized token set.

    Returns lowercase, whitespace-trimmed, non-empty entries.
    """
    raw = os.environ.get(name, "")
    return {entry.strip().lower() for entry in raw.split(",") if entry.strip()}
