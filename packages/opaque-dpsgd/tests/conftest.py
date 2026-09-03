"""Pytest hooks for DP-SGD execution tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DISTRIBUTED = str(Path(__file__).resolve().parent / "execution" / "torch" / "ddp")
if _DISTRIBUTED not in sys.path:
    sys.path.insert(0, _DISTRIBUTED)
_previous_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_DISTRIBUTED, *([_previous_pythonpath] if _previous_pythonpath else [])]
)
