"""Pytest hooks for the opaque-engine test tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DDP_PATHS = (
    Path(__file__).resolve().parent / "ddp",
    Path(__file__).resolve().parent / "dpsgd" / "ddp",
    Path(__file__).resolve().parent / "dpftrl",
    Path(__file__).resolve().parent / "dpftrl" / "ddp",
)
for _ddp_path in _DDP_PATHS:
    _ddp = str(_ddp_path)
    if _ddp not in sys.path:
        sys.path.insert(0, _ddp)
_prev_pp = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [*(str(path) for path in _DDP_PATHS), *([_prev_pp] if _prev_pp else [])]
)
