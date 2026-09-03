"""Pytest hooks for DP-FTRL execution tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TORCH_EXECUTION = Path(__file__).resolve().parent / "execution" / "torch"
_TEST_HELPERS = (_TORCH_EXECUTION, _TORCH_EXECUTION / "ddp")
for helper_path in _TEST_HELPERS:
    helper = str(helper_path)
    if helper not in sys.path:
        sys.path.insert(0, helper)
_previous_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        *(str(path) for path in _TEST_HELPERS),
        *([_previous_pythonpath] if _previous_pythonpath else []),
    ]
)
