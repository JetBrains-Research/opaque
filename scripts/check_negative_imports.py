#!/usr/bin/env python3
"""Assert legacy module paths are no longer importable.

This is a CI guardrail: if any of these imports succeeds, the modularization
has regressed.
"""
from __future__ import annotations

import importlib
import sys

FORBIDDEN = ["opaque_accounting", "opaque.compat"]

errors: list[str] = []
for name in FORBIDDEN:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        continue
    except ImportError:
        continue
    errors.append(name)

if errors:
    for e in errors:
        print(f"ERROR: legacy module is still importable: {e}")
    sys.exit(1)

print("OK: legacy modules are not importable.")
