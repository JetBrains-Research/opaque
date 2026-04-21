#!/usr/bin/env python3
"""CI lint enforcing the Option B namespace layout.

Rules enforced:

1. Only the umbrella distribution (``packages/opaque``) may ship
   ``src/opaque/__init__.py``. Every other sub-package must leave
   ``opaque/`` as a PEP 420 namespace.
2. Forbidden tokens must not appear in the repo (excluding CHANGELOG.md
   and this script itself): ``opaque_accounting`` (as a top-level Python
   import), ``opaque.compat``, ``opaque.sampling.b_min_sep``,
   ``opaque.sampling.truncated_poisson``, ``opaque.clipping.adaptive``,
   ``opaque.clipping.auto``.
3. Legacy Python modules must not be importable at runtime:
   ``opaque_accounting``, ``opaque.compat``.

Exit code 0 on success, 1 on failure.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"

# --- Rule 1: No stray src/opaque/__init__.py outside the umbrella ----------
UMBRELLA = "opaque"
STRAY_INITS: list[Path] = []
for pkg_dir in sorted(PACKAGES_DIR.iterdir()):
    if not pkg_dir.is_dir():
        continue
    init = pkg_dir / "src" / "opaque" / "__init__.py"
    if init.exists() and pkg_dir.name != UMBRELLA:
        STRAY_INITS.append(init)

# --- Rule 2: Forbidden legacy tokens ---------------------------------------
FORBIDDEN = [
    "opaque_accounting",
    r"opaque\.compat",
    r"opaque\.sampling\.b_min_sep",
    r"opaque\.sampling\.truncated_poisson",
    r"opaque\.sampling\.cyclic_poisson",
    r"opaque\.sampling\.balls_in_bins",
    r"opaque\.sampling\.sequential",
    r"opaque\.clipping\.adaptive",
    r"opaque\.clipping\.auto",
]
FORBIDDEN_RE = re.compile("|".join(FORBIDDEN))

SCAN_ROOTS = [PACKAGES_DIR, REPO_ROOT / "examples", REPO_ROOT / "docs"]
# Files where these tokens are allowed (history/rename notes).
ALLOWED_FILES = {
    REPO_ROOT / "CHANGELOG.md",
    Path(__file__).resolve(),
    REPO_ROOT / ".junie" / "plans" / "modularize-opaque-option-b.md",
    # Rust crate is still named `opaque_accounting` (valid Rust identifier);
    # the README documents Rust-side usage.
    PACKAGES_DIR / "opaque-accounting" / "README.md",
    # The PyO3 extension is mounted at ``opaque.accounting.opaque_accounting``
    # (the ``.so`` file name matches the Rust crate); the package's
    # ``__init__`` aliases it as ``_native`` for internal use. The stub
    # alongside it is documentation for that same module.
    PACKAGES_DIR / "opaque-accounting" / "src" / "opaque" / "accounting" / "__init__.py",
    PACKAGES_DIR / "opaque-accounting" / "src" / "opaque" / "accounting" / "opaque_accounting.pyi",
    PACKAGES_DIR / "opaque-accounting" / "pyproject.toml",
}
# Directories that host the Rust/PyO3 internals where the legacy crate name
# is still a Rust identifier (not a Python import path).
ALLOWED_DIR_PREFIXES = [
    PACKAGES_DIR / "opaque-accounting" / "src" / "lib.rs",
    PACKAGES_DIR / "opaque-accounting" / "src" / "numerics",
    PACKAGES_DIR / "opaque-accounting" / "src" / "pld",
    PACKAGES_DIR / "opaque-accounting" / "src" / "matrix_factorization",
    PACKAGES_DIR / "opaque-accounting" / "src" / "python",
    PACKAGES_DIR / "opaque-accounting" / "Cargo.toml",
]

SKIP_DIRS = {"__pycache__", ".egg-info", "target", ".venv", "node_modules"}
SKIP_EXTS = {".so", ".pyc", ".lock", ".pdb"}

violations: list[tuple[Path, int, str]] = []


def _is_allowed(path: Path) -> bool:
    if path in ALLOWED_FILES:
        return True
    for prefix in ALLOWED_DIR_PREFIXES:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            pass
        if path == prefix:
            return True
    return False


def _iter_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.suffix in SKIP_EXTS:
            continue
        yield p


for root in SCAN_ROOTS:
    for path in _iter_files(root):
        if _is_allowed(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            m = FORBIDDEN_RE.search(line)
            if m:
                violations.append((path, lineno, m.group(0)))

# --- Rule 3: Legacy modules must not be importable at runtime -------------
FORBIDDEN_IMPORTS = ["opaque_accounting", "opaque.compat"]
importable: list[str] = []
for name in FORBIDDEN_IMPORTS:
    try:
        importlib.import_module(name)
    except (ModuleNotFoundError, ImportError):
        continue
    importable.append(name)

exit_code = 0
if STRAY_INITS:
    exit_code = 1
    print("ERROR: stray src/opaque/__init__.py in non-umbrella sub-package(s):")
    for p in STRAY_INITS:
        print(f"  {p.relative_to(REPO_ROOT)}")

if violations:
    exit_code = 1
    print("ERROR: forbidden legacy import tokens detected:")
    for path, lineno, token in violations:
        print(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {token}")

if importable:
    exit_code = 1
    print("ERROR: legacy modules are still importable at runtime:")
    for name in importable:
        print(f"  {name}")

if exit_code == 0:
    print("OK: namespace layout, legacy tokens, and runtime imports all clean.")

sys.exit(exit_code)
