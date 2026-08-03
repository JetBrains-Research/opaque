"""Contract test: every package ships LICENSE + NOTICE; all NOTICE paths resolve.

Validates:
1. Every ``packages/<pkg>/`` directory contains LICENSE and NOTICE files.
   setuptools and maturin auto-discover ``LICEN[SC]E*`` and ``NOTICE*`` from the
   project root and include them in wheels, so their presence here is the
   pre-requisite for correct wheel contents.
2. Every source path listed in the root NOTICE resolves to an existing file.
3. Every in-file NOTICE backreference (``../../../../../NOTICE`` style) that
   appears in kernel source files resolves to an existing NOTICE file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"
ROOT_NOTICE = REPO_ROOT / "NOTICE"

PACKAGES = [p.name for p in sorted(PACKAGES_DIR.iterdir()) if p.is_dir()]


# ---------------------------------------------------------------------------
# 1. Every package directory has LICENSE and NOTICE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_has_license(pkg: str) -> None:
    """packages/<pkg>/LICENSE must exist so wheels ship the Apache-2.0 text."""
    assert (PACKAGES_DIR / pkg / "LICENSE").is_file(), (
        f"packages/{pkg}/LICENSE is missing — add it so the wheel ships the "
        "Apache-2.0 license text."
    )


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_has_notice(pkg: str) -> None:
    """packages/<pkg>/NOTICE must exist so wheels ship attribution information."""
    assert (PACKAGES_DIR / pkg / "NOTICE").is_file(), (
        f"packages/{pkg}/NOTICE is missing — add a NOTICE file so the wheel "
        "ships copyright and attribution information."
    )


# ---------------------------------------------------------------------------
# 2. Every source path listed in the root NOTICE resolves
# ---------------------------------------------------------------------------


def _extract_notice_file_paths(notice_path: Path) -> list[str]:
    """Return the indented file paths listed in a NOTICE file."""
    paths: list[str] = []
    for line in notice_path.read_text(encoding="utf-8").splitlines():
        # NOTICE convention: paths are indented with two spaces
        m = re.match(r"^  (packages/.+\.py)$", line)
        if m:
            paths.append(m.group(1))
    return paths


_ROOT_NOTICE_PATHS = _extract_notice_file_paths(ROOT_NOTICE)


@pytest.mark.parametrize("rel_path", _ROOT_NOTICE_PATHS)
def test_root_notice_source_path_exists(rel_path: str) -> None:
    """Every source file listed in the root NOTICE must exist in the repo."""
    full = REPO_ROOT / rel_path
    assert full.is_file(), (
        f"Root NOTICE lists '{rel_path}' but the file does not exist. "
        "Update the NOTICE to match the current source layout."
    )


# ---------------------------------------------------------------------------
# 3. In-file NOTICE backreferences resolve
# ---------------------------------------------------------------------------


def _find_notice_backrefs() -> list[tuple[str, str]]:
    """Scan kernel source files for relative NOTICE paths.

    Returns a list of (relative_source_path, relative_notice_ref) pairs where
    the source path is relative to the repo root.
    """
    results: list[tuple[str, str]] = []
    kernel_dir = PACKAGES_DIR / "opaque-patches" / "src"
    for py_file in kernel_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for m in re.finditer(r"See (\.\.[\./]+NOTICE)", text):
            rel_source = py_file.relative_to(REPO_ROOT).as_posix()
            results.append((rel_source, m.group(1)))
    return results


_NOTICE_BACKREFS = _find_notice_backrefs()


@pytest.mark.parametrize(("src_rel", "notice_rel"), _NOTICE_BACKREFS)
def test_kernel_notice_backref_resolves(src_rel: str, notice_rel: str) -> None:
    """Relative NOTICE paths in kernel source files must resolve to real files."""
    src_file = REPO_ROOT / src_rel
    resolved = (src_file.parent / notice_rel).resolve()
    assert resolved.is_file(), (
        f"{src_rel} references '{notice_rel}' which resolves to "
        f"'{resolved}' — that file does not exist. "
        "Fix the relative path to point to the package-level NOTICE."
    )

