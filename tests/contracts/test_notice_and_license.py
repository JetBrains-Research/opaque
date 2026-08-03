"""Contract test: every package ships LICENSE + NOTICE; all NOTICE paths resolve.

Validates:
1. Every ``packages/<pkg>/`` directory contains LICENSE and NOTICE files.
   setuptools and maturin auto-discover ``LICEN[SC]E*`` and ``NOTICE*`` from the
   project root and include them in wheels, so their presence here is the
   pre-requisite for correct wheel contents.
2. Every source path listed in the root NOTICE resolves to an existing file.
3. Every in-file NOTICE backreference (``../../../../../NOTICE`` style) that
   appears in package source files resolves to an existing NOTICE file.
4. The provenance inventory is well-formed and every NOTICE-required
   provenance path is listed in root NOTICE or package NOTICE.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"
ROOT_NOTICE = REPO_ROOT / "NOTICE"
PROVENANCE_TOML = REPO_ROOT / "third_party_provenance.toml"

PACKAGES = [p.name for p in sorted(PACKAGES_DIR.iterdir()) if p.is_dir()]


# ---------------------------------------------------------------------------
# 1. Every package directory has LICENSE and NOTICE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_has_license(pkg: str) -> None:
    """packages/<pkg>/ must contain a LICEN[SC]E* file for wheel auto-discovery."""
    pkg_dir = PACKAGES_DIR / pkg
    matches = list(pkg_dir.glob("LICEN[SC]E*"))
    assert matches, (
        f"packages/{pkg}/ has no LICEN[SC]E* file — add one so the wheel ships "
        "the Apache-2.0 license text."
    )


@pytest.mark.parametrize("pkg", PACKAGES)
def test_package_has_notice(pkg: str) -> None:
    """packages/<pkg>/ must contain a NOTICE* file for wheel auto-discovery."""
    pkg_dir = PACKAGES_DIR / pkg
    matches = list(pkg_dir.glob("NOTICE*"))
    assert matches, (
        f"packages/{pkg}/ has no NOTICE* file — add one so the wheel ships "
        "copyright and attribution information."
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


def _extract_package_notice_file_paths(notice_path: Path, package: str) -> list[str]:
    """Return indented source paths from a package NOTICE as repo-relative paths."""
    paths: list[str] = []
    for line in notice_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^  (opaque/.+\.py)$", line)
        if m:
            paths.append(f"packages/{package}/src/{m.group(1)}")
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
    for src_dir in PACKAGES_DIR.glob("*/src"):
        for py_file in src_dir.rglob("*.py"):
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


def test_provenance_inventory_is_well_formed() -> None:
    """third_party_provenance.toml must parse and point to existing files."""
    data = tomllib.loads(PROVENANCE_TOML.read_text(encoding="utf-8"))
    assert "upstreams" in data
    assert isinstance(data["upstreams"], list)
    for upstream in data["upstreams"]:
        assert isinstance(upstream["name"], str)
        assert upstream["name"]
        assert isinstance(upstream["homepage"], str)
        assert upstream["homepage"]
        assert isinstance(upstream["license"], str)
        assert upstream["license"]
        assert upstream["disposition"] in {"copied", "adapted", "inspired"}
        assert isinstance(upstream["requires_notice"], bool)
        assert isinstance(upstream["files"], list)
        assert upstream["files"]
        for rel in upstream["files"]:
            assert (REPO_ROOT / rel).is_file(), (
                f"third_party_provenance.toml lists '{rel}' but it does not exist."
            )


def test_notice_required_provenance_entries_are_listed_in_notice_files() -> None:
    """Every provenance entry requiring NOTICE must be listed in root or package NOTICE."""
    data = tomllib.loads(PROVENANCE_TOML.read_text(encoding="utf-8"))
    root_paths = set(_extract_notice_file_paths(ROOT_NOTICE))
    package_paths: set[str] = set()
    for pkg in PACKAGES:
        pkg_notice = PACKAGES_DIR / pkg / "NOTICE"
        package_paths.update(_extract_package_notice_file_paths(pkg_notice, pkg))
    listed = root_paths | package_paths

    for upstream in data["upstreams"]:
        if not upstream["requires_notice"]:
            continue
        for rel in upstream["files"]:
            assert rel in listed, (
                f"NOTICE-required provenance file '{rel}' is not listed in root NOTICE "
                "or its package NOTICE."
            )
