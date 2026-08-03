#!/usr/bin/env python3
"""Validate opaque-accounting artifact packaging policy from built artifacts."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRANSIENT_BYTECODE_PATTERNS = ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")


def _is_transient(path: str) -> bool:
    """Check if a path matches any of the transient bytecode patterns."""
    return any(fnmatch(path, pattern) for pattern in TRANSIENT_BYTECODE_PATTERNS)


def _check_pyproject_config() -> list[str]:
    """Verify pyproject.toml has correct artifact policy settings."""
    pyproject = REPO_ROOT / "packages" / "opaque-accounting" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    errors: list[str] = []
    maturin = data["tool"]["maturin"]

    # Verify exclusion patterns are configured and cover transient bytecode
    exclude_patterns = list(maturin.get("exclude", []))
    test_paths = [
        "src/pkg/__pycache__/module.cpython-312.pyc",
        "src/pkg/temp.pyc",
        "src/pkg/temp.pyo",
    ]
    for test_path in test_paths:
        if not any(fnmatch(test_path, pattern) for pattern in exclude_patterns):
            errors.append(f"exclude patterns don't cover: {test_path}")

    # Verify macOS deployment target
    target = maturin.get("target", {}).get("aarch64-apple-darwin", {})
    deployment_target = float(target.get("macos-deployment-target", 0))
    if deployment_target < 11.0:
        errors.append(f"macOS deployment target {deployment_target} is below 11.0")

    return errors


def _check_cargo_config() -> list[str]:
    """Verify Cargo.toml package.exclude has transient bytecode patterns."""
    cargo = REPO_ROOT / "packages" / "opaque-accounting" / "Cargo.toml"
    data = tomllib.loads(cargo.read_text(encoding="utf-8"))

    errors: list[str] = []
    exclude_patterns = data.get("package", {}).get("exclude", [])

    has_pycache = any("__pycache__" in str(pattern) for pattern in exclude_patterns)
    has_pyc = any("*.pyc" in str(pattern) for pattern in exclude_patterns)
    if not (has_pycache and has_pyc):
        errors.append("Cargo.toml package.exclude missing transient bytecode patterns (__pycache__ or *.pyc)")

    return errors


def _validate_wheel_artifact(wheel_path: Path) -> list[str]:
    """Check wheel artifact doesn't contain transient files."""
    with zipfile.ZipFile(wheel_path) as wheel:
        entries = wheel.namelist()

    errors: list[str] = []
    for entry in entries:
        if _is_transient(entry):
            errors.append(f"wheel {wheel_path.name} contains transient file: {entry}")

    return errors


def _validate_sdist_artifact(sdist_path: Path) -> list[str]:
    """Check sdist artifact doesn't contain transient files."""
    with tarfile.open(sdist_path, "r:gz") as sdist:
        entries = [member.name for member in sdist.getmembers() if member.isfile()]

    errors: list[str] = []
    for entry in entries:
        if _is_transient(entry):
            errors.append(f"sdist {sdist_path.name} contains transient file: {entry}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        help="Directory containing built wheel files.",
    )
    args = parser.parse_args()

    errors: list[str] = []

    # Check configuration files
    errors.extend(_check_pyproject_config())
    errors.extend(_check_cargo_config())

    # Check built artifacts if provided
    if args.wheel_dir:
        for wheel_path in sorted(args.wheel_dir.glob("*.whl")):
            errors.extend(_validate_wheel_artifact(wheel_path))

        for sdist_path in sorted(args.wheel_dir.glob("*.tar.gz")):
            errors.extend(_validate_sdist_artifact(sdist_path))

    if errors:
        print("Accounting artifact policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Accounting artifact policy validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
