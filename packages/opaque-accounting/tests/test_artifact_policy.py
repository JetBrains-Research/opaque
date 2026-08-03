from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "packages" / "opaque-accounting"
PYPROJECT = PACKAGE_DIR / "pyproject.toml"


def _copy_repo_subset(repo_dir: Path) -> Path:
    (repo_dir / "packages").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "Cargo.toml", repo_dir / "Cargo.toml")
    package_dir = repo_dir / "packages" / "opaque-accounting"
    shutil.copytree(
        PACKAGE_DIR,
        package_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return package_dir


def _build_artifacts(
    package_dir: Path, out_dir: Path, target_dir: Path
) -> tuple[Path, Path]:
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(out_dir)],
        cwd=package_dir,
        capture_output=True,
        text=True,
        env={**os.environ, "CARGO_TARGET_DIR": str(target_dir)},
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    return wheels[0], sdists[0]


def _wheel_entries(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        return sorted(wheel.namelist())


def _sdist_entries(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as sdist:
        entries = sorted(
            member.name
            for member in sdist.getmembers()
            if member.isfile() and "/" in member.name
        )
    top_levels = {name.split("/", 1)[0] for name in entries}
    assert len(top_levels) == 1
    prefix = next(iter(top_levels))
    return sorted(name.removeprefix(f"{prefix}/") for name in entries)


def _assert_no_transient_bytecode(entries: list[str]) -> None:
    assert not any("__pycache__" in entry for entry in entries)
    assert not any(entry.endswith((".pyc", ".pyo")) for entry in entries)


def _is_excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch(path, pattern) for pattern in patterns)


def test_maturin_exclude_patterns_cover_transient_bytecode_shapes() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    maturin = data["tool"]["maturin"]
    exclude_patterns = list(maturin["exclude"])
    assert _is_excluded("src/pkg/__pycache__/module.cpython-312.pyc", exclude_patterns)
    assert _is_excluded("src/pkg/temp.pyc", exclude_patterns)
    assert _is_excluded("src/pkg/temp.pyo", exclude_patterns)

    target = data["tool"]["maturin"]["target"]["aarch64-apple-darwin"]
    assert float(target["macos-deployment-target"]) >= 11.0


@pytest.mark.slow
def test_clean_and_dirty_builds_ship_the_same_accounting_files(tmp_path: Path) -> None:
    clean_dir = _copy_repo_subset(tmp_path / "clean")
    dirty_dir = _copy_repo_subset(tmp_path / "dirty")

    dirty_cache = (
        dirty_dir / "src" / "opaque" / "api" / "accounting" / "core" / "__pycache__"
    )
    dirty_cache.mkdir(parents=True, exist_ok=True)
    (dirty_cache / "leak.cpython-312.pyc").write_bytes(b"opaque")
    (dirty_dir / "src" / "opaque" / "accounting" / "temp.pyo").write_bytes(b"opaque")

    clean_wheel, clean_sdist = _build_artifacts(
        clean_dir,
        tmp_path / "clean-dist",
        tmp_path / "cargo-target",
    )
    dirty_wheel, dirty_sdist = _build_artifacts(
        dirty_dir,
        tmp_path / "dirty-dist",
        tmp_path / "cargo-target",
    )

    clean_wheel_entries = _wheel_entries(clean_wheel)
    dirty_wheel_entries = _wheel_entries(dirty_wheel)
    clean_sdist_entries = _sdist_entries(clean_sdist)
    dirty_sdist_entries = _sdist_entries(dirty_sdist)

    _assert_no_transient_bytecode(clean_wheel_entries)
    _assert_no_transient_bytecode(dirty_wheel_entries)
    _assert_no_transient_bytecode(clean_sdist_entries)
    _assert_no_transient_bytecode(dirty_sdist_entries)

    assert dirty_wheel_entries == clean_wheel_entries
    assert dirty_sdist_entries == clean_sdist_entries
