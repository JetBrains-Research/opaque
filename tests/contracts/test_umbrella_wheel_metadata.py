from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / ".github/scripts/set_build_versions.sh"
TEST_VERSION = "9.8.7"


def _copy_build_fixture(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "build-fixture"
    (fixture_root / ".github/scripts").mkdir(parents=True)

    for relative_path in ("pyproject.toml", "README.md", "Cargo.toml"):
        source = REPO_ROOT / relative_path
        destination = fixture_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    shutil.copy2(
        BUILD_SCRIPT,
        fixture_root / BUILD_SCRIPT.relative_to(REPO_ROOT),
    )

    for source in sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")):
        destination = fixture_root / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return fixture_root


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _wheel_metadata_text(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return wheel.read(metadata_name).decode("utf-8")


def test_umbrella_wheel_metadata_pins_subpackages(tmp_path: Path) -> None:
    fixture_root = _copy_build_fixture(tmp_path)

    preflight = _run(
        ["bash", ".github/scripts/set_build_versions.sh", TEST_VERSION],
        cwd=fixture_root,
    )
    assert preflight.returncode == 0, preflight.stderr or preflight.stdout

    build_env = {
        **os.environ,
        "SETUPTOOLS_SCM_PRETEND_VERSION": TEST_VERSION,
    }
    build = _run(
        ["uv", "build", "--wheel", "--out-dir", "dist"],
        cwd=fixture_root,
        env=build_env,
    )
    assert build.returncode == 0, build.stderr or build.stdout

    wheel_path = next((fixture_root / "dist").glob("opaque-*.whl"))
    metadata = _wheel_metadata_text(wheel_path)

    assert f"Requires-Dist: opaque-engine=={TEST_VERSION}" in metadata
    assert f"Requires-Dist: opaque-accounting=={TEST_VERSION}" in metadata
    assert f"Requires-Dist: opaque-dpsgd=={TEST_VERSION}" in metadata
    assert f"Requires-Dist: opaque-optimizers=={TEST_VERSION}" in metadata
    assert f"Requires-Dist: opaque-patches=={TEST_VERSION}" in metadata
    assert (
        f'Requires-Dist: opaque-auditing=={TEST_VERSION}; extra == "auditing"'
        in metadata
    )
    assert (
        f'Requires-Dist: opaque-dpftrl=={TEST_VERSION}; extra == "dpftrl"'
        in metadata
    )
    assert (
        f'Requires-Dist: opaque-alignment=={TEST_VERSION}; extra == "alignment"'
        in metadata
    )
    assert (
        f'Requires-Dist: opaque-transformers=={TEST_VERSION}; extra == "transformers"'
        in metadata
    )
    assert (
        f'Requires-Dist: opaque-patches[transformers]=={TEST_VERSION}; extra == "transformers"'
        in metadata
    )


def test_build_preflight_fails_when_expected_sentinel_is_missing(tmp_path: Path) -> None:
    fixture_root = _copy_build_fixture(tmp_path)
    pyproject_path = fixture_root / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    pyproject_path.write_text(
        pyproject_text.replace(
            '"opaque-engine>=0.0.0.dev0"',
            '"opaque-engine"',
            1,
        ),
        encoding="utf-8",
    )

    preflight = _run(
        ["bash", ".github/scripts/set_build_versions.sh", TEST_VERSION],
        cwd=fixture_root,
    )

    assert preflight.returncode != 0
    assert "expected 'opaque-engine>=0.0.0.dev0' exactly once" in (
        preflight.stderr or preflight.stdout
    )
