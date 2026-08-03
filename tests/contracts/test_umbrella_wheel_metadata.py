from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / ".github/scripts/set_build_versions.sh"
TEST_VERSION = "9.8.7"
SENTINEL_SUFFIX = ">=0.0.0.dev0"

WHEEL_METADATA_TARGETS = (
    (
        "opaque",
        Path(),
        (
            "Requires-Dist: opaque-engine==9.8.7",
            "Requires-Dist: opaque-accounting==9.8.7",
            'Requires-Dist: opaque-auditing==9.8.7; extra == "auditing"',
        ),
    ),
    (
        "opaque-accounting",
        Path("packages/opaque-accounting"),
        ("Requires-Dist: opaque-base==9.8.7",),
    ),
    (
        "opaque-dpsgd",
        Path("packages/opaque-dpsgd"),
        (
            "Requires-Dist: opaque-engine==9.8.7",
            "Requires-Dist: opaque-accounting==9.8.7",
            'Requires-Dist: opaque-optimizers==9.8.7; extra == "optimizers"',
        ),
    ),
    (
        "opaque-transformers",
        Path("packages/opaque-transformers"),
        (
            "Requires-Dist: opaque-engine==9.8.7",
            "Requires-Dist: opaque-patches==9.8.7",
            "Requires-Dist: opaque-dpsgd==9.8.7",
            "Requires-Dist: opaque-dpftrl==9.8.7",
            "Requires-Dist: opaque-accounting==9.8.7",
            "Requires-Dist: opaque-optimizers==9.8.7",
            "Requires-Dist: opaque-alignment==9.8.7",
        ),
    ),
)


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

    for source in sorted((REPO_ROOT / "packages").iterdir()):
        if source.is_dir():
            shutil.copytree(
                source,
                fixture_root / source.relative_to(REPO_ROOT),
            )

    return fixture_root


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _build_env() -> dict[str, str]:
    return {
        **os.environ,
        "SETUPTOOLS_SCM_PRETEND_VERSION": TEST_VERSION,
    }


def _wheel_metadata_text(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        return wheel.read(metadata_name).decode("utf-8")


def _internal_requirements(pyproject_path: Path) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]

    requirements = [
        requirement
        for requirement in project.get("dependencies", [])
        if requirement.startswith("opaque-")
    ]
    for extra_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(
            requirement
            for requirement in extra_requirements
            if requirement.startswith("opaque-")
        )
    return requirements


def _build_wheel(fixture_root: Path, build_dir: Path, dist_name: str) -> Path:
    wheel_dir = fixture_root / "dist" / dist_name
    wheel_dir.mkdir(parents=True, exist_ok=True)
    build = _run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=fixture_root / build_dir,
        env=_build_env(),
    )
    assert build.returncode == 0, build.stderr or build.stdout
    return next(wheel_dir.glob("*.whl"))


def test_build_preflight_rewrites_all_internal_opaque_requirements(
    tmp_path: Path,
) -> None:
    fixture_root = _copy_build_fixture(tmp_path)

    preflight = _run(
        ["bash", ".github/scripts/set_build_versions.sh", TEST_VERSION],
        cwd=fixture_root,
    )
    assert preflight.returncode == 0, preflight.stderr or preflight.stdout

    pyproject_paths = (
        fixture_root / "pyproject.toml",
        *sorted((fixture_root / "packages").glob("*/pyproject.toml")),
    )
    internal_requirements = [
        requirement
        for pyproject_path in pyproject_paths
        for requirement in _internal_requirements(pyproject_path)
    ]

    assert internal_requirements
    assert all(
        requirement.endswith(f"=={TEST_VERSION}")
        for requirement in internal_requirements
    )
    assert all(
        SENTINEL_SUFFIX not in requirement
        for requirement in internal_requirements
    )


@pytest.mark.parametrize(
    ("dist_name", "build_dir", "expected_requirements"),
    WHEEL_METADATA_TARGETS,
)
def test_built_wheel_metadata_pins_internal_opaque_requirements(
    tmp_path: Path,
    dist_name: str,
    build_dir: Path,
    expected_requirements: tuple[str, ...],
) -> None:
    fixture_root = _copy_build_fixture(tmp_path)

    preflight = _run(
        ["bash", ".github/scripts/set_build_versions.sh", TEST_VERSION],
        cwd=fixture_root,
    )
    assert preflight.returncode == 0, preflight.stderr or preflight.stdout

    wheel_path = _build_wheel(fixture_root, build_dir, dist_name)
    metadata = _wheel_metadata_text(wheel_path)

    for requirement in expected_requirements:
        assert requirement in metadata


def test_build_preflight_fails_when_cross_package_sentinel_is_missing(
    tmp_path: Path,
) -> None:
    fixture_root = _copy_build_fixture(tmp_path)
    pyproject_path = fixture_root / "packages/opaque-dpsgd/pyproject.toml"
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
    assert "expected rewriteable sentinel for internal dependency" in (
        preflight.stderr or preflight.stdout
    )
