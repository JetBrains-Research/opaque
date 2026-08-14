"""Package metadata preserves the provider ownership graph."""

from __future__ import annotations

import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"

PROVIDERS = frozenset({"opaque-torch", "opaque-jax", "opaque-mlx"})
TORCH_CONSUMERS = frozenset(
    {
        "opaque-alignment",
        "opaque-auditing",
        "opaque-dpftrl",
        "opaque-dpsgd",
        "opaque-patches",
        "opaque-transformers",
    }
)


def _metadata(path: pathlib.Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _project(path: pathlib.Path) -> dict[str, object]:
    return _metadata(path)["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    return {
        re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0].lower()
        for requirement in requirements
    }


def _package_project(distribution: str) -> dict[str, object]:
    return _project(PACKAGES_DIR / distribution / "pyproject.toml")


def _package_metadata(distribution: str) -> dict[str, object]:
    return _metadata(PACKAGES_DIR / distribution / "pyproject.toml")


def test_engine_metadata_and_exports_are_framework_neutral() -> None:
    engine_path = PACKAGES_DIR / "opaque-engine"
    engine = _metadata(engine_path / "pyproject.toml")
    dependencies = _requirement_names(engine["project"]["dependencies"])

    assert not dependencies & {"torch", "jax", "jaxlib", "mlx"}
    assert not dependencies & PROVIDERS
    assert not (engine_path / "src" / "opaque" / "device").exists()
    assert not (engine_path / "src" / "opaque" / "api" / "engine" / "device").exists()

    exported_packages = set(engine["tool"]["setuptools"]["packages"]["find"]["include"])
    assert "opaque.clipping*" not in exported_packages
    assert "opaque.device*" not in exported_packages
    assert "opaque.torch*" not in exported_packages


def test_provider_metadata_declares_only_its_native_framework() -> None:
    expected_frameworks = {
        "opaque-torch": "torch",
        "opaque-jax": "jax",
        "opaque-mlx": "mlx",
    }
    for provider, framework in expected_frameworks.items():
        metadata = _package_metadata(provider)
        project = metadata["project"]
        dependencies = _requirement_names(project["dependencies"])
        sources = metadata["tool"]["uv"]["sources"]

        assert {"opaque-engine", framework} <= dependencies
        assert not (dependencies & PROVIDERS - {provider})
        assert sources == {"opaque-engine": {"workspace": True}}
        assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == [
            f"opaque.api.{provider.removeprefix('opaque-')}*",
            f"opaque.{provider.removeprefix('opaque-')}*",
        ]
        assert metadata["tool"]["setuptools"]["package-data"] == {"*": ["py.typed"]}


def test_torch_consumers_depend_on_the_torch_provider_directly() -> None:
    for consumer in TORCH_CONSUMERS:
        metadata = _package_metadata(consumer)
        project = metadata["project"]
        dependencies = _requirement_names(project["dependencies"])

        assert "opaque-torch" in dependencies
        assert metadata["tool"]["uv"]["sources"]["opaque-torch"] == {"workspace": True}


def test_workspace_and_lockfile_register_every_provider() -> None:
    root = _metadata(REPO_ROOT / "pyproject.toml")
    workspace_members = {
        pathlib.PurePosixPath(member).name
        for member in root["tool"]["uv"]["workspace"]["members"]
    }
    workspace_sources = set(root["tool"]["uv"]["sources"])
    package_names = {
        _project(path)["name"] for path in PACKAGES_DIR.glob("*/pyproject.toml")
    }
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_members = set(lock["manifest"]["members"])

    assert workspace_members >= PROVIDERS
    assert package_names == workspace_sources
    assert package_names | {"opaque"} == locked_members


def test_umbrella_defaults_to_torch_and_exposes_other_providers_as_extras() -> None:
    root = _metadata(REPO_ROOT / "pyproject.toml")
    dependencies = _requirement_names(root["project"]["dependencies"])
    extras = root["project"]["optional-dependencies"]

    assert "opaque-torch" in dependencies
    assert not ({"opaque-jax", "opaque-mlx"} & dependencies)
    assert _requirement_names(extras["jax"]) == {"opaque-jax"}
    assert _requirement_names(extras["mlx"]) == {"opaque-mlx"}
