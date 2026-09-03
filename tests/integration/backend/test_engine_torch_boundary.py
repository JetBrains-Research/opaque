"""``opaque-engine`` remains entirely independent of PyTorch."""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tomllib

from tests._support.package_metadata import assert_portable_backend_test_matrix

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ENGINE_ROOT = REPO_ROOT / "packages" / "opaque-engine"
ENGINE_SOURCE = ENGINE_ROOT / "src"
ENGINE_PROJECT = ENGINE_ROOT / "pyproject.toml"


def _imports_torch(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                name.name == "torch" or name.name.startswith("torch.")
                for name in node.names
            ):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "torch" or node.module.startswith("torch."))
        ):
            return True
        elif isinstance(node, ast.Call) and node.args:
            target = node.func
            is_dynamic_import = (
                isinstance(target, ast.Name) and target.id == "__import__"
            ) or (isinstance(target, ast.Attribute) and target.attr == "import_module")
            module = node.args[0]
            if (
                is_dynamic_import
                and isinstance(module, ast.Constant)
                and isinstance(module.value, str)
                and (module.value == "torch" or module.value.startswith("torch."))
            ):
                return True
    return False


def _requirement_name(requirement: str) -> str:
    name = re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def test_engine_source_does_not_import_torch() -> None:
    violations = [
        str(path.relative_to(ENGINE_SOURCE))
        for pattern in ("*.py", "*.pyi")
        for path in ENGINE_SOURCE.rglob(pattern)
        if _imports_torch(path)
    ]

    assert not violations, (
        "opaque-engine must not import Torch; move Torch integration to "
        "opaque-torch:\n" + "\n".join(f"  - {path}" for path in violations)
    )


def test_engine_metadata_does_not_depend_on_torch() -> None:
    project = tomllib.loads(ENGINE_PROJECT.read_text(encoding="utf-8"))["project"]
    requirements = list(project.get("dependencies", ()))
    for extra_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(extra_requirements)

    torch_requirements = [
        requirement
        for requirement in requirements
        if _requirement_name(requirement) == "torch"
    ]
    assert not torch_requirements, (
        "opaque-engine must not depend on Torch; declare it in opaque-torch instead: "
        + ", ".join(torch_requirements)
    )


def test_engine_tests_declare_the_portable_provider_matrix() -> None:
    groups = tomllib.loads(ENGINE_PROJECT.read_text(encoding="utf-8")).get(
        "dependency-groups", {}
    )

    assert_portable_backend_test_matrix(groups.get("test", []))
    assert "transformers" in {
        _requirement_name(requirement) for requirement in groups.get("test", [])
    }


def test_importing_engine_does_not_load_a_framework() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import opaque.api.engine, sys; "
            "leaked = sorted({'torch', 'jax', 'jaxlib', 'mlx'} & sys.modules.keys()); "
            "assert not leaked, leaked",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_engine_does_not_ship_torch_specific_public_surfaces() -> None:
    # Directories left empty by a branch switch ship nothing, so look for
    # modules rather than for the directory itself.
    for candidate in (
        ENGINE_SOURCE / "opaque" / "device",
        ENGINE_SOURCE / "opaque" / "api" / "engine" / "device",
    ):
        assert not sorted(candidate.glob("*.py")), candidate

    functional = (
        ENGINE_SOURCE / "opaque" / "api" / "engine" / "functional" / "__init__.py"
    ).read_text(encoding="utf-8")
    exported = next(
        node
        for node in ast.parse(functional).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    names = ast.literal_eval(exported.value)
    assert "make_functional" not in names


def test_torch_wheel_owns_provider_specific_conveniences() -> None:
    torch_source = REPO_ROOT / "packages" / "opaque-torch" / "src"
    expected = (
        "opaque/api/torch/device/__init__.py",
        "opaque/api/torch/functional/__init__.py",
        "opaque/torch/device/__init__.py",
        "opaque/torch/functional/__init__.py",
    )
    assert all((torch_source / path).is_file() for path in expected)
