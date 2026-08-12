"""``opaque-engine`` remains entirely independent of PyTorch."""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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
