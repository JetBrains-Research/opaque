"""CI provider-matrix policy stays aligned with supported platforms."""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = (
    REPO_ROOT / ".github/workflows/pr.yml",
    REPO_ROOT / ".github/workflows/ci.yml",
    REPO_ROOT / ".github/workflows/prepare-release-implementation.yml",
)
REUSABLE_WORKFLOW = REPO_ROOT / ".github/workflows/python-tests.yml"
SETUP_SCRIPT = REPO_ROOT / ".github/scripts/setup_python_test_environment.sh"
ROOT_PROJECT = REPO_ROOT / "pyproject.toml"
_PYTHON_TEST_JOB = re.compile(
    r"^  (?P<name>python-tests-[\w-]+):\n(?P<body>.*?)(?=^  \w[\w-]*:|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _python_test_jobs(workflow: Path) -> list[tuple[str, str]]:
    return [
        (match["name"], match["body"])
        for match in _PYTHON_TEST_JOB.finditer(workflow.read_text(encoding="utf-8"))
        if "python-tests.yml" in match["body"]
    ]


def _input_value(job: str, name: str) -> str | None:
    match = re.search(rf"^      {name}: (?P<value>.+)$", job, re.MULTILINE)
    return None if match is None else match["value"]


def test_every_python_test_caller_declares_the_platform_provider_matrix() -> None:
    for workflow in WORKFLOWS:
        jobs = _python_test_jobs(workflow)
        assert jobs, (
            f"Expected Python-test callers in {workflow.relative_to(REPO_ROOT)}"
        )
        for name, job in jobs:
            providers = _input_value(job, "expected-providers")
            assert providers is not None, f"{workflow.name}:{name} omits providers"
            if providers == "torch,mlx":
                assert "macos-arm64" in name
                assert _input_value(job, "runner") == "macos-latest"
                assert _input_value(job, "runtime-test-paths") == "packages/opaque-mlx"
            else:
                assert providers == "torch"
                assert "opaque-mlx" not in job


def test_reusable_workflow_exports_the_expected_provider_matrix() -> None:
    workflow = REUSABLE_WORKFLOW.read_text(encoding="utf-8")

    assert "expected-providers:" in workflow
    assert "OPAQUE_EXPECTED_PROVIDERS: ${{ inputs.expected-providers }}" in workflow
    assert "setup_python_test_environment.sh" in workflow


def test_setup_installs_test_groups_without_umbrella_extras_and_validates_providers(
    tmp_path: Path,
) -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "sync_args=(--group dev --group test --all-packages)" in setup
    assert "--extra all" not in setup
    assert "Unknown expected provider" in setup
    assert '"torch": ("torch", "opaque.api.torch")' in setup
    assert '"mlx": ("mlx.core", "opaque.api.mlx")' in setup

    uv_log = tmp_path / "uv.log"
    uv_stub = tmp_path / "uv"
    uv_stub.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$UV_LOG"\n',
        encoding="utf-8",
    )
    uv_stub.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "DEPENDENCY_SELECTION": "locked",
            "OPAQUE_EXPECTED_PROVIDERS": "torch,mlx",
            "PATH": f"{tmp_path}{os.pathsep}{environment['PATH']}",
            "UV_LOG": str(uv_log),
        }
    )

    result = subprocess.run(
        ["/bin/bash", str(SETUP_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert uv_log.read_text(encoding="utf-8").splitlines() == [
        "sync --locked --group dev --group test --all-packages",
        "run python - torch mlx",
    ]


def test_shared_dev_tools_do_not_hide_package_oracles() -> None:
    with ROOT_PROJECT.open("rb") as file:
        shared_tools = tomllib.load(file)["dependency-groups"]["dev"]

    shared_names = {
        requirement.split("=", 1)[0].split(">", 1)[0] for requirement in shared_tools
    }
    assert shared_names.isdisjoint(
        {"dp-accounting", "random-allocation", "riskcal", "torchopt"}
    )
