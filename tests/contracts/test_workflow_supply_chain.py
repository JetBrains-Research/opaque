"""Publishing and self-hosted runners remain fail-closed."""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _workflow(name: str) -> str:
    return (REPO_ROOT / ".github/workflows" / name).read_text()


def _job(workflow: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    assert match is not None, f"missing workflow job {name!r}"
    return match.group(1)


def test_main_publish_requires_test_jobs() -> None:
    publish = _job(_workflow("ci.yml"), "publish-dev-wheels")
    assert "needs: [merge-wheels, rust-tests, python-tests]" in publish


def test_release_publish_requires_release_tests() -> None:
    workflow = _workflow("release.yml")
    tests = _job(workflow, "release-tests")
    publish = _job(workflow, "publish")
    assert "cargo test --manifest-path" in tests
    assert 'pytest -m "not cuda and not mps"' in tests
    assert "needs: [merge-wheels, release-tests]" in publish


def test_self_hosted_gpu_job_rejects_fork_pull_requests() -> None:
    workflow = _workflow("pr.yml")
    gpu = _job(workflow, "python-tests-gpu")
    hosted = _job(workflow, "python-tests")
    assert workflow.count("runs-on: GPU-runner-001") == 1
    assert "runs-on: GPU-runner-001" in gpu
    assert "github.event.pull_request.head.repo.full_name == github.repository" in gpu
    assert "github.event_name == 'workflow_dispatch'" in gpu
    assert "GPU-runner-001" not in hosted
