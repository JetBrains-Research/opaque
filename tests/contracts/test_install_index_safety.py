"""Published install commands do not fall back to public PyPI."""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_INDEX = (
    "https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/"
    "jbr-fed-python/simple/"
)


@pytest.mark.parametrize(
    "path",
    ["README.md", "docs/getting-started/installation.md"],
)
def test_pip_uses_artifact_registry_as_primary_index(path: str) -> None:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert "--extra-index-url" not in text
    assert f"--index-url {ARTIFACT_INDEX}" in text
