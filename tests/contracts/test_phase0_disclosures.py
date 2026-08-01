"""Phase-0 randomness and telemetry disclosures remain visible."""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_limitations_disclose_randomness_threat_model() -> None:
    text = " ".join(
        (REPO_ROOT / "docs/limitations.md")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )
    assert "## randomness and the threat model" in text
    assert "not cryptographically secure" in text
    assert "strip the random and noise state" in text


def test_limitations_disclose_unaccounted_telemetry() -> None:
    text = " ".join(
        (REPO_ROOT / "docs/limitations.md")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )
    assert "## telemetry outside the guarantee" in text
    assert "un-noised mean loss" in text
    assert "does not cover exact diagnostics" in text
    assert "`runs/`" in text


@pytest.mark.parametrize(
    "path",
    [
        "packages/opaque-alignment/src/opaque/api/alignment/metric/__init__.py",
        "packages/opaque-alignment/src/opaque/api/alignment/metric/_token.py",
        "packages/opaque-alignment/src/opaque/api/alignment/dpo/metric/__init__.py",
        "packages/opaque-alignment/src/opaque/api/alignment/dpo/metric/_reward.py",
        "packages/opaque-alignment/src/opaque/alignment/metric/__init__.py",
    ],
)
def test_metric_docs_disclose_unnoised_private_values(path: str) -> None:
    text = " ".join(
        (REPO_ROOT / path).read_text(encoding="utf-8").lower().split()
    )
    assert "un-noised" in text
    assert "outside opaque's dp accounting" in text
    assert "not for release" not in text
    assert "private internal state" not in text
