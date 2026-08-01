"""Phase-0 safety claims stay withdrawn until their fixes land."""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("path", "forbidden"),
    [
        ("docs/user-guide/noise.md", "post-processing on the standard"),
        ("docs/mechanisms/dp-sgd/gaussian.md", "post-processing on the standard"),
        ("docs/user-guide/precision.md", "skipped steps consume zero privacy budget"),
        (
            "packages/opaque-engine/src/opaque/api/engine/precision/_loss_scaler.py",
            "skipped steps consume zero privacy budget",
        ),
        ("docs/user-guide/auditing.md", "every dp mechanism opaque ships"),
        ("docs/user-guide/auditing.md", "regardless of which method you pick"),
        (
            "packages/opaque-alignment/src/opaque/api/alignment/dpo/loss/"
            "_squarechipo.py",
            "first optimal-rate dp-dpo",
        ),
        ("packages/opaque-alignment/README.md", "nan-injection verified"),
        (
            "packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/"
            "_mf_gaussian_noise.py",
            "the one-time build is sub-second",
        ),
        (
            "packages/opaque-dpsgd/src/opaque/api/dpsgd/clipping/_adaptive.py",
            "automatically detects if distributed",
        ),
    ],
)
def test_unsafe_claim_is_absent(path: str, forbidden: str) -> None:
    text = " ".join(
        (REPO_ROOT / path).read_text(encoding="utf-8").lower().split()
    )
    assert forbidden not in text


@pytest.mark.parametrize(
    "path",
    [
        "docs/user-guide/noise.md",
        "docs/mechanisms/dp-sgd/gaussian.md",
        "packages/opaque-dpsgd/src/opaque/api/dpsgd/noise/_gaussian.py",
    ],
)
def test_bounded_gaussian_discloses_missing_accounting(path: str) -> None:
    text = " ".join(
        (REPO_ROOT / path).read_text(encoding="utf-8").lower().split()
    )
    assert "experimental" in text
    assert "does not" in text
    assert "account" in text


@pytest.mark.parametrize(
    "path",
    [
        "packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/_base.py",
        "packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/"
        "amplification/_b_min_sep/__init__.py",
        "packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/"
        "amplification/_balls_in_bins.py",
    ],
)
def test_monte_carlo_pld_discloses_point_estimate(path: str) -> None:
    text = " ".join(
        (REPO_ROOT / path).read_text(encoding="utf-8").lower().split()
    )
    assert "point estimate" in text
    assert "not an upper confidence bound" in text
