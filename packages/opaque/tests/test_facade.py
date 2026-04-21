"""Tests for the ``opaque`` umbrella curated facade.

Covers:

- ``opaque.__version__`` and ``opaque.patch_all`` are exposed.
- ``patch_all()`` dispatches to sub-packages when installed.
- ``OPAQUE_SKIP_COMPAT_PATCHES`` (and the ``skip`` argument) correctly gate
  the performance and huggingface sub-systems.
- Invalid skip tokens raise ``ValueError``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import opaque


def test_umbrella_exposes_version_and_patch_all() -> None:
    assert isinstance(opaque.__version__, str)
    assert callable(opaque.patch_all)


def test_patch_all_invalid_token_raises() -> None:
    with pytest.raises(ValueError):
        opaque.patch_all(skip="bogus")


def test_patch_all_skip_all_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # With skip={"all"} no sub-package hook must fire; we assert by swapping
    # the module-level hooks with sentinels that would raise if called.
    calls: list[str] = []
    monkeypatch.setattr(
        opaque,
        "_patch_performance",
        lambda: calls.append("performance"),
        raising=False,
    )
    monkeypatch.setattr(
        opaque,
        "_patch_huggingface",
        lambda: calls.append("huggingface"),
        raising=False,
    )
    opaque.patch_all(skip="all")
    assert calls == []


def test_patch_all_skip_huggingface_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        opaque,
        "_patch_performance",
        lambda: calls.append("performance"),
        raising=False,
    )
    monkeypatch.setattr(
        opaque,
        "_patch_huggingface",
        lambda: calls.append("huggingface"),
        raising=False,
    )
    opaque.patch_all(skip={"huggingface"})
    assert calls == ["performance"]


def test_patch_all_env_var_skip_all_does_not_import_transformers() -> None:
    """With ``OPAQUE_SKIP_COMPAT_PATCHES=all`` importing opaque + patch_all
    must not pull in ``transformers``.
    """
    code = (
        "import sys, opaque; opaque.patch_all(); "
        "assert 'transformers' not in sys.modules, list(sys.modules)"
    )
    env = os.environ.copy()
    env["OPAQUE_SKIP_COMPAT_PATCHES"] = "all"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_version_matches_core() -> None:
    from opaque.core import __version__ as core_version

    assert opaque.__version__ == core_version
