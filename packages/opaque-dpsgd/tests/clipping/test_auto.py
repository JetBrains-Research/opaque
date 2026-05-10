"""Tests that :mod:`opaque.dpsgd.clipping` re-exports match :mod:`opaque._clipping`."""

from __future__ import annotations

import opaque.dpsgd  # noqa: F401  (registers package)


def test_auto_clipped_grad_is_internal_impl():
    from opaque._clipping import auto_clipped_grad as internal
    from opaque.dpsgd.clipping import auto_clipped_grad as public

    assert internal is public


def test_auto_clipped_grad_root_hoist():
    from opaque.dpsgd import auto_clipped_grad as root
    from opaque.dpsgd.clipping import auto_clipped_grad as mod

    assert root is mod


def test_auto_clipped_fun_is_internal_impl():
    from opaque._clipping.fun import auto_clipped_fun as internal
    from opaque.dpsgd.clipping.fun import auto_clipped_fun as public

    assert internal is public


def test_auto_types_match_internal():
    from opaque._clipping.types import (
        AutoClipState as IntState,
        AutoClippedFunAux as IntFunAux,
        AutoClippedGradAux as IntGradAux,
    )
    from opaque.dpsgd.clipping.types import (
        AutoClipState as PubState,
        AutoClippedFunAux as PubFunAux,
        AutoClippedGradAux as PubGradAux,
    )

    assert IntState is PubState
    assert IntFunAux is PubFunAux
    assert IntGradAux is PubGradAux


def test_auto_state_default_marker_equality():
    from opaque.dpsgd.clipping.types import AutoClipState

    assert AutoClipState() == AutoClipState()
