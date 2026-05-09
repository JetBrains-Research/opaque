"""Backward-compat tests for the AUTO-S re-exports under :mod:`opaque.dpsgd.clipping`.

The canonical home for AUTO-S clipping is :mod:`opaque.clipping` (see the
unit tests in ``packages/opaque-core/tests/clipping/test_auto.py``).  This
module exists only to lock the legacy import paths so that downstream code
written against ``opaque.dpsgd.clipping[...]`` keeps working unchanged.
"""

from __future__ import annotations

import opaque.dpsgd  # noqa: F401  (top-level re-export of auto_clipped_grad)


class TestAutoClippingBackwardCompat:
    """Lock the legacy DP-SGD import paths to the new opaque-core canonical home."""

    def test_auto_clipped_grad_is_shared(self):
        from opaque.clipping import auto_clipped_grad as canonical
        from opaque.dpsgd.clipping import auto_clipped_grad as legacy
        from opaque.dpsgd import auto_clipped_grad as legacy_top_level

        assert canonical is legacy
        assert canonical is legacy_top_level

    def test_auto_clipped_fun_is_shared(self):
        from opaque.clipping.fun import auto_clipped_fun as canonical
        from opaque.dpsgd.clipping.fun import auto_clipped_fun as legacy

        assert canonical is legacy

    def test_auto_state_and_aux_are_shared(self):
        from opaque.clipping.types import (
            AutoClipState as CanonState,
            AutoClippedFunAux as CanonFunAux,
            AutoClippedGradAux as CanonGradAux,
        )
        from opaque.dpsgd.clipping.types import (
            AutoClipState as LegacyState,
            AutoClippedFunAux as LegacyFunAux,
            AutoClippedGradAux as LegacyGradAux,
        )

        assert CanonState is LegacyState
        assert CanonFunAux is LegacyFunAux
        assert CanonGradAux is LegacyGradAux

    def test_auto_state_default_marker_equality(self):
        """Two default-constructed AutoClipState instances compare equal."""
        from opaque.dpsgd.clipping.types import AutoClipState

        assert AutoClipState() == AutoClipState()
